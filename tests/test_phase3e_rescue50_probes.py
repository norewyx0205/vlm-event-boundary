"""Synthetic grouped-probe tests; no real model, remote service or GPU."""

import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts import phase3e_rescue50_probes as core
from scripts import run_phase3e_rescue50_probes as runner
from scripts.phase3c_core import atomic_write, digest, file_hash, read_json
from scripts.phase3d_core import audit_pairs, base_folds
from scripts.prepare_phase3e_rescue50 import task_manifest
from scripts.run_phase3d_probes import save_npz
from tests.test_phase3d import pairs


def fixture():
    annotations = pairs()
    for pair in annotations.values():
        for row in pair.values():
            row.update(video_path="unchanged.mp4", phase3c_video_sha256="frozen_video")
    cases = audit_pairs(annotations)
    tasks, examples, contrasts = task_manifest(annotations, cases)
    plan = {"schema": "phase3e_rescue50_option_preparation_v1", "cohort": "original_frozen_rescue50",
        "case_metadata": cases, "outer_folds": base_folds(cases), "tasks": tasks,
        "statement_examples": examples, "boundary_pairs": contrasts, "readouts": list(core.READOUTS)}
    plan["fingerprint"] = digest(plan)
    enriched = []
    lookup = {c["pair_id"]: c for c in cases}
    for row in examples:
        option = row["option"] if row["view"] == "archived" else {"A": "B", "B": "A"}[row["option"]]
        semantic = lookup[row["pair_id"]]["option_semantics"][option]
        enriched.append({**row, "scene_seed": row["base_sample_id"], **semantic,
                         "relation_word": semantic["relation"]})
    rng = np.random.default_rng(5)
    values = rng.normal(size=(400, 5))
    values[:, 0] += [3*(2*r["y_statement_true"]-1) for r in examples]
    return plan, enriched, values


class Rescue50ProbeTest(unittest.TestCase):
    def setUp(self):
        self.plan, self.examples, self.values = fixture()
        self.folds = core.folds_for(self.plan)
        self.selection = core.selections(self.examples, self.plan)
        self.settings = {"C_grid": [0.01], "bootstrap_replicates": 20}

    def test_historical_folds_budgets_and_complementary_views(self):
        self.assertEqual(self.selection, core.selections(self.examples, self.plan))
        self.assertEqual(Counter(self.selection["base_phase"].values()), {0: 25, 1: 25})
        for old, fold in zip(self.plan["outer_folds"], self.folds):
            self.assertEqual(old["test_base_ids"], fold["test_scene_seeds"])
            self.assertEqual(len(fold["train_scene_seeds"]), 40)
        for regime in core.REGIMES:
            audit = core.training_audit(self.examples, self.folds, self.selection, regime, self.plan)
            self.assertTrue(all(r["train"]["statement_rows"] == (320 if regime == "mixed_full" else 160)
                                for r in audit["outer_models"]))
        chosen = set(self.selection["selected_example_ids"]["mixed_matched"])
        for base in range(1, 51):
            rows = [r for r in self.examples if r["base_sample_id"] == base and r["example_id"] in chosen]
            self.assertEqual(len(rows), 4)
            self.assertEqual(len({r["view"] for r in rows}), 2)
        broken = copy.deepcopy(self.selection)
        broken["selected_example_ids"]["low_only"] = broken["selected_example_ids"]["mixed_full"]
        with self.assertRaises(ValueError):
            core.training_audit(self.examples, self.folds, broken, "low_only", self.plan)

    def test_other_condition_and_outer_test_cannot_affect_training(self):
        for regime, excluded in (("low_only", core.CONDITIONS[1]), ("temporal_only", core.CONDITIONS[0])):
            first = core.fit_source(self.examples, self.values, self.folds, self.selection, regime, settings=self.settings)
            changed = self.values.copy()
            changed[[i for i, r in enumerate(self.examples) if r["condition"] == excluded]] += 1000
            second = core.fit_source(self.examples, changed, self.folds, self.selection, regime, settings=self.settings)
            self.assertEqual(len(first["predictions"]), 400)
            for left, right in zip(first["models"], second["models"]):
                self.assertEqual(left["inner_scores"], right["inner_scores"])
                np.testing.assert_array_equal(left["model"].named_steps["probe"].coef_, right["model"].named_steps["probe"].coef_)
        first = core.fit_source(self.examples, self.values, self.folds, self.selection, "mixed_full", settings=self.settings)
        changed = self.values.copy()
        changed[[i for i, r in enumerate(self.examples) if r["base_sample_id"] in self.folds[0]["test_scene_seeds"]]] += 500
        second = core.fit_source(self.examples, changed, self.folds, self.selection, "mixed_full", settings=self.settings)
        self.assertEqual(first["models"][0]["inner_scores"], second["models"][0]["inner_scores"])
        np.testing.assert_array_equal(first["models"][0]["model"].named_steps["probe"].coef_,
                                      second["models"][0]["model"].named_steps["probe"].coef_)

    def test_view_specific_levels_and_within_model_auc(self):
        assignment = {b: f["outer_fold"] for f in self.folds for b in f["test_scene_seeds"]}
        predictions = []
        for row in self.examples:
            label = 2*row["y_statement_true"]-1
            score = 100*assignment[row["base_sample_id"]]+label*(3 if row["condition"] == core.CONDITIONS[1] else 2)
            predictions.append({**row, "model_id": f"outer_{assignment[row['base_sample_id']]}",
                "probe_logit": score, "probe_prediction": int(score > 0), "true_statement_evidence": label*score})
        measured = core.evaluation(predictions, self.plan, self.settings)
        primary = measured["archived"]
        self.assertEqual(primary["metrics"][0]["mean_within_fold_roc_auc"], 1.)
        self.assertLess(roc_auc_score([r["y_statement_true"] for r in predictions], [r["probe_logit"] for r in predictions]), 1.)
        paired = primary["paired_summary"]
        self.assertEqual(paired["evidence_low"]["mean"], 2.)
        self.assertEqual(paired["evidence_temporal"]["mean"], 3.)
        self.assertEqual(paired["evidence_temporal_minus_low"]["mean"], 1.)
        self.assertEqual(paired["evidence_temporal_minus_low"]["base_count"], 50)

    def test_numeric_and_lexical_export_replay_and_tamper(self):
        for categorical in (False, True):
            values = core.lexical_features(self.examples, self.plan) if categorical else self.values.astype(np.float32)
            fitted = core.fit_source(self.examples, values, self.folds, self.selection, "low_only", categorical, self.settings)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runner.write_job(root, fitted, core.training_audit(self.examples, self.folds, self.selection,
                    "low_only", self.plan), core.evaluation(fitted["predictions"], self.plan, self.settings))
                result = runner.verify_saved(root, self.examples, values, self.folds, self.plan,
                    self.selection, "low_only", categorical, self.settings)
                if not categorical:
                    self.assertEqual(result["direct_D_projection_replay"]["boundary_contrasts"], 200)
                predictions = read_json(root/"predictions.json")
                predictions[0]["probe_logit"] += 1
                atomic_write(root/"predictions.json", predictions)
                with self.assertRaisesRegex(ValueError, "checksum"):
                    runner.verify_saved(root, self.examples, values, self.folds, self.plan,
                        self.selection, "low_only", categorical, self.settings)

    def test_complete_store_contract_rejects_changed_labels_folds_and_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decisions = {}
            for task in self.plan["tasks"]:
                correct = task["condition"] == core.CONDITIONS[1]
                decisions[task["eval_id"]] = {"correct_option": task["correct_option"], "correct_logit": 2 if correct else 0,
                    "incorrect_logit": 1, "margin": 1 if correct else -1, "is_correct": correct,
                    "prediction": task["correct_option"] if correct else {"A": "B", "B": "A"}[task["correct_option"]]}
            proof = {k: True for k in ("complete", "stage_verified", "original_baseline_exact", "capture_noop_exact", "option_alignment")}
            proof.update(prompt_cells=100, config_sha256="config", new_KO_or_patch=False)
            for name, payload in (("plan", self.plan), ("examples", self.plan["statement_examples"]),
                                  ("decisions", decisions), ("capture_verification", proof)):
                atomic_write(root/(name+".json"), payload)
            save_npz(root/"features.npz", features=np.zeros((400, 4, 4096), dtype=np.float32))

            def seal():
                manifest = {"schema": "phase3e_rescue50_option_store_v1", "artifact_type": "real", "complete": True,
                    "base_count": 50, "prompt_cells": 200, "statement_rows": 400, "shape": [400, 4, 4096],
                    "feature_names": list(core.READOUTS), "preparation_fingerprint": self.plan["fingerprint"],
                    "config_sha256": "config", "files": {name: file_hash(root/name) for name in
                        ("plan.json", "examples.json", "decisions.json", "features.npz", "capture_verification.json")}}
                manifest["fingerprint"] = digest(manifest)
                atomic_write(root/"manifest.json", manifest)
            seal()
            self.assertEqual(core.load_store(root/"manifest.json")[3].shape, (400, 4, 4096))
            examples = read_json(root/"examples.json")
            examples[0]["y_statement_true"] ^= 1
            atomic_write(root/"examples.json", examples)
            seal()
            with self.assertRaisesRegex(ValueError, "Statement rows"):
                core.load_store(root/"manifest.json")

    def test_end_to_end_synthetic_resume_and_verify_without_refit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root/"source/manifest.json"
            atomic_write(manifest, {"synthetic": True})
            decisions = {}
            for task in self.plan["tasks"]:
                correct = task["condition"] == core.CONDITIONS[1]
                decisions[task["eval_id"]] = {"correct_option": task["correct_option"], "correct_logit": int(correct)+1,
                    "incorrect_logit": int(not correct)+1, "margin": 1 if correct else -1, "is_correct": correct}
            features = np.stack([self.values for _ in range(4)], axis=1)
            source = ({"synthetic": True}, self.plan, self.examples, features, decisions)
            with patch.object(core, "load_store", return_value=source):
                result = runner.run(manifest, root/"output", settings=self.settings)
                self.assertEqual(result["new_outer_models"], 100)
                with patch.object(core, "fit_source", side_effect=AssertionError("must not refit")):
                    self.assertEqual(result, runner.run(manifest, root/"output", True, self.settings))
                    self.assertEqual(result, runner.run(manifest, root/"output", settings=self.settings))
                (root/"output/summary.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "Aggregate"):
                    runner.run(manifest, root/"output", True, self.settings)


if __name__ == "__main__":
    unittest.main()
