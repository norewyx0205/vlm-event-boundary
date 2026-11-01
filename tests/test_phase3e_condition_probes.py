import copy
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts import phase3e_condition_probes as core
from scripts import phase3e_option_probes as original_probes
from scripts import run_phase3e_condition_probes as runner
from scripts import run_phase3e_option_probes as original_runner
from scripts.phase3c_core import atomic_write, file_hash, read_json
from scripts.phase3e_option_store import GATES, write_store
from scripts.phase3e_option_features import FEATURE_NAMES
from scripts.phase3e_paired_design import build_plan, group_folds, labels_and_pairs
from scripts.phase3e_preflight import factorial_rows


class ConditionSelectionTest(unittest.TestCase):
    def setUp(self):
        _, self.rows = factorial_rows("/tmp/unused_condition_probes", list(range(10)))
        self.examples = labels_and_pairs(self.rows)[0]
        self.folds = group_folds(list(range(10)))
        self.selection = core.selections(self.examples, self.rows)
        self.settings = {"C_grid": [0.1, 1.], "bootstrap_replicates": 20}

    def test_budget_balance_determinism_and_preserved_order_pairs(self):
        self.assertEqual(self.selection, core.selections(self.examples, self.rows))
        self.assertEqual(Counter(self.selection["family_phase"].values()), {0: 5, 1: 5})
        lookup = {row["eval_id"]: row for row in self.rows}
        for regime, identities in self.selection["selected_example_ids"].items():
            expected = 32 if regime == "mixed_full" else 16
            self.assertEqual(len(identities), 10*expected)
            selected = [row for row in self.examples if row["example_id"] in set(identities)]
            for seed in range(10):
                subset = [row for row in selected if row["scene_seed"] == seed]
                self.assertEqual(Counter(row["y_statement_true"] for row in subset), {0: expected//2, 1: expected//2})
                self.assertEqual(Counter(row["option"] for row in subset), {"A": expected//2, "B": expected//2})
            if regime == "mixed_matched":
                groups = defaultdict(set)
                for example in selected:
                    row = lookup[example["eval_id"]]
                    groups[(row["scene_seed"], row["condition"], row["subject_id"], row["layout"])].add(row["first_object_id"])
                self.assertTrue(all(orders == {1, 2} for orders in groups.values()))
            audit = core.training_audit(self.examples, self.folds, self.selection, regime)
            self.assertEqual(audit["outer_models"][0]["train"]["statement_rows"], 8*expected)
            self.assertEqual(audit["test_rows"], 320)
        changed = copy.deepcopy(self.examples)
        changed[0]["y_statement_true"] ^= 1
        with self.assertRaisesRegex(ValueError, "original"):
            core.selections(changed, self.rows)

    def test_source_only_fits_scalers_hyperparameters_and_family_weights(self):
        values = np.asarray([[2*row["y_statement_true"]-1, row["scene_seed"],
                              int(row["condition"] == "temporal_boundary"), i]
                             for i, row in enumerate(self.examples)], dtype=float)
        calls, validations = [], []
        make_model = core.model

        def recording_model(C, settings):
            pipe = make_model(C, settings)
            fit, predict = pipe.fit, pipe.predict

            def recording_fit(x, y, **kwargs):
                indices = x[:, -1].astype(int).tolist()
                calls.append(indices)
                counts = Counter(self.examples[i]["scene_seed"] for i in indices)
                weights = kwargs["probe__sample_weight"]
                for family in counts:
                    self.assertAlmostEqual(sum(w for i, w in zip(indices, weights)
                                               if self.examples[i]["scene_seed"] == family), 1.)
                return fit(x, y, **kwargs)

            def recording_predict(x):
                validations.extend(x[:, -1].astype(int).tolist())
                return predict(x)

            pipe.fit, pipe.predict = recording_fit, recording_predict
            return pipe

        with patch.object(core, "model", side_effect=recording_model):
            fitted = core.crossfit_condition(self.examples, values, self.folds, self.selection, "low_only", self.settings)
        self.assertEqual(len(calls), 5*(2*4+1))
        self.assertTrue(all(self.examples[i]["condition"] == "low_boundary" for group in calls for i in group))
        self.assertTrue(all(self.examples[i]["condition"] == "low_boundary" for i in validations))
        self.assertEqual(len(fitted["predictions"]), len(self.examples))
        for item, fold in zip(fitted["models"], self.folds):
            indices = [i for i, row in enumerate(self.examples) if row["scene_seed"] in fold["train_scene_seeds"]
                       and row["condition"] == "low_boundary"]
            np.testing.assert_allclose(item["model"].named_steps["scale"].mean_, values[indices].mean(0))
            self.assertFalse(set(item["train_scene_seeds"]) & set(item["test_scene_seeds"]))

    def test_other_condition_cannot_change_source_training(self):
        rng = np.random.default_rng(4)
        values = rng.normal(size=(len(self.examples), 3))
        values[:, 0] += np.asarray([4*row["y_statement_true"] for row in self.examples])
        for regime, excluded in (("low_only", "temporal_boundary"), ("temporal_only", "low_boundary")):
            original = core.crossfit_condition(self.examples, values, self.folds, self.selection, regime, self.settings)
            changed = values.copy()
            changed[[i for i, row in enumerate(self.examples) if row["condition"] == excluded]] += 1000
            repeated = core.crossfit_condition(self.examples, changed, self.folds, self.selection, regime, self.settings)
            for left, right in zip(original["models"], repeated["models"]):
                self.assertEqual(left["chosen_C"], right["chosen_C"])
                self.assertEqual(left["inner_scores"], right["inner_scores"])
                np.testing.assert_array_equal(left["model"].named_steps["probe"].coef_, right["model"].named_steps["probe"].coef_)
                np.testing.assert_array_equal(left["model"].named_steps["scale"].mean_, right["model"].named_steps["scale"].mean_)

    def test_auc_is_within_model_not_pooled_and_levels_are_preserved(self):
        predictions = []
        for row in self.examples:
            model_index = row["scene_seed"]//2
            magnitude = 3 if row["condition"] == "temporal_boundary" else 2
            score = 100*model_index + magnitude*(2*row["y_statement_true"]-1)
            predictions.append({**row, "model_id": f"outer_{model_index}", "probe_logit": score,
                "probe_prediction": int(score > 0), "true_statement_evidence": (2*row["y_statement_true"]-1)*score})
        result = core.evaluation(predictions, self.rows, self.settings)
        self.assertEqual(result["metrics"][0]["mean_within_fold_roc_auc"], 1.)
        self.assertLess(roc_auc_score([row["y_statement_true"] for row in predictions],
                                     [row["probe_logit"] for row in predictions]), 1.)
        evidence = result["boundary_summary"]
        self.assertEqual(evidence["evidence_low"]["mean"], 2.)
        self.assertEqual(evidence["evidence_temporal"]["mean"], 3.)
        self.assertEqual(evidence["evidence_temporal_minus_low"]["mean"], 1.)
        self.assertEqual(evidence["evidence_temporal_minus_low"]["base_count"], 10)

    def test_invalid_folds_values_regimes_are_rejected(self):
        folds = copy.deepcopy(self.folds)
        folds[0]["inner_folds"][0]["train_scene_seeds"].append(folds[0]["test_scene_seeds"][0])
        with self.assertRaisesRegex(ValueError, "leak"):
            core.training_audit(self.examples, folds, self.selection, "low_only")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            core.training_audit(self.examples, self.folds, self.selection, "other")
        selection = copy.deepcopy(self.selection)
        selection["selected_example_ids"]["low_only"][0] = selection["selected_example_ids"]["temporal_only"][0]
        with self.assertRaisesRegex(ValueError, "Source-condition"):
            core.training_audit(self.examples, self.folds, selection, "low_only")
        values = np.full((len(self.examples), 2), np.nan)
        with self.assertRaisesRegex(ValueError, "finite"):
            core.crossfit_condition(self.examples, values, self.folds, self.selection, "low_only")


class ConditionRunnerTest(unittest.TestCase):
    def test_complete_run_reference_reuse_resume_replay_and_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan(root / "materials")
            examples = plan["statement_examples"]
            values = np.zeros((1280, len(FEATURE_NAMES), 3), dtype=np.float32)
            values[:, :, 0] = np.asarray([2*row["y_statement_true"]-1 for row in examples])[:, None]
            values[:, :, 1] = np.asarray([row["subject_id"] for row in examples])[:, None]
            values[:, :, 2] = np.asarray([int(row["condition"] == "temporal_boundary") for row in examples])[:, None]
            decisions = {row["eval_id"]: {"correct_option": row["correct_option"], "prediction": row["correct_option"],
                "is_correct": True, "correct_logit": 2., "incorrect_logit": 1., "margin": 1.} for row in plan["rows"]}
            write_store(root / "store", plan, examples, values, decisions,
                {**{name: True for name in GATES}, "prompt_cells": 640}, {"synthetic_test_fixture": True})
            settings = {"C_grid": [1.], "bootstrap_replicates": 20}
            with patch.dict(original_probes.SETTINGS, settings), patch.object(original_runner, "DEFAULT_NAMES", core.READOUTS):
                original_runner.run(root / "store/manifest.json", root / "reference")
                before = {p.relative_to(root / "reference").as_posix(): file_hash(p)
                          for p in (root / "reference").rglob("*") if p.is_file()}
                result = runner.run(root / "store/manifest.json", root / "reference", root / "followup")
                with patch.object(runner, "crossfit_condition", side_effect=AssertionError("must not refit")):
                    self.assertEqual(result, runner.run(root / "store/manifest.json", root / "reference", root / "followup", True))
                    self.assertEqual(result, runner.run(root / "store/manifest.json", root / "reference", root / "followup"))
                self.assertEqual(result["new_saved_outer_models"], 60)
                self.assertEqual(result["reused_outer_models"], 20)
                self.assertFalse(result["new_GPU_inference"])
                for name, regimes in result["results"].items():
                    for regime, measured in regimes.items():
                        self.assertEqual(measured["metrics"][0]["accuracy"], 1.)
                        counts = {row["train"]["statement_rows"] for row in measured["training_audit"]["outer_models"]}
                        self.assertEqual(counts, {1024 if regime == "mixed_full" else 512})
                    for relative in read_json(root / "reference" / name.replace(":", "__") / "result.json")["files"]:
                        self.assertEqual(file_hash(root / "reference" / name.replace(":", "__") / relative),
                            file_hash(root / "followup" / name.replace(":", "__") / "mixed_full" / relative))
                self.assertEqual(before, {p.relative_to(root / "reference").as_posix(): file_hash(p)
                    for p in (root / "reference").rglob("*") if p.is_file()})
                report = (root / "followup/report.md").read_text()
                self.assertIn("Low evidence", report)
                self.assertIn("Temporal evidence", report)
                self.assertIn("within held-out folds", report)
                for output in (root / "store", root / "store/subdir", root / "reference", root):
                    with self.assertRaisesRegex(ValueError, "separate"):
                        runner.run(root / "store/manifest.json", root / "reference", output)
                altered = read_json(root / "followup/selection_manifest.json")
                altered["selected_example_ids"]["low_only"].pop()
                atomic_write(root / "followup/selection_manifest.json", altered)
                with self.assertRaisesRegex(ValueError, "selections"):
                    runner.run(root / "store/manifest.json", root / "reference", root / "followup", True)
                atomic_write(root / "followup/selection_manifest.json", core.selections(examples, plan["rows"]))
                partial = root / "partial/reference_last__L16_block_output/low_only"
                partial.mkdir(parents=True)
                atomic_write(partial / "predictions.json", {})
                with self.assertRaisesRegex(ValueError, "Partial"):
                    runner.run(root / "store/manifest.json", root / "reference", root / "partial")
                job = root / "followup/reference_last__L18_block_output/low_only"
                with (job / "outer_0.npz").open("ab") as handle:
                    handle.write(b"changed")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    runner.run(root / "store/manifest.json", root / "reference", root / "followup", True)
                reference = root / "reference/reference_last__L16_block_output/outer_0.npz"
                with reference.open("ab") as handle:
                    handle.write(b"changed")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    runner.run(root / "store/manifest.json", root / "reference", root / "new_followup")


if __name__ == "__main__":
    unittest.main()
