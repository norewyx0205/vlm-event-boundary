import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import phase3d_knockout as ko
from scripts import phase3d_probes as probes
from scripts import run_phase3d_probes as runner
from scripts.phase3c_core import atomic_write, digest, file_hash
from scripts.phase3d_core import audit_pairs, base_folds
from scripts.run_phase3d_probes import frozen_numeric_scores, load_dataset, saved_result, write_result
from tests.test_phase3d import pairs


def fixture(count=20):
    cases = audit_pairs(pairs(count), expected_cases=count)
    rows, features = [], []
    for case in cases:
        base, y = case["base_sample_id"], case["y"]
        for view in ("archived", "option_order_permuted"):
            for condition in ("low_boundary", "temporal_boundary"):
                temporal = condition == "temporal_boundary"
                rows.append({"base_sample_id": base, "y": y, "condition": condition, "view": view,
                             "intervention": "baseline", "answer_margin": 1 if temporal else -1})
                features.append([(2*y-1)*(1+.5*temporal), base/100, temporal, np.sin(base)])
        for condition, intervention in (("low_boundary", "temporal_to_low_L16"),
                                        ("temporal_boundary", "low_to_temporal_L16")):
            rows.append({"base_sample_id": base, "y": y, "condition": condition, "view": "archived",
                         "intervention": intervention, "answer_margin": .3 if condition == "low_boundary" else .5})
            features.append([(2*y-1)*1.4, base/100, .5, np.sin(base)])
    return rows, np.asarray(features), cases, base_folds(cases)


SETTINGS = {**probes.SETTINGS, "inner_folds": 2, "C_grid": [.001, .01],
            "bootstrap_replicates": 100, "label_permutations": 2}


class LinearProbeTest(unittest.TestCase):
    def test_crossfit_primary_pairing_and_complete_predictions(self):
        rows, values, cases, folds = fixture()
        probes.validate_rows(rows, cases, folds, require_complete=True)
        result = probes.crossfit(rows, values, cases, folds, SETTINGS)
        self.assertEqual(len(result["predictions"]), len(rows))
        self.assertTrue(all(metric["balanced_accuracy"] > .9 for metric in probes.prediction_metrics(result["predictions"])))
        evidence = probes.case_evidence(result["predictions"])
        self.assertEqual(len(evidence), 40)
        self.assertTrue(all(row["delta_S_boundary"] > 0 for row in evidence))
        self.assertTrue(all(row["delta_S_temporal_to_low_L16"] > 0 for row in evidence if row["view"] == "archived"))

    def test_mirrored_and_patched_rows_never_enter_fitting_or_hyperparameter_selection(self):
        rows, values, cases, folds = fixture()
        first = probes.crossfit(rows, values, cases, folds, SETTINGS)
        changed = values.copy()
        for i, row in enumerate(rows):
            if row["view"] != "archived" or row["intervention"] != "baseline":
                changed[i] = 100000
        second = probes.crossfit(rows, changed, cases, folds, SETTINGS)
        for a, b in zip(first["models"], second["models"]):
            self.assertEqual(a["chosen_C"], b["chosen_C"])
            self.assertTrue(np.array_equal(a["model"].named_steps["scale"].mean_, b["model"].named_steps["scale"].mean_))
            self.assertTrue(np.array_equal(a["model"].named_steps["probe"].coef_, b["model"].named_steps["probe"].coef_))

    def test_outer_test_data_cannot_change_its_frozen_probe(self):
        rows, values, cases, folds = fixture()
        first = probes.crossfit(rows, values, cases, folds, SETTINGS)
        changed = values.copy()
        test = set(folds[0]["test_base_ids"])
        for i, row in enumerate(rows):
            if row["base_sample_id"] in test:
                changed[i] *= 1000
        second = probes.crossfit(rows, changed, cases, folds, SETTINGS)
        a, b = first["models"][0], second["models"][0]
        self.assertEqual(a["chosen_C"], b["chosen_C"])
        self.assertTrue(np.array_equal(a["model"].named_steps["probe"].coef_, b["model"].named_steps["probe"].coef_))

    def test_condition_and_shortcut_controls_are_distinct_from_relation_evidence(self):
        rows, values, cases, folds = fixture()
        result = probes.crossfit(rows, values, cases, folds, SETTINGS, target="condition")
        self.assertTrue(all(row["true_relation_evidence"] is None for row in result["predictions"]))
        self.assertTrue(all(metric["base_count"] == 20 and metric["condition_rows"] == 40
                            for metric in probes.prediction_metrics(result["predictions"])))
        for kind in ("static_attributes", "option_lexical"):
            control = probes.categorical_controls(rows, cases, kind)
            self.assertFalse(any("correct_option" in key or "first_object" in key or key == "y"
                                 for value in control for key in value))
            actual = probes.crossfit(rows, control, cases, folds, SETTINGS, categorical=True)
            self.assertEqual(len(actual["predictions"]), len(rows))

    def test_group_leakage_and_duplicate_rows_are_rejected(self):
        rows, values, cases, folds = fixture()
        changed = copy.deepcopy(folds)
        changed[0]["train_base_ids"].append(changed[0]["test_base_ids"][0])
        with self.assertRaises(ValueError):
            probes.validate_rows(rows, cases, changed)
        with self.assertRaises(ValueError):
            probes.validate_rows(rows+[rows[0]], cases, folds)

    def test_export_is_checksummed_resumable_and_contains_no_pickle(self):
        rows, values, cases, folds = fixture()
        values = values.astype(np.float32)
        result = probes.crossfit(rows, values, cases, folds, SETTINGS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            written = write_result(root, result, "fixed", SETTINGS)
            self.assertEqual(saved_result(root, "fixed"), written)
            with np.load(root/"outer_0.npz", allow_pickle=False) as payload:
                self.assertIn("scaler_mean", payload.files)
                self.assertIn("coefficient", payload.files)
            model = result["models"][0]["model"]
            self.assertTrue(np.allclose(frozen_numeric_scores(root/"outer_0.npz", values),
                                        model.decision_function(values.astype(np.float64)), rtol=1e-12, atol=1e-12))
            atomic_write(root/"models.json", {"tampered": True})
            with self.assertRaises(ValueError):
                saved_result(root, "fixed")

    def test_mock_feature_manifest_cannot_produce_formal_probe_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"features.json"
            atomic_write(path, {"schema": "phase3d_feature_dataset_v1", "artifact_type": "mock", "complete": True})
            with self.assertRaises(ValueError):
                load_dataset(path)
            gate = path.with_name("gates.json")
            atomic_write(gate, {"schema": "phase3d_capture_gate_summary_v1", "complete": True,
                                "full_logit_parity": False})
            atomic_write(path, {"schema": "phase3d_feature_dataset_v1", "artifact_type": "real", "complete": True,
                                "gate_summary": {"path": gate.name, "sha256": file_hash(gate)}})
            with self.assertRaisesRegex(ValueError, "capture gates"):
                load_dataset(path)

    def test_complete_store_contract_and_changed_prompt_fingerprints(self):
        rows, values, cases, folds = fixture(50)
        for value in rows:
            value["correct_option"] = "A" if value["view"] == "archived" else "B"
            value["input_fingerprint"] = digest([value["base_sample_id"], value["condition"], value["view"]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Contract-only temporary synthetic fixture; no scientific analysis is generated.
            atomic_write(root/"cases.json", cases)
            atomic_write(root/"folds.json", folds)
            atomic_write(root/"rows.json", rows)
            labels = {case["base_sample_id"]: case["y"] for case in cases}
            gates = {"schema": "phase3d_capture_gate_summary_v1", "complete": True,
                     "base_count": 50, "case_labels_sha256": digest(sorted(labels.items())),
                     **{name: True for name in runner.GATES}}
            atomic_write(root/"gates.json", gates)
            runner.save_npz(root/"arrays.npz", features=values[:, None, :])
            def reference(name):
                return {"path": name, "sha256": file_hash(root/name)}
            manifest = {"schema": runner.DATASET_SCHEMA, "artifact_type": "real", "complete": True,
                        "case_metadata": reference("cases.json"), "outer_folds": reference("folds.json"),
                        "row_metadata": reference("rows.json"), "gate_summary": reference("gates.json"),
                        "features": reference("arrays.npz"), "feature_names": ["option_pair:L20_output"]}
            atomic_write(root/"manifest.json", manifest)
            self.assertEqual(load_dataset(root/"manifest.json")[-1].shape, (300, 1, 4))
            control = next(value for value in rows if value["view"] == "option_order_permuted")
            original = next(value for value in rows if value["base_sample_id"] == control["base_sample_id"] and
                            value["condition"] == control["condition"] and value["view"] == "archived")
            control["input_fingerprint"] = original["input_fingerprint"]
            atomic_write(root/"rows.json", rows)
            manifest["row_metadata"] = reference("rows.json")
            atomic_write(root/"manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "old prompt"):
                load_dataset(root/"manifest.json")

    def test_output_lock_prevents_duplicate_cpu_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with runner.probe_lock(root):
                with self.assertRaises(RuntimeError):
                    with runner.probe_lock(root):
                        self.fail("Duplicate owner entered the probe root")

    def test_cpu_training_cli_checkpoints_resume_without_refitting(self):
        rows, values, cases, folds = fixture(50)
        names = ["option_pair:L20_output", "source_pair:L16_output"]
        features = np.stack([values, values], axis=1)
        settings = {**SETTINGS, "C_grid": [.001], "label_permutations": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root/"store/manifest.json"
            atomic_write(manifest, {"artifact_type": "unit_test_fixture_not_scientific_evidence"})
            args = SimpleNamespace(manifest_path=manifest, output_dir=root/"probes",
                                   feature_names=["option_pair:L20_output"])
            # Only the in-memory unit fixture bypasses the real-store gate; no public mock flag exists.
            with patch.object(runner, "load_dataset", return_value=({}, rows, cases, folds, names, features)), \
                    patch.object(runner, "SETTINGS", settings), redirect_stdout(io.StringIO()):
                result = runner.run(args)
                self.assertEqual(len(result["results"]), 6)
                self.assertEqual(result["base_count"], 50)
                with patch.object(runner, "crossfit", side_effect=AssertionError("Completed fits must be reused")):
                    self.assertEqual(result, runner.run(args))


class Knockout50Test(unittest.TestCase):
    def test_full_cohort_equal_weight_and_all_routes(self):
        summary = {"complete": True, "artifact_type": "real", "case_count": 25, "paired_rows": 2925}
        intent = ko.intent(pairs(), summary)
        tasks = ko.planned_tasks(intent)
        self.assertEqual(intent["paired_task_count"], 5850)
        self.assertEqual(intent["condition_forward_count"], 11700)
        self.assertEqual(intent["validated_reused_tasks"], 0)
        self.assertEqual(len({task["task_id"] for task in tasks}), 5850)
        self.assertEqual(len([task for task in tasks if task["key_group"] == "whole_event2"]), 450)
        self.assertEqual({sum(task["pair_id"] == key for task in tasks) for key in intent["case_ids"]}, {117})
        self.assertNotIn("cohort_groups", intent)
        changed = copy.deepcopy(intent)
        changed["case_ids"] = changed["case_ids"][:-1]
        with self.assertRaises(ValueError):
            ko.planned_tasks(changed)
        with self.assertRaises(ValueError):
            ko.intent(pairs(), {**summary, "complete": False})


if __name__ == "__main__":
    unittest.main()
