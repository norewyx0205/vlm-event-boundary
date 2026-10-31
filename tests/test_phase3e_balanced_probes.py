import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import reanalyze_phase3d_balanced_probes as runner
from scripts.phase3c_core import atomic_write, digest, file_hash, read_json
from tests.test_phase3d_probes import fixture


SETTINGS = {**runner.SETTINGS, "inner_folds": 2, "C_grid": [.001], "bootstrap_replicates": 80}


def sample(count=20):
    rows, values, cases, folds = fixture(count)
    for row in rows:
        row["correct_option"] = "B" if row["view"] == "option_order_permuted" else "A"
        row["input_fingerprint"] = digest([row["base_sample_id"], row["view"], row["condition"]])
    return rows, values, cases, folds


class BalancedProbeTest(unittest.TestCase):
    def test_complete_grid_and_grouped_outer_folds(self):
        rows, values, cases, folds = sample(50)
        self.assertEqual(runner.balanced_grid(rows, cases, folds),
                         {condition: {"A": 50, "B": 50} for condition in runner.CONDITIONS})
        corrupted = copy.deepcopy(rows)
        swapped = next(row for row in corrupted if row["view"] == "option_order_permuted")
        swapped["correct_option"] = "A"
        with self.assertRaisesRegex(ValueError, "permutation"):
            runner.balanced_grid(corrupted, cases, folds)
        corrupted_folds = copy.deepcopy(folds)
        corrupted_folds[0]["train_base_ids"].append(corrupted_folds[0]["test_base_ids"][0])
        with self.assertRaisesRegex(ValueError, "leak"):
            runner.balanced_grid(rows, cases, corrupted_folds)

    def test_both_baseline_views_fit_but_patches_and_outer_test_do_not(self):
        rows, values, cases, folds = sample()
        first = runner.crossfit_balanced(rows, values, cases, folds, SETTINGS)
        for record in first["models"]:
            training = [i for i, row in enumerate(rows) if row["base_sample_id"] in record["train_base_ids"]
                        and row["intervention"] == "baseline"]
            self.assertEqual(len(training), 4 * len(record["train_base_ids"]))
            self.assertEqual(record["baseline_weight_per_row"], .25)
            np.testing.assert_allclose(record["model"].named_steps["scale"].mean_, values[training].mean(0))
        changed = values.copy()
        for i, row in enumerate(rows):
            if row["intervention"] != "baseline":
                changed[i] = 10000
        second = runner.crossfit_balanced(rows, changed, cases, folds, SETTINGS)
        for a, b in zip(first["models"], second["models"]):
            np.testing.assert_array_equal(a["model"].named_steps["probe"].coef_, b["model"].named_steps["probe"].coef_)
        for i, row in enumerate(rows):
            if row["base_sample_id"] in folds[0]["test_base_ids"]:
                changed[i] = -9999
        third = runner.crossfit_balanced(rows, changed, cases, folds, SETTINGS)
        np.testing.assert_array_equal(first["models"][0]["model"].named_steps["probe"].coef_,
                                      third["models"][0]["model"].named_steps["probe"].coef_)
        self.assertEqual(first["models"][0]["chosen_C"], third["models"][0]["chosen_C"])

    def test_delta_is_temporal_minus_low_and_scores_cancel_intercept(self):
        rows, values, cases, folds = sample()
        drows, delta = runner.delta_features(rows, values)
        self.assertEqual(len(drows), 40)
        self.assertTrue(all(row["condition"] == "paired_delta" for row in drows))
        self.assertEqual(set(row["intervention"] for row in drows), {"baseline"})
        self.assertTrue(all(np.isclose(vector[0], .5 * (2 * row["y"] - 1))
                            for row, vector in zip(drows, delta)))
        result = runner.crossfit_balanced(rows, values, cases, folds, SETTINGS)
        pipeline = result["models"][0]["model"]
        scaler, probe = pipeline.named_steps["scale"], pipeline.named_steps["probe"]
        low, high = values[0:1], values[1:2]
        np.testing.assert_allclose(pipeline.decision_function(high) - pipeline.decision_function(low),
                                  ((high - low) / scaler.scale_) @ probe.coef_[0], atol=1e-12)
        evidence = runner.paired_evidence(result["predictions"])
        self.assertEqual(len(evidence), 40)
        self.assertTrue(all(row["delta_signed_evidence_temporal_minus_low"] > 0 for row in evidence))

    def test_always_A_rule_balances_views_not_the_canonical_labels(self):
        rows, values, cases, folds = sample(50)
        cases = copy.deepcopy(cases)
        for case in cases[:4]:
            a, b = case["option_semantics"].values()
            case["option_semantics"] = {"A": b, "B": a}
        for view, expected in (("archived", .92), ("option_order_permuted", .08)):
            actual = np.mean([runner.option_a_heuristic(case, view) == case["y"] for case in cases])
            self.assertAlmostEqual(actual, expected)

    def test_bootstrap_and_model_export_preserve_base_units(self):
        rows, values, cases, folds = sample()
        result = runner.crossfit_balanced(rows, values, cases, folds, SETTINGS)
        metrics = runner.cluster_metrics(result["predictions"], SETTINGS)
        pooled = [row for row in metrics if row["view"] == "both_views"]
        self.assertTrue(all(row["n_bases"] == 20 and row["n_rows"] == 40 for row in pooled))
        self.assertTrue(all(row["accuracy"] == 1 for row in pooled))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "new_fit"
            runner.export_fit(root, result, SETTINGS, paired=True)
            self.assertEqual(len(read_json(root / "held_out_predictions.json")), 120)
            with np.load(root / "outer_0.npz", allow_pickle=False) as saved:
                self.assertIn("coefficient", saved.files)
                self.assertIn("scaler_scale", saved.files)
            with self.assertRaises(FileExistsError):
                runner.export_fit(root, result, SETTINGS, paired=True)

    def test_missing_real_resources_report_needs_input_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "store/manifest.json"
            resource = {"path": "missing.json", "sha256": "0" * 64}
            atomic_write(source, {"schema": runner.DATASET_SCHEMA, "complete": True,
                                  "artifact_type": "real", **{key: resource for key in runner.RESOURCES}})
            before = file_hash(source)
            with patch.object(runner, "load_dataset", side_effect=AssertionError("Must not fit")):
                result = runner.run(source, root / "new", audit_only=True)
            self.assertEqual(result["status"], "needs_input")
            self.assertFalse(result["analysis_performed"])
            self.assertEqual(file_hash(source), before)
            self.assertFalse((root / "new/summary.json").exists())
            for output in (source.parent, source.parent / "nested", root, root / "new"):
                with self.assertRaises(ValueError):
                    runner.isolated_output(source, output)

    def test_cpu_end_to_end_exports_are_isolated_and_hash_bound(self):
        rows, values, cases, folds = sample(50)
        names = runner.PRIMARY + runner.SECONDARY
        features = np.stack([values] * len(names), axis=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = root / "store"
            store.mkdir()
            files = {"case_metadata": "cases.json", "outer_folds": "folds.json", "row_metadata": "rows.json",
                     "gate_summary": "gates.json", "features": "features.npz"}
            for key, payload in (("case_metadata", cases), ("outer_folds", folds), ("row_metadata", rows)):
                atomic_write(store / files[key], payload)
            atomic_write(store / "gates.json", {"synthetic_test_only": True})
            runner.save_npz(store / "features.npz", features=features)
            manifest = {"schema": runner.DATASET_SCHEMA, "artifact_type": "real", "complete": True,
                        **{key: {"path": name, "sha256": file_hash(store / name)} for key, name in files.items()}}
            atomic_write(store / "manifest.json", manifest)
            # Explicit test-only loader substitution; CLI has no mock-data bypass.
            with patch.object(runner, "load_dataset", return_value=(manifest, rows, cases, folds, names, features)), \
                    patch.object(runner, "SETTINGS", SETTINGS), patch.object(runner, "PRIMARY", runner.PRIMARY[:1]), \
                    patch.object(runner, "SECONDARY", []), redirect_stdout(io.StringIO()):
                result = runner.run(store / "manifest.json", root / "out")
            self.assertEqual(result["status"], "complete")
            self.assertTrue(result["source_bytes_unchanged"])
            self.assertIn("delta_source_pair:L18", result["results"])
            self.assertIn("option_lexical", result["results"])
            for item in read_json(root / "out/output_manifest.json")["files"]:
                self.assertEqual(file_hash(root / "out" / item["path"]), item["sha256"])


if __name__ == "__main__":
    unittest.main()
