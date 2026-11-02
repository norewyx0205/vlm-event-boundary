"""Synthetic direct-difference tests; no GPU, SURF or scientific result fitting."""

import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import phase3e_order_materials as materials
from scripts import phase3e_order_probes as core
from scripts import run_phase3e_order_probes as runner
from scripts.phase3c_core import atomic_write, read_json
from tests.test_phase3e_rescue50_probes import fixture


def order_fixture():
    plan, original, _ = fixture()
    rng = np.random.default_rng(19)
    endpoints, values = [], []
    for row in original:
        common = rng.normal(size=(4, 5))
        d = rng.normal(scale=.1, size=(4, 5))
        d[:, 0] += (2*row["y_statement_true"]-1)*(3 if row["condition"] == core.CONDITIONS[1] else 2)
        endpoints.extend([{**row, "order_variant": "original", "parent_original_example_id": row["example_id"]},
            {**row, "example_id": row["example_id"]+"__reversed", "order_variant": "order_reversed",
                "parent_original_example_id": row["example_id"], "y_statement_true": 1-row["y_statement_true"]}])
        values.extend([common+d, common])
    return plan, endpoints, np.stack(values)


class DirectDifferenceTest(unittest.TestCase):
    def setUp(self):
        self.plan, self.endpoints, self.features = order_fixture()
        self.rows, self.values, self.left, self.right = core.paired_dataset(self.plan, self.endpoints, self.features)
        self.folds = core.previous.folds_for(self.plan)
        self.selection = core.selections(self.rows, self.plan)
        self.settings = {"C_grid": [0.01], "bootstrap_replicates": 20}

    def test_input_is_D_and_matching_is_same_object_statement(self):
        self.assertEqual(self.values.shape, (800, 4, 5))
        np.testing.assert_array_equal(self.values, self.left-self.right)
        np.testing.assert_array_equal(self.values[::2], -self.values[1::2])
        self.assertTrue(all(self.rows[i]["y_statement_true"] == 1-self.rows[i+1]["y_statement_true"]
                            for i in range(0, 800, 2)))
        for key in ("subject_id", "reference_id", "condition", "option_text"):
            bad = copy.deepcopy(self.endpoints)
            bad[1][key] = "incorrect"
            with self.assertRaises(ValueError):
                core.paired_dataset(self.plan, bad, self.features)
        bad = copy.deepcopy(self.endpoints)
        bad[1]["y_statement_true"] = bad[0]["y_statement_true"]
        with self.assertRaisesRegex(ValueError, "truth"):
            core.paired_dataset(self.plan, bad, self.features)
        with self.assertRaisesRegex(ValueError, "endpoint"):
            core.paired_dataset(self.plan, self.endpoints[:-1], self.features[:-1])

    def test_training_budget_and_source_condition_isolation(self):
        for regime in core.REGIMES:
            selected = set(self.selection["selected_example_ids"][regime])
            expected = 16 if regime == "mixed_full" else 8
            self.assertEqual(Counter(r["base_sample_id"] for r in self.rows if r["example_id"] in selected),
                             {base: expected for base in range(1, 51)})
        arguments = (self.rows, self.values[:, 0], self.left[:, 0], self.right[:, 0], self.folds, self.selection)
        first = core.crossfit(*arguments, "low_only", self.settings)
        left, right = self.left[:, 0].copy(), self.right[:, 0].copy()
        indices = [i for i, r in enumerate(self.rows) if r["condition"] == core.CONDITIONS[1]]
        left[indices] += 300
        second = core.crossfit(self.rows, left-right, left, right, self.folds, self.selection, "low_only", self.settings)
        self.assertEqual(len(first["predictions"]), 800)
        for a, b in zip(first["models"], second["models"]):
            self.assertEqual(a["train_directed_rows"], 320)
            self.assertEqual(a["inner_scores"], b["inner_scores"])
            np.testing.assert_array_equal(a["model"].named_steps["probe"].coef_, b["model"].named_steps["probe"].coef_)
        with self.assertRaisesRegex(ValueError, "actual endpoint difference"):
            core.crossfit(self.rows, self.left[:, 0], self.left[:, 0], self.right[:, 0],
                          self.folds, self.selection, "low_only", self.settings)

    def test_antisymmetry_endpoint_levels_replay_and_tamper(self):
        values, left, right = self.values[:, 0], self.left[:, 0], self.right[:, 0]
        fitted = core.crossfit(self.rows, values, left, right, self.folds, self.selection, "mixed_full", self.settings)
        for record in fitted["models"]:
            self.assertFalse(record["model"].named_steps["probe"].fit_intercept)
            self.assertFalse(record["model"].named_steps["scale"].with_mean)
            np.testing.assert_array_equal(record["model"].decision_function(values),
                                          -record["model"].decision_function(-values))
        measured = core.evaluation(fitted["predictions"], self.plan, self.settings)
        self.assertEqual(measured["independent_bases"], 50)
        self.assertEqual(measured["archived"]["metrics"][0]["statement_rows"], 100)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner.write_job(root, fitted, measured)
            replay = runner.verify_saved(root, self.rows, values, left, right, self.folds,
                self.selection, "mixed_full", self.plan, {**core.SETTINGS, **self.settings})
            self.assertEqual(replay, measured)
            changed = read_json(root/"predictions.json")
            changed[0]["left_endpoint_projection"] += 1
            atomic_write(root/"predictions.json", changed)
            with self.assertRaisesRegex(ValueError, "checksum"):
                runner.verify_saved(root, self.rows, values, left, right, self.folds,
                    self.selection, "mixed_full", self.plan, {**core.SETTINGS, **self.settings})

    def test_old_single_video_store_cannot_train_order_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"manifest.json"
            atomic_write(path, {"schema": "phase3e_rescue50_option_store_v1", "complete": True})
            with self.assertRaisesRegex(ValueError, "order-reversal"):
                core.load_store(path)


class OrderMaterialsTest(unittest.TestCase):
    def test_reversal_keeps_paths_appearance_text_and_distractors(self):
        from tests.test_phase3d import row
        value = row()
        value.update(feature_variant="full", fps=15, duration_sec=18, total_frames=270,
            event_timing={"first_event_start_frame": 30, "first_event_end_frame": 60,
                "second_event_start_frame": 60, "second_event_end_frame": 90,
                "unrelated_event_start_frame": None, "unrelated_event_end_frame": None},
            boundary_timing={"boundary_start_frame": 60, "boundary_end_frame": 60},
            distractors=[{"id": 2, "shape": "triangle", "color": "green", "direction": "right",
                "from": [300, 300], "to": [400, 300], "radius": None, "motion_kind": "unrelated_motion",
                "motion_timing": "near_targets"}])
        for obj in value["target_objects"]:
            obj.update(direction="right", **{"from": [100*obj["id"], 100], "to": [100*obj["id"]+30, 100]})
        original, reversed_ = materials.render_spec(value), materials.render_spec(value, True)
        self.assertEqual(original["distractors"], reversed_["distractors"])
        self.assertNotIn("radius", original["distractors"][0])
        for a, b in zip(original["objects"], reversed_["objects"]):
            self.assertEqual({k: v for k, v in a.items() if k != "motion"}, {k: v for k, v in b.items() if k != "motion"})
            self.assertNotEqual(a["motion"]["start_frame"], b["motion"]["start_frame"])
            self.assertEqual(a["motion"]["from"], b["motion"]["from"])
        new = materials.reversed_row(value, "new.mp4")
        self.assertEqual((new["option_A"], new["option_B"]), (value["option_A"], value["option_B"]))
        self.assertNotEqual(new["correct_option"], value["correct_option"])


if __name__ == "__main__":
    unittest.main()
