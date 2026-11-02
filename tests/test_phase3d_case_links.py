"""CPU-only checks for the descriptive, frozen-export case join."""

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_phase3d_case_links as links


def fixture():
    cases, observations, evidence, knockout = [], [], [], []
    for base in range(10):
        pair = f"case_{base}"
        y = base % 2
        cases.append({"base_sample_id": base, "pair_id": pair, "y": y,
                      "correct_option": "A", "first_object_id": y + 1})
        for condition, intervention, margin in (("low_boundary", "baseline", -1.0),
                ("temporal_boundary", "baseline", 1.0),
                ("low_boundary", "temporal_to_low_L16", base / 4.0 - .5)):
            observations.append({"base_sample_id": base, "pair_id": pair, "y": y,
                "correct_option": "A", "view": "archived", "condition": condition,
                "intervention": intervention, "answer_margin": margin, "input_fingerprint": f"input_{base}"})
        evidence.append({"base_sample_id": base, "y": y, "view": "archived",
            "model_id": f"outer_{base//5}", "S_low": 1.0, "S_temporal": 2.0,
            "delta_S_boundary": 1.0, "delta_M_temporal_to_low_L16": base / 4.0 + .5,
            "delta_S_temporal_to_low_L16": base / 10.0})
        knockout.append({"base_sample_id": base, "pair_id": pair, "window_start": 18,
            "window_end": 18, "layer_count": 1, "query_group": "options_all", "key_group": "whole_event2",
            "correct_option": "A", "first_object_id": y + 1, "low_key_count": 512,
            "temporal_key_count": 512, "low_base_margin": -1.0, "temporal_base_margin": 1.0,
            "low_ko_margin": -1.5,
            "temporal_ko_margin": 1.0-base/4.0, "delta_M_temporal": -base/4.0,
            "delta_M_low": -.5, "compression": base/4.0-.5,
            "temporal_strict_correct_to_incorrect": base > 4, "temporal_non_ab_first_token": False})
    folds = [{"outer_fold": i, "test_base_ids": list(range(i*5, i*5+5)),
              "train_base_ids": list(range((1-i)*5, (1-i)*5+5))} for i in range(2)]
    models = [{"model_id": f"outer_{f['outer_fold']}", "test_base_ids": f["test_base_ids"],
               "train_base_ids": f["train_base_ids"]} for f in folds]
    return cases, observations, evidence, knockout, folds, models


class CaseLinksTest(unittest.TestCase):
    def joined(self, data=None):
        return links.join_cases(*(fixture() if data is None else data), expected_count=10)

    def test_one_row_per_base_exact_join(self):
        rows = self.joined()
        self.assertEqual(len(rows), 10)
        self.assertEqual(len({r["base_sample_id"] for r in rows}), 10)
        self.assertEqual(rows[8]["ko_temporal_damage"], 2)

    def test_probe_permutations_are_not_pooled(self):
        data = fixture()
        extra = {**data[2][0], "view": "option_order_permuted", "delta_S_temporal_to_low_L16": 999}
        data[2].append(extra)
        self.assertEqual(len(self.joined(data)), 10)

    def test_joint_reference_is_not_L18_singleton(self):
        data = fixture()
        data[3].append({**data[3][0], "window_start": 16, "window_end": 19, "layer_count": 4})
        self.assertEqual(len(self.joined(data)), 10)

    def test_duplicates_missing_and_identity_fail_closed(self):
        for field, mutate in (
            ("probe", lambda d: d[2].append(copy.deepcopy(d[2][0]))),
            ("ko", lambda d: d[3].pop()),
            ("pair", lambda d: d[3][0].update(pair_id="other")),
            ("observation", lambda d: d[1].append(copy.deepcopy(d[1][0]))),
            ("support", lambda d: d[3][0].update(temporal_key_count=511)),
            ("route", lambda d: d[3][0].update(query_group="query_all")),
        ):
            with self.subTest(field=field):
                data = fixture()
                mutate(data)
                with self.assertRaises(ValueError):
                    self.joined(data)

    def test_changed_baselines_and_arithmetic_fail_closed(self):
        for mutate in (lambda d: d[3][0].update(low_base_margin=-2),
                       lambda d: d[2][0].update(delta_M_temporal_to_low_L16=20),
                       lambda d: d[3][0].update(temporal_ko_margin=5),
                       lambda d: d[3][0].update(compression=5),
                       lambda d: d[1][2].update(input_fingerprint="different")):
            data = fixture()
            mutate(data)
            with self.assertRaises(ValueError):
                self.joined(data)

    def test_leakage_and_wrong_fold_fail_closed(self):
        for mutate in (lambda d: d[5][0]["train_base_ids"].append(0),
                       lambda d: d[4][1]["test_base_ids"].append(0),
                       lambda d: d[2][0].update(model_id="outer_1")):
            data = fixture()
            mutate(data)
            with self.assertRaises(ValueError):
                self.joined(data)

    def test_ties_are_not_strict_flips(self):
        rows = self.joined()
        self.assertEqual(rows[2]["patch_margin"], 0)
        self.assertTrue(rows[2]["patch_zero_margin_tie"])
        self.assertFalse(rows[2]["patch_strict_rescue"])
        self.assertTrue(rows[4]["ko_zero_margin_tie"])
        self.assertFalse(rows[4]["ko_strict_temporal_flip"])

    def test_nonfinite_probe_score_is_rejected(self):
        data = fixture()
        data[2][0]["delta_S_temporal_to_low_L16"] = float("nan")
        with self.assertRaises(ValueError):
            self.joined(data)

    def test_average_ranks_and_vectorized_bootstrap(self):
        x = np.array([1, 1, 2, 3.0])
        y = np.array([3, 2, 2, 1.0])
        self.assertAlmostEqual(float(links.rank_correlation(x, y)), -5/6)
        result = links.rank_correlation(np.stack([x, x]), np.stack([y, -y]), axis=1)
        np.testing.assert_allclose(result, [-5/6, 5/6])
        self.assertTrue(np.isnan(links.rank_correlation(np.ones(4), y)))

    def test_repeatable_case_bootstrap_and_all_comparisons(self):
        a = links.summarize(self.joined(), resamples=200, seed=42)
        b = links.summarize(self.joined(), resamples=200, seed=42)
        self.assertEqual(a, b)
        self.assertEqual(len(a["associations"]), 3)
        self.assertEqual(sum(r["case_count"] for r in a["transitions"]), 10)
        self.assertFalse(a["probe_refitting"])
        self.assertFalse(a["gpu_inference"])
        self.assertAlmostEqual(a["associations"][0]["spearman_rho"], 1)

    def test_manifest_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "rows.json"
            path.write_text("[]\n")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"files": [{"path": "run/rows.json",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}))
            before = path.read_bytes()
            self.assertEqual(len(links.sealed_inputs(root, manifest, ["rows.json"])), 1)
            self.assertEqual(path.read_bytes(), before)
            path.write_text("[1]\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                links.sealed_inputs(root, manifest, ["rows.json"])


if __name__ == "__main__":
    unittest.main()
