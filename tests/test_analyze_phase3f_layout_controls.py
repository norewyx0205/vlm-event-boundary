"""Synthetic tests for the verified Phase 3F re-analysis; no GPU, VM or source runs."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import analyze_phase3f_layout_controls as verified
from scripts import phase3f_layout_controls as core

IDS = [f"base_{i:03d}" for i in range(50)]


def decision(margin, correct, prediction=None):
    if prediction is None:
        prediction = correct if margin > 0 else ("B" if correct == "A" else "A")
    return {"margin": margin, "prediction": prediction, "correct_option": correct, "is_correct": prediction == correct}


class TransitionTests(unittest.TestCase):
    def test_zero_margin_tie_is_a_flip_but_not_a_strict_crossing(self):
        move = verified.transition(decision(-1.0, "A"), decision(0.0, "A", prediction="A"))
        self.assertTrue(move["answer_changed"] and move["incorrect_to_correct"] and move["zero_margin_after"])
        self.assertFalse(move["strict_sign_crossing"])

    def test_non_ab_output_is_not_a_strict_crossing(self):
        move = verified.transition(decision(1.0, "A"), decision(-1.0, "A", prediction="other"))
        self.assertTrue(move["correct_to_incorrect"] and move["non_ab_after"])
        self.assertFalse(move["strict_sign_crossing"])


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"fingerprint": "f", "case_ids": IDS}
        letter = {core.ARCHIVED: "A", core.PERMUTED: "B"}
        bias = {core.ARCHIVED: -2.0, core.PERMUTED: 2.0}
        base = {core.LOW: 0.5, core.TEMPORAL: 2.0}
        self.margin = lambda view, condition, delta=0.0: base[condition] + bias[view] + delta
        self.records = {p: {v: {c: {"baseline": {"decision": decision(self.margin(v, c), letter[v])}}
                                for c in core.CONDITIONS} for v in letter} for p in IDS}
        effect = {"patch": 2.0, "ko_event2": -1.0, "ko_control": -0.2, "patch_ko": 0.0}
        self.rows = []
        for spec in core.specs(self.plan, "gate") + core.specs(self.plan, "intervention"):
            key = "ko_" + ("event2" if spec["route"] == "event2" else "control") if spec["kind"] == "ko" else spec["kind"]
            delta = 0.0 if spec["kind"] == "gate" else effect[key]
            self.rows.append({"spec": spec, "decision": decision(self.margin(spec["view"], spec["condition"], delta),
                                                                 letter[spec["view"]])})
        self.singles = {p: {c: {w: decision(self.margin(core.ARCHIVED, c, -1.0), "A") for w in ("16", "17", "18", "19")}
                            for c in core.CONDITIONS} for p in IDS}
        self.joint = {p: {c: decision(self.margin(core.ARCHIVED, c, -1.0), "A") for c in core.CONDITIONS} for p in IDS}
        self.patch = {p: {c: decision(self.margin(core.ARCHIVED, c, 2.0), "A") for c in core.CONDITIONS} for p in IDS}
        self.frozen = {"event2_ko": {p: {c: {**{w: {"margin": d["margin"], "prediction": d["prediction"]}
                                                for w, d in self.singles[p][c].items()},
                                             "16-19": self.joint[p][c]} for c in core.CONDITIONS} for p in IDS},
                       "patch": {p: {c: dict(self.patch[p][c]) for c in core.CONDITIONS} for p in IDS}}

    def run_analysis(self, frozen=None):
        tmp = tempfile.mkdtemp()
        run, out = Path(tmp, "run"), Path(tmp, "out")
        run.mkdir()
        (run / "source_records.json").write_text(json.dumps(self.records))
        (run / "references.json").write_text(json.dumps(frozen or self.frozen))
        with mock.patch.object(core, "load_context", return_value=(self.plan, {}, {})), \
                mock.patch.object(core, "verify_rows", return_value=self.rows), \
                mock.patch.object(verified, "verified_singleton", return_value=(self.singles, 500)), \
                mock.patch.object(verified, "verified_phase3d", return_value=(self.joint, self.patch, {"reuse_rebuilt": 100})):
            verified.analyze(run, out)
        return json.loads((out / "results.json").read_text()), (out / "report.md").read_text()

    def test_endpoints_flips_and_symmetry(self):
        result, text = self.run_analysis()
        low = result["patch"][core.LOW]
        self.assertAlmostEqual(low["archived"]["before"]["mean"], -1.5)
        self.assertAlmostEqual(low["archived"]["after"]["mean"], 0.5)
        self.assertEqual((low["archived"]["correct_before"], low["archived"]["correct_after"]), (0, 50))
        self.assertEqual(low["archived"]["incorrect_to_correct"], 50)
        self.assertAlmostEqual(low["symmetric"]["before"]["mean"], 0.5)
        self.assertAlmostEqual(low["symmetric"]["change"]["mean"], 2.0)
        ko = result["event2_ko"]["16-19"]
        self.assertAlmostEqual(ko[core.TEMPORAL]["symmetric"]["change"]["mean"], -1.0)
        self.assertAlmostEqual(ko["compression"]["symmetric"]["mean"], 0.0)
        cross = result["patch_x_ko"]["17-23"]
        self.assertAlmostEqual(cross["patch_under_ko"]["symmetric"]["change"]["mean"], 1.0)
        self.assertIn("→", text)
        self.assertIn("strict", text)

    def test_frozen_reference_mismatch_is_fatal(self):
        frozen = json.loads(json.dumps(self.frozen))
        frozen["event2_ko"][IDS[0]][core.LOW]["17"]["margin"] += 0.5
        with self.assertRaises(ValueError):
            self.run_analysis(frozen)

    def test_refuses_output_inside_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                verified.analyze(tmp, Path(tmp, "analysis_v2"))


if __name__ == "__main__":
    unittest.main()
