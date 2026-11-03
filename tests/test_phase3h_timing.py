"""CPU tests for Phase 3H timing materials, processor-alignment gate, lifecycle and analysis; no GPU or VM."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts import analyze_phase3h_timing as analysis
from scripts import phase3e_order_materials as order_materials
from scripts import phase3h_timing_materials as materials
from scripts import run_phase3h_timing_behaviour as behaviour
from scripts import run_phase3h_timing_unattended as lifecycle
from scripts.phase3c_core import digest

ANNOTATIONS = Path(__file__).resolve().parents[1]/"data/l5_feature_ablation_v1/L5_full/annotations.jsonl"


def source_rows():
    rows = [json.loads(line) for line in ANNOTATIONS.read_text().splitlines() if line.strip()]
    base = rows[0]["base_sample_id"]
    return {r["condition"]: r for r in rows if r["base_sample_id"] == base and r["prompt_variant"] == "original"
            and r["condition"] in materials.CONDITIONS}


class ScheduleTests(unittest.TestCase):
    def test_cells_follow_the_frozen_design(self):
        rows = source_rows()
        expected = {"low": (30, 60, 60, 90, 0), "low_mid": (60, 90, 90, 120, 0), "low_late": (75, 105, 105, 135, 0),
                    "temporal_early": (15, 45, 90, 120, 45), "temporal": (30, 60, 105, 135, 45)}
        for cell, frames in expected.items():
            event, boundary = materials.schedule(rows[materials.CELLS[cell][0]], cell)
            self.assertEqual((event["first_event_start_frame"], event["first_event_end_frame"],
                              event["second_event_start_frame"], event["second_event_end_frame"], boundary["gap_frames"]), frames)
        self.assertEqual({materials.native_check(r) for r in rows.values()}, {"low", "temporal"})

    def test_cell_must_use_its_own_source_condition(self):
        rows = source_rows()
        with self.assertRaises(ValueError):
            materials.schedule(rows["low_boundary"], "temporal_early")
        with self.assertRaises(ValueError):
            materials.native_check({**rows["low_boundary"], "boundary_timing": {**rows["low_boundary"]["boundary_timing"], "gap_frames": 15}})

    def test_native_spec_equals_phase3e_reconstruction_spec(self):
        for row in source_rows().values():
            self.assertEqual(digest(materials.render_spec(row, materials.native_check(row))),
                             digest(order_materials.render_spec(row)))

    def test_retiming_moves_targets_and_near_target_distractors_together(self):
        row = source_rows()["low_boundary"]
        spec = materials.render_spec(row, "low_late")
        first = {o["id"]: o["motion"] for o in spec["objects"]}[row["first_object_id"]]
        self.assertEqual((first["start_frame"], first["end_frame"]), (75, 105))
        near = [d["motion"]["start_frame"] for d in spec["distractors"] if d.get("motion_timing") == "near_targets"]
        self.assertTrue(set(near) <= {75, 105})
        self.assertEqual(spec["timing"]["total_frames"], row["total_frames"])

    def test_cell_row_keeps_text_and_answer(self):
        row = source_rows()["temporal_boundary"]
        new = materials.cell_row(row, "temporal_early", "videos/x.mp4")
        for key in ("option_A", "option_B", "correct_option", "question", "distractors"):
            self.assertEqual(new.get(key), row.get(key))
        self.assertEqual(new["timing_cell"], "temporal_early")
        self.assertTrue(new["eval_id"].endswith("__timing_temporal_early"))
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "videos").mkdir()
            Path(tmp, "videos/x.mp4").write_bytes(b"video")
            archived, permuted = materials.prompts(row, "temporal_early", Path("videos/x.mp4"), 1, "fresh", tmp)
        self.assertEqual([archived["view"], permuted["view"]], list(materials.VIEWS))
        self.assertEqual(permuted["row"]["option_A"], archived["row"]["option_B"])
        self.assertNotEqual(permuted["row"]["correct_option"], archived["row"]["correct_option"])
        self.assertEqual(permuted["row"]["timing_cell"], "temporal_early")

    def test_render_and_timing_audit_pass_for_a_new_cell(self):
        row = source_rows()["low_boundary"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"low_mid.mp4"
            spec = materials.render(path, row, "low_mid")
            audit = materials.timing_audit(path, spec)
            self.assertTrue(audit["passed"], audit)
            self.assertEqual(audit["token_bins"]["event_2"], [6, 8])
            wrong = copy.deepcopy(spec)
            for obj in wrong["objects"]:
                obj["motion"]["start_frame"] += 15
                obj["motion"]["end_frame"] += 15
            self.assertFalse(materials.timing_audit(path, wrong)["passed"])


def record(cohort, base, view, cell, prompt="p", e1=(2, 3), e2=(4, 5)):
    return {"cell": {"cohort": cohort, "base_sample_id": base, "view": view, "timing_cell": cell},
            "processor": {"prompt_input_ids_sha256": prompt, "visual_positions_sha256": "v",
                          "event_token_bins": {"event_1": list(e1), "event_2": list(e2)}}}


def aligned(cohort="fresh", base=1, view="archived"):
    shifts = {"low": (0, 0), "low_mid": (2, 2), "low_late": (3, 3), "temporal_early": (-1, 2), "temporal": (0, 3)}
    return [record(cohort, base, view, cell, e1=(2+a, 3+a), e2=(4+b, 5+b)) for cell, (a, b) in shifts.items()]


class AlignmentTests(unittest.TestCase):
    def test_scheduled_shifts_pass(self):
        self.assertEqual(behaviour.alignment(aligned()+aligned(view="option_order_permuted"), None), 2)

    def test_wrong_shift_or_prompt_tokens_fail(self):
        records = aligned()
        records[2]["processor"]["event_token_bins"]["event_2"] = [6, 7]
        with self.assertRaises(ValueError):
            behaviour.alignment(records, None)
        records = aligned()
        records[1]["processor"]["prompt_input_ids_sha256"] = "other"
        with self.assertRaises(ValueError):
            behaviour.alignment(records, None)

    def test_backup_refuses_unexpected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "private.json").write_text("{}")
            with self.assertRaises(ValueError):
                behaviour.public_files(tmp)

    def test_lifecycle_runs_preflight_then_full_export_verify(self):
        class Args:
            project_root, output_root, stage = "/repo", "/data/run", "full"
        steps = [c[c.index("--action")+1] for c in lifecycle.commands(Args)]
        self.assertEqual(steps, ["run", "run", "export", "verify"])
        self.assertTrue(all("run_phase3h_timing_behaviour.py" in c[2] for c in lifecycle.commands(Args)))


def result_rows(cohort, effects, bases=30):
    """Margins whose layout-symmetric S per cell equals `effects[cell]` (+ base noise); B layout gets a +2 offset."""
    rows = []
    for base in range(bases):
        noise = (base % 7 - 3)/10
        for cell in analysis.CELLS:
            for correct, offset in (("A", -2.0), ("B", 2.0)):
                margin = effects[cell]+noise+offset
                rows.append({"cohort": cohort, "base_sample_id": base, "timing_cell": cell, "view": correct,
                             "correct_option": correct, "margin": margin, "is_correct": margin > 0})
    return rows


class AnalysisTests(unittest.TestCase):
    def test_s_and_b_cancel_the_letter_offset(self):
        table = analysis.per_base(result_rows("fresh", dict.fromkeys(analysis.CELLS, 1.0)), "fresh")
        self.assertAlmostEqual(table[3]["low"]["S"], 1.0)
        self.assertAlmostEqual(table[3]["low"]["b"], 2.0)

    def test_reading_rule(self):
        gap = {"low": 0, "low_mid": 0, "low_late": 0, "temporal_early": 1.5, "temporal": 1.5}
        timing = {"low": 0, "low_mid": 0.9, "low_late": 1.5, "temporal_early": 0.9, "temporal": 1.5}
        null = dict.fromkeys(analysis.CELLS, 0.0)
        null["low_late"] = 1.0
        negative = {"low": 1.0, "low_mid": 1.0, "low_late": 0.5, "temporal_early": 0.0, "temporal": 0.0}
        for effects, expected in ((gap, "gap_contrast_dominant"), (timing, "sequence_shift_dominant"),
                                  (null, "no_temporal_advantage"), (negative, "temporal_disadvantage")):
            found = analysis.contrasts(analysis.per_base(result_rows("fresh", effects), "fresh"))
            self.assertAlmostEqual(found["Total"]["mean"], found["G105"]["mean"]+found["T0"]["mean"])
            self.assertEqual(analysis.reading(found), expected)
            share = analysis.share(found)
            self.assertEqual(share is None, expected in ("no_temporal_advantage", "temporal_disadvantage"))

    def test_missing_layout_is_refused(self):
        rows = [r for r in result_rows("fresh", dict.fromkeys(analysis.CELLS, 0.0)) if not (r["base_sample_id"] == 0 and
                r["timing_cell"] == "low_mid" and r["correct_option"] == "B")]
        with self.assertRaises(ValueError):
            analysis.per_base(rows, "fresh")


if __name__ == "__main__":
    unittest.main()
