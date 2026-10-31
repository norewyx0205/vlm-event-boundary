import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import audit_phase3e_interactions as audit
from scripts import run_phase3e_paired_capture as capture
from scripts.phase3c_core import atomic_write, digest, read_json
from scripts.phase3e_paired_design import build_plan
from scripts.phase3e_preflight import factorial_rows


class InteractionTest(unittest.TestCase):
    def setUp(self):
        _, self.rows = factorial_rows("/tmp/unused_interaction_audit", [2026103101])

    def test_all_target_roles_and_prompt_deduplication(self):
        result = audit.audit({"rows": self.rows})
        self.assertEqual(result["videos"], 4)
        self.assertEqual(result["target_events"], 8)
        self.assertEqual(result["event_distractor_comparisons"], 32)
        groups = {(r["condition"], r["event_role"]): r for r in result["co_motion_by_condition_and_role"]}
        self.assertEqual(groups["low_boundary", "second"]["mean_co_motion_fraction"], 1.)
        self.assertEqual(groups["temporal_boundary", "second"]["mean_co_motion_fraction"], 1.)
        self.assertEqual(groups["temporal_boundary", "first"]["mean_co_motion_fraction"], 1.)
        self.assertFalse(result["systematic_second_event_co_motion_change"])
        self.assertTrue(result["event_relative_schedule_verified"])
        self.assertFalse(result["model_inference_performed"])

    def test_same_target_pairs_keep_endpoints_and_independent_family_unit(self):
        result = audit.audit({"rows": self.rows})
        low = result["paired_summary"]["order_first_to_second"]["low_boundary"]["metrics"]["co_motion_fraction"]
        self.assertEqual((low["left"], low["right"], low["right_minus_left"]), (1., 1., 0.))
        temporal = result["paired_summary"]["order_first_to_second"]["temporal_boundary"]
        value = temporal["metrics"]["co_motion_fraction"]
        self.assertEqual((value["left"], value["right"], value["right_minus_left"]), (1., 1., 0.))
        self.assertEqual(temporal["target_pairs"], 2)
        self.assertEqual(temporal["independent_scene_families"], 1)
        self.assertEqual(value["families_zero"], 1)
        self.assertTrue(all(p["active_identity_set_changed"] for p in result["order_pairs"]))

    def test_actual_motion_intervals_are_end_exclusive_and_static_objects_not_active(self):
        row = copy.deepcopy(self.rows[0])
        target = row["target_objects"][0]
        start, end = target["start_frame"], target["end_frame"]
        dist = row["distractors"][2]
        dist.update(start_frame=start+15, end_frame=end+15)
        row["distractors"] = [dist, row["distractors"][0]]
        result = audit.target_event(row, target)
        self.assertEqual(result["co_motion_fraction"], .5)
        self.assertEqual(result["co_motion_frames"], list(range(start+15, end)))
        self.assertEqual(result["distractors"][1]["active_fraction"], 0.)
        self.assertNotIn(end, result["co_motion_frames"])

    def test_overlap_matches_full_generator_raster_for_all_shapes_and_clipped_edges(self):
        for shape_one in ("circle", "square", "triangle"):
            for shape_two in ("circle", "square", "triangle"):
                for a, b in (((100, 100), (125, 110)), ((50, 50), (300, 300)),
                             ((5, 5), (20, 10)), ((504, 505), (490, 500))):
                    masks = []
                    for shape, centre, radius in ((shape_one, a, 28), (shape_two, b, 23)):
                        mask = np.full((512, 512, 3), 255, dtype=np.uint8)
                        audit.generator.draw_shape(mask, shape, "black", *centre, size=radius)
                        masks.append(mask[:, :, 0] == 0)
                    expected = int(np.count_nonzero(masks[0] & masks[1]))
                    actual = audit.footprint_overlap({"shape": shape_one}, a, {"shape": shape_two}, b)
                    self.assertEqual(actual, expected, (shape_one, shape_two, a, b))

    def test_metadata_is_not_mutated_and_changed_prompt_view_is_rejected(self):
        before = copy.deepcopy(self.rows)
        audit.audit({"rows": self.rows})
        self.assertEqual(before, self.rows)
        changed = copy.deepcopy(self.rows)
        changed[0]["distractors"][0]["from"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "distractor"):
            audit.audit({"rows": changed})

    def test_full_fixed_plan_audit_binding_and_no_new_design_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan(root)
            fingerprint = plan["fingerprint"]
            result = audit.save(root, plan)
            self.assertEqual(plan["fingerprint"], fingerprint)
            self.assertEqual(result["videos"], 160)
            self.assertEqual(result["target_events"], 320)
            self.assertEqual(result["event_distractor_comparisons"], 1280)
            self.assertEqual(capture.interaction_record(root, plan)["record_sha256"], result["record_sha256"])
            content = read_json(root / "interaction_audit.json")
            content["target_events"] = 640
            atomic_write(root / "interaction_audit.json", content)
            with self.assertRaisesRegex(ValueError, "Interaction audit"):
                capture.interaction_record(root, plan)

    def test_audit_retains_actual_active_distractor_identity(self):
        result = audit.audit({"rows": self.rows})
        low = [e for e in result["events"] if e["condition"] == "low_boundary"]
        self.assertEqual({tuple(e["active_distractor_ids"]) for e in low if e["event_role"] == "first"}, {(4,)})
        self.assertEqual({tuple(e["active_distractor_ids"]) for e in low if e["event_role"] == "second"}, {(3,)})
        report = audit.report_text(result)
        self.assertIn("Left | Right | Delta", report)
        self.assertIn("not model inference", report)

    def test_boundary_matching_includes_geometry_and_every_distractor_slot(self):
        result = audit.audit({"rows": self.rows})
        for role, summary in result["paired_summary"]["boundary_low_to_temporal"].items():
            self.assertEqual(summary["active_identity_sets_changed"], 0, role)
            for metric in summary["metrics"].values():
                self.assertEqual(metric["target_pairs_changed"], 0, (role, metric))
                self.assertEqual(metric["left"], metric["right"])
        for pair in result["boundary_pairs"]:
            self.assertEqual(pair["left_active_distractor_ids"], pair["right_active_distractor_ids"])

    def test_historical_missing_windows_are_unknown_not_reconstructed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.jsonl"
            rows = copy.deepcopy(self.rows)
            for row in rows:
                for dist in row["distractors"]:
                    dist.pop("start_frame")
                    dist.pop("end_frame")
            atomic_write(path, rows, jsonl=True)
            before = path.read_bytes()
            result = audit.historical_audit([path])
            self.assertEqual(path.read_bytes(), before)
            source = result["sources"][0]
            self.assertEqual(source["videos"], 4)
            for counts in source["by_condition"].values():
                self.assertEqual(counts["videos_missing_moving_windows"], 2)
                self.assertEqual(counts["events_with_known_windows"]["second"], 0)
            self.assertFalse(result["missing_times_reconstructed"])

    def test_historical_saved_windows_are_measured_with_prompt_deduplication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "annotations.jsonl"
            atomic_write(path, self.rows, jsonl=True)
            result = audit.save_history(root, [path])
            for counts in result["sources"][0]["by_condition"].values():
                self.assertEqual(counts["videos_with_saved_moving_windows"], 2)
                self.assertEqual(counts["events_with_co_motion"]["second"], 2)
            self.assertTrue((root / "historical_annotation_audit.md").is_file())


if __name__ == "__main__":
    unittest.main()
