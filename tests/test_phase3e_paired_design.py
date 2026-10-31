import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts import phase3e_paired_design as design
from scripts.prepare_phase3e_paired_design import prepare


class PairedOptionDesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = design.build_plan("/tmp/phase3e_design_test_unused")

    def test_full_material_counts_and_scope(self):
        plan = self.plan
        audit = design.validate_plan(plan)
        self.assertEqual(plan["counts"]["videos"], 160)
        self.assertEqual(len(plan["rows"]), 640)
        self.assertEqual(len(plan["statement_examples"]), 1280)
        self.assertEqual(len(plan["order_pairs"]), 1280)
        self.assertEqual(len(plan["boundary_pairs"]), 640)
        self.assertEqual(Counter(r["y_statement_true"] for r in plan["statement_examples"]), {0: 640, 1: 640})
        self.assertTrue(audit["second_event_mover_is_valid_task_evidence"])
        self.assertFalse(plan["alignment_with_prior_ko"]["requires_new_ko_forwards"])
        self.assertEqual(plan["counts"]["new_ko_forwards"], 0)
        self.assertFalse(plan["gpu_launch_authorized_by_this_plan"])

    def test_order_pairs_reverse_labels_not_names_or_paths(self):
        examples = {e["example_id"]: e for e in self.plan["statement_examples"]}
        pairs = self.plan["order_pairs"]
        by_pair = {}
        for pair in pairs:
            left, right = examples[pair["left_example_id"]], examples[pair["right_example_id"]]
            self.assertEqual(left["option_text"], right["option_text"])
            self.assertEqual(left["reference_label"], right["reference_label"])
            self.assertEqual(left["group_name"], right["group_name"])
            self.assertEqual(left["y_statement_true"], 1-right["y_statement_true"])
            by_pair.setdefault(pair["unordered_pair_id"], []).append(pair)
        self.assertEqual(len(by_pair), 640)
        for directions in by_pair.values():
            self.assertEqual(len(directions), 2)
            self.assertEqual(directions[0]["left_example_id"], directions[1]["right_example_id"])
            self.assertEqual(directions[0]["right_example_id"], directions[1]["left_example_id"])

    def test_named_label_is_independent_of_correct_letter(self):
        row = copy.deepcopy(self.plan["rows"][0])
        before = [design.statement_truth(row, option) for option in ("A", "B")]
        row["correct_option"] = "B" if row["correct_option"] == "A" else "A"
        self.assertEqual(before, [design.statement_truth(row, option) for option in ("A", "B")])
        changed = copy.deepcopy(self.plan["rows"])
        changed[0] = row
        with self.assertRaises(ValueError):
            design.validate_materials(changed, self.plan["scene_seeds"])

    def test_all_conditions_stay_in_family_folds(self):
        plan = self.plan
        for fold in plan["folds"]:
            train, test = set(fold["train_scene_seeds"]), set(fold["test_scene_seeds"])
            self.assertEqual((len(train), len(test)), (32, 8))
            test_rows = [r for r in plan["rows"] if r["scene_seed"] in test]
            self.assertEqual(len(test_rows), 128)
            for inner in fold["inner_folds"]:
                self.assertEqual(len(inner["validation_scene_seeds"]), 8)
                self.assertFalse(test & set(inner["train_scene_seeds"]))
        changed = copy.deepcopy(plan)
        changed["folds"][0]["inner_folds"][0]["train_scene_seeds"][0] = plan["folds"][0]["test_scene_seeds"][0]
        with self.assertRaisesRegex(ValueError, "Inner folds"):
            design.validate_plan(changed)

    def test_changed_correspondence_and_incomplete_factorial_rejected(self):
        changed = copy.deepcopy(self.plan)
        changed["order_pairs"][0]["group_name"] = "option_A_t1_wrong"
        with self.assertRaisesRegex(ValueError, "pairing"):
            design.validate_plan(changed)
        rows = copy.deepcopy(self.plan["rows"])
        rows.pop()
        with self.assertRaises(ValueError):
            design.validate_materials(rows, self.plan["scene_seeds"])

    def test_paired_question_and_boundary_schedule_cannot_change(self):
        rows = copy.deepcopy(self.plan["rows"])
        rows[0]["question"] = "A different question"
        with self.assertRaisesRegex(ValueError, "question/option"):
            design.validate_materials(rows, self.plan["scene_seeds"])
        rows = copy.deepcopy(self.plan["rows"])
        rows[0]["event_timing"]["gap_frames"] = 999
        with self.assertRaisesRegex(ValueError, "timing schedule"):
            design.validate_materials(rows, self.plan["scene_seeds"])
        rows = copy.deepcopy(self.plan["rows"])
        obj = rows[0]["target_objects"][0]
        x, y = obj["from"]
        obj["from"] = (x+1, y)
        with self.assertRaises(ValueError):
            design.validate_materials(rows, self.plan["scene_seeds"])

    def test_cpu_manifest_output_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "new_design"
            result = prepare(root)
            self.assertEqual(result["status"], "cpu_design_audit_passed")
            self.assertFalse(result["new_gpu_inference_performed"])
            self.assertTrue((root / "order_pairs.jsonl").is_file())
            with self.assertRaises(ValueError):
                prepare(root)


if __name__ == "__main__":
    unittest.main()
