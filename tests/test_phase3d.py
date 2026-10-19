import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts import phase3d_core as protocol
from scripts import prepare_phase3d as preparation


def row(base=1, first=1, subject=1, correct="A", condition="low_boundary"):
    relation = "before" if first == subject else "after"
    opposite = "after" if relation == "before" else "before"
    labels = {1: "the red triangle", 2: "the blue circle"}
    sentence = lambda word: f"{labels[subject]} moves {word} {labels[3-subject]}."
    gap = 45 if condition == "temporal_boundary" else 0
    targets = [{"id": key, "label": labels[key], "color": "red" if key == 1 else "blue",
                "shape": "triangle" if key == 1 else "circle",
                "start_frame": 30 if key == first else 60+gap,
                "end_frame": 60 if key == first else 90+gap} for key in (1, 2)]
    return {"base_sample_id": base, "condition": condition, "target_objects": targets,
            "eval_id": f"base_{base}_{condition}", "first_object_id": first,
            "second_object_id": 3-first, "prompt_variant": "original",
            "correct_relation": relation, "correct_sentence": sentence(relation),
            "correct_option": correct, "option_A": sentence(relation if correct == "A" else opposite),
            "option_B": sentence(relation if correct == "B" else opposite),
            "phase3c_analysis_stratum": "rescue", "archived_prediction": "A",
            "archived_input_metadata": {"do_not_reuse_for_permuted_prompt": True}}


def pairs(count=50):
    return {f"pair_{base}": {condition: row(base, 1 if base % 2 else 2,
                condition=condition) for condition in protocol.CONDITIONS}
            for base in range(1, count+1)}


class CanonicalLabelsTest(unittest.TestCase):
    def test_ground_truth_uses_timing_not_sentence_subject_or_answer_label(self):
        for first in (1, 2):
            for subject in (1, 2):
                for correct in ("A", "B"):
                    value = protocol.audit_row(row(first=first, subject=subject, correct=correct))
                    self.assertEqual(value["y"], int(first == 1))
                    self.assertEqual(value["sentence_relation"], "before" if first == subject else "after")
        value = protocol.audit_row(row(first=1, subject=2, correct="B"))
        self.assertEqual(value["canonical_relation"], "before")
        self.assertEqual(value["sentence_relation"], "after")

    def test_rejects_inconsistent_answers_timing_and_templates(self):
        changes = ({"correct_option": "B"}, {"correct_relation": "after"},
                   {"first_object_id": 2}, {"option_A": "An unsupported question."},
                   {"correct_sentence": "different"})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                protocol.audit_row({**row(), **change})
        for end in (math.nan, 70):
            value = row()
            value["target_objects"][0]["end_frame"] = end
            with self.assertRaises(ValueError):
                protocol.audit_row(value)

    def test_option_order_control_is_pure_permutation_without_stale_outputs(self):
        original = row(first=1, subject=2, correct="B")
        before = copy.deepcopy(original)
        control = protocol.option_order_control(original)
        self.assertEqual(original, before)
        self.assertEqual(control["option_A"], original["option_B"])
        self.assertEqual(control["option_B"], original["option_A"])
        self.assertEqual(control["correct_option"], "A")
        self.assertEqual(control["y"], 1)
        self.assertTrue(control["requires_fresh_tokenization_and_baseline"])
        self.assertFalse(control["require_rescue_behavior"])
        self.assertNotIn("archived_prediction", control)
        self.assertNotIn("archived_input_metadata", control)

    def test_case_independence_and_cross_condition_consistency(self):
        values = pairs()
        self.assertEqual(len(protocol.audit_pairs(values)), 50)
        for change in ("duplicate_base", "changed_prompt", "changed_static_attribute", "wrong_condition"):
            altered = copy.deepcopy(values)
            target = altered["pair_2"]["temporal_boundary"]
            if change == "duplicate_base":
                for condition in protocol.CONDITIONS:
                    altered["pair_2"][condition]["base_sample_id"] = 1
            elif change == "changed_prompt":
                target["prompt_variant"] = "swapped"
            elif change == "changed_static_attribute":
                target["target_objects"][0]["color"] = "green"
            else:
                target["condition"] = "low_boundary"
            with self.subTest(change=change), self.assertRaises(ValueError):
                protocol.audit_pairs(altered)
        with self.assertRaises(ValueError):
            protocol.audit_pairs(pairs(49))


class FoldsAndEvidenceTest(unittest.TestCase):
    def test_deterministic_stratification_is_at_base_level(self):
        cases = protocol.audit_pairs(pairs())
        folds = protocol.base_folds(cases)
        self.assertEqual(folds, protocol.base_folds(list(reversed(cases))))
        seen = []
        for fold in folds:
            train, test = set(fold["train_base_ids"]), set(fold["test_base_ids"])
            self.assertFalse(train & test)
            self.assertEqual(len(train), 40)
            self.assertEqual(len(test), 10)
            self.assertEqual(fold["test_class_counts"], {"before": 5, "after": 5})
            seen.extend(test)
        self.assertEqual(sorted(seen), list(range(1, 51)))
        self.assertNotEqual(folds, protocol.base_folds(cases, seed=43))
        with self.assertRaises(ValueError):
            protocol.base_folds(cases[:3], n_folds=5)

    def test_signed_evidence_handles_both_relation_labels(self):
        for label, logits in ((1, (-.5, 1.5, 1)), (0, (.5, -1.5, -1))):
            value = protocol.evidence_changes(label, *logits)
            self.assertEqual(value["S_low"], -.5)
            self.assertEqual(value["delta_S_boundary"], 2)
            self.assertEqual(value["delta_S_patch"], 1.5)
        self.assertGreater(protocol.relation_evidence(1, .3), 0)
        with self.assertRaises(ValueError):
            protocol.relation_evidence(1, math.inf)

    def test_observed_hooks_do_not_prove_saved_tensors(self):
        index = {"site_count": 7, "hook_audit": {"block_output_sites": 36,
                 "observed_sites": [[i, "block_output"] for i in range(0, 36, 4)]}}
        self.assertIsNone(protocol.declared_saved_sites(index))
        index["hook_audit"]["saved_sites"] = [[12, "block_output"], [16, "block_output"]]
        with self.assertRaises(ValueError):
            protocol.declared_saved_sites(index)
        index["site_count"] = 2
        self.assertEqual(protocol.declared_saved_sites(index), {(12, "block_output"), (16, "block_output")})


class PreparationTest(unittest.TestCase):
    def test_visual_only_stores_do_not_cover_downstream_text(self):
        source = pairs(1)
        mappings = {"pair_1": {"processor_records": {condition: {"group_positions": {
            "option_target_1_mentions": [7], "option_target_2_mentions": [8]}}
            for condition in protocol.CONDITIONS}, "support_audit": {"supports": {
                "whole_event2": {"low_positions": [1, 2], "temporal_positions": [3, 4]}}}}}
        indexes = [{"artifact_type": "real", "eval_id": source["pair_1"][condition]["eval_id"],
                    "site_count": 1, "positions": positions,
                    "hook_audit": {"saved_sites": [[16, "block_output"]]}}
                   for condition, positions in zip(protocol.CONDITIONS, ([1, 2], [3, 4]))]
        result = preparation.capture_coverage(source, mappings, indexes)
        self.assertEqual(result["candidate_condition_counts"]["L16_output_visual"], 2)
        self.assertEqual(result["candidate_condition_counts"]["L20_option_mentions"], 0)
        self.assertFalse(any(r["tensor_hash_and_live_replay_verified"] for r in result["rows"]))
        mappings["pair_1"]["processor_records"]["low_boundary"]["group_positions"]["option_target_1_mentions"] = []
        with self.assertRaises(ValueError):
            preparation.capture_coverage(source, mappings, indexes)

    def test_metadata_only_idempotence_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"old/pairs.json"
            source.parent.mkdir()
            source.write_text(json.dumps(pairs()))
            original = source.read_bytes()
            args = SimpleNamespace(pairs_path=source, expected_cases=50, folds=5, seed=42,
                                   mappings_path=None, capture_root=[], source_capture_refs=None,
                                   output_dir=root/"new")
            result = preparation.prepare(args)
            self.assertFalse(result["gpu_preflight_passed"])
            self.assertFalse(result["probe_trained"])
            self.assertFalse(result["experiment_complete"])
            self.assertEqual(result, preparation.prepare(args))
            self.assertEqual(source.read_bytes(), original)
            args.seed = 43
            with self.assertRaises(ValueError):
                preparation.prepare(args)
            args.output_dir = source.parent/"inside_old"
            with self.assertRaises(ValueError):
                preparation.prepare(args)


if __name__ == "__main__":
    unittest.main()
