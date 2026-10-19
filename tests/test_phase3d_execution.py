"""Synthetic ledgers exercise orchestration, not scientific model evidence."""

import copy
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from scripts import phase3d_execution as ex
from scripts import phase3d_knockout as ko
from scripts import run_phase3d_vm as vm
from scripts.phase3c_core import atomic_write, digest, file_hash, input_ids_hash, read_json
from scripts.phase3d_core import audit_pairs, base_folds
from scripts.run_phase3c_preflight import save_tensors
from scripts.run_phase3d_probes import DEFAULT_READOUTS, load_dataset
from tests.test_phase3d import pairs
from tests.test_phase3d_features import CharacterTokenizer
from tests import test_phase3d_features as feature_tests
from tests.test_phase3c_preflight import ToyText, toy_inputs


EXACT = {"exact_match": True, "logits_allclose": True, "max_abs_diff": 0.0, "passed": True}


def decision(logits, correct="A"):
    good, bad = (0, 1) if correct == "A" else (1, 0)
    predicted = int(logits.argmax())
    return {"correct_option": correct, "correct_token_id": good, "incorrect_token_id": bad,
            "correct_logit": float(logits[good]), "incorrect_logit": float(logits[bad]),
            "margin": float(logits[good]-logits[bad]), "predicted_first_token_id": predicted,
            "prediction": "A" if predicted == 0 else "B" if predicted == 1 else "other",
            "is_correct": predicted == good}


def ledger_fixture(root, count=50):
    """Small hidden dimension, but the real frozen row/support counts."""
    source_pairs = pairs(count)
    plan = {"fingerprint": digest(["synthetic test only", count]), "case_ids": list(source_pairs),
            "preflight_ids": list(source_pairs)[:2]}
    mappings = {key: {"support_audit": {"supports": {"whole_event2": {
        "one_to_one": True, "low_positions": list(range(512)), "temporal_positions": list(range(512))}}}}
        for key in source_pairs}
    context = plan, source_pairs, mappings
    refs = {}
    for key, pair in source_pairs.items():
        for condition, annotation in pair.items():
            logits = torch.tensor([1., 2., -1.]) if condition == "low_boundary" else torch.tensor([3., 1., -1.])
            vectors = torch.full((512, 4), 1. if condition == "low_boundary" else 2.)
            for view in ex.VIEWS:
                audit_spec = next(s for s in ex.feature_tasks(plan, "audit", [key]) if s["condition"] == condition and s["view"] == view)
                ids = [151656]*512 + [10, 20 if view == "archived" else 21]
                hashes = {"input_ids": {"dtype": "torch.int64", "shape": [1, len(ids)], "sha256": input_ids_hash(ids)}}
                record = {"input_ids": ids, "prompt_input_ids_sha256": input_ids_hash(ids), "input_tensor_sha256": hashes}
                groups = {"whole_event2": list(range(512))}
                audit = ex.commit(root, plan, audit_spec, {"processor_record": record, "groups": groups,
                    "input_fingerprint": digest(hashes)})
                capture_spec = next(s for s in ex.feature_tasks(plan, "capture", [key]) if s["condition"] == condition and s["view"] == view)
                actual_logits = logits if view == "archived" else torch.tensor([1., 4., -1.])
                correct = "A" if view == "archived" else "B"
                measured = decision(actual_logits, correct)
                data = {"features": torch.full((len(ex.FEATURE_NAMES), 4), .1), "logits": actual_logits,
                        "plain_logits": actual_logits.clone(), "standard_logits": actual_logits.clone(),
                        "timing": {"upstream": "synthetic exact", "input16": "synthetic exact"}}
                result = {"input_fingerprint": audit["input_fingerprint"], "audit_record_sha256": audit["record_sha256"],
                          "decision": measured, "gates": {"capture_noop": EXACT, "standard": {**EXACT, "first_token_match": True}},
                          "hook_audit": {"block_output_sites": 36, "post_deepstack_layers": [0, 1, 2],
                              "patch_applied_count": 0, "L16_input_matches_L15_output_at_planned_positions": True}}
                if view == "archived":
                    source = root.parent / "source" / annotation["eval_id"] / "states.pt"
                    state = {"layer": 16, "location": "block_output", "positions": list(range(512)), "vectors": vectors}
                    save_tensors(source, {"states": {(16, "block_output"): state}, "logits": logits})
                    index = source.with_name("index.json")
                    atomic_write(index, {"artifact_type": "real", "eval_id": annotation["eval_id"], "site_count": 1,
                        "hook_audit": {"saved_sites": [[16, "block_output"]]}, "input_tensor_sha256": hashes,
                        "vectors_path": str(source), "vectors_sha256": file_hash(source)})
                    ref = ex.seal({"eval_id": annotation["eval_id"], "capture_index_path": str(index),
                        "capture_index_sha256": file_hash(index), "input_tensor_sha256": hashes, "decision": measured})
                    refs[annotation["eval_id"]] = ref
                    result.update(source_replay={"logits": EXACT, "states_exact": True, "inputs_exact": True, "decision_exact": True},
                                  source_record_sha256=ref["record_sha256"], source_index_sha256=file_hash(index))
                    result["gates"].update(identity=EXACT, disabled_ko=EXACT)
                    data.update(donor16=state, identity_logits=logits.clone(), disabled_ko_logits=logits.clone(),
                                identity_features=data["features"].clone(), identity_timing=data["timing"])
                ex.commit(root, plan, capture_spec, result, data)
    atomic_write(root / "source_baselines.json", refs)
    cases = audit_pairs(source_pairs, count)
    atomic_write(root / "case_metadata.json", cases)
    if count >= 10:
        atomic_write(root / "outer_folds.json", base_folds(cases))
    for spec in ex.feature_tasks(plan, "patch"):
        before, data = ex.capture_reference(root, context, spec["pair_id"], spec["condition"])
        other = "temporal_boundary" if spec["condition"] == "low_boundary" else "low_boundary"
        donor, donor_data = ex.capture_reference(root, context, spec["pair_id"], other)
        logits = torch.tensor([2.5, 1., -1.])
        ex.commit(root, plan, spec, {"input_fingerprint": before["input_fingerprint"],
            "audit_record_sha256": before["audit_record_sha256"], "baseline_record_sha256": before["record_sha256"],
            "donor_record_sha256": donor["record_sha256"], "decision": decision(logits), "recipient_replay": EXACT,
            "same_block_timing_exact": True, "observed_donor_replacement_exact": True,
            "recipient_positions": list(range(512)), "donor_positions": list(range(512)),
            "hook_audit": {"block_output_sites": 36, "post_deepstack_layers": [0, 1, 2],
                "patch_applied_count": 1, "L16_input_matches_L15_output_at_planned_positions": True}},
            {"features": data["features"]+.2, "logits": logits, "timing": data["timing"],
             "patched_visual16": donor_data["donor16"]})
    return context


class ExecutionTest(unittest.TestCase):
    def test_default_cpu_probe_jobs_run_and_verify_from_complete_synthetic_export(self):
        from scripts import run_phase3d_probes as probe_runner
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)/"run"
            context = ledger_fixture(root)
            ex.export_features(root, context)
            args = SimpleNamespace(manifest_path=root/"feature_store/manifest.json", output_dir=root/"probes", feature_names=None)
            result = probe_runner.run(args)
            self.assertEqual(len(result["results"]), 107)
            self.assertEqual(result, probe_runner.verify_outputs(root/"probes", args.manifest_path))

    def test_frozen_plan_is_idempotent_preserves_source_and_detects_snapshot_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source = parent/"old_run"
            context = ledger_fixture(source)
            _, annotations, mappings = context
            video = parent/"frozen.mp4"
            video.write_bytes(b"synthetic plan fixture")
            for pair in annotations.values():
                for row in pair.values():
                    row.update(video_path=str(video), phase3c_video_sha256=file_hash(video))
            cases = audit_pairs(annotations)
            plan = {"artifact_type": "real", "case_ids": [c["pair_id"] for c in cases],
                    "pairs_sha256": digest(annotations), "mappings_sha256": digest(mappings),
                    "settings": {}, "source_runtime": {}}
            plan["fingerprint"] = digest(plan)
            for name, value in (("plan", plan), ("pairs", annotations), ("mappings", mappings)):
                atomic_write(source/f"{name}.json", value)
            refs = read_json(source/"source_baselines.json")
            manifest, rows = [], []
            for key, pair in annotations.items():
                for condition, annotation in pair.items():
                    spec = {"pair_id": key, "condition": condition, "task_id": digest([key, condition])}
                    manifest.append(spec)
                    rows.append(ex.seal({**refs[annotation["eval_id"]], "task_id": spec["task_id"],
                        "spec": spec, "plan_fingerprint": plan["fingerprint"], "passed": True}))
            atomic_write(source/"baseline/task_manifest.jsonl", manifest, jsonl=True)
            atomic_write(source/"baseline/rows.jsonl", rows, jsonl=True)
            atomic_write(source/"baseline/summary.json", {"complete": True, "completed": 100, "failed": 0,
                "plan_fingerprint": plan["fingerprint"], "rows_sha256": file_hash(source/"baseline/rows.jsonl"),
                "manifest_sha256": file_hash(source/"baseline/task_manifest.jsonl")})
            before = {str(p): file_hash(p) for p in source.rglob("*") if p.is_file()}
            args = SimpleNamespace(source_run=source, output_root=parent/"new_run", ko_source=None,
                                   project_root=str(Path(__file__).resolve().parents[1]))
            first = ex.freeze_plan(args)
            self.assertEqual(first, ex.freeze_plan(args))
            self.assertEqual(first, ex.load_context(args.output_root))
            self.assertEqual(before, {str(p): file_hash(p) for p in source.rglob("*") if p.is_file()})
            altered = copy.deepcopy(mappings)
            altered["pair_1"]["support_audit"]["supports"]["whole_event2"]["low_positions"][0] = 999
            atomic_write(args.output_root/"mappings.json", altered)
            with self.assertRaisesRegex(ValueError, "snapshot"):
                ex.load_context(args.output_root)

    def test_changed_prompt_drops_archived_inputs_outputs_and_keeps_true_relation(self):
        original = pairs(1)["pair_1"]["low_boundary"]
        actual = ex.permuted_row(original)
        self.assertFalse(any(k.startswith(("archived_", "phase3")) for k in actual))
        self.assertEqual(actual["option_A"], original["option_B"])
        self.assertEqual(actual["correct_option"], "B")
        self.assertEqual(actual["target_objects"], original["target_objects"])

    def test_task_counts_unique_and_old_new_primary_groups_not_used(self):
        plan = {"fingerprint": "new", "case_ids": list(pairs())}
        self.assertEqual(len(ex.feature_tasks(plan, "audit")), 200)
        self.assertEqual(len(ex.feature_tasks(plan, "capture")), 200)
        self.assertEqual(len(ex.feature_tasks(plan, "patch")), 100)
        specs = ko.condition_specs(plan, pairs())
        self.assertEqual(len(specs), 11700)
        self.assertEqual(len({s["task_id"] for s in specs}), 11700)
        self.assertEqual(len(DEFAULT_READOUTS), 83)
        self.assertEqual({f"decision:L{l}_output" for l in (16, 20, 24, 28, 32, 35)} & set(DEFAULT_READOUTS),
                         {f"decision:L{l}_output" for l in (16, 20, 24, 28, 32, 35)})

    def test_verified_resume_does_not_execute_again_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            context = ({"fingerprint": "test", "case_ids": ["one"]}, {}, {})
            specs = ex.feature_tasks(context[0], "audit")
            execute = Mock(return_value=({"note": "test"}, None))
            ex.run_tasks(root, context, specs, execute)
            self.assertEqual(execute.call_count, 4)
            ex.run_tasks(root, context, specs, Mock(side_effect=AssertionError("must not run")))
            path = ex.checkpoint_path(root, specs[0])
            record = read_json(path)
            record["passed"] = False
            atomic_write(path, record)
            with self.assertRaises(ValueError):
                ex.saved_checkpoint(root, context[0], specs[0])

    def test_failed_task_requires_explicit_retry_and_no_partial_checkpoint_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            context = ({"fingerprint": "test", "case_ids": ["one"]}, {}, {})
            spec = ex.feature_tasks(context[0], "audit")[:1]
            with self.assertRaisesRegex(RuntimeError, "synthetic fail"):
                ex.run_tasks(root, context, spec, Mock(side_effect=RuntimeError("synthetic fail")))
            with self.assertRaisesRegex(RuntimeError, "retry_failed"):
                ex.run_tasks(root, context, spec, Mock())
            ex.run_tasks(root, context, spec, Mock(return_value=({}, None)), retry_failed=True)
            self.assertIsNotNone(ex.saved_checkpoint(root, context[0], spec[0]))

    def test_independent_export_complete_store_and_hash_rejection(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory) / "run"
            context = ledger_fixture(root)
            manifest = ex.export_features(root, context)
            _, rows, cases, _, names, features = load_dataset(root / "feature_store/manifest.json")
            self.assertEqual((len(rows), len(cases)), (300, 50))
            self.assertEqual(features.shape, (300, 289, 4))
            self.assertIn("option_A_relation:L20_output", names)
            self.assertEqual(manifest, ex.export_features(root, context))
            (root / "feature_store/features.npz").write_bytes(b"altered")
            with self.assertRaises(ValueError):
                load_dataset(root / "feature_store/manifest.json")

    def test_gate_recomputed_from_raw_logits_not_claimed_true_flag(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory) / "run"
            context = ledger_fixture(root, 2)
            spec = ex.feature_tasks(context[0], "capture")[0]
            row = ex.saved_checkpoint(root, context[0], spec)
            data = ex.payload(root, row)
            data["identity_logits"] = data["identity_logits"] + 1
            path = ex.payload_path(root, row)
            save_tensors(path, data)
            row["payload"]["sha256"] = file_hash(path)
            atomic_write(ex.checkpoint_path(root, spec), ex.seal(row))
            with self.assertRaisesRegex(ValueError, "identity"):
                ex.verify_features(root, context)

    def test_patch_mapping_or_timing_cannot_be_marked_passed_by_summary(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory) / "run"
            context = ledger_fixture(root, 2)
            spec = ex.feature_tasks(context[0], "patch")[0]
            record = ex.saved_checkpoint(root, context[0], spec)
            data = ex.payload(root, record)
            data["patched_visual16"]["vectors"][0] += 1
            path = ex.payload_path(root, record)
            save_tensors(path, data)
            record["payload"]["sha256"] = file_hash(path)
            atomic_write(ex.checkpoint_path(root, spec), ex.seal(record))
            with self.assertRaisesRegex(ValueError, "replacement"):
                ex.verify_features(root, context)

    def test_incomplete_grid_blocks_export_and_preflight_missing_blocks_full(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            context = ledger_fixture(root, 2)
            spec = ex.feature_tasks(context[0], "patch")[0]
            ex.checkpoint_path(root, spec).unlink()
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Incomplete"):
                ex.export_features(root, context)
            with self.assertRaises(FileNotFoundError):
                vm.require_preflight(SimpleNamespace(output_root=str(root)), context)

    def test_root_lock_excludes_duplicate_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            with vm.lock(directory):
                with self.assertRaises(RuntimeError):
                    with vm.lock(directory):
                        pass


class KnockoutVerifierTest(unittest.TestCase):
    def test_complete_ko50_pairing_uniform_analysis_and_missing_condition_rejection(self):
        from scripts.phase3c_knockout25 import positions
        from scripts.analyze_phase3c import aggregate as real_aggregate
        annotations = pairs()
        plan = {"fingerprint": "test", "case_ids": list(annotations)}
        mapping = {"processor_records": {condition: {"input_ids": list(range(8)), "group_positions": {
            "options_all": [6], "query_all": [5]}} for condition in ex.CONDITIONS},
            "support_audit": {"supports": {"whole_event2": {"low_positions": [1, 2, 3, 4], "temporal_positions": [1, 2, 3, 4]}},
            "knockout_controls": {condition: {query: {name: {"background": {"query_positions": queries,
                "target_key_positions": keys, "control_key_positions": controls}}
                for name, keys, controls in (("target_1", [1], [3]), ("target_2", [2], [4]), ("both_targets", [1, 2], [3, 4]))}
                for query, queries in (("query_all", [5]), ("options_all", [6]))} for condition in ex.CONDITIONS}}}
        context = plan, annotations, {key: mapping for key in annotations}
        def reference(_root, _context, pair, condition):
            return {"record_sha256": digest([pair, condition]), "input_fingerprint": "input", "audit_record_sha256": "audit",
                    "decision": decision(torch.tensor([1., 2.]) if condition == "low_boundary" else torch.tensor([3., 1.]))}, None
        def checkpoint(root, plan, spec):
            before, _ = reference(root, None, spec["pair_id"], spec["condition"])
            queries, keys, budget = positions(mapping, spec["condition"], spec)
            return {"baseline_record_sha256": before["record_sha256"], "input_fingerprint": "input", "audit_record_sha256": "audit",
                    "query_positions": queries, "key_positions": keys, "edge_budget": budget,
                    "decision": decision(torch.tensor([1.5, 2.]) if spec["condition"] == "low_boundary" else torch.tensor([2., 1.])),
                    "mask_audit": {"enabled": True, "all_heads": 32, "pre_softmax_mask": "negative_infinity", "renormalized": True,
                        "layers": {str(layer): {"budget": budget, "max_blocked_probability": 0., "max_row_sum_error": 0.}
                                   for layer in spec["window"]}}}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.object(ex, "capture_reference", side_effect=reference), patch.object(ex, "saved_checkpoint", side_effect=checkpoint), \
                patch("scripts.analyze_phase3c.aggregate", side_effect=lambda r, k, m: real_aggregate(r, k, m, repeats=2)):
            summary = ko.analyze_complete(directory, context)
            self.assertEqual(summary["paired_rows"], 5850)
            self.assertEqual(summary["condition_rows"], 11700)
            cells = read_json(Path(directory)/"knockout/analysis/aggregate_cells.json")
            self.assertEqual(len(cells), 117)
            self.assertEqual({cell["case_count"] for cell in cells}, {50})
            contrasts = read_json(Path(directory)/"knockout/analysis/matched_control_contrasts.json")
            self.assertEqual(len(contrasts), 2700)
            self.assertFalse(any(row["key_group"] == "whole_event2" for row in contrasts))
            first = ko.condition_specs(plan, annotations)[0]["task_id"]
            with patch.object(ex, "saved_checkpoint", side_effect=lambda root, plan, spec:
                    None if spec["task_id"] == first else checkpoint(root, plan, spec)):
                with self.assertRaisesRegex(ValueError, "11,700"):
                    ko.analyze_complete(directory, context)

    def row(self):
        spec = {"pair_id": "pair_1", "condition": "low_boundary", "query_group": "options_all",
                "key_group": "whole_event2", "control": "whole_event_reference", "window": [16, 17, 18, 19]}
        mapping = {"processor_records": {"low_boundary": {"input_ids": list(range(8)),
            "group_positions": {"options_all": [5, 6]}}}, "support_audit": {"supports": {
                "whole_event2": {"low_positions": [1, 2]}}}}
        from scripts.phase3c_knockout25 import positions
        queries, keys, budget = positions(mapping, "low_boundary", spec)
        before = {"record_sha256": "baseline", "input_fingerprint": "input", "audit_record_sha256": "audit",
                  "decision": decision(torch.tensor([1., 2.]))}
        row = {"baseline_record_sha256": "baseline", "input_fingerprint": "input", "audit_record_sha256": "audit",
               "query_positions": queries, "key_positions": keys, "edge_budget": budget,
               "decision": decision(torch.tensor([2., 1.])), "mask_audit": {
                   "enabled": True, "all_heads": 32, "pre_softmax_mask": "negative_infinity", "renormalized": True,
                   "layers": {str(layer): {"budget": budget, "max_blocked_probability": 0., "max_row_sum_error": 0.}
                              for layer in spec["window"]}}}
        return row, spec, mapping, before

    def test_mask_edges_inputs_and_raw_ab_decision_are_verified(self):
        row, spec, mapping, before = self.row()
        ko.verify_condition(row, spec, mapping, before)
        for change in ("mask", "budget", "input", "margin"):
            altered = copy.deepcopy(row)
            if change == "mask":
                altered["mask_audit"]["layers"]["16"]["max_blocked_probability"] = .01
            elif change == "budget":
                altered["key_positions"] = [2, 3]
            elif change == "input":
                altered["input_fingerprint"] = "wrong"
            else:
                altered["decision"]["margin"] += 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                ko.verify_condition(altered, spec, mapping, before)

    def test_semantic_matching_does_not_use_different_plan_task_ids(self):
        _, spec, _, _ = self.row()
        self.assertEqual(ko.semantic_key({**spec, "task_id": "old"}), ko.semantic_key({**spec, "task_id": "new"}))
        self.assertNotEqual(ko.semantic_key(spec), ko.semantic_key({**spec, "window": [12, 13, 14, 15]}))

    def test_semantic_reuse_checks_inputs_runtime_baseline_masks_and_output_chain(self):
        actual, spec, mapping, baseline = self.row()
        old_baseline = ex.seal({"decision": baseline["decision"]})
        old_result = {**actual, "input_tensor_sha256": {"test": "same"},
                      "baseline_record_sha256": old_baseline["record_sha256"]}
        outcome = {f"low_base_{name}": baseline["decision"][name] for name in
                   ("margin", "correct_logit", "incorrect_logit", "prediction")}
        outcome.update({f"low_ko_{name}": actual["decision"][name] for name in
                        ("margin", "correct_logit", "incorrect_logit", "prediction")})
        old_record = ex.seal({"spec": {**spec, "task_id": "historical_plan"},
                              "conditions": {"low_boundary": old_result}, "paired_outcome": outcome})
        source = {"source_root": "/old/read_only", "source_runtime": {"engine": "same"}, "candidates": {
            digest(ko.semantic_key(spec)): {"source_record": old_record, "baseline": old_baseline, "result": old_result}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_write(root / "ko_reuse_source.json", source)
            context = ({"ko_reuse_source_sha256": digest(source)}, {}, {"pair_1": mapping})
            audit = {"processor_record": {"input_tensor_sha256": {"test": "same"}}, "input_fingerprint": "input",
                     "record_sha256": "audit"}
            reused = ko.reuse_condition(root, context, spec, baseline, audit, {"engine": "same"})
            self.assertEqual(reused["decision"], actual["decision"])
            self.assertEqual(reused["reuse"]["source_record_sha256"], old_record["record_sha256"])
            atomic_write(root/"reused.json", reused)
            self.assertEqual(read_json(root/"reused.json")["reuse"], reused["reuse"])
            for runtime, change in (({"engine": "different"}, None), ({"engine": "same"}, "different input")):
                altered = copy.deepcopy(audit)
                if change:
                    altered["processor_record"]["input_tensor_sha256"] = {"test": "changed"}
                with self.assertRaises(ValueError):
                    ko.reuse_condition(root, context, spec, baseline, altered, runtime)


class ToyExecutorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_real_capture_and_patch_functions_on_toy_decoder(self):
        model = ToyText()
        inputs = {condition: toy_inputs() for condition in ex.CONDITIONS}
        inputs[ex.CONDITIONS[1]][0][0, 1:3] += .4
        with torch.inference_mode():
            scores = [float(model(*inputs[c])[0, -1, 0]) for c in ex.CONDITIONS]
        midpoint = sum(scores)/2
        def forward(live):
            value = model(*live["inputs"])[0, -1, 0]-midpoint
            return torch.tensor([float(value), 0., -100.])
        engine = SimpleNamespace(torch=torch, text=model, processor=None, model=None,
                                 runtime={"engine": "toy"}, forward=forward,
                                 decide=lambda logits, row: decision(logits, row["correct_option"]))
        groups = feature_tests.CompactCaptureTest().groups()
        groups["query_all"] = [3]
        record = {"input_ids": list(range(6)), "visual_positions": [1, 2]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = pairs(1)
            plan = {"fingerprint": "toy", "case_ids": ["pair_1"]}
            context = plan, annotations, {"pair_1": {"support_audit": {"supports": {
                "whole_event2": {"low_positions": [1, 2], "temporal_positions": [1, 2]}}},
                "processor_records": {condition: {**record, "group_positions": {"options_all": [3, 4]}}
                                      for condition in ex.CONDITIONS}}}
            audits, refs, old_data = {}, {}, {}
            for condition in ex.CONDITIONS:
                audit_spec = next(s for s in ex.feature_tasks(plan, "audit") if s["condition"] == condition and s["view"] == "archived")
                audits[condition] = ex.commit(root, plan, audit_spec, {"processor_record": record, "groups": groups,
                    "input_fingerprint": digest([condition, "toy"]), "input_tensor_sha256": {"toy": condition}})
                live = {"row": annotations["pair_1"][condition], "inputs": inputs[condition],
                        "input_tensor_sha256": {"toy": condition}}
                controller, data = ex.observe(engine, live, audits[condition])
                refs[live["row"]["eval_id"]] = ex.seal({"eval_id": live["row"]["eval_id"], "decision": engine.decide(data["logits"], live["row"]),
                    "input_tensor_sha256": live["input_tensor_sha256"], "capture_index_sha256": "toy_index"})
                old_data[live["row"]["eval_id"]] = {"states": {(16, "block_output"): controller.donor16}, "logits": data["logits"]}
            atomic_write(root / "source_baselines.json", refs)
            def live_for(_root, _context, _engine, spec):
                condition = spec["condition"]
                return {"row": annotations["pair_1"][condition], "inputs": inputs[condition],
                        "input_tensor_sha256": {"toy": condition}}, audits[condition]
            with patch.object(ex, "audited_live", side_effect=live_for), \
                    patch.object(ex, "checked_source", side_effect=lambda ref, **kw: old_data[ref["eval_id"]]), \
                    patch("scripts.probe_attention_roi.standard_first_token", side_effect=lambda _m, x:
                        (int(forward({"inputs": x}).argmax()), forward({"inputs": x}))):
                for spec in ex.feature_tasks(plan, "capture"):
                    if spec["view"] != "archived":
                        continue
                    result, data = ex.capture_execute(root, context, engine, spec)
                    ex.commit(root, plan, spec, result, data)
                spec = ex.feature_tasks(plan, "patch")[0]
                result, patched = ex.patch_execute(root, context, engine, spec)
                baseline, before = ex.capture_reference(root, context, "pair_1", "low_boundary")
                self.assertTrue(result["same_block_timing_exact"])
                self.assertGreater(result["decision"]["margin"], baseline["decision"]["margin"])
                self.assertTrue(torch.equal(before["features"][ex.FEATURE_NAMES.index("option_pair:L16_output")],
                                            patched["features"][ex.FEATURE_NAMES.index("option_pair:L16_output")]))
                self.assertFalse(torch.equal(before["features"][ex.FEATURE_NAMES.index("decision:L17_output")],
                                             patched["features"][ex.FEATURE_NAMES.index("decision:L17_output")]))
                ko_spec = {"pair_id": "pair_1", "condition": "low_boundary", "view": "archived",
                           "query_group": "options_all", "key_group": "whole_event2", "control": "whole_event_reference",
                           "window": [16, 17, 18, 19]}
                masked, _ = ko.execute_condition(root, context, engine, ko_spec, {})
                ko.verify_condition(masked, ko_spec, context[2]["pair_1"], baseline)
                self.assertTrue(all(value["max_blocked_probability"] == 0 for value in masked["mask_audit"]["layers"].values()))
                self.assertTrue(all(not block._forward_hooks and not block.self_attn._forward_hooks for block in model.layers))

    def test_fresh_processor_audit_of_pure_option_permutation(self):
        from scripts.phase3b_core import build_text_groups
        tokenizer = CharacterTokenizer()
        class Inputs(dict):
            def __getattr__(self, key):
                return self[key]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "frozen.mp4"
            video.write_bytes(b"synthetic fixture, not real video")
            annotations = pairs(1)
            def prepare(row, *unused, **kwargs):
                text = ("Which statement correctly describes the order of events?\n\n"
                        f"A: {row['option_A']}\nB: {row['option_B']}\n\nAnswer with only A or B.")
                ids = [151656, 151656] + tokenizer(text)["input_ids"]
                groups = {**build_text_groups(row, ids, tokenizer), "video_t1_e1": [0], "video_t1_e2": [0],
                          "video_t2_e1": [1], "video_t2_e2": [1]}
                inputs = Inputs(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                                pixel_values_videos=torch.ones(2, 4), video_grid_thw=torch.tensor([[1, 2, 2]]))
                return {"row": row, "inputs": inputs, "groups": groups, "video_metadata": {"fps": 15}, "video_path": str(video)}
            original = annotations["pair_1"]["low_boundary"]
            record = ex.processor_record(prepare(original))
            mapping = {"processor_records": {"low_boundary": {**record}}, "support_audit": {"supports": {
                "whole_event2": {"low_positions": [0, 1]}}}}
            context = {"fingerprint": "test", "settings": {"video_fps": None, "video_num_frames": None,
                "video_max_pixels": None, "roi_padding": 8}, "project_root": directory, "source_path_map": {}}, annotations, {"pair_1": mapping}
            atomic_write(root / "source_baselines.json", {original["eval_id"]: {"input_tensor_sha256": record["input_tensor_sha256"]}})
            spec = {"pair_id": "pair_1", "condition": "low_boundary", "view": "option_order_permuted"}
            with patch("scripts.phase3b_core.prepare_example", side_effect=prepare):
                result, _ = ex.audit_execute(root, context, SimpleNamespace(tokenizer=tokenizer), spec)
            self.assertFalse(result["require_rescue_behavior"])
            self.assertNotEqual(result["input_fingerprint"], digest(record["input_tensor_sha256"]))
            self.assertEqual(result["processor_record"]["input_tensor_sha256"]["pixel_values_videos"], record["input_tensor_sha256"]["pixel_values_videos"])
            self.assertTrue(result["groups"]["option_A_relation"])


class LifecycleTest(unittest.TestCase):
    def exercise(self, code=0, backup_ok=True, idle_failure=False, stale_attempt=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name) / "run"
        root.mkdir()
        args = SimpleNamespace(output_root=str(root), storage_root=directory.name, backup_dir=str(Path(directory.name)/"backups"),
            project_root=str(Path(__file__).resolve().parents[1]), gpus="0,1", stage="full", attempt_id="new-attempt",
            pause_after=True, email_notify=True, retry_failed=False, probe_python="python", preflight_subset=False)
        events = []
        notifier, client = Mock(), Mock()
        notifier.send.side_effect = lambda *a: events.append("email") or {"accepted": True}
        client.request_pause.side_effect = lambda: events.append("pause") or {"accepted": True}
        def child(*unused):
            events.append("child")
            atomic_write(root / "last_attempt.json", {"stage": "full", "complete": True,
                "attempt_id": "old-attempt" if stale_attempt else "new-attempt"})
            return code
        def pack(*unused, **kwargs):
            events.append("backup")
            return Path(directory.name) / "bundle"
        def idle():
            if idle_failure and "backup" in events:
                raise RuntimeError("another job is active")
        manifest = {"schema": "phase3d_backup_v1", "complete": backup_ok, "reports_only": False,
                    "activation_tensors_included": True}
        with patch.object(vm.execution, "load_context", return_value=({"fingerprint": "plan"}, {}, {})), \
                patch.object(vm, "storage_gate"), patch.object(vm, "verify_complete"), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = vm.job(args, child=child, pack=pack, verify=lambda _: manifest,
                            idle=idle, service_factory=lambda _: (notifier, client))
        return result, events, read_json(vm.lifecycle_dir(args) / "job_status.json")

    def test_success_orders_verified_backup_then_email_then_pause(self):
        result, events, status = self.exercise()
        self.assertEqual(result, 0)
        self.assertEqual(events, ["child", "backup", "email", "pause"])
        self.assertTrue(status["backup_verified"])
        self.assertFalse(status["local_backup_confirmed"])
        self.assertTrue(status["pause_not_confirmed"])

    def test_failure_also_backs_up_and_notifies_before_optional_pause(self):
        result, events, status = self.exercise(code=1)
        self.assertEqual(result, 1)
        self.assertEqual(events, ["child", "backup", "email", "pause"])
        self.assertFalse(status["experiment_success"])

    def test_bad_backup_or_nonidle_vm_blocks_pause(self):
        for options in ({"backup_ok": False}, {"idle_failure": True}):
            with self.subTest(options=options):
                result, events, status = self.exercise(**options)
                self.assertEqual(result, 1)
                self.assertNotIn("pause", events)
                self.assertTrue(status["pause_not_confirmed"])

    def test_old_success_is_not_success_for_this_attempt(self):
        result, _, status = self.exercise(stale_attempt=True)
        self.assertEqual(result, 1)
        self.assertFalse(status["experiment_success"])

    def test_real_multipart_backup_preserves_evidence_rejects_private_files_and_symlinks(self):
        from scripts.backup_phase3b import create_backup, verify_backup
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)/"run"
            root.mkdir()
            video = Path(directory)/"source.mp4"
            video.write_bytes(b"synthetic fixture")
            plan = {"schema": ex.SCHEMA, "artifact_type": "real", "video_sha256": {str(video): file_hash(video)}}
            plan["fingerprint"] = digest(plan)
            atomic_write(root/"plan.json", plan)
            save_tensors(root/"capture/payloads/test.pt", {"features": torch.ones(2, 3)})
            bundle = create_backup(root, Path(directory)/"backups", part_bytes=300,
                bundle_prefix="phase3c_backup", file_provider=vm.backup_files, metadata={"schema": "phase3d_backup_v1"})
            verified = verify_backup(bundle)
            self.assertEqual(verified["schema"], "phase3d_backup_v1")
            self.assertTrue(any(row["path"].endswith("test.pt") for row in verified["files"]))
            self.assertTrue(any(row["path"].startswith("source_videos/") for row in verified["files"]))
            atomic_write(root/"private_config.json", {"synthetic": "must not be archived"})
            with self.assertRaisesRegex(ValueError, "private"):
                vm.backup_files(root, False)
            (root/"private_config.json").unlink()
            (root/"capture/link").symlink_to(video)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                vm.backup_files(root, False)


if __name__ == "__main__":
    unittest.main()
