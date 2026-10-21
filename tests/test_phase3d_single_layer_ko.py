"""Synthetic ledgers and a toy attention model; not scientific experiment results."""

import copy
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from scripts import phase3d_execution as ex
from scripts import phase3d_single_layer_ko as followup
from scripts import phase3d_knockout as ko
from scripts import run_phase3d_single_layer_ko as vm
from scripts.analyze_phase3c import aggregate
from scripts.phase3c_core import atomic_write, digest, file_hash, read_json
from scripts.phase3c_knockout25 import positions
from scripts.run_phase3c_preflight import save_tensors
from tests.test_phase3d import pairs
from tests.test_phase3d_execution import decision, ledger_fixture
from tests.test_phase3c_preflight import ToyText, toy_inputs


def result_for(spec, mapping, baseline, logits, enabled):
    queries, keys, budget = positions(mapping, spec["condition"], spec)
    return {"decision": decision(logits), "baseline_record_sha256": baseline["record_sha256"],
        "input_fingerprint": baseline["input_fingerprint"], "audit_record_sha256": baseline["audit_record_sha256"],
        "query_positions": queries, "key_positions": keys, "edge_budget": budget,
        "mask_audit": {"enabled": enabled, "all_heads": 32, "pre_softmax_mask": "negative_infinity", "renormalized": True,
            "orientation": "text_queries_to_visual_keys",
            "layers": {str(layer): {"budget": budget, "max_blocked_probability": 0., "max_row_sum_error": 0.}
                       for layer in spec["window"]}}}


def source_fixture(root):
    context = ledger_fixture(root)
    plan, annotations, mappings = context
    plan.update(settings={"synthetic": True}, source_runtime={"synthetic": True},
                project_root=str(root.parent), source_path_map={}, video_sha256={})
    for pair_id in plan["case_ids"]:
        mappings[pair_id]["processor_records"] = {condition: {
            "input_ids": [151656]*512+[10, 20], "visual_positions": list(range(512)),
            "group_positions": {"options_all": [512]}} for condition in ex.CONDITIONS}
    for name, value in (("plan", plan), ("pairs", annotations), ("mappings", mappings),
                        ("runtime", {"plan_fingerprint": plan["fingerprint"], "runtime": {"synthetic": True}})):
        atomic_write(root / f"{name}.json", value)
    joint_specs = [s for s in ko.condition_specs(plan, annotations) if s["window"] == list(followup.LAYERS)
                   and all(s[k] == v for k, v in followup.ROUTE.items())]
    for spec in joint_specs:
        baseline, _ = ex.capture_reference(root, context, spec["pair_id"], spec["condition"])
        ex.commit(root, plan, spec, result_for(spec, mappings[spec["pair_id"]], baseline,
            torch.tensor([1., 2., -1.]), True))
    table = root / "knockout/analysis/case_decomposition.json"
    rows = []
    for pair_id in plan["case_ids"]:
        baselines = {c: ex.capture_reference(root, context, pair_id, c)[0] for c in ex.CONDITIONS}
        matching = [s for s in joint_specs if s["pair_id"] == pair_id]
        joint = {s["condition"]: ex.saved_checkpoint(root, plan, s) for s in matching}
        rows.append(followup.paired_outcome({"pair_id": pair_id, "window": list(followup.LAYERS)}, joint, baselines, annotations[pair_id]))
    atomic_write(table, rows)
    summary = {"complete": True, "artifact_type": "real", "plan_fingerprint": plan["fingerprint"],
        "case_count": 50, "condition_rows": 11700, "paired_rows": 5850, "missing_count": 0,
        "output_sha256": {table.name: file_hash(table)}}
    atomic_write(root / "knockout/analysis/aggregate_summary.json", summary)
    atomic_write(root / "aggregate_summary.json", {"complete": True, "artifact_type": "real", "base_count": 50,
        "plan_fingerprint": plan["fingerprint"], "ko_condition_rows": 11700, "ko_paired_rows": 5850,
        "knockout_summary_sha256": file_hash(root / "knockout/analysis/aggregate_summary.json")})
    return context


def complete_singletons(root, context):
    records = read_json(root / "source_records.json")
    original = torch.load(root / "source_logits.pt", map_location="cpu", weights_only=True)
    atomic_write(root/"runtime.json", {"runtime": context[0]["source_runtime"], "plan_fingerprint": context[0]["fingerprint"]})
    gates = {}
    for spec in followup.specs(context[0], "baseline_gate") + followup.specs(context[0]):
        key = spec["pair_id"], spec["condition"]
        baseline = records[key[0]][key[1]]["baseline"]
        is_gate = spec["stage"] == "baseline_gate"
        logits = original[key[0]][key[1]] if is_gate else torch.tensor(
            [1.5, 2., -1.] if key[1] == "low_boundary" else [1.5, 1., -1.])
        result = result_for(spec, context[2][key[0]], baseline, logits, not is_gate)
        payload = {"logits": logits}
        if is_gate:
            payload["baseline_logits"] = logits.clone()
        else:
            result["gate_record_sha256"] = gates[key]["record_sha256"]
        row = ex.commit(root, context[0], spec, result, payload)
        if is_gate:
            gates[key] = row


class GridTest(unittest.TestCase):
    def test_exact_scope_grouped_conditions_and_no_joint_rerun(self):
        annotations = pairs()
        plan = {"fingerprint": "test", "case_ids": list(annotations)}
        tasks = followup.specs(plan)
        self.assertEqual(len(tasks), 400)
        self.assertEqual(len({s["task_id"] for s in tasks}), 400)
        self.assertEqual({tuple(s["window"]) for s in tasks}, {(16,), (17,), (18,), (19,)})
        self.assertEqual({(s["query_group"], s["key_group"], s["control"]) for s in tasks},
                         {("options_all", "whole_event2", "whole_event_reference")})
        self.assertEqual(len(followup.specs(plan, "baseline_gate")), 100)
        self.assertEqual(len(followup.specs(plan, pair_ids=plan["case_ids"][:2])), 16)
        self.assertEqual([s["condition"] for s in tasks[:4]], ["low_boundary"]*4)
        for invalid in ([], ["unknown"], ["pair_1", "pair_1"]):
            with self.assertRaises(ValueError):
                followup.specs(plan, pair_ids=invalid)

    def test_actual_budget_not_old_hardcoded_four_and_outcome_decomposition(self):
        pair = pairs(1)["pair_1"]
        mapping = {"processor_records": {c: {"input_ids": list(range(8)), "group_positions": {"options_all": [5, 6]}}
            for c in ex.CONDITIONS}, "support_audit": {"supports": {"whole_event2": {
                "low_positions": [1, 2], "temporal_positions": [1, 2]}}}}
        before = {c: {"decision": decision(torch.tensor([1., 2., -1.]) if c == "low_boundary" else torch.tensor([3., 1., -1.])),
            "record_sha256": c, "input_fingerprint": "input", "audit_record_sha256": "audit"} for c in ex.CONDITIONS}
        for window in ([16], [16, 17, 18, 19]):
            spec = {"pair_id": "pair_1", "window": window, **followup.ROUTE}
            after = {c: {**result_for({**spec, "condition": c}, mapping, before[c], torch.tensor([2., 1., -1.]), True),
                "record_sha256": digest([c, window])} for c in ex.CONDITIONS}
            item = followup.paired_outcome(spec, after, before, pair)
            self.assertEqual(item["delta_M_low"], 2.)
            self.assertEqual(item["delta_M_temporal"], -1.)
            self.assertEqual(item["compression"], 3.)
            self.assertEqual(item["low_causal_edges_all_heads_window"], len(window)*2*2*32)
            self.assertTrue(item["low_strict_incorrect_to_correct"])
        after["temporal_boundary"]["decision"] = decision(torch.tensor([1., 1., 4.]))
        item = followup.paired_outcome(spec, after, before, pair)
        self.assertTrue(item["temporal_non_ab_first_token"])
        self.assertTrue(item["temporal_zero_margin_tie"])
        self.assertFalse(item["temporal_strict_correct_to_incorrect"])

    def test_full_logit_validation_and_real_non_ab_parser_convention(self):
        logits = torch.tensor([2., 1., 4.])
        measured = decision(logits)
        followup.check_logits(logits, measured)
        measured.update(predicted_first_token_text="?", prediction="UNKNOWN")
        followup.check_logits(logits, measured)
        for field, value in (("margin", 99), ("predicted_first_token_id", 0), ("is_correct", True)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                followup.check_logits(logits, {**measured, field: value})


class SourceAndAnalysisTest(unittest.TestCase):
    def test_source_is_frozen_readonly_and_complete_grid_is_independently_verified(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source, root = Path(directory)/"source_run", Path(directory)/"followup"
            source.mkdir()
            old = source_fixture(source)
            before = {str(p.relative_to(source)): file_hash(p) for p in source.rglob("*") if p.is_file()}
            with patch.object(ex, "load_context", return_value=old):
                context = followup.freeze_plan(source, root)
                self.assertEqual(context, followup.load_context(root))
                self.assertEqual(context, followup.freeze_plan(source, root))
                complete_singletons(root, context)
                self.assertEqual(len(followup.verify_rows(root, context)), 500)
                with patch.object(followup, "aggregate", side_effect=lambda r, k, m: aggregate(r, k, m, repeats=8)):
                    summary = followup.analyze(root, context)
                self.assertEqual(summary["new_condition_rows"], 400)
                self.assertEqual(summary["paired_rows_with_reference"], 250)
                cells = read_json(root/"analysis/aggregate_cells.json")
                self.assertEqual(len(cells), 5)
                self.assertEqual({cell["case_count"] for cell in cells}, {50})
                self.assertTrue((root/"analysis/single_layer_vs_joint.png").stat().st_size > 1000)
                table = read_json(root/"analysis/case_decomposition.json")
                singleton, joint = [next(r for r in table if r["measurement"] == kind) for kind in
                                    ("new_single_layer", "joint_reference")]
                self.assertEqual(4*singleton["low_causal_edges_all_heads_window"], joint["low_causal_edges_all_heads_window"])
                first = followup.specs(context[0])[0]
                path = ex.checkpoint_path(root, first)
                original = read_json(path)
                for change in ("mask", "input", "gate", "decision"):
                    altered = copy.deepcopy(original)
                    if change == "mask":
                        altered["mask_audit"]["layers"]["16"]["max_blocked_probability"] = .1
                    elif change == "input":
                        altered["input_fingerprint"] = "foreign"
                    elif change == "gate":
                        altered["gate_record_sha256"] = "foreign"
                    else:
                        altered["decision"]["margin"] = 123
                    atomic_write(path, ex.seal(altered))
                    with self.subTest(change=change), self.assertRaises(ValueError):
                        followup.verify_rows(root, context)
                atomic_write(path, original)
                path.unlink()
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    followup.verify_rows(root, context)
                atomic_write(path, original)
                atomic_write(source/"runtime.json", {"altered": True})
                with self.assertRaisesRegex(ValueError, "bytes changed"):
                    followup.load_context(root)
                atomic_write(source/"runtime.json", {"plan_fingerprint": old[0]["fingerprint"], "runtime": {"synthetic": True}})
            after = {str(p.relative_to(source)): file_hash(p) for p in source.rglob("*") if p.is_file()}
            self.assertEqual(before, after)

    def test_source_partial_or_missing_joint_and_failed_rescue_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source = Path(directory)/"source_run"
            source.mkdir()
            context = source_fixture(source)
            with patch.object(ex, "load_context", return_value=context):
                summary = read_json(source/"aggregate_summary.json")
                atomic_write(source/"aggregate_summary.json", {**summary, "complete": False})
                with self.assertRaisesRegex(ValueError, "complete"):
                    followup.source_snapshot(source)
                atomic_write(source/"aggregate_summary.json", summary)
                spec = next(s for s in ko.condition_specs(context[0], context[1]) if s["window"] == list(followup.LAYERS)
                            and all(s[k] == v for k, v in followup.ROUTE.items()))
                path = ex.checkpoint_path(source, spec)
                row = read_json(path)
                path.unlink()
                with self.assertRaisesRegex(ValueError, "Missing original joint"):
                    followup.source_snapshot(source)
                atomic_write(path, row)
                baseline, data = ex.capture_reference(source, context, spec["pair_id"], spec["condition"])
                changed = copy.deepcopy(baseline)
                changed["decision"] = decision(torch.tensor([3., 1., -1.]))
                atomic_write(ex.checkpoint_path(source, baseline["spec"]), ex.seal(changed))
                with self.assertRaises(ValueError):
                    followup.source_snapshot(source)


class ToyExecutionTest(unittest.TestCase):
    def test_real_singleton_hook_and_disabled_noop_resume_and_failure_ledger(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        model = ToyText()
        x = toy_inputs()
        def forward(live):
            with torch.inference_mode():
                value = float(model(*live["inputs"])[0, -1, 0])
            return torch.tensor([value, 0., -100.])
        engine = SimpleNamespace(torch=torch, text=model, runtime={"synthetic": True},
            forward=Mock(side_effect=forward), decide=lambda logits, row: decision(logits))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            annotations = pairs(1)
            plan = {"fingerprint": "toy only", "case_ids": ["pair_1"], "source_root": str(root/"source"),
                    "source_runtime": {"synthetic": True}}
            atomic_write(root/"runtime.json", {"runtime": plan["source_runtime"], "plan_fingerprint": plan["fingerprint"]})
            mapping = {"processor_records": {c: {"input_ids": list(range(6)), "group_positions": {"options_all": [3, 4]}}
                for c in ex.CONDITIONS}, "support_audit": {"supports": {"whole_event2": {
                    "low_positions": [1, 2], "temporal_positions": [1, 2]}}}}
            context = plan, annotations, {"pair_1": mapping}
            records, original, audits = {"pair_1": {}}, {"pair_1": {}}, {}
            for c in ex.CONDITIONS:
                logits = forward({"inputs": x})
                audit = ex.seal({"processor_record": {"synthetic": c}})
                baseline = ex.seal({"decision": decision(logits), "input_fingerprint": c, "audit_record_sha256": audit["record_sha256"]})
                records["pair_1"][c] = {"baseline": baseline, "audit": audit}
                original["pair_1"][c], audits[c] = logits, audit
            atomic_write(root/"source_records.json", records)
            save_tensors(root/"source_logits.pt", original)
            def live_for(_root, _context, _engine, spec):
                return {"row": annotations["pair_1"][spec["condition"]], "inputs": x}, audits[spec["condition"]]
            with patch.object(ex, "load_context", return_value=context), patch.object(ex, "audited_live", side_effect=live_for):
                execute = followup.Executor(root, context, engine)
                task = followup.specs(plan)[0]
                with self.assertRaisesRegex(ValueError, "gate"):
                    execute(task)
                self.assertEqual(engine.forward.call_count, 0)
                ex.run_tasks(root, context, followup.specs(plan, "baseline_gate"), execute)
                ex.run_tasks(root, context, followup.specs(plan), execute)
                rows = followup.verify_rows(root, context)
                for row in rows[2:]:
                    self.assertEqual(set(row["mask_audit"]["layers"]), {str(row["spec"]["window"][0])})
                calls = engine.forward.call_count
                ex.run_tasks(root, context, followup.specs(plan), execute)
                self.assertEqual(engine.forward.call_count, calls)
                self.assertEqual(read_json(root/"progress_status.json")["resumed"], 8)
                self.assertTrue(all(not layer.self_attn._forward_hooks and not layer.self_attn._forward_pre_hooks for layer in model.layers))
                ex.checkpoint_path(root, task).unlink()
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    ex.run_tasks(root, context, [task], Mock(side_effect=RuntimeError("synthetic failure")))
                with self.assertRaisesRegex(RuntimeError, "Earlier failure"):
                    ex.run_tasks(root, context, [task], execute)
                ex.run_tasks(root, context, [task], execute, retry_failed=True)
                self.assertTrue((root/"failures"/(task["task_id"]+".json")).exists())


class RunnerTest(unittest.TestCase):
    def fixture(self, root):
        annotations = pairs()
        plan = {"fingerprint": "synthetic runner", "source_plan_fingerprint": "synthetic source",
            "case_ids": list(annotations), "preflight_ids": ["pair_1", "pair_2"], "source_runtime": {"synthetic": True}}
        mapping = {"processor_records": {c: {"input_ids": list(range(8)), "group_positions": {"options_all": [6]}}
            for c in ex.CONDITIONS}, "support_audit": {"supports": {"whole_event2": {
                "low_positions": [1, 2, 3, 4], "temporal_positions": [1, 2, 3, 4]}}}}
        context = plan, annotations, {key: copy.deepcopy(mapping) for key in annotations}
        records, logits = {}, {}
        for pair_id in plan["case_ids"]:
            records[pair_id], logits[pair_id] = {}, {}
            for c in ex.CONDITIONS:
                measured = torch.tensor([1., 2., -1.] if c == "low_boundary" else [3., 1., -1.])
                baseline = ex.seal({"decision": decision(measured), "input_fingerprint": digest([pair_id, c]),
                    "audit_record_sha256": digest([pair_id, c, "audit"])})
                spec = {"pair_id": pair_id, "condition": c, "window": list(followup.LAYERS), **followup.ROUTE}
                joint = ex.seal({**result_for(spec, mapping, baseline, torch.tensor([1., 2., -1.]), True), "spec": spec})
                records[pair_id][c] = {"baseline": baseline, "joint": joint}
                logits[pair_id][c] = measured
        atomic_write(root/"source_records.json", records)
        save_tensors(root/"source_logits.pt", logits)
        atomic_write(root/"runtime.json", {"plan_fingerprint": plan["fingerprint"], "runtime": plan["source_runtime"]})
        return context, records, logits

    def test_preflight_then_full_exactly_400_new_kos_with_no_duplicate_resume(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)/"run"
            root.mkdir()
            context, records, original = self.fixture(root)
            args = SimpleNamespace(output_root=str(root), stage="full", attempt_id="test", retry_failed=False)
            called = []
            def execute(spec):
                called.append(spec)
                key, c = spec["pair_id"], spec["condition"]
                gate = spec["stage"] == "baseline_gate"
                logits = original[key][c] if gate else torch.tensor([1.5, 2., -1.] if c == "low_boundary" else [1.5, 1., -1.])
                row = result_for(spec, context[2][key], records[key][c]["baseline"], logits, not gate)
                data = {"logits": logits}
                if gate:
                    data["baseline_logits"] = logits
                else:
                    gate_spec = next(s for s in followup.specs(context[0], "baseline_gate", [key]) if s["condition"] == c)
                    row["gate_record_sha256"] = ex.saved_checkpoint(root, context[0], gate_spec)["record_sha256"]
                return row, data
            with patch.object(followup, "load_context", return_value=context), patch.object(vm, "storage_gate"), \
                    patch.object(vm, "make_engine") as engine, patch.object(followup, "Executor", return_value=execute), \
                    patch.object(followup, "aggregate", side_effect=lambda r, k, m: aggregate(r, k, m, repeats=8)):
                with self.assertRaisesRegex(ValueError, "preflight first"):
                    vm.run(args)
                engine.assert_not_called()
                args.stage = "preflight"
                summary = vm.run(args)
                self.assertFalse(summary["is_full_experiment"])
                self.assertEqual(len(called), 20)
                args.stage = "full"
                summary = vm.run(args)
                self.assertEqual(summary["new_condition_rows"], 400)
                self.assertEqual(sum(s["stage"] == "knockout" for s in called), 400)
                self.assertEqual(sum(s["stage"] == "baseline_gate" for s in called), 100)
                self.assertEqual(len({s["task_id"] for s in called}), 500)
                vm.run(args)
                self.assertEqual(len(called), 500)
                self.assertEqual(engine.call_count, 2)

    def test_failed_independent_gate_blocks_all_interventional_tasks(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)/"run"
            root.mkdir()
            context, records, original = self.fixture(root)
            args = SimpleNamespace(output_root=str(root), stage="preflight", attempt_id="test", retry_failed=False)
            calls = []
            def bad_gate(spec):
                calls.append(spec)
                key, c = spec["pair_id"], spec["condition"]
                logits = torch.tensor([9., 1., -1.])
                return result_for(spec, context[2][key], records[key][c]["baseline"], logits, False), {
                    "logits": logits, "baseline_logits": original[key][c]}
            with patch.object(followup, "load_context", return_value=context), patch.object(vm, "storage_gate"), \
                    patch.object(vm, "make_engine"), patch.object(followup, "Executor", return_value=bad_gate):
                with self.assertRaisesRegex(ValueError, "disabled-hook"):
                    vm.run(args)
                self.assertEqual({s["stage"] for s in calls}, {"baseline_gate"})
                self.assertFalse((root/"last_attempt.json").exists())


class LifecycleTest(unittest.TestCase):
    def exercise(self, code=0, good_backup=True, stale=False, busy=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)/"run"
        root.mkdir()
        args = SimpleNamespace(output_root=str(root), storage_root=temp.name, backup_dir=str(Path(temp.name)/"backup"),
            project_root=str(Path(__file__).resolve().parents[1]), stage="full", attempt_id="current", gpus="0,1",
            retry_failed=False, pause_after=True, email_notify=True)
        events, notifier, client = [], Mock(), Mock()
        notifier.send.side_effect = lambda *a: events.append("email") or {}
        client.request_pause.side_effect = lambda: events.append("pause") or {}
        def child(*unused):
            events.append("child")
            atomic_write(root/"last_attempt.json", {"complete": True, "stage": "full", "plan_fingerprint": "plan",
                "attempt_id": "old" if stale else "current"})
            return code
        def pack(*unused, **kwargs):
            events.append("backup")
            self.assertEqual(kwargs["bundle_prefix"], "phase3c_backup")
            return Path(temp.name)/"bundle"
        def idle():
            if busy and "backup" in events:
                raise RuntimeError("another task is active")
        with patch.object(followup, "load_context", return_value=({"fingerprint": "plan"}, {}, {})), \
                patch.object(vm, "require_preflight"), patch.object(vm, "storage_gate"), \
                patch.object(followup, "analyze"), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = vm.job(args, child=child, pack=pack, idle=idle, service_factory=lambda _: (notifier, client),
                verify=lambda _: {"schema": followup.SCHEMA+"_backup", "complete": good_backup,
                                  "reports_only": False, "activation_tensors_included": True})
        return result, events, read_json(vm.lifecycle_dir(args)/"job_status.json")

    def test_success_and_failure_back_up_notify_then_pause(self):
        for code in (0, 1):
            with self.subTest(code=code):
                result, events, status = self.exercise(code=code)
                self.assertEqual(result, code)
                self.assertEqual(events, ["child", "backup", "email", "pause"])
                self.assertEqual(status["experiment_success"], code == 0)
                self.assertFalse(status["local_backup_confirmed"])
                self.assertTrue(status["pause_not_confirmed"])

    def test_stale_success_invalid_backup_and_busy_vm(self):
        result, _, status = self.exercise(stale=True)
        self.assertEqual(result, 1)
        self.assertFalse(status["experiment_success"])
        for options in ({"good_backup": False}, {"busy": True}):
            with self.subTest(options=options):
                result, events, _ = self.exercise(**options)
                self.assertEqual(result, 1)
                self.assertNotIn("pause", events)

    def test_real_backup_checksums_and_private_file_rejection(self):
        from scripts.backup_phase3b import create_backup, verify_backup
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)/"run"
            root.mkdir()
            video = Path(directory)/"video.mp4"
            video.write_bytes(b"synthetic video")
            atomic_write(root/"source_plan.json", {"video_sha256": {str(video): file_hash(video)}})
            save_tensors(root/"source_logits.pt", {"synthetic": torch.ones(3)})
            bundle = create_backup(root, Path(directory)/"backup", part_bytes=300, file_provider=vm.backup_files,
                bundle_prefix="phase3c_backup", metadata={"schema": followup.SCHEMA+"_backup"})
            manifest = verify_backup(bundle)
            self.assertTrue(manifest["complete"])
            self.assertTrue(any(f["path"].endswith("source_logits.pt") for f in manifest["files"]))
            atomic_write(root/"email.json", {"synthetic": "not a credential"})
            with self.assertRaisesRegex(ValueError, "private"):
                vm.backup_files(root, False)
            (root/"email.json").unlink()
            (root/"link").symlink_to(video)
            with self.assertRaisesRegex(ValueError, "symlink"):
                vm.backup_files(root, False)

    def test_low_disk_space_fails_without_deleting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"run"
            root.mkdir()
            retained = root/"keep.txt"
            retained.write_text("retain")
            args = SimpleNamespace(output_root=str(root), storage_root=directory, backup_dir=str(Path(directory)/"backup"))
            with patch("os.path.ismount", return_value=True), patch("shutil.disk_usage", return_value=SimpleNamespace(free=1)):
                with self.assertRaisesRegex(RuntimeError, "No evidence deleted"):
                    vm.storage_gate(args)
            self.assertEqual(retained.read_text(), "retain")


if __name__ == "__main__":
    unittest.main()
