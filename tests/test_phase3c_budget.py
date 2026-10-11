import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import phase3c_budget as protocol
from scripts import run_phase3c_budget as runner
from scripts.phase3c_core import CONDITIONS, atomic_write, digest, event_layout, read_json, support_record
from tests.test_phase3c_support import fixture, result_fixture


def big_fixture():
    pair, mapping = fixture()
    for condition, offset in zip(CONDITIONS, (100, 700)):
        record = mapping["processor_records"][condition]
        record["video_metadata"].update(merged_video_grid_thw=[2, 16, 16], visual_token_count=512)
        record.update(visual_positions=list(range(offset, offset + 512)), prompt_token_count=2000)
    whole = support_record(list(range(100, 612)), list(range(700, 1212)), list(range(100, 612)), list(range(700, 1212)), "whole")
    target = support_record([100, 356, 357], [700, 956, 957], [100, 356, 357], [700, 956, 957], "targets")
    mapping["support_audit"]["supports"].update(whole_event2=whole, both_targets_event2=target)
    return pair, mapping


class BudgetSupportTest(unittest.TestCase):
    def test_exact_counts_target_membership_and_nested_support(self):
        pair, mapping = big_fixture()
        old = copy.deepcopy(mapping)
        audit = protocol.audit_supports("case_1", pair, mapping)
        self.assertEqual(mapping, old)
        self.assertEqual(len(audit["supports"]), 18)
        targets = mapping["support_audit"]["supports"]["both_targets_event2"]
        for seed in protocol.SEEDS:
            for family in protocol.FAMILIES:
                before = set()
                for budget in protocol.BUDGETS:
                    name = protocol.support_name(family, budget, seed)
                    record = audit["supports"][name]
                    self.assertEqual(record["mapped_token_count"], budget)
                    selected = set(record["low_positions"])
                    self.assertLessEqual(before, selected)
                    before = selected
                    if family == "context_only":
                        self.assertFalse(selected & set(targets["low_positions"]))
                    else:
                        self.assertLessEqual(set(targets["low_positions"]), selected)
                    self.assertEqual([x - 100 for x in record["low_positions"]], [x - 700 for x in record["temporal_positions"]])

    def test_quota_matched_within_each_temporal_bin(self):
        pair, mapping = big_fixture()
        audit = protocol.audit_supports("case_1", pair, mapping)
        layout = event_layout(pair[CONDITIONS[0]], mapping["processor_records"][CONDITIONS[0]]["video_metadata"],
            mapping["processor_records"][CONDITIONS[0]]["visual_positions"], 2000)
        for row in audit["token_budget_by_bin"]:
            for family in protocol.FAMILIES:
                support = audit["supports"][protocol.support_name(family, row["budget"], row["seed"])]
                count = sum(layout["lookup"][position][0] == row["low_temporal_index"] for position in support["low_positions"])
                self.assertEqual(count, row["quota_each_family"])

    def test_seed_selection_is_deterministic_and_effect_blind(self):
        pair, mapping = big_fixture()
        before = protocol.audit_supports("case_1", pair, mapping)
        pair[CONDITIONS[0]]["margin"] = -1000
        pair[CONDITIONS[1]]["patch_effect"] = 1000
        self.assertEqual(before, protocol.audit_supports("case_1", pair, mapping))
        seed_sets = [before["supports"][protocol.support_name("context_only", 64, seed)]["low_positions"] for seed in protocol.SEEDS]
        self.assertNotEqual(seed_sets[0], seed_sets[1])

    def test_target_inclusion_is_not_silently_reduced(self):
        pair, mapping = fixture()
        with self.assertRaisesRegex(ValueError, "quota"):
            protocol.audit_supports("case_1", pair, mapping, budgets=(2,), seeds=(42,))

    def test_context_capacity_shortfall_is_not_filled_with_targets(self):
        pair, mapping = fixture()
        with self.assertRaisesRegex(ValueError, "quota"):
            protocol.audit_supports("case_1", pair, mapping, budgets=(16,), seeds=(42,))

    def test_invalid_or_duplicate_budgets_rejected(self):
        pair, mapping = fixture()
        for budgets in ((4, 4), (0,), (17,)):
            with self.subTest(budgets=budgets), self.assertRaises(ValueError):
                protocol.audit_supports("case_1", pair, mapping, budgets=budgets, seeds=(42,))

    def test_temporal_correspondence_not_inferred_from_shape(self):
        pair, mapping = fixture()
        whole = mapping["support_audit"]["supports"]["whole_event2"]
        whole["temporal_positions"] = whole["temporal_positions"][8:] + whole["temporal_positions"][:8]
        mapping["support_audit"]["supports"]["both_targets_event2"]["temporal_positions"] = [148, 140, 141]
        with self.assertRaisesRegex(ValueError, "temporal-bin"):
            protocol.audit_supports("case_1", pair, mapping, budgets=(4,), seeds=(42,))

    def test_remainder_quota_allocation_is_exact_and_deterministic(self):
        self.assertEqual(protocol.allocate_budget(5, {(0, 0): 4, (1, 1): 6}), {(0, 0): 2, (1, 1): 3})
        self.assertEqual(protocol.allocate_budget(3, {(0, 0): 8, (1, 1): 8}), {(0, 0): 2, (1, 1): 1})

    def test_full_grid_unique_counts_no_old_support_rerun(self):
        pair, mapping = big_fixture()
        audit = protocol.audit_supports("case_1", pair, mapping)
        config = {"case_ids": [f"case_{i}" for i in range(12)], "preflight_case_ids": ["case_0", "case_1"],
            "support_audit": {f"case_{i}": audit for i in range(12)}}
        for stage, expected in (("replay", 24), ("preflight", 144), ("patch", 864)):
            tasks = protocol.task_grid(config, stage)
            self.assertEqual(len(tasks), expected)
            self.assertEqual(len({task["task_id"] for task in tasks}), expected)
            if stage != "replay":
                self.assertEqual({row["layer"] for row in tasks}, {12, 16})
                self.assertFalse({row["support"] for row in tasks} & {"whole_event2", "both_targets_event2"})


def seed_rows():
    return [{"pair_id": "case_1", "base_sample_id": 1, "analysis_stratum": "rescue", "first_object_id": 2,
        "direction": "temporal_to_low", "layer": 16, "family": "context_only", "budget": 64,
        "sampling_seed": seed, "margin_delta": value, "source_aligned_margin_delta": value,
        "recovery": value / 3, "strict_incorrect_to_correct": value > 1, "strict_sign_crossing": value > 1,
        "categorical_flip": value > 1, "replaced_token_count": 64}
        for seed, value in zip(protocol.SEEDS, (0.0, 1.0, 2.0))]


class SeedStatisticsTest(unittest.TestCase):
    def test_seeds_collapse_to_one_case_before_bootstrap(self):
        result = protocol.collapse_seeds(seed_rows())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["seed_count"], 3)
        self.assertEqual(result[0]["margin_delta"], 1)
        self.assertEqual(result[0]["strict_incorrect_to_correct"], 1 / 3)
        summary = runner.shared.aggregate(result, ("family", "budget", "layer"), ("margin_delta",))
        self.assertEqual(summary[0]["case_count"], 1)

    def test_incomplete_or_duplicate_seeds_fail(self):
        for rows in (seed_rows()[:2], seed_rows() + seed_rows()[:1]):
            with self.assertRaises(ValueError):
                protocol.collapse_seeds(rows)

    def test_null_recovery_denominator_preserved(self):
        rows = seed_rows()
        for row in rows:
            row["recovery"] = None
        self.assertIsNone(protocol.collapse_seeds(rows)[0]["recovery"])
        rows[0]["recovery"] = 1
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            protocol.collapse_seeds(rows)

    def test_full_matrix_keeps_eight_rescue_and_four_stable_units(self):
        rows = []
        for base in range(12):
            for direction in ("temporal_to_low", "low_to_temporal"):
                for layer in protocol.LAYERS:
                    for family in protocol.FAMILIES:
                        for budget in protocol.BUDGETS:
                            rows.extend({**row, "pair_id": f"case_{base}", "base_sample_id": base,
                                "analysis_stratum": "rescue" if base < 8 else "stable",
                                "direction": direction, "layer": layer, "family": family, "budget": budget,
                                "replaced_token_count": budget} for row in seed_rows())
        self.assertEqual(len(rows), 864)
        cases = protocol.collapse_seeds(rows)
        self.assertEqual(len(cases), 288)
        groups = runner.shared.aggregate(cases, ("analysis_stratum", "direction", "layer", "family", "budget"), ("margin_delta",))
        self.assertEqual(len(groups), 48)
        for row in groups:
            self.assertEqual(row["case_count"], 8 if row["analysis_stratum"] == "rescue" else 4)


def runtime_fixture():
    return {"torch": "pinned", "precision": "fp16", "model_device_map": {"layer0": "0"},
        "gpu_hardware": {"cuda_runtime": "12.8", "device_count": 2, "devices": [
            {"name": "A10", "compute_capability": [8, 6], "total_memory_bytes": 100},
            {"name": "A10", "compute_capability": [8, 6], "total_memory_bytes": 100}]}}


def replay_fixture():
    config = {"fingerprint": "frozen", "case_ids": ["case_1"], "preflight_case_ids": ["case_1"],
              "expected_replay_signatures": {}}
    pairs = {"case_1": {condition: {"eval_id": condition} for condition in CONDITIONS}}
    baseline = {"decision": {"margin": -1, "prediction": "B"}, "input_tensor_sha256": {"input": "original"}}
    baselines = {condition: copy.deepcopy(baseline) for condition in CONDITIONS}
    signature = {"logits": {"sha256": "logits"}, "positions": [1, 2], "sites": {"12": {"sha256": "vectors12"}, "16": {"sha256": "vectors16"}}}
    config["expected_replay_signatures"] = {condition: copy.deepcopy(signature) for condition in CONDITIONS}
    task = protocol.task_grid(config, "replay")[0]
    row = {"passed": True, "spec": task, "task_id": task["task_id"], "support_fingerprint": "frozen",
        "replay_signatures": copy.deepcopy(signature), "decision": copy.deepcopy(baseline["decision"]),
        "input_tensor_sha256": baseline["input_tensor_sha256"], "is_primary_effect_estimate": False,
        "hook_audit": {"block_output_sites": 36, "post_deepstack_layers": [0, 1, 2]}}
    return config, pairs, baselines, task, row


class ReplayAndResumeTest(unittest.TestCase):
    def test_budget_identity_uses_actual_frozen_positions_and_zero_difference(self):
        _, _, baselines, _, row = result_fixture()
        pair, mapping = big_fixture()
        config = {"fingerprint": "frozen", "case_ids": ["case_1"], "preflight_case_ids": ["case_1"],
            "support_audit": {"case_1": protocol.audit_supports("case_1", pair, mapping)}}
        task = protocol.task_grid(config, "preflight")[0]
        selected = config["support_audit"]["case_1"]["supports"][task["support"]]
        row.update(spec=task, task_id=task["task_id"], support_audit=selected,
                   donor_positions=selected["low_positions"], recipient_positions=selected["low_positions"])
        self.assertEqual(runner.result_failures(row, task, config, {"case_1": pair}, baselines), [])
        row["recipient_positions"] = [1]
        self.assertTrue(runner.result_failures(row, task, config, {"case_1": pair}, baselines))
        row["recipient_positions"] = selected["low_positions"]
        row["noop_parity"] = {"exact_match": False, "max_abs_diff": 0.001}
        self.assertIn("failed_exact_noop", runner.result_failures(row, task, config, {"case_1": pair}, baselines))
    def test_memory_capacity_only_is_recorded_not_other_runtime_changes(self):
        before = runtime_fixture()
        after = copy.deepcopy(before)
        after["gpu_hardware"]["devices"][1]["total_memory_bytes"] = 110
        check = protocol.runtime_compatibility(before, after)
        self.assertTrue(check["exact_replay_required"])
        self.assertEqual(check["memory_capacity_changes"][0]["device"], 1)
        self.assertEqual(after["gpu_hardware"]["devices"][1]["total_memory_bytes"], 110)
        after["model_device_map"] = {"layer0": "1"}
        with self.assertRaisesRegex(ValueError, "placement"):
            protocol.runtime_compatibility(before, after)

    def test_gpu_name_and_compute_capability_change_rejected(self):
        before = runtime_fixture()
        for field, value in (("name", "A100"), ("compute_capability", [8, 0])):
            after = copy.deepcopy(before)
            after["gpu_hardware"]["devices"][1][field] = value
            with self.assertRaises(ValueError):
                protocol.runtime_compatibility(before, after)

    def test_actual_replay_requires_exact_logits_and_vectors(self):
        config, pairs, baselines, task, row = replay_fixture()
        self.assertEqual(runner.result_failures(row, task, config, pairs, baselines), [])
        for mutate in (lambda r: r["replay_signatures"]["sites"]["16"].update(sha256="changed"),
                       lambda r: r["replay_signatures"]["logits"].update(sha256="changed"),
                       lambda r: r.update(input_tensor_sha256={}), lambda r: r["decision"].update(margin=0)):
            changed = copy.deepcopy(row)
            mutate(changed)
            self.assertTrue(runner.result_failures(changed, task, config, pairs, baselines))

    def test_passed_flag_cannot_override_corrupt_replay(self):
        config, pairs, baselines, task, row = replay_fixture()
        row["replay_signatures"]["sites"]["12"]["sha256"] = "changed"
        with tempfile.TemporaryDirectory() as root:
            atomic_write(Path(root) / "replay/task_checkpoints" / f"{task['task_id']}.json", row)
            with self.assertRaisesRegex(ValueError, "claimed-passed"):
                runner.saved_rows(root, "replay", config, pairs, baselines)

    def test_incomplete_replay_blocks_preflight(self):
        config, pairs, baselines, _, _ = replay_fixture()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                runner.run_stage(SimpleNamespace(output_root=root), (config, {}, pairs, {}, baselines), "preflight")

    def test_complete_resume_does_not_load_model(self):
        config, pairs, baselines, task, row = replay_fixture()
        with tempfile.TemporaryDirectory() as root:
            atomic_write(Path(root) / "replay/task_checkpoints" / f"{task['task_id']}.json", row)
            with patch.object(protocol, "task_grid", return_value=[task]), patch.object(runner.shared, "Engine") as engine:
                self.assertTrue(runner.run_stage(SimpleNamespace(output_root=root), (config, {}, pairs, {}, baselines), "replay")["complete"])
                engine.assert_not_called()

    def test_other_budget_worker_blocks_pause(self):
        idle = runner.shared
        from scripts import run_phase3c_unattended as module
        process = SimpleNamespace(stdout="999999 python scripts/run_phase3c_budget.py --action run")
        with patch.object(module, "legacy_idle"), patch.object(module.subprocess, "run", return_value=process), \
             patch.object(module.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "active"):
                idle.assert_quiescent()


class BudgetLifecycleTest(unittest.TestCase):
    def run_job(self, root, child_code=0, stale=False, backup_fails=False):
        output = Path(root) / "new_budget"
        lifecycle = Path(str(output) + "_lifecycle")
        lifecycle.mkdir()
        (lifecycle / "job.log").write_text("public log", encoding="ascii")
        config = {"fingerprint": "frozen", "source_backup_manifest_sha256": "sourcehash"}
        args = SimpleNamespace(output_root=str(output), source_backup="source_backup", source_followup="prior",
            backup_dir=str(Path(root) / "backups"), pause_after=True, retry_failed=False, stage="full",
            project_root="repository", plan_dir="plan", source_run="source", storage_root=root,
            gpus="0,1", email_notify=True)
        order = []
        notifier, client = Mock(), Mock()
        notifier.send.side_effect = lambda *_: order.append("email") or {"smtp_accepted": True}
        client.request_pause.side_effect = lambda: order.append("pause") or {"request_accepted": True}

        def child(*_):
            order.append("child")
            if child_code == 0 and not stale:
                atomic_write(output / "last_attempt.json", {"attempt_id": args.attempt_id, "complete": True, "support_fingerprint": "frozen"})
            return child_code

        def backup(*_, **__):
            order.append("backup")
            if backup_fails:
                raise RuntimeError("no space")
            return Path(root) / "package"

        with patch.object(runner, "context", return_value=(config, {}, {}, {}, {})), \
             patch.object(runner, "saved_rows"), patch.object(runner, "private_services", return_value=(notifier, client)), \
             patch.object(runner.shared, "assert_quiescent", side_effect=lambda: order.append("idle")), \
             patch.object(runner.shared, "run_child", side_effect=child), patch.object(runner, "verify_analysis"), \
             patch.object(runner.shared, "create_backup", side_effect=backup), \
             patch.object(runner.shared, "verify_backup", side_effect=lambda *_: order.append("verify_backup") or
                          {"complete": True, "reports_only": False, "support_fingerprint": "frozen"}):
            code = runner.job(args)
        return code, order, read_json(lifecycle / "job_status.json")

    def test_success_backup_email_pause_order(self):
        with tempfile.TemporaryDirectory() as root:
            code, order, status = self.run_job(root)
            self.assertEqual(code, 0)
            self.assertEqual(order, ["idle", "child", "backup", "verify_backup", "idle", "email", "pause"])
            self.assertTrue(status["experiment_success"])

    def test_failure_preserves_evidence_and_pauses_only_after_backup(self):
        with tempfile.TemporaryDirectory() as root:
            code, order, status = self.run_job(root, child_code=1)
            self.assertEqual(code, 1)
            self.assertFalse(status["experiment_success"])
            self.assertLess(order.index("verify_backup"), order.index("pause"))

    def test_failed_backup_blocks_pause(self):
        with tempfile.TemporaryDirectory() as root:
            code, order, status = self.run_job(root, backup_fails=True)
            self.assertEqual(code, 1)
            self.assertNotIn("pause", order)
            self.assertEqual(status["state"], "needs_attention")

    def test_stale_analysis_cannot_claim_new_success(self):
        with tempfile.TemporaryDirectory() as root:
            code, _, status = self.run_job(root, stale=True)
            self.assertEqual(code, 1)
            self.assertFalse(status["experiment_success"])


if __name__ == "__main__":
    unittest.main()
