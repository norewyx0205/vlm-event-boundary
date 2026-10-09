import copy
import subprocess
import sys
import tempfile
import unittest
import torch
from pathlib import Path
from unittest.mock import patch

import test_phase3c_preflight as fixtures
from scripts import analyze_phase3c as analysis, phase3c_core as core, phase3c_execution as execution
from scripts import phase3c_primary as primary, run_phase3c_preflight as runner


def decision(margin):
    return {"margin": margin, "correct_logit": 2.0 + margin, "incorrect_logit": 2.0,
            "prediction": "A" if margin >= 0 else "B", "correct_option": "A", "is_correct": margin >= 0}


def synthetic_result(prepared, mapping, task):
    """Test-only predictions; never model evidence or on-disk scientific fixtures."""
    annotation = prepared["row"]
    condition = task["condition"]
    before = decision(1.0 if condition == core.CONDITIONS[1] or annotation["phase3c_analysis_stratum"] == "stable" else -1.0)
    result = {"spec": task, "passed": True, "baseline_decision": before, "decision": before,
        "prompt_token_count": 60, "input_tensor_sha256": {"input_ids": {"sha256": "a" * 64}},
        "is_primary_effect_estimate": task["kind"] not in ("routing_baseline", "identity", "transplant_smoke", "disabled_knockout", "knockout_smoke")}
    if task["kind"] in ("visual_patch", "identity", "transplant_smoke"):
        donor_condition = condition if task["kind"] == "identity" else core.CONDITIONS[1] if condition == core.CONDITIONS[0] else core.CONDITIONS[0]
        support = mapping["support_audit"]["supports"][task["support"]]
        side = "low_positions" if condition == core.CONDITIONS[0] else "temporal_positions"
        donor_side = "low_positions" if donor_condition == core.CONDITIONS[0] else "temporal_positions"
        result.update({"support_audit": support, "recipient_positions": support[side], "donor_positions": support[donor_side],
            "donor_condition": donor_condition, "recipient_condition": condition,
            "donor_capture_location": task["location"], "recipient_patch_location": task["location"],
            "hook_audit": {"block_output_sites": 36, "post_deepstack_layers": [0, 1, 2], "patch_applied_count": 1}})
        if task["kind"] != "identity":
            result["decision"] = decision(before["margin"] + (0.25 if condition == core.CONDITIONS[0] else -0.25))
        if task["kind"] == "visual_patch":
            donor = decision(1.0 if donor_condition == core.CONDITIONS[1] or annotation["phase3c_analysis_stratum"] == "stable" else -1.0)
            result["donor_baseline_decision"] = donor
            result.update(primary.patch_outcomes(before["margin"], result["decision"]["margin"], donor["margin"]))
    elif task["kind"] == "routing_baseline":
        result["routes"] = []
        for route in primary.routes(mapping, condition):
            queries, keys, budget = primary.route_positions(mapping, condition, route)
            result["routes"].append({**route, "query_positions": queries, "key_positions": keys, "edge_budget": budget,
                "mask_audit": {"enabled": False, "all_heads": 32, "layers": {str(layer): {
                    "budget": budget, "mean_selected_edge_mass_per_query_head": 0.05,
                    "max_row_sum_error": 0, "max_blocked_probability": 0.01} for layer in range(36)}}})
    else:
        queries, keys, budget = primary.route_positions(mapping, condition, task)
        enabled = task["kind"] != "disabled_knockout"
        result.update({"query_positions": queries, "key_positions": keys, "edge_budget": budget,
            "mask_audit": {"enabled": enabled, "all_heads": 32, "orientation": "text_queries_to_visual_keys", "renormalized": True,
                "layers": {str(layer): {"budget": budget, "max_row_sum_error": 0,
                    "max_blocked_probability": 0 if enabled else 0.01} for layer in task["window"]}}})
        if enabled:
            change = -0.25 if condition == core.CONDITIONS[1] else 0.05
            if task["control"] == "background":
                change /= 5
            result["decision"] = decision(before["margin"] + change)
    result["margin_delta"] = result["decision"]["margin"] - before["margin"]
    if task["kind"] in ("identity", "disabled_knockout", "routing_baseline"):
        result["noop_parity"] = {"exact_match": True, "max_abs_diff": 0}
    return result


class Phase3CPrimaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.prior_threads)

    def setup_run(self, directory):
        fixture = fixtures.Phase3CGateTest()
        plan = fixture.frozen_fixture(directory)
        frozen, pairs, mappings = execution.load_selection(plan)
        root = Path(directory) / "execution"
        config = runner.bind_execution(root, frozen, {"type": "test_fixture", "execution_mode": "model_parallel",
            "gpu_weight_budget_gib": 10, "visible_devices": "0,1"}, directory, {})
        engine = fixture.baseline_engine_fixture()
        original_prepare = engine.prepare

        def prepare(row, mapping):
            return {**original_prepare(row, mapping), "input_tensor_sha256": {"input_ids": {"sha256": "a" * 64}}}

        engine.prepare = prepare
        engine.torch.load = lambda *_args, **_kwargs: {}
        engine.technical = lambda prepared, mapping, task, _captures: synthetic_result(prepared, mapping, task)
        engine.primary = engine.technical
        return plan, root, frozen, pairs, mappings, config, engine

    def gates(self, data):
        _, root, frozen, pairs, mappings, config, engine = data
        self.assertTrue(runner.run_stage(root, frozen, pairs, mappings, engine, config, "baseline")["passed"])
        self.assertTrue(runner.run_stage(root, frozen, pairs, mappings, engine, config, "preflight")["passed"])

    def test_fixed_grid_counts_both_directions_and_no_duplicate_layer_zero_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, frozen, pairs, mappings, _, _ = self.setup_run(directory)
            for stage, expected in (("patch", 696), ("routing", 24), ("knockout", 2592)):
                tasks = primary.primary_tasks(frozen, mappings, stage)
                self.assertEqual(len(tasks), expected)
                self.assertEqual(len({task["task_id"] for task in tasks}), expected)
                self.assertEqual({task["pair_id"] for task in tasks}, set(pairs))
            tasks = primary.primary_tasks(frozen, mappings, "patch")
            self.assertEqual(sum(task["support"] == "whole_event2" and task["layer"] == 0 and task["location"] == "block_output"
                                 for task in tasks), 24)
            self.assertEqual(sum(task["location"] == "post_deepstack" for task in tasks), 72)
            mapping = next(iter(mappings.values()))
            mapping["support_audit"]["knockout_controls"][core.CONDITIONS[0]]["query_all"]["target_1"]["background"]["eligible"] = False
            with self.assertRaisesRegex(ValueError, "shrink"):
                primary.primary_tasks(frozen, mappings, "knockout")

    def test_patch_recovery_uses_directional_denominator_and_ties_are_not_rescues(self):
        forward = primary.patch_outcomes(-2, 1, 2)
        reverse = primary.patch_outcomes(2, -1, -2)
        self.assertEqual(forward["recovery"], 0.75)
        self.assertEqual(reverse["recovery"], 0.75)
        self.assertEqual(forward["source_aligned_margin_delta"], reverse["source_aligned_margin_delta"])
        self.assertIsNone(primary.patch_outcomes(1, 2, 1)["recovery"])
        self.assertIsNone(primary.patch_outcomes(1, 2, 1)["source_aligned_margin_delta"])
        tie = primary.patch_outcomes(-1, 0, 1)
        self.assertTrue(tie["zero_margin_tie"])
        self.assertFalse(tie["strict_sign_crossing"])
        self.assertFalse(tie["strict_incorrect_to_correct"])

    def test_primary_stages_require_preflight_and_knockout_requires_intact_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.setup_run(directory)
            _, root, frozen, pairs, mappings, config, engine = data
            runner.run_stage(root, frozen, pairs, mappings, engine, config, "baseline")
            before = len(engine.calls)
            with self.assertRaises(FileNotFoundError):
                runner.run_stage(root, frozen, pairs, mappings, engine, config, "patch")
            self.assertEqual(len(engine.calls), before)
            runner.run_stage(root, frozen, pairs, mappings, engine, config, "preflight")
            before = len(engine.calls)
            with self.assertRaises(FileNotFoundError):
                runner.run_stage(root, frozen, pairs, mappings, engine, config, "knockout")
            self.assertEqual(len(engine.calls), before)

    def test_resume_skips_tasks_even_if_killed_before_rows_consolidation_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.setup_run(directory)
            self.gates(data)
            _, root, frozen, pairs, mappings, config, engine = data
            first = runner.run_stage(root, frozen, pairs, mappings, engine, config, "patch", max_tasks=2)
            self.assertFalse(first["passed"])
            self.assertEqual(first["completed"], 2)
            original = engine.primary
            engine.primary = lambda *_args: self.fail("Completed primary tasks must not be recomputed")
            runner.run_stage(root, frozen, pairs, mappings, engine, config, "patch", max_tasks=0)
            engine.primary = original
            tasks = primary.primary_tasks(frozen, mappings, "patch")
            task = tasks[2]
            prepared = engine.prepare(pairs[task["pair_id"]][task["condition"]], mappings[task["pair_id"]])
            saved = {**synthetic_result(prepared, mappings[task["pair_id"]], task), "task_id": task["task_id"],
                     "execution_fingerprint": config["execution_fingerprint"]}
            path = root / "patch/task_checkpoints" / f"{task['task_id']}.json"
            core.atomic_write(path, saved)
            # Simulate interruption after atomic per-task save, before consolidated JSONL update.
            runner.run_stage(root, frozen, pairs, mappings, engine, config, "patch", max_tasks=0)
            self.assertEqual(len(core.read_jsonl(root / "patch/rows.jsonl")), 3)
            saved["recipient_positions"] = [0]
            core.atomic_write(path, saved)
            with self.assertRaisesRegex(ValueError, "claimed-passed"):
                runner.run_stage(root, frozen, pairs, mappings, engine, config, "patch")

    def test_primary_failure_stops_and_preserves_error_history_on_explicit_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.setup_run(directory)
            self.gates(data)
            _, root, frozen, pairs, mappings, config, engine = data
            original = engine.primary
            def failed(*_args):
                raise RuntimeError("Test-only primary failure")
            engine.primary = failed
            first = runner.run_stage(root, frozen, pairs, mappings, engine, config, "patch")
            self.assertFalse(first["passed"])
            self.assertEqual(first["completed"], 1)
            engine.primary = original
            blocked = runner.run_stage(root, frozen, pairs, mappings, engine, config, "patch")
            self.assertEqual(blocked["completed"], 1)
            result = runner.run_stage(root, frozen, pairs, mappings, engine, config, "patch", max_tasks=1, retry_failed=True)
            self.assertEqual(result["completed"], 1)
            self.assertEqual(len(core.read_json(root / "patch/errors.json")), 1)

    def test_complete_cpu_fixture_analysis_decomposes_advantage_and_keeps_strata_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.setup_run(directory)
            self.gates(data)
            plan, root, frozen, pairs, mappings, config, engine = data
            for stage in primary.PRIMARY_STAGES:
                self.assertTrue(runner.run_stage(root, frozen, pairs, mappings, engine, config, stage)["passed"])
            summary = analysis.analyze(plan, root, repeats=100, plots=False)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["primary_patch_rows"], 696)
            self.assertEqual(summary["primary_knockout_rows"], 2592)
            table = core.read_json(root / "analysis/case_knockout_decomposition.json")
            row = table[0]
            self.assertAlmostEqual(row["compression"], row["delta_M_low"] - row["delta_M_temporal"])
            self.assertAlmostEqual(row["compression"], 0.30)
            contrasts = core.read_json(root / "analysis/case_matched_control_contrasts.json")
            self.assertAlmostEqual(contrasts[0]["compression_target_minus_background"], 0.24)
            summary_rows = core.read_json(root / "analysis/knockout_summary.json")
            self.assertEqual({row["case_count"] for row in summary_rows if row["analysis_stratum"] == "rescue"}, {8})
            self.assertEqual({row["case_count"] for row in summary_rows if row["analysis_stratum"] == "stable"}, {4})
            argv = ["run_phase3c.py", "--stage", "patch", "--plan_dir", str(plan), "--output_dir", str(root),
                    "--storage_root", directory, "--project_root", directory]
            with patch.object(sys, "argv", argv), patch.dict("os.environ"), patch.object(runner, "Engine") as gpu_engine:
                runner.main(stages=primary.PRIMARY_STAGES)
                gpu_engine.assert_not_called()
            # Synthetic norm indices exercise all eight figure layouts, not GPU capture content.
            plot_baselines = copy.deepcopy(analysis.verify_inputs(plan, root)[3])
            for pair_id in frozen["technical_preflight_case_ids"]:
                for side, condition in enumerate(core.CONDITIONS):
                    row = pairs[pair_id][condition]
                    path = root / "test_plot_indices" / f"{row['eval_id']}.json"
                    core.atomic_write(path, {"group_mean_norms_by_site": {f"block_output:L{layer}": {
                        "whole_event2": layer / 10 + side, "options_all": layer / 20 + side} for layer in range(36)}})
                    plot_baselines[row["eval_id"]]["capture_index_path"] = str(path)
            from scripts.visualize_phase3c import figures
            table_names = ("patch_summary", "knockout_summary")
            plots = figures(root / "test_figures", {name: core.read_json(root / f"analysis/{name}.json") for name in table_names},
                            frozen, pairs, plot_baselines)
            self.assertEqual(len(plots), 8)
            import matplotlib.image as mpimg
            for path in plots:
                pixels = mpimg.imread(path)
                self.assertGreater(pixels[..., :3].std(), 0.05)
                self.assertGreater(pixels.shape[0], 400)
            # Duplicate/task-file mutation cannot be blessed by an old aggregate summary.
            path = next((root / "knockout/task_checkpoints").glob("*.json"))
            changed = core.read_json(path)
            changed["edge_budget"]["key_count"] += 1
            core.atomic_write(path, changed)
            with self.assertRaisesRegex(ValueError, "per-task"):
                analysis.verify_inputs(plan, root)

    def test_paired_control_and_base_unit_validators_reject_pseudoreplication(self):
        rows = [{"base_sample_id": 1, "first_object_id": 1, "stratum": "rescue", "value": 1.0}]
        with self.assertRaisesRegex(ValueError, "independent base"):
            analysis.aggregate(rows * 2, ("stratum",), ("value",), repeats=100)
        first = analysis.aggregate(rows + [{**rows[0], "base_sample_id": 2, "value": 2.0}], ("stratum",), ("value",), repeats=100)
        self.assertEqual(first, analysis.aggregate(rows + [{**rows[0], "base_sample_id": 2, "value": 2.0}], ("stratum",), ("value",), repeats=100))
        row = {"pair_id": "case", "query_group": "options_all", "key_group": "target_1", "window_start": 0, "window_end": 3,
               "control": "target"}
        with self.assertRaisesRegex(ValueError, "Missing paired"):
            analysis.control_contrasts([row])

    def test_new_cli_help_and_analysis_imports_are_weight_free(self):
        result = subprocess.run([sys.executable, "-c", "import sys; from scripts import run_phase3c, analyze_phase3c; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("run_phase3c.py", "analyze_phase3c.py"):
            result = subprocess.run([sys.executable, f"scripts/{name}", "--help"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_engine_primary_uses_real_hooks_and_collects_all_intact_routes_in_one_noop(self):
        from scripts.phase3c_interventions import ResidualSites
        class CPUEngine(runner.Engine):
            def __init__(self):
                self.torch, self.text = torch, fixtures.ToyText()

            def forward(self, prepared):
                return self.text(*prepared["toy_inputs"])[0, -1].detach()

            def decide(self, logits, _row):
                return {"margin": float(logits[0] - logits[1]), "correct_logit": float(logits[0]),
                        "incorrect_logit": float(logits[1]), "prediction": "A" if logits[0] > logits[1] else "B"}

        engine = CPUEngine()
        captures, prepared = {}, {}
        mapping = {"processor_records": {}, "support_audit": {"supports": {}, "knockout_controls": {}}}
        for side, condition in enumerate(core.CONDITIONS):
            inputs = fixtures.toy_inputs()
            inputs[0].add_(side * 0.2)
            prepared[condition] = {"row": {"correct_option": "A"}, "toy_inputs": inputs, "input_tensor_sha256": {}}
            controller = ResidualSites(engine.text, 6, [1, 2], [1, 2])
            with controller.installed():
                logits = engine.forward(prepared[condition])
            controller.validate()
            captures[condition] = {"states": controller.states, "logits": logits}
            mapping["processor_records"][condition] = {"input_ids": [0] * 6, "visual_positions": [1, 2]}
            controls = mapping["support_audit"]["knockout_controls"].setdefault(condition, {})
            for query in ("options_all", "query_all"):
                controls[query] = {}
                for key, positions in (("target_1", [1]), ("target_2", [2]), ("both_targets", [1, 2])):
                    budget = core.edge_budget([4, 5], positions, 6)
                    controls[query][key] = {"background": {"eligible": True, "query_positions": [4, 5],
                        "target_key_positions": positions, "control_key_positions": [0] if len(positions) == 1 else [0, 3],
                        "target_budget": budget, "control_budget": budget}}
        mapping["support_audit"]["supports"]["whole_event2"] = core.support_record([1, 2], [1, 2], [1, 2], [1, 2], "test")
        for kind in ("visual_patch", "routing_baseline", "attention_knockout"):
            spec = {"pair_id": "test", "condition": core.CONDITIONS[0], "kind": kind}
            if kind == "visual_patch":
                spec.update({"layer": 0, "location": "post_deepstack", "support": "whole_event2"})
            elif kind == "attention_knockout":
                spec.update({"window": [0, 1, 2, 3], "query_group": "options_all", "key_group": "both_targets", "control": "target"})
            spec["task_id"] = core.digest(spec)
            result = engine.primary(prepared[core.CONDITIONS[0]], mapping, spec, captures)
            result["task_id"] = spec["task_id"]
            self.assertEqual(primary.primary_failures(result), [])
            if kind == "routing_baseline":
                self.assertEqual(len(result["routes"]), 12)
                self.assertTrue(result["noop_parity"]["exact_match"])
                self.assertEqual(len(result["routes"][0]["mask_audit"]["layers"]), 36)
        self.assertTrue(all(not block.self_attn._forward_hooks and not block.self_attn._forward_pre_hooks for block in engine.text.layers))


if __name__ == "__main__":
    unittest.main()
