import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts import phase3c_core as core, phase3c_execution as execution
from scripts import run_phase3c_preflight as runner
from scripts.phase3c_interventions import (
    AttentionKnockout, ResidualSites, knockout_mask, selected_state, transplant,
)


class ToyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="eager")

    def forward(self, hidden_states, attention_mask, past_key_values=None):
        scores = hidden_states @ hidden_states.transpose(1, 2) / 4
        weights = torch.softmax(scores[:, None] + attention_mask, dim=-1).expand(-1, 32, -1, -1)
        output = weights[:, 0] @ hidden_states / 100
        return output, weights


class ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = ToyAttention()

    def forward(self, hidden_states, attention_mask):
        update, _ = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        return hidden_states + update


class ToyText(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([ToyBlock() for _ in range(36)])
        self.calls = []

    def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
        self.calls.append(visual_embeds.clone())
        result = hidden_states.clone()
        result[visual_pos_masks] += visual_embeds.to(result)
        return result

    def forward(self, hidden, visual, embeds, mask):
        for layer, block in enumerate(self.layers):
            hidden = block(hidden, attention_mask=mask)
            if layer < 3:
                hidden = self._deepstack_process(hidden, visual, embeds[layer])
        return hidden


def toy_inputs():
    hidden = torch.arange(24, dtype=torch.float32).view(1, 6, 4) / 20
    visual = torch.tensor([[False, True, True, False, False, False]])
    embeds = [torch.full((2, 4), (layer + 1) / 10) for layer in range(3)]
    mask = torch.zeros(1, 1, 6, 6).masked_fill(torch.triu(torch.ones(6, 6, dtype=torch.bool), 1), float("-inf"))
    return hidden, visual, embeds, mask


class Phase3CInterventionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.prior_threads)

    def capture(self, model, inputs):
        controller = ResidualSites(model, 6, [1, 2], [1, 2, 5])
        with controller.installed():
            output = model(*inputs)
        controller.validate()
        return controller, output

    def test_all_layer_capture_has_exact_noop_and_correct_post_addition_content(self):
        model, inputs = ToyText(), toy_inputs()
        plain = model(*inputs)
        controller, observed = self.capture(model, inputs)
        self.assertTrue(torch.equal(plain, observed))
        self.assertEqual(len(controller.states), 39)
        for layer in range(3):
            pre = selected_state(controller.states, layer, "block_output", [1, 2])["vectors"]
            post = selected_state(controller.states, layer, "post_deepstack", [1, 2])["vectors"]
            self.assertTrue(torch.equal(pre + inputs[2][layer], post))
        self.assertNotIn("_deepstack_process", model.__dict__)
        self.assertTrue(all(not block._forward_hooks for block in model.layers))

    def test_identity_is_exact_at_all_planned_sites(self):
        model, inputs = ToyText(), toy_inputs()
        capture, baseline = self.capture(model, inputs)
        sites = [(layer, "block_output") for layer in (0, 1, 2, 4, 8, 12, 16, 20)]
        sites += [(layer, "post_deepstack") for layer in (0, 1, 2)]
        for layer, location in sites:
            with self.subTest(layer=layer, location=location):
                donor = selected_state(capture.states, layer, location, [1, 2])
                controller = ResidualSites(model, 6, [1, 2], patch={"layer": layer, "location": location,
                    "recipient_positions": [1, 2], "donor": donor})
                with controller.installed():
                    result = model(*inputs)
                self.assertEqual(controller.validate()["patch_applied_count"], 1)
                self.assertTrue(torch.equal(baseline, result))

    def test_wrong_donor_site_is_rejected_and_hooks_are_cleaned_on_failure(self):
        model, inputs = ToyText(), toy_inputs()
        capture, _ = self.capture(model, inputs)
        donor = selected_state(capture.states, 0, "block_output", [1, 2])
        controller = ResidualSites(model, 6, [1, 2], patch={"layer": 0, "location": "post_deepstack",
            "recipient_positions": [1, 2], "donor": donor})
        with self.assertRaisesRegex(ValueError, "semantics"):
            with controller.installed():
                model(*inputs)
        self.assertNotIn("_deepstack_process", model.__dict__)
        self.assertTrue(all(not block._forward_hooks for block in model.layers))

    def test_pre_and_post_patch_use_distinct_observed_donor_states(self):
        donor_model, recipient_model = ToyText(), ToyText()
        donor_inputs, recipient_inputs = toy_inputs(), toy_inputs()
        donor_inputs[0].add_(0.2)
        donor_inputs[2][0].add_(0.4)
        captured, _ = self.capture(donor_model, donor_inputs)
        for location in ("block_output", "post_deepstack"):
            donor = selected_state(captured.states, 0, location, [1, 2])
            controller = ResidualSites(recipient_model, 6, [1, 2], [1, 2], patch={
                "layer": 0, "location": location, "recipient_positions": [1, 2], "donor": donor})
            with controller.installed():
                recipient_model(*recipient_inputs)
            controller.validate()
            pre = controller.states[(0, "block_output")]["vectors"]
            post = controller.states[(0, "post_deepstack")]["vectors"]
            if location == "block_output":
                self.assertTrue(torch.equal(pre, donor["vectors"]))
                self.assertTrue(torch.equal(post, donor["vectors"] + recipient_inputs[2][0]))
            else:
                self.assertTrue(torch.equal(post, donor["vectors"]))

    def test_rejects_missing_deepstack_wrong_visual_mask_and_cached_shapes(self):
        model, inputs = ToyText(), toy_inputs()
        for mode in ("mask", "length", "missing"):
            with self.subTest(mode=mode):
                controller = ResidualSites(model, 6, [1, 2], [1, 2])
                changed = copy.deepcopy(inputs)
                if mode == "mask":
                    changed[1][0, 1] = False
                elif mode == "length":
                    changed = (changed[0][:, :1], changed[1], changed[2], changed[3][:, :, :1, :1])
                with self.assertRaises((ValueError, RuntimeError)):
                    with controller.installed():
                        if mode == "missing":
                            model.layers[0](changed[0], attention_mask=changed[3])
                        else:
                            model(*changed)
                    controller.validate()

    def test_precision_duplicate_and_nonfinite_donor_vectors_rejected(self):
        hidden = toy_inputs()[0]
        donor = {"layer": 0, "location": "block_output", "positions": [1, 2], "vectors": hidden[0, [1, 2]].clone()}
        for mode in ("dtype", "nonfinite", "duplicate", "count"):
            with self.subTest(mode=mode):
                changed = copy.deepcopy(donor)
                if mode == "dtype":
                    changed["vectors"] = changed["vectors"].half()
                elif mode == "nonfinite":
                    changed["vectors"][0, 0] = float("nan")
                elif mode == "duplicate":
                    changed["positions"] = [1, 1]
                else:
                    changed["positions"] = [1]
                with self.assertRaises(ValueError):
                    transplant(hidden, [1, 2], changed, 0, "block_output")

    def test_knockout_changes_only_selected_causal_edges_without_changing_original(self):
        mask = toy_inputs()[3]
        original = mask.clone()
        knocked, budget = knockout_mask(mask, [4, 5], [1, 2], 6)
        expected = original.clone()
        expected[:, :, 4:6, 1:3] = float("-inf")
        self.assertTrue(torch.equal(knocked, expected))
        self.assertTrue(torch.equal(mask, original))
        self.assertEqual(budget["all_head_visible_causal_edges_per_layer"], 128)

    def test_actual_causal_mask_is_not_inferred_from_equal_shape(self):
        mask = toy_inputs()[3]
        for changed in (None, torch.ones(1, 6), torch.zeros_like(mask), mask.clone()):
            if torch.is_tensor(changed) and changed.shape == mask.shape and bool(torch.isinf(changed).any()):
                changed[0, 0, 4, 1] = float("-inf")
            with self.assertRaises(ValueError):
                knockout_mask(changed, [4, 5], [1, 2], 6)

    def test_disabled_mask_is_bitwise_noop_and_enabled_edges_have_zero_probability(self):
        model, inputs = ToyText(), toy_inputs()
        plain = model(*inputs)
        disabled = AttentionKnockout(model.layers, [0, 1, 2, 3], [4, 5], [1, 2], 6, enabled=False)
        with disabled.installed():
            same = model(*inputs)
        self.assertTrue(torch.equal(plain, same))
        self.assertGreater(disabled.validate()["layers"]["0"]["mean_selected_edge_mass_per_query_head"], 0)
        enabled = AttentionKnockout(model.layers, [0, 1, 2, 3], [4, 5], [1, 2], 6)
        with enabled.installed():
            changed = model(*inputs)
        self.assertFalse(torch.equal(plain, changed))
        self.assertTrue(all(item["max_blocked_probability"] == 0 for item in enabled.validate()["layers"].values()))
        self.assertTrue(all(not layer.self_attn._forward_pre_hooks and not layer.self_attn._forward_hooks for layer in model.layers))

    def test_attention_hooks_are_removed_after_exception(self):
        model, inputs = ToyText(), toy_inputs()
        inputs[3][0, 0, 5, 2] = float("-inf")
        controller = AttentionKnockout(model.layers, [0], [4, 5], [1, 2], 6)
        with self.assertRaises(ValueError):
            with controller.installed():
                model(*inputs)
        self.assertFalse(model.layers[0].self_attn._forward_pre_hooks)
        self.assertFalse(model.layers[0].self_attn._forward_hooks)

    def test_positional_attention_arguments_are_supported(self):
        module = ToyAttention()
        layer = SimpleNamespace(self_attn=module)
        hidden, _, _, mask = toy_inputs()
        controller = AttentionKnockout([layer], [0], [4, 5], [1, 2], 6)
        with controller.installed():
            module(hidden, mask)
        self.assertEqual(controller.validate()["layers"]["0"]["max_blocked_probability"], 0)

    def test_logits_parity_rejects_nan_and_preserves_full_vocab_check(self):
        baseline = torch.tensor([1.0, 2.0, 3.0])
        changed = baseline.clone()
        changed[0] += 0.01
        self.assertFalse(runner.logits_parity(baseline, changed, exact=True)["passed"])
        self.assertTrue(runner.logits_parity(baseline, changed)["passed"])
        changed[2] = float("nan")
        self.assertFalse(runner.logits_parity(changed, changed)["passed"])

    def test_engine_technical_uses_matching_donor_and_actual_mask_end_to_end_on_cpu(self):
        class CPUEngine(runner.Engine):
            def __init__(self):
                self.torch = torch
                self.text = ToyText()

            def forward(self, prepared):
                return self.text(*prepared["toy_inputs"])[0, -1].detach()

            def decide(self, logits, row):
                return {"margin": float(logits[0] - logits[1]), "correct_logit": float(logits[0]),
                        "incorrect_logit": float(logits[1]), "prediction": "A" if logits[0] > logits[1] else "B"}

        engine = CPUEngine()
        prepared, captures = {}, {}
        mapping = {"processor_records": {}, "support_audit": {"supports": {}, "knockout_controls": {}}}
        for side, condition in enumerate(core.CONDITIONS):
            inputs = toy_inputs()
            if side:
                inputs[0].add_(0.1)
                inputs[2][0].add_(0.2)
            prepared[condition] = {"row": {"correct_option": "A"}, "toy_inputs": inputs}
            observed = ResidualSites(engine.text, 6, [1, 2], [1, 2, 5])
            with observed.installed():
                logits = engine.forward(prepared[condition])
            observed.validate()
            captures[condition] = {"states": observed.states, "logits": logits}
            mapping["processor_records"][condition] = {"input_ids": [0] * 6, "visual_positions": [1, 2]}
            mapping["support_audit"]["knockout_controls"][condition] = {"options_all": {"both_targets": {"background": {
                "query_positions": [4, 5], "target_key_positions": [1, 2], "control_key_positions": [0, 3],
                "target_budget": core.edge_budget([4, 5], [1, 2], 6)}}}}
        mapping["support_audit"]["supports"]["whole_event2"] = core.support_record([1, 2], [1, 2], [1, 2], [1, 2], "test")
        for condition in core.CONDITIONS:
            for kind in ("identity", "transplant_smoke", "disabled_knockout", "knockout_smoke"):
                for location in ("block_output", "post_deepstack") if kind in ("identity", "transplant_smoke") else (None,):
                    with self.subTest(condition=condition, kind=kind, location=location):
                        task = {"pair_id": "test", "condition": condition, "kind": kind}
                        if location:
                            task.update({"layer": 0, "location": location, "support": "whole_event2"})
                        else:
                            task.update({"window": [0, 1, 2, 3], "query_group": "options_all",
                                         "key_group": "both_targets", "control": "target"})
                        task["task_id"] = core.digest(task)
                        result = engine.technical(prepared[condition], mapping, task, captures)
                        result["task_id"] = task["task_id"]
                        self.assertEqual(execution.technical_failures(result), [])

    def test_random_tiny_qwen_architecture_cpu_not_pinned_gpu_validation(self):
        try:
            from transformers import Qwen3VLTextConfig
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
        except ImportError:
            self.skipTest("Local Qwen3-VL implementation is unavailable.")
        # Random CPU weights exercise library hooks; they are not 8B model evidence.
        torch.manual_seed(42)
        config = Qwen3VLTextConfig(hidden_size=192, intermediate_size=192, num_hidden_layers=36,
            num_attention_heads=32, num_key_value_heads=8, head_dim=6, vocab_size=32,
            rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 1]},
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [1, 1, 1]})
        config._attn_implementation = "eager"
        model = Qwen3VLTextModel(config).eval()
        inputs = {"inputs_embeds": torch.randn(1, 6, 192), "attention_mask": torch.ones(1, 6),
                  "visual_pos_masks": torch.tensor([[False, True, True, False, False, False]]),
                  "deepstack_visual_embeds": [torch.randn(2, 192) / 20 for _ in range(3)], "use_cache": False}
        with torch.inference_mode():
            plain = model(**inputs).last_hidden_state
            capture = ResidualSites(model, 6, [1, 2], [1, 2, 5])
            with capture.installed():
                captured = model(**inputs).last_hidden_state
            capture.validate()
            self.assertTrue(torch.equal(plain, captured))
            donor = selected_state(capture.states, 2, "post_deepstack", [1, 2])
            identity = ResidualSites(model, 6, [1, 2], patch={"layer": 2, "location": "post_deepstack",
                "recipient_positions": [1, 2], "donor": donor})
            with identity.installed():
                same = model(**inputs).last_hidden_state
            identity.validate()
            self.assertTrue(torch.equal(plain, same))
            knockout = AttentionKnockout(model.layers, [0, 1, 2, 3], [4, 5], [1, 2], 6)
            with knockout.installed():
                changed = model(**inputs).last_hidden_state
            knockout.validate()
            self.assertFalse(torch.equal(plain, changed))


class Phase3CGateTest(unittest.TestCase):
    def frozen_fixture(self, directory):
        from test_phase3c import fixture_archive, RUNTIME
        from scripts import prepare_phase3c, audit_phase3c_mappings
        source, live = fixture_archive(directory)
        output = Path(directory) / "plan"
        prepare_phase3c.audit_archive(source, output)
        with patch.object(audit_phase3c_mappings, "processor_condition", side_effect=lambda candidate, condition, *_:
                          live[candidate["pair_id"]][condition]):
            audit_phase3c_mappings.run_audit(output, object(), RUNTIME, directory, {})
        prepare_phase3c.freeze_selection(output)
        return output

    def baseline_engine_fixture(self):
        from test_phase3c import fixture_pair
        class FixtureEngine:
            def __init__(self):
                self.torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
                self.load_sec, self.calls, self.fail = 0, [], False

            def prepare(self, row, mapping):
                self.calls.append(row["eval_id"])
                return {"row": row}

            def baseline(self, prepared, mapping, root, fingerprint):
                if self.fail:
                    raise RuntimeError("test-only failure")
                row = prepared["row"]
                candidate, _ = fixture_pair(row["base_sample_id"], row["first_object_id"], row["phase3c_analysis_stratum"])
                saved = copy.deepcopy(candidate["capture_indices"][row["condition"]])
                path = Path(root) / "captures" / row["eval_id"] / "test_fixture.json"
                core.atomic_write(path, {"artifact_type": "test_fixture", "value": 1})
                index = path.parent / "index.json"
                hashes = {"input_ids": {"sha256": "a" * 64}}
                core.atomic_write(index, {"vectors_path": str(path), "vectors_sha256": core.file_hash(path),
                    "execution_fingerprint": fingerprint, "eval_id": row["eval_id"], "site_count": 39,
                    "input_tensor_sha256": hashes})
                saved.update({"passed": True, "capture_index_path": str(index), "capture_index_sha256": core.file_hash(index),
                    "hook_audit": {"block_output_sites": 36, "post_deepstack_layers": [0, 1, 2]},
                    "capture_noop_parity": {"exact_match": True}, "input_tensor_sha256": hashes})
                return saved
        return FixtureEngine()

    def test_baseline_checkpoint_resume_skips_completed_rows_and_detects_modified_captures(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.frozen_fixture(directory)
            frozen, pairs, mappings = execution.load_selection(plan)
            root = Path(directory) / "execution"
            config = runner.bind_execution(root, frozen, {"type": "test_fixture"}, directory, {})
            engine = self.baseline_engine_fixture()
            first = runner.run_stage(root, frozen, pairs, mappings, engine, config, "baseline", max_tasks=2)
            self.assertFalse(first["passed"])
            self.assertEqual(len(engine.calls), 2)
            second = runner.run_stage(root, frozen, pairs, mappings, engine, config, "baseline")
            self.assertTrue(second["passed"])
            self.assertEqual(len(engine.calls), 24)
            expected = {row["eval_id"] for pair in pairs.values() for row in pair.values()}
            rows = execution.require_stage(root, config, pairs, "baseline", expected)
            item = next(iter(rows.values()))
            capture = core.read_json(item["capture_index_path"])
            core.atomic_write(capture["vectors_path"], {"changed": True})
            with self.assertRaisesRegex(ValueError, "capture"):
                execution.require_stage(root, config, pairs, "baseline", expected)

    def test_failure_stops_tasks_preserves_history_and_requires_explicit_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.frozen_fixture(directory)
            frozen, pairs, mappings = execution.load_selection(plan)
            root = Path(directory) / "execution"
            config = runner.bind_execution(root, frozen, {"type": "test_fixture"}, directory, {})
            engine = self.baseline_engine_fixture()
            engine.fail = True
            first = runner.run_stage(root, frozen, pairs, mappings, engine, config, "baseline")
            self.assertFalse(first["passed"])
            self.assertEqual(len(engine.calls), 1)
            self.assertEqual(len(core.read_json(root / "baseline/errors.json")), 1)
            engine.fail = False
            runner.run_stage(root, frozen, pairs, mappings, engine, config, "baseline")
            self.assertEqual(len(engine.calls), 1)
            retried = runner.run_stage(root, frozen, pairs, mappings, engine, config, "baseline", retry_failed=True)
            self.assertTrue(retried["passed"])
            self.assertEqual(len(engine.calls), 25)
            self.assertEqual(len(core.read_json(root / "baseline/errors.json")), 1)

    def test_preflight_is_blocked_without_complete_baseline_before_new_forwards(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.frozen_fixture(directory)
            frozen, pairs, mappings = execution.load_selection(plan)
            root = Path(directory) / "execution"
            config = runner.bind_execution(root, frozen, {"type": "test_fixture"}, directory, {})
            engine = self.baseline_engine_fixture()
            with self.assertRaises(FileNotFoundError):
                runner.run_stage(root, frozen, pairs, mappings, engine, config, "preflight")
            self.assertEqual(engine.calls, [])

    def test_frozen_preflight_cases_and_task_grid_cover_both_orders_without_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.frozen_fixture(directory)
            frozen, pairs, mappings = execution.load_selection(root)
            self.assertEqual(frozen["technical_preflight_case_ids"], ["phase3c_base_005_original", "phase3c_base_006_original"])
            tasks = execution.technical_tasks(frozen, mappings)
            self.assertEqual(len(tasks), 152)
            self.assertEqual({task["condition"] for task in tasks}, set(core.CONDITIONS))
            self.assertEqual({task["location"] for task in tasks if "location" in task}, {"block_output", "post_deepstack"})
            self.assertEqual(len({task["task_id"] for task in tasks}), len(tasks))
            self.assertTrue(all(task["pair_id"] in frozen["technical_preflight_case_ids"] for task in tasks))

    def test_baseline_recheck_does_not_silently_relax_behavior_or_technical_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.frozen_fixture(directory)
            frozen, pairs, mappings = execution.load_selection(root)
            baselines = {}
            for pair_id, pair in pairs.items():
                mapping = mappings[pair_id]
                for condition, row in pair.items():
                    from test_phase3c import fixture_pair
                    candidate, _ = fixture_pair(row["base_sample_id"], row["first_object_id"], row["phase3c_analysis_stratum"])
                    saved = copy.deepcopy(candidate["capture_indices"][condition])
                    saved["hook_audit"] = {"block_output_sites": 36, "post_deepstack_layers": [0, 1, 2]}
                    saved["capture_noop_parity"] = {"exact_match": True}
                    saved["input_tensor_sha256"] = {"input_ids": {"sha256": "a" * 64}}
                    baselines[row["eval_id"]] = saved
            self.assertEqual(execution.baseline_reasons(pairs, baselines), [])
            first = next(iter(baselines.values()))
            first["capture_noop_parity"]["exact_match"] = False
            self.assertTrue(any("changed_logits" in reason for reason in execution.baseline_reasons(pairs, baselines)))
            first["decision"]["margin"] = 0
            self.assertTrue(execution.baseline_reasons(pairs, baselines))

    def test_duplicate_foreign_and_stale_task_checkpoints_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            row = {"task_id": "task1", "execution_fingerprint": "expected", "passed": True}
            for rows in ([row, row], [{**row, "task_id": "foreign"}], [{**row, "execution_fingerprint": "stale"}]):
                core.atomic_write(path, rows, jsonl=True)
                with self.assertRaises(ValueError):
                    execution.checkpoint_rows(path, "expected", {"task1"})

    def test_gate_rechecks_technical_controls_instead_of_file_existence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"execution_fingerprint": "expected"}
            core.atomic_write(root / "execution_config.json", config)
            pairs = {"case": {condition: {"eval_id": condition} for condition in core.CONDITIONS}}
            output = root / "preflight"
            spec = {"pair_id": "case", "condition": "low_boundary", "kind": "identity",
                    "layer": 0, "location": "post_deepstack", "support": "whole_event2"}
            key = core.digest(spec)
            spec["task_id"] = key
            decision = {"margin": 1.0, "correct_logit": 3.0, "incorrect_logit": 2.0}
            row = {"task_id": key, "execution_fingerprint": "expected", "passed": True,
                   "spec": spec, "is_primary_effect_estimate": False, "decision": decision,
                   "baseline_decision": decision, "noop_parity": {"exact_match": True, "max_abs_diff": 0},
                   "hook_audit": {"block_output_sites": 36, "post_deepstack_layers": [0, 1, 2], "patch_applied_count": 1},
                   "donor_capture_location": "post_deepstack", "recipient_patch_location": "post_deepstack"}
            core.atomic_write(output / "task_manifest.jsonl", [spec], jsonl=True)
            core.atomic_write(output / "rows.jsonl", [row], jsonl=True)
            execution.save_stage_summary(output, config, pairs, {key: row}, {key}, "preflight")
            self.assertEqual(len(execution.require_stage(root, config, pairs, "preflight", {key})), 1)
            core.atomic_write(output / "rows.jsonl", [{**row, "passed": False}], jsonl=True)
            with self.assertRaises(ValueError):
                execution.require_stage(root, config, pairs, "preflight", {key})
            # A claimed pass with a nonzero identity effect is also rejected after hashes are refreshed.
            row["noop_parity"]["max_abs_diff"] = 0.1
            core.atomic_write(output / "rows.jsonl", [row], jsonl=True)
            summary = execution.save_stage_summary(output, config, pairs, {key: row}, {key}, "preflight")
            self.assertFalse(summary["passed"])

    def test_execution_config_is_bound_to_code_runtime_placement_and_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = {"selection_fingerprint": "frozen"}
            runtime = {"model_device_map": {"model": "cuda:0"}}
            config = runner.bind_execution(root, frozen, runtime, directory, {})
            self.assertEqual(runner.bind_execution(root, frozen, runtime, directory, {}), config)
            with self.assertRaisesRegex(ValueError, "Frozen"):
                runner.bind_execution(root, frozen, {"model_device_map": {"model": "cuda:1"}}, directory, {})

    def test_changed_execution_request_is_blocked_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = {"selection_fingerprint": "frozen"}
            runtime = {"model_device_map": {"model": "cuda:0"}, "execution_mode": "model_parallel",
                       "gpu_weight_budget_gib": 10, "visible_devices": "0,1"}
            runner.bind_execution(root, frozen, runtime, directory, {})
            runner.check_execution_request(root, frozen, directory, {}, "model_parallel", 10, ["0", "1"])
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                runner.check_execution_request(root, frozen, directory, {}, "model_parallel", 10, ["1", "0"])

    def test_entry_point_help_does_not_import_torch_or_load_weights(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; from scripts import run_phase3c_preflight, phase3c_execution; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
