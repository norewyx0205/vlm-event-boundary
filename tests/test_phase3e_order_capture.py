"""Mocked order capture/lifecycle tests. Never contact SURF or SMTP."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch

from scripts import phase3e_order_probes as probes
from scripts import run_phase3e_order_capture as capture
from scripts import run_phase3e_order_unattended as lifecycle
from scripts.phase3c_core import atomic_write, file_hash, read_json
from scripts.run_phase3c_preflight import save_tensors
from tests.test_phase3e_order_probes import order_fixture


class CaptureScopeTest(unittest.TestCase):
    def input_records(self):
        original = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "visual_positions": [0],
            "video_metadata": {"visual_token_count": 1000, "source_frame_groups": [[0, 1], [2, 3]],
                               "video_grid_thw": [2, 4, 4], "ambiguous_visual_cells": 911},
            "group_positions": {"video_t1_e2": [0], "options_all": [1, 2]},
            "input_tensor_sha256": {"input_ids": {"sha256": "tokens"},
                "pixel_values_videos": {"shape": [4, 8], "dtype": "torch.float32", "sha256": "original"}}}
        observed = copy.deepcopy(original)
        observed["video_metadata"]["ambiguous_visual_cells"] = 909
        observed["group_positions"]["video_t1_e2"] = []
        observed["input_tensor_sha256"]["pixel_values_videos"]["sha256"] = "reversed"
        return original, observed

    def test_motion_dependent_roi_diagnostic_is_not_sampling_metadata(self):
        original, observed = self.input_records()
        result = capture.reversed_input_audit(original, observed, {"reference": 2}, {"reference": 2})
        self.assertTrue(result["same_sampling_and_grid"])
        self.assertEqual(result["roi_assignment_diagnostic"], {
            "original_ambiguous_visual_cells": 911, "reversed_ambiguous_visual_cells": 909})

    def test_reversal_rejects_changed_sampling_grid_or_invalid_roi_counts(self):
        for key, value in (("source_frame_groups", [[1, 2], [3, 4]]), ("video_grid_thw", [2, 8, 8]),
                           ("ambiguous_visual_cells", -1), ("ambiguous_visual_cells", 1001)):
            original, observed = self.input_records()
            observed["video_metadata"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                capture.reversed_input_audit(original, observed, {}, {})

    def test_reversal_rejects_changed_text_mask_or_reference_positions(self):
        for field, value in (("input_ids", [1, 2, 4]), ("attention_mask", [1, 1, 0]),
                             ("visual_positions", [1]), ("group_positions", {"options_all": [2]})):
            original, observed = self.input_records()
            observed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                capture.reversed_input_audit(original, observed, {}, {})
        original, observed = self.input_records()
        with self.assertRaises(ValueError):
            capture.reversed_input_audit(original, observed, {"reference": 1}, {"reference": 2})

    def test_reversal_requires_changed_pixels_but_same_shape_dtype_and_other_tensors(self):
        for name, key, value in (("pixel_values_videos", "sha256", "original"),
                                 ("pixel_values_videos", "shape", [8, 8]),
                                 ("pixel_values_videos", "dtype", "torch.float16"),
                                 ("input_ids", "sha256", "changed")):
            original, observed = self.input_records()
            observed["input_tensor_sha256"][name][key] = value
            with self.subTest(name=name, key=key), self.assertRaises(ValueError):
                capture.reversed_input_audit(original, observed, {}, {})

    def test_prepare_retains_original_weight_budget_not_new_family_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            material, source, store = root/"materials", root/"source", root/"store"
            source.mkdir()
            atomic_write(material/"plan.json", {"settings": {"original": True}, "fingerprint": "material",
                "source_runtime": {"gpu_weight_budget_gib": 10.0}})
            atomic_write(store/"manifest.json", {"synthetic": True})
            with patch.object(capture, "space", return_value=31*1024**3), patch.object(capture, "context"), \
                    patch.object(capture, "code_hashes", return_value={"synthetic": "test"}):
                capture.prepare(root/"output", material, source, store/"manifest.json")
            self.assertEqual(read_json(root/"output/config.json")["gpu_weight_budget_gib"], 10.0)

    def test_eight_preflight_cells_and_full_two_hundred(self):
        rows = [{"pair_id": f"base_{base}", "condition": condition, "view": view} for base in range(50)
                for condition in ("low_boundary", "temporal_boundary") for view in ("archived", "option_order_permuted")]
        context = ({"preflight_ids": ["base_0", "base_1"]}, {}, {})
        self.assertEqual(len(capture.selected({"rows": rows}, context, True)), 8)
        self.assertEqual(len(capture.selected({"rows": rows}, context, False)), 200)

    def test_reversed_video_path_must_be_inside_material_videos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/"materials/videos/v.mp4"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"fixture-video")
            spec = {"row": {"video_path": "videos/v.mp4"}, "video_sha256": file_hash(path)}
            self.assertEqual(capture.safe_video(root, spec), path.resolve())
            spec["row"]["video_path"] = "../../private.json"
            with self.assertRaises(ValueError):
                capture.safe_video(root, spec)

    def test_full_capture_cannot_load_model_before_preflight(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(capture, "context", return_value=({}, {}, {}, {})), \
                patch.object(capture, "Engine") as engine:
            with self.assertRaises(FileNotFoundError):
                capture.run(directory)
            engine.assert_not_called()

    def test_export_constructs_800_endpoints_not_derived_fake_h(self):
        plan, endpoint_rows, values = order_fixture()
        originals = [r for r in endpoint_rows if r["order_variant"] == "original"]
        matrix = values[::2]
        specs = []
        for task in plan["tasks"]:
            specs.append({"pair_id": task["pair_id"], "condition": task["condition"], "view": task["view"],
                "original_eval_id": task["eval_id"], "row": {"eval_id": task["eval_id"]+"__new_reversed"}})
        reverse_values = {t["eval_id"]: values[[i for i, r in enumerate(endpoint_rows)
            if r["eval_id"] == t["eval_id"] and r["order_variant"] == "order_reversed"]] for t in plan["tasks"]}

        def verified(_root, _config, _history, spec):
            return {"decision": {"synthetic": True}}, {"features": torch.from_numpy(reverse_values[spec["original_eval_id"]])}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proof = {"complete": True, "synthetic": True}
            atomic_write(root/"capture_summary.json", proof)
            context = ({}, {"rows": specs, "fingerprint": "fixture"}, {}, ({}, plan, originals, matrix, {}))
            with patch.object(capture, "context", return_value=context), patch.object(capture, "verify", return_value=proof), \
                    patch.object(capture, "verify_one", side_effect=verified), patch.object(probes, "load_store"):
                result = capture.export(root)
            self.assertEqual(result["shape"], [800, 4, 5])
            self.assertEqual(len(read_json(root/"feature_store/endpoints.json")), 800)
            with np.load(root/"feature_store/features.npz", allow_pickle=False) as store:
                np.testing.assert_array_equal(store["features"][:400], matrix)
            self.assertEqual(result["base_count"], 50)

    def test_original_replay_requires_exact_logits_and_reference_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {"pair_id": "base", "condition": "low_boundary", "view": "archived", "original_eval_id": "original"}
            expected = torch.arange(40, dtype=torch.float32).reshape(2, 4, 5)
            logits = torch.tensor([1., 2., 3.])
            tokens = {o: {"reference_last_position": i+1} for i, o in enumerate(("A", "B"))}
            audit = {"record_sha256": "audit", "processor_record": {"input_ids": [1, 2, 3]}}
            baseline = {"record_sha256": "baseline"}
            original = ({}, {}, [{"eval_id": "original", "option": o} for o in ("A", "B")], expected.numpy(), {})
            engine = SimpleNamespace(processor=Mock(), model=SimpleNamespace(device="cpu"), text=Mock(), forward=Mock(return_value=logits))
            collector = MagicMock()
            collector.features.return_value = expected
            task = {"condition": "low_boundary", "view": "archived"}
            with patch.object(capture, "old_audit", return_value=(audit, baseline, {}, tokens)), \
                    patch.object(capture.source, "feature_tasks", return_value=[task]), \
                    patch.object(capture.source, "prepare_live", return_value=({"inputs": torch.ones(1)}, audit["processor_record"])), \
                    patch.object(capture.source, "payload", return_value={"logits": logits}), \
                    patch.object(capture, "OptionBlocks", return_value=collector):
                first = capture.original_replay(root, {"source_root": "synthetic"}, ({}, {}, {}), original, spec, engine)
                self.assertEqual(first["original_eval_id"], "original")
                engine.forward.assert_called_once()
                self.assertEqual(first, capture.original_replay(root, {"source_root": "synthetic"}, ({}, {}, {}), original, spec))
                changed = list(original)
                changed[3] = expected.numpy()+1
                with self.assertRaisesRegex(ValueError, "changed"):
                    capture.original_replay(root, {"source_root": "synthetic"}, ({}, {}, {}), changed, spec)


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)/"run"
        root.mkdir()
        atomic_write(root/"config.json", {"synthetic": True})
        (root/"captures").mkdir()
        (root/"captures/fixture.pt").write_bytes(b"synthetic only")
        self.args = SimpleNamespace(output_root=str(root), storage_root=self.temp.name, backup_dir=str(Path(self.temp.name)/"backups"),
            project_root=str(Path(__file__).resolve().parents[1]), stage="full", gpus="0,1", email_notify=True,
            pause_after=True, confirm_exclusive_workspace=True)
        self.notifier, self.client = Mock(), Mock()
        self.notifier.send.return_value = {"smtp_accepted": True}
        self.client.request_pause.return_value = {"request_accepted": True, "observed_status": "pausing"}

    def execute(self, child=None, **kwargs):
        with patch.object(capture, "context", return_value=({}, {}, {}, {})), patch.object(capture, "space"), \
                patch.object(capture, "verify"), patch.object(capture, "export"), patch.object(lifecycle, "snapshot"):
            code = lifecycle.job(self.args, child=child or Mock(return_value=0),
                get_services=lambda _: (self.notifier, self.client), check_idle=kwargs.pop("check_idle", Mock()), **kwargs)
        return code, read_json(lifecycle.directory(self.args)/"job_status.json")

    def test_success_backup_email_then_pause(self):
        events = []
        self.notifier.send.side_effect = lambda *_: events.append("email") or {"smtp_accepted": True}
        self.client.request_pause.side_effect = lambda: events.append("pause") or {"request_accepted": True}
        code, status = self.execute()
        self.assertEqual(code, 0)
        self.assertTrue(status["backup_verified"])
        self.assertTrue(status["experiment_success"])
        self.assertEqual(events, ["email", "pause"])

    def test_failed_capture_stops_grid_but_backs_up_mails_and_pauses(self):
        child = Mock(return_value=1)
        code, status = self.execute(child)
        self.assertEqual(code, 1)
        self.assertEqual(child.call_count, 1)
        self.assertFalse(status["experiment_success"])
        self.assertTrue(status["backup_verified"])
        self.client.request_pause.assert_called_once()
        self.assertIn("FAILED", self.notifier.send.call_args.args[0])

    def test_backup_failure_is_disclosed_but_idle_vm_still_pauses(self):
        code, status = self.execute(pack=Mock(side_effect=RuntimeError("synthetic backup failure")))
        self.assertEqual(code, 1)
        self.assertFalse(status["backup_verified"])
        self.assertEqual(status["backup_error_type"], "RuntimeError")
        self.client.request_pause.assert_called_once()

    def test_active_worker_blocks_pause(self):
        code, status = self.execute(check_idle=Mock(side_effect=RuntimeError("synthetic active worker")))
        self.assertEqual(code, 1)
        self.assertEqual(status["state"], "pause_needs_attention")
        self.client.request_pause.assert_not_called()


if __name__ == "__main__":
    unittest.main()
