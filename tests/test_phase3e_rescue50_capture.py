"""Toy hooks and mocked services; no live model, SMTP, SURF or SSH."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from scripts import run_phase3e_rescue50_capture as capture
from scripts import run_phase3e_rescue50_unattended as lifecycle
from scripts.phase3c_core import atomic_write, read_json
from scripts.prepare_phase3e_rescue50 import task_manifest
from scripts.phase3d_core import audit_pairs
from tests.test_phase3d import pairs


class ToyText(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(36)])

    def forward(self, values):
        for layer in self.layers:
            values = layer(values)
        return values


class ScopeAndHookTest(unittest.TestCase):
    def test_only_original_pure_permutations_and_four_preflight_cells(self):
        annotations = pairs()
        for pair in annotations.values():
            for row in pair.values():
                row.update(video_path="original.mp4", phase3c_video_sha256="sha")
        cases = audit_pairs(annotations)
        prompts, _, _ = task_manifest(annotations, cases)
        context = ({"case_ids": [c["pair_id"] for c in cases], "fingerprint": "old",
                    "preflight_ids": [cases[0]["pair_id"], cases[1]["pair_id"]]}, annotations, {})
        specs = capture.tasks(context, {"tasks": prompts})
        self.assertEqual(len(specs), 100)
        self.assertEqual({s["view"] for s in specs}, {"option_order_permuted"})
        self.assertEqual(len(capture.select(specs, context, True)), 4)
        self.assertEqual(len(capture.select(specs, context, False)), 100)
        self.assertEqual({s["stage"] for s in specs}, {"capture"})

    def test_read_only_hooks_exact_positions_and_cleanup(self):
        torch.manual_seed(5)
        model, values = ToyText(), torch.randn(1, 7, 4)
        expected, states = model(values), {}
        x = values
        for index, layer in enumerate(model.layers):
            x = layer(x)
            if index in capture.LAYERS:
                states[index] = x[0, [2, 5]].detach().float()
        collector = capture.OptionBlocks(model, 7, [2, 5])
        with collector.installed():
            observed = model(values)
        self.assertTrue(torch.equal(observed, expected))
        wanted = torch.stack([states[layer] for layer in capture.LAYERS]).permute(1, 0, 2)
        self.assertTrue(torch.equal(collector.features(), wanted))
        self.assertTrue(all(not layer._forward_hooks for layer in model.layers))
        with self.assertRaises(RuntimeError):
            with capture.OptionBlocks(model, 7, [2, 5]).installed():
                raise RuntimeError("toy error")
        self.assertTrue(all(not layer._forward_hooks for layer in model.layers))
        with self.assertRaises(ValueError):
            capture.OptionBlocks(model, 7, [2, 2])

    def test_full_cannot_load_model_without_passed_preflight(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(capture, "input_context", return_value=({}, {}, {}, None)), \
                patch.object(capture, "tasks", return_value=[]), patch.object(capture, "select", return_value=[]), \
                patch.object(capture, "Engine") as engine:
            with self.assertRaises(FileNotFoundError):
                capture.run(temp, False)
            engine.assert_not_called()


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)/"run"
        root.mkdir()
        atomic_write(root/"config.json", {"synthetic": True})
        (root/"captures").mkdir()
        (root/"captures/example.pt").write_bytes(b"synthetic fixture, not model states")
        self.args = SimpleNamespace(output_root=str(root), storage_root=self.temp.name,
            backup_dir=str(Path(self.temp.name)/"backups"), project_root=str(Path(__file__).resolve().parents[1]),
            stage="full", gpus="0,1", email_notify=True, pause_after=True, confirm_exclusive_workspace=True)
        self.notifier, self.client = Mock(), Mock()
        self.notifier.send.return_value = {"smtp_accepted": True}
        self.client.request_pause.return_value = {"accepted": True}

    def execute(self, child=None, **kwargs):
        with patch.object(lifecycle, "precheck", return_value=({"synthetic": True}, {})), \
                patch.object(lifecycle, "snapshot"), patch.object(capture, "space"), \
                patch.object(capture, "verify", return_value={"synthetic": True}), \
                patch.object(capture, "export", return_value={"complete": True, "statement_rows": 400}):
            code = lifecycle.job(self.args, child=child or Mock(return_value=0), idle=Mock(),
                get_services=lambda args: (self.notifier, self.client), **kwargs)
        return code, read_json(lifecycle.directory(self.args)/"job_status.json")

    def test_full_sequence_backups_emails_then_pauses(self):
        events = []
        self.notifier.send.side_effect = lambda *args: events.append("email") or {"smtp_accepted": True}
        self.client.request_pause.side_effect = lambda: events.append("pause") or {"accepted": True}
        child = Mock(return_value=0)
        code, status = self.execute(child)
        self.assertEqual(code, 0)
        self.assertEqual(events, ["email", "pause"])
        self.assertTrue(status["experiment_success"])
        self.assertTrue(status["backup_verified"])
        self.assertEqual(child.call_count, 4)
        commands = [c.args[1] for c in child.call_args_list]
        self.assertIn("preflight", commands[0])
        self.assertIn("full", commands[1])
        self.assertIn("export", commands[2])
        self.assertIn("verify", commands[3])
        self.assertTrue(all("knockout" not in c and "probes" not in c for c in commands))
        self.assertIn("original 50 rescue pairs", self.notifier.send.call_args.args[1])

    def test_preflight_failure_stops_full_and_preserves_partial_backup(self):
        child = Mock(return_value=7)
        code, status = self.execute(child)
        self.assertEqual(code, 7)
        self.assertFalse(status["experiment_success"])
        self.assertTrue(status["backup_verified"])
        self.assertEqual(child.call_count, 1)
        self.client.request_pause.assert_called_once()

    def test_backup_failure_never_pauses(self):
        code, status = self.execute(pack=Mock(side_effect=RuntimeError("synthetic backup failure")))
        self.assertEqual(code, 1)
        self.assertFalse(status["backup_verified"])
        self.client.request_pause.assert_not_called()
        self.notifier.send.assert_called_once()

    def test_email_failure_still_pauses_after_verified_backup(self):
        self.notifier.send.side_effect = RuntimeError("synthetic email failure")
        code, status = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(status["email"]["error_type"], "RuntimeError")
        self.assertTrue(status["backup_verified"])
        self.client.request_pause.assert_called_once()

    def test_public_backup_refuses_unknown_files_and_symlinks(self):
        root = Path(self.args.output_root)
        (root/"unknown.json").write_text("{}")
        with self.assertRaises(ValueError):
            capture.public_files(root)
        (root/"unknown.json").unlink()
        (root/"captures/link.pt").symlink_to(root/"config.json")
        with self.assertRaises(ValueError):
            capture.public_files(root)


if __name__ == "__main__":
    unittest.main()
