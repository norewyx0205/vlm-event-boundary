import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from scripts import phase3e_preflight as core
from scripts import run_phase3e_preflight as runner
from scripts.export_phase3e_preflight_contrasts import paired_contrasts


class Attention(nn.Module):
    def forward(self, hidden_states):
        return hidden_states*.125, None


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.post_attention_layernorm = nn.Identity()

    def forward(self, hidden_states):
        x = hidden_states+self.self_attn(hidden_states)[0]
        return x+self.post_attention_layernorm(x)*.03125


class Text(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block() for _ in range(21)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PreflightTest(unittest.TestCase):
    def test_factorial_balance_fixed_geometry_and_rng_isolation(self):
        state = random.getstate()
        scenes, rows = core.factorial_rows("/tmp/unused_phase3e")
        self.assertEqual(state, random.getstate())
        self.assertEqual(len(scenes), 2)
        audit = core.audit_factorial(rows)
        self.assertEqual(audit["correct_option_counts"], {"A": 16, "B": 16})
        self.assertEqual(audit["unique_videos"], 8)
        self.assertEqual(len(rows), 32)
        self.assertEqual(scenes, core.factorial_rows("/tmp/unused_phase3e")[0])
        for change in ("label", "path", "missing", "duplicate"):
            changed = copy.deepcopy(rows)
            if change == "label":
                changed[0]["correct_option"] = "B"
            elif change == "path":
                x, y = changed[0]["target_objects"][0]["from"]
                changed[0]["target_objects"][0]["from"] = (x+1, y)
            elif change == "missing":
                changed.pop()
            else:
                changed[-1] = copy.deepcopy(changed[0])
            with self.assertRaises(ValueError):
                core.audit_factorial(changed)

    def test_event_relative_distractors_keep_identity_path_duration_and_slot(self):
        scenes, rows = core.factorial_rows("/tmp/unused_phase3e")
        self.assertTrue(all("motion" not in d for s in scenes for d in s["distractors"]))
        self.assertTrue(all(d.get("motion_timing") == "near_targets" for s in scenes for d in s["distractors"]
                            if d["motion_kind"] != "static"))
        for row in rows:
            rendered = {d["id"]: d for d in row["render_distractors"]}
            for dist in row["distractors"]:
                motion = rendered[dist["id"]]["motion"]
                self.assertTrue(all(dist[k] == v for k, v in motion.items()))
                if dist["motion_kind"] == "static":
                    self.assertEqual((dist["start_frame"], dist["end_frame"]), (0, 0))
                    continue
                slot = "first" if dist["id"] % 2 == 0 else "second"
                self.assertEqual(dist["start_frame"], row["event_timing"][slot+"_event_start_frame"])
                self.assertEqual(dist["end_frame"], row["event_timing"][slot+"_event_end_frame"])
                self.assertEqual(dist["end_frame"]-dist["start_frame"], 30)

    def test_low_timing_cannot_leak_into_temporal_and_slot_swap_rejected(self):
        _, rows = core.factorial_rows("/tmp/unused_phase3e")
        for kind in ("fixed_low", "swap_slots", "one_frame_shift"):
            changed = copy.deepcopy(rows)
            for row in changed:
                if row["condition"] != "temporal_boundary":
                    continue
                movers = [d for d in row["distractors"] if d["motion_kind"] != "static"]
                if kind == "fixed_low":
                    for dist in movers:
                        start, end = core.generator.moving_distractor_window(dist, core.timing("low_boundary"))
                        dist.update(start_frame=start, end_frame=end)
                elif kind == "swap_slots":
                    a, b = movers
                    a["start_frame"], b["start_frame"] = b["start_frame"], a["start_frame"]
                    a["end_frame"], b["end_frame"] = b["end_frame"], a["end_frame"]
                else:
                    movers[0]["start_frame"] += 1
            with self.assertRaisesRegex(ValueError, "event-relative"):
                core.audit_factorial(changed)

    def test_legacy_absolute_annotations_require_explicit_read_only_validation(self):
        _, rows = core.factorial_rows("/tmp/unused_phase3e")
        for row in rows:
            for dist in row["distractors"]:
                if dist["motion_kind"] != "static":
                    start, end = core.generator.moving_distractor_window(dist, core.timing("low_boundary"))
                    dist.update(start_frame=start, end_frame=end)
                dist.pop("motion_timing")
        with self.assertRaises(ValueError):
            core.audit_factorial(rows)
        audit = core.audit_factorial(rows, legacy_absolute=True)
        self.assertEqual(audit["distractor_schedule"], "absolute frames fixed within each scene across all variants")

    def test_actual_read_only_sites_and_cleanup(self):
        model = Text()
        values = torch.randn(1, 7, 4, dtype=torch.float16)
        baseline = model(values)
        sites = core.AttentionSites(model, 7, {"whole_event2": [0, 1], "option_A_t1": [4]})
        with sites.installed():
            captured = model(values)
        self.assertTrue(torch.equal(baseline, captured))
        self.assertEqual(len(sites.validate()["sites"]), 12)
        self.assertEqual(sites.payload()["union_positions"], [0, 1, 4])
        for module in model.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)
        sites.raw["L18_attention_contribution"][0, 0] += 1
        with self.assertRaisesRegex(RuntimeError, "residual"):
            sites.validate()

    def test_hooks_removed_on_failure_and_missing_sites_rejected(self):
        model = Text()
        sites = core.AttentionSites(model, 7, {"query": [4]})
        with self.assertRaises(RuntimeError):
            with sites.installed():
                raise RuntimeError("intentional")
        for module in model.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)
        with self.assertRaises(RuntimeError):
            sites.validate()
        for bad in ([], [-1], [8], [1, 1]):
            with self.assertRaises(ValueError):
                core.AttentionSites(model, 7, {"bad": bad})

    def test_video_render_decode_and_content_check(self):
        with tempfile.TemporaryDirectory() as directory:
            _, rows = core.factorial_rows(directory)
            row = rows[0]
            path = Path(row["video_path"])
            path.parent.mkdir()
            core.generator.render_video(path, row["condition"], 15, 18,
                row["render_objects"], row["render_distractors"], row["event_timing"])
            audit = core.video_audit(path)
            self.assertEqual(audit["decoded_frames"], 270)
            self.assertGreater(audit["changing_frame_pairs"], 0)
            with self.assertRaises(ValueError):
                core.video_audit(path, expected_frames=5)

    def test_actual_video_tensor_budget_gate(self):
        _, rows = core.factorial_rows("/tmp/unused_phase3e")
        plan = {"rows": rows, "videos": {r["video_id"]: {} for r in rows}}
        record = {"processor_record": {"visual_positions": [0, 1, 2, 3],
            "input_tensor_sha256": {"pixel_values_videos": {"sha256": "fixed"}}},
            "capture_groups": {"whole_event2": [2, 3]}}
        records = {r["eval_id"]: copy.deepcopy(record) for r in rows}
        self.assertEqual(runner.require_same_video_inputs(records, plan)["whole_event2_positions_per_prompt"], 2)
        records[rows[0]["eval_id"]]["processor_record"]["input_tensor_sha256"]["pixel_values_videos"]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "video tensor"):
            runner.require_same_video_inputs(records, plan)
        records = {r["eval_id"]: copy.deepcopy(record) for r in rows}
        records[rows[0]["eval_id"]]["capture_groups"]["whole_event2"].pop()
        with self.assertRaisesRegex(ValueError, "support counts"):
            runner.require_same_video_inputs(records, plan)

    def test_backup_rejects_private_files_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "videos").mkdir()
            (root / "videos/test.mp4").touch()
            (root / ".pipeline.lock").touch()
            self.assertEqual(list(runner.public_files(root)), ["run/videos/test.mp4"])
            (root / "email.json").touch()
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                runner.public_files(root)
            (root / "email.json").unlink()
            (root / "videos/link").symlink_to(root / "videos/test.mp4")
            with self.assertRaisesRegex(ValueError, "symlinks"):
                runner.public_files(root)

    def test_pause_refused_when_gpu_process_remains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backup = {"schema": "phase3e_preflight_backup_v1", "source_run_root": str(root.resolve())}
            with patch("scripts.backup_phase3b.verify_backup", return_value=backup), \
                    patch.object(runner, "verify", return_value={}), \
                    patch.object(runner.subprocess, "run") as command, \
                    patch("scripts.surf_workspace.SurfClient") as client, \
                    patch("scripts.phase3b_email.EmailNotifier") as mail:
                command.return_value.stdout = "12345\n"
                with self.assertRaisesRegex(RuntimeError, "GPU compute"):
                    runner.finish(root, root / "backup")
                client.assert_not_called()
                mail.assert_not_called()

    def test_both_contrasts_keep_original_values_and_exact_pairing(self):
        _, rows = core.factorial_rows("/tmp/unused_phase3e")
        means = {}
        for row in rows:
            temporal = row["condition"] == "temporal_boundary"
            means[row["eval_id"]] = {"L18_block_output": {
                "source_t1": torch.tensor([4. if temporal else 1.]),
                "source_t2": torch.tensor([2. if temporal else 3.]),
                "whole_event2": torch.tensor([7. if temporal else 5.])}}
        result = paired_contrasts({"rows": rows}, means)
        self.assertEqual(len(result), 16)
        for pair in result:
            site = pair["sites"]["L18_block_output"]
            self.assertEqual(float(site["D_low"]), -2)
            self.assertEqual(float(site["D_temporal"]), 2)
            self.assertEqual(float(site["delta_D"]), 4)
            self.assertEqual(float(site["B_low"]), 5)
            self.assertEqual(float(site["B_temporal"]), 7)
            self.assertEqual(float(site["delta_B"]), 2)
        with self.assertRaises(ValueError):
            paired_contrasts({"rows": rows[:-1]}, means)


if __name__ == "__main__":
    unittest.main()
