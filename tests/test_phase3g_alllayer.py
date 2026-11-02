"""CPU tests for the all-layer capture helpers and layerwise analysis primitives; no GPU, VM or SURF."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from scripts import phase3g_layerwise_probes as layerwise
from scripts import run_phase3g_alllayer_capture as capture
from scripts import run_phase3g_alllayer_unattended as lifecycle


class FakeLayer(torch.nn.Module):
    def __init__(self, shift):
        super().__init__()
        self.shift = shift

    def forward(self, hidden):
        return (hidden + self.shift,)


class FakeText:
    def __init__(self):
        self.layers = torch.nn.ModuleList(FakeLayer(float(i)) for i in range(36))

    def __call__(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return hidden


class CaptureTests(unittest.TestCase):
    def test_all_blocks_reads_every_layer_without_changing_output(self):
        text, hidden = FakeText(), torch.randn(1, 10, 4096)
        plain = text(hidden)
        collector = capture.AllBlocks(text, 10, [3, 7])
        with collector.installed():
            observed = text(hidden)
        features = collector.features()
        self.assertTrue(torch.equal(plain, observed))
        self.assertEqual(tuple(features.shape), (2, 36, 4096))
        cumulative = hidden[0, [3, 7]] + sum(range(17))
        self.assertTrue(torch.allclose(features[:, 16], cumulative))

    def test_repeated_visit_is_rejected(self):
        text, hidden = FakeText(), torch.randn(1, 10, 4096)
        collector = capture.AllBlocks(text, 10, [1, 2])
        with self.assertRaises(RuntimeError):
            with collector.installed():
                text(hidden)
                text(hidden)

    def test_payload_gate_requires_exact_logits_and_four_layers(self):
        features = torch.randn(2, 36, 4096)
        reference = {"positions": [5, 9], "logits": torch.randn(100), "four": features[:, [16, 17, 18, 20]].clone()}
        payload = {"features": features, "logits": reference["logits"].clone(), "positions": [5, 9]}
        capture.check_payload(payload, reference)
        changed_logits = payload["logits"].clone()
        changed_logits[0] += 1
        for broken in ({**payload, "logits": changed_logits}, {**payload, "positions": [5, 10]}):
            with self.assertRaises(ValueError):
                capture.check_payload(broken, reference)
        shifted = features.clone()
        shifted[0, 17, 0] += 1
        with self.assertRaises(ValueError):
            capture.check_payload({**payload, "features": shifted}, reference)

    def test_cells_pair_each_reversed_prompt_with_its_original(self):
        rows = [{"original_eval_id": f"orig_{i}", "pair_id": f"p{i//4}", "condition": "low_boundary",
                 "view": "archived", "row": {"eval_id": f"rev_{i}"}} for i in range(200)]
        result = capture.cells((None, {"rows": rows}, None, None))
        self.assertEqual(len(result), 400)
        self.assertEqual(sum(c["kind"] == "original" for c in result), 200)
        with self.assertRaises(ValueError):
            capture.cells((None, {"rows": rows[:199]}, None, None))

    def test_backup_refuses_unexpected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "private.json").write_text("{}")
            with self.assertRaises(ValueError):
                capture.public_files(tmp)

    def test_lifecycle_runs_preflight_then_full_export_verify(self):
        class Args:
            project_root, output_root, stage = "/repo", "/data/run", "full"
        steps = [c[c.index("--action")+1] for c in lifecycle.commands(Args)]
        self.assertEqual(steps, ["run", "run", "export", "verify"])
        self.assertTrue(all("run_phase3g_alllayer_capture.py" in c[2] for c in lifecycle.commands(Args)))


class AnalysisTests(unittest.TestCase):
    def test_auc_matches_sklearn_with_ties(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, 300)
        scores = np.round(rng.normal(size=300) + labels, 1)
        self.assertAlmostEqual(layerwise.auc(labels, scores), roc_auc_score(labels, scores))

    def test_contrast_is_zero_for_identical_tables(self):
        rows = [{"base": b, "fold": f"outer_{b % 5}", "condition": c, "correct": (b+k) % 2,
                 "single": [(float(b+k), (b+k) % 2)]} for b in range(20) for c in ("low_boundary", "temporal_boundary")
                for k in range(2)]
        layerwise.DRAWS = 50
        result = layerwise.contrast(rows, "low_boundary", rows, "temporal_boundary", seed=0)
        self.assertAlmostEqual(result["accuracy"]["difference"], 0.0)
        self.assertLessEqual(result["accuracy"]["ci_low"], 0.0)


if __name__ == "__main__":
    unittest.main()
