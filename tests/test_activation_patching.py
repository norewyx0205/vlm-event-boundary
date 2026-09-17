import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts import activation_patching_core as core
from scripts import analyze_activation_patching as analysis
from scripts import select_activation_patching_candidates as candidates
from scripts import select_activation_patching_cases as cases


class Inputs(dict):
    __getattr__ = dict.__getitem__


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return {"A": [1], "B": [2]}.get(text, [3])


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def batch_decode(self, token_ids, **_kwargs):
        values = token_ids.tolist() if torch.is_tensor(token_ids) else token_ids
        token_id = values[0][0]
        return [{1: "A", 2: "B"}.get(token_id, "?")]


class MixingLayer(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states + hidden_states.mean(dim=1, keepdim=True)


class FakeLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([MixingLayer(), MixingLayer()])


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeLanguageModel()

    def forward(self, input_ids, **_kwargs):
        hidden = torch.nn.functional.one_hot(input_ids % 4, num_classes=4).float()
        for layer in self.model.layers:
            hidden = layer(hidden)
        logits = torch.zeros(hidden.shape[0], hidden.shape[1], 4)
        logits[..., 1] = hidden[..., 1] + hidden[..., 3]
        logits[..., 2] = hidden[..., 2]
        return SimpleNamespace(logits=logits)


def annotation(base_id, condition, variant, correct_option="A"):
    return {
        "eval_id": f"l5_full_sample_{base_id:03d}_{condition}_{variant}",
        "base_sample_id": base_id,
        "feature_variant": "full",
        "condition": condition,
        "prompt_variant": variant,
        "question": "Which?",
        "option_A": "The orange circle moves before the blue square.",
        "option_B": "The orange circle moves after the blue square.",
        "correct_option": correct_option,
        "correct_sentence": "before",
        "incorrect_sentence": "after",
        "fps": 15,
        "duration_sec": 18,
        "total_frames": 270,
        "target_objects": [{
            "id": 1,
            "shape": "circle",
            "color": "orange",
            "from": [1, 1],
            "to": [2, 1],
            "start_frame": 30,
            "end_frame": 60 if condition == "low_boundary" else 105,
        }],
        "distractors": [],
    }


class ActivationPatchingTest(unittest.TestCase):
    def test_case_selection_enforces_rescue_and_stable_controls(self):
        annotations = []
        results = []
        for base_id, category in ((5, "rescue"), (1, "stable")):
            for condition in ("low_boundary", "temporal_boundary"):
                for variant in ("original", "swapped"):
                    row = annotation(base_id, condition, variant)
                    annotations.append(row)
                    is_correct = not (
                        category == "rescue"
                        and variant == "original"
                        and condition == "low_boundary"
                    )
                    results.append({
                        "eval_id": row["eval_id"],
                        "prediction": "A" if is_correct else "B",
                        "is_correct": is_correct,
                    })

        selected, audits = cases.select_cases(
            annotations, results, rescue_bases=(5,), control_bases=(1,)
        )

        self.assertEqual(len(selected), 8)
        self.assertEqual({row["phase3_case_category"] for row in selected}, {
            "temporal_rescue", "stable_both_correct"
        })
        self.assertEqual(len(audits), 2)

    def test_case_selection_validates_archived_model_config(self):
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "raw_results.jsonl"
            result_path.write_text("", encoding="utf-8")
            (Path(directory) / "config.json").write_text(
                json.dumps({
                    "model_name": "Qwen/Qwen3-VL-8B-Instruct",
                    "model_revision": "revision-1",
                    "video_sampling_request": {"fps": None},
                }),
                encoding="utf-8",
            )

            audit = cases.validate_main_run_config(
                result_path,
                expected_model_name="Qwen/Qwen3-VL-8B-Instruct",
                expected_model_revision="revision-1",
            )

        self.assertTrue(audit["available"])
        self.assertEqual(audit["model_revision"], "revision-1")

    def test_candidate_selection_is_pairwise_and_group_diverse(self):
        rows = []
        for pair in ("pair_a", "pair_b"):
            for index in range(8):
                rows.append({
                    "phase3_pair_id": pair,
                    "status": "ok",
                    "layer": index,
                    "token_group": f"group_{index % 4}",
                    "cosine_distance": 8 - index,
                    "relative_l2": 4 + (index % 3),
                })

        selected, audits = candidates.select_candidates(
            rows, top_k_per_pair=3, max_per_token_group=1
        )

        self.assertEqual(len(selected), 6)
        for pair in ("pair_a", "pair_b"):
            groups = [row["token_group"] for row in selected if row["phase3_pair_id"] == pair]
            self.assertEqual(len(groups), len(set(groups)))
        self.assertEqual(len(audits), 2)

    def test_residual_patch_and_identity_noop(self):
        model = FakeModel()
        processor = FakeProcessor()
        low = {
            "row": {"correct_option": "A"},
            "inputs": Inputs(input_ids=torch.tensor([[2, 0, 3]])),
            "groups": {"visual_all": [0]},
        }
        temporal = {
            "row": {"correct_option": "A"},
            "inputs": Inputs(input_ids=torch.tensor([[1, 0, 3]])),
            "groups": {"visual_all": [0]},
        }
        group_map = {0: {"visual_all": [0]}}
        low_run = core.run_baseline_capture(
            model, processor, low, layer_indices=[0], groups_by_layer=group_map,
            max_tokenwise_vectors=8, verify_standard=False,
        )
        temporal_run = core.run_baseline_capture(
            model, processor, temporal, layer_indices=[0], groups_by_layer=group_map,
            max_tokenwise_vectors=8, verify_standard=False,
        )

        identity = core.run_patched_forward(
            model, processor, low, 0, "visual_all",
            low_run["captures"][0]["visual_all"],
            low_run["captures"][0]["visual_all"],
        )
        patched = core.run_patched_forward(
            model, processor, low, 0, "visual_all",
            temporal_run["captures"][0]["visual_all"],
            low_run["captures"][0]["visual_all"],
        )

        self.assertTrue(torch.equal(identity["logits"], low_run["logits"]))
        self.assertEqual(identity["patch_method"], "positionwise_replace")
        self.assertNotEqual(patched["decision"]["margin"], low_run["decision"]["margin"])

    def test_divergence_metrics_report_missing_and_tokenwise_values(self):
        missing = core.divergence_metrics(
            {"mean": None, "values": None},
            {"mean": torch.ones(2), "values": None},
        )
        available = core.divergence_metrics(
            {"mean": torch.tensor([1.0, 0.0]), "values": torch.tensor([[1.0, 0.0]])},
            {"mean": torch.tensor([0.0, 1.0]), "values": torch.tensor([[0.0, 1.0]])},
        )

        self.assertEqual(missing["status"], "missing_token_group")
        self.assertAlmostEqual(available["cosine_distance"], 1.0)
        self.assertAlmostEqual(available["relative_l2"], 2 ** 0.5)
        self.assertAlmostEqual(available["tokenwise_cosine_mean"], 1.0)

    def test_positionwise_patch_requires_matching_sequence_positions(self):
        source = {
            "positions": [3, 4],
            "values": torch.ones(2, 4),
            "mean": torch.ones(4),
        }
        target = {
            "positions": [5, 6],
            "values": torch.zeros(2, 4),
            "mean": torch.zeros(4),
        }

        self.assertEqual(core.patch_method(source, target), "pooled_mean_delta")
        target["positions"] = [3, 4]
        self.assertEqual(core.patch_method(source, target), "positionwise_replace")

    def test_archived_processor_metadata_must_match(self):
        metadata = {
            "video_grid_thw": [[8, 32, 32]],
            "visual_token_count_from_grid_thw": 8192,
            "video_token_count_from_mm_token_type_ids": 2048,
            "video_inputs": [{"shape": [16, 3, 512, 512]}],
            "pixel_values_videos": {"shape": [8192, 1176]},
            "input_ids": {"shape": [1, 2100]},
        }
        prepared = {
            "row": {"archived_input_metadata": dict(metadata)},
            "input_metadata": dict(metadata),
        }

        result = core.validate_archived_input_metadata(prepared)
        self.assertTrue(result["matches"])
        prepared["input_metadata"] = dict(metadata, video_grid_thw=[[7, 32, 32]])
        with self.assertRaisesRegex(RuntimeError, "processor inputs differ"):
            core.validate_archived_input_metadata(prepared)

    def test_spearman_uses_tied_ranks(self):
        self.assertAlmostEqual(analysis.spearman([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(analysis.spearman([1, 2, 3], [6, 4, 2]), -1.0)


if __name__ == "__main__":
    unittest.main()
