import unittest
from types import SimpleNamespace

import torch

from scripts import probe_attention_roi as probe


class Inputs(dict):
    __getattr__ = dict.__getitem__


class FakeCache:
    def __init__(self, length):
        self.length = length

    def get_seq_length(self):
        return self.length


class FakeProcessor:
    def batch_decode(self, token_ids, **_kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return ["".join("A" if token_id == 1 else "" for token_id in token_ids)]


class FakeTransformers5Model:
    def __init__(self):
        self.generation_config = SimpleNamespace(eos_token_id=2)

    def _prepare_position_ids_for_generation(self, input_ids, _kwargs):
        positions = torch.arange(input_ids.shape[1]).view(1, 1, -1)
        return positions.expand(4, input_ids.shape[0], -1).clone()

    def prepare_inputs_for_generation(self, input_ids, next_sequence_length=None, **kwargs):
        if next_sequence_length != 1:
            raise AssertionError("Expected a one-token cached decode step.")
        return {
            "input_ids": input_ids[:, -1:],
            "past_key_values": kwargs["past_key_values"],
            "attention_mask": kwargs["attention_mask"],
            "position_ids": kwargs["position_ids"][..., -1:],
            # Transformers may leave this full-length; the probe must slice it.
            "mm_token_type_ids": kwargs["mm_token_type_ids"],
        }

    def generate(self, input_ids, **_kwargs):
        sequences = torch.cat([input_ids, torch.tensor([[1]])], dim=-1)
        scores = (torch.tensor([[0.0, 1.0, 0.0]]),)
        return SimpleNamespace(sequences=sequences, scores=scores)

    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        if input_ids.shape[1] == 4:
            if kwargs["position_ids"].shape != (4, 1, 4):
                raise AssertionError("Prefill M-RoPE positions were not prepared.")
            logits = torch.zeros(1, 4, 3)
            return SimpleNamespace(logits=logits, past_key_values=FakeCache(4))

        if input_ids.shape != (1, 1):
            raise AssertionError("The prompt was forwarded again during cached decoding.")
        if kwargs["position_ids"].shape != (4, 1, 1):
            raise AssertionError("Cached M-RoPE positions were not sliced to one token.")
        if kwargs["mm_token_type_ids"].shape != (1, 1):
            raise AssertionError("Cached modality ids were not sliced to one token.")
        cache_length = kwargs["past_key_values"].get_seq_length()
        logits = torch.zeros(1, 1, 3)
        if cache_length == 4:
            if kwargs["attention_mask"].shape != (1, 5):
                raise AssertionError("The decision attention mask has the wrong length.")
            logits[:, -1, 1] = 1
            attentions = (torch.full((1, 2, 1, 5), 1 / 5),) * 2
            next_cache = FakeCache(5)
        elif cache_length == 5:
            if kwargs["attention_mask"].shape != (1, 6):
                raise AssertionError("The continuation attention mask has the wrong length.")
            logits[:, -1, 2] = 1
            attentions = None
            next_cache = FakeCache(6)
        else:
            raise AssertionError(f"Unexpected cache length: {cache_length}")
        return SimpleNamespace(
            logits=logits,
            past_key_values=next_cache,
            attentions=attentions,
        )


class AttentionCacheTest(unittest.TestCase):
    def test_transformers5_cached_step_keeps_single_token_query(self):
        inputs = Inputs(
            input_ids=torch.tensor([[9, 8, 7, 6, 5]]),
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            mm_token_type_ids=torch.tensor([[2, 2, 2, 0, 0]]),
            pixel_values_videos=torch.ones(2, 3),
        )
        original_cache_api = probe.generation_cache_api
        probe.generation_cache_api = lambda: "next_sequence_length"
        try:
            response, attentions, metadata = probe.greedy_generate_with_decision_attention(
                FakeTransformers5Model(),
                FakeProcessor(),
                inputs,
                max_new_tokens=3,
            )
        finally:
            probe.generation_cache_api = original_cache_api

        self.assertEqual(response, "A")
        self.assertEqual(attentions[0].shape, (1, 2, 1, 5))
        self.assertEqual(metadata["cache_api"], "next_sequence_length")
        self.assertEqual(metadata["prefill_position_ids_shape"], [4, 1, 5])
        self.assertEqual(metadata["prefix_cache_length"], 4)
        self.assertTrue(metadata["standard_first_token_match"])
        self.assertTrue(metadata["standard_logits_allclose"])
        self.assertEqual(metadata["standard_top10_token_overlap"], 1.0)
        self.assertEqual(metadata["standard_logits_max_abs_diff"], 0.0)
        self.assertEqual(
            metadata["attention_semantics"],
            "prompt_final_position_decision_query",
        )

    def test_overlap_assignment_keeps_small_object_at_cell_edge(self):
        row = {
            "target_objects": [{
                "id": 1,
                "shape": "circle",
                "radius": 14,
                "from": [31, 31],
                "to": [31, 31],
                "start_frame": 0,
                "end_frame": 0,
            }],
            "distractors": [],
            "total_frames": 1,
            "boundary_timing": {"visual_marker": "none"},
        }
        label_map = probe.spatial_roi_label_map(row, 0, 0, 512, 512)
        weights = probe.cell_roi_weights(label_map, 0, 0, 16, 16, 512, 512)

        self.assertGreater(weights.get("target_1", 0.0), 0.0)
        self.assertEqual(
            probe.spatial_roi(row, 16, 16, 0, 0, 512, 512),
            "background",
        )

    def test_temporal_patch_preserves_fractional_boundary_phases(self):
        row = {
            "event_timing": {
                "first_event_start_frame": 30,
                "first_event_end_frame": 60,
                "second_event_start_frame": 105,
                "second_event_end_frame": 135,
            },
            "boundary_timing": {
                "boundary_start_frame": 60,
                "boundary_end_frame": 105,
            },
        }
        groups = probe.source_frame_groups(
            {"frames_indices": [59, 61]},
            grid_t=1,
            total_frames=270,
        )
        weights = probe.temporal_group_phase_weights(row, groups[0])

        self.assertEqual(groups, [[59, 61]])
        self.assertEqual(weights, {"boundary": 0.5, "event_1": 0.5})

    def test_fallback_selection_preserves_pairs_and_spreads_conditions(self):
        rows = []
        for base_id in (1, 2):
            for condition in ("low_boundary", "temporal_boundary"):
                for variant in ("original", "swapped"):
                    rows.append({
                        "eval_id": f"{base_id}_{condition}_{variant}",
                        "pairing_id": f"{base_id}_{condition}",
                        "base_sample_id": base_id,
                        "condition": condition,
                        "prompt_variant": variant,
                    })

        selected = probe.select_probe_rows(rows, 4)

        self.assertEqual({row["condition"] for row in selected}, {"low_boundary", "temporal_boundary"})
        self.assertEqual(
            {row["prompt_variant"] for row in selected},
            {"original", "swapped"},
        )


if __name__ == "__main__":
    unittest.main()
