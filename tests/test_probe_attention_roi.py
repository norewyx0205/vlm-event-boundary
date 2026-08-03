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

    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        if input_ids.shape[1] == 5:
            if kwargs["position_ids"].shape != (4, 1, 5):
                raise AssertionError("Prefill M-RoPE positions were not prepared.")
            logits = torch.zeros(1, 5, 3)
            logits[:, -1, 1] = 1
            return SimpleNamespace(logits=logits, past_key_values=FakeCache(5))

        if input_ids.shape != (1, 1):
            raise AssertionError("The prompt was forwarded again during cached decoding.")
        if kwargs["position_ids"].shape != (4, 1, 1):
            raise AssertionError("Cached M-RoPE positions were not sliced to one token.")
        if kwargs["mm_token_type_ids"].shape != (1, 1):
            raise AssertionError("Cached modality ids were not sliced to one token.")
        if kwargs["attention_mask"].shape != (1, 6):
            raise AssertionError("The cached attention mask has the wrong length.")
        if kwargs["past_key_values"].get_seq_length() != 5:
            raise AssertionError("The prefill cache has the wrong length.")

        logits = torch.zeros(1, 1, 3)
        logits[:, -1, 2] = 1
        attentions = (torch.full((1, 2, 1, 6), 1 / 6),) * 2
        return SimpleNamespace(
            logits=logits,
            past_key_values=FakeCache(6),
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
            response, attentions, metadata = probe.greedy_generate_with_answer_attention(
                FakeTransformers5Model(),
                FakeProcessor(),
                inputs,
                max_new_tokens=3,
            )
        finally:
            probe.generation_cache_api = original_cache_api

        self.assertEqual(response, "A")
        self.assertEqual(attentions[0].shape, (1, 2, 1, 6))
        self.assertEqual(metadata["cache_api"], "next_sequence_length")
        self.assertEqual(metadata["prefill_position_ids_shape"], [4, 1, 5])


if __name__ == "__main__":
    unittest.main()
