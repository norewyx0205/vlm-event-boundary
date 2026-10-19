import copy
import unittest

import torch

from scripts.phase3b_core import build_text_groups
from scripts.phase3d_features import CompactSites, assert_patch_timing, mapped_l16_patch, option_spans
from tests.test_phase3c_preflight import ToyText, toy_inputs
from tests.test_phase3d import row


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(char)+256 for char in text],
                "offset_mapping": [(i, i+1) for i in range(len(text))]}


class OptionSpansTest(unittest.TestCase):
    def test_actual_option_sections_and_literal_identities(self):
        value, tokenizer = row(), CharacterTokenizer()
        text = ("system\nWhich statement correctly describes the order of events?\n\n"
                f"A: {value['option_A']}\nB: {value['option_B']}\n\nAnswer with only A or B.")
        ids = tokenizer(text)["input_ids"]
        old = build_text_groups(value, ids, tokenizer)
        groups = option_spans(value, ids, tokenizer, old)
        self.assertEqual(len(groups), 6)
        self.assertLess(max(groups["option_A_t2"]), min(groups["option_B_t1"]))
        self.assertFalse(set(groups["option_A_t1"]) & set(groups["option_B_t1"]))
        altered = copy.deepcopy(old)
        altered["option_target_1_mentions"] = altered["option_target_1_mentions"][:-1]
        with self.assertRaises(ValueError):
            option_spans(value, ids, tokenizer, altered)


class CompactCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def groups(self):
        return {"source_t1": [1], "source_t2": [2], "whole_event2": [1, 2],
                "option_A_t1": [3], "option_A_t2": [4], "option_B_t1": [3], "option_B_t2": [4],
                "option_A_relation": [3], "option_B_relation": [4], "query_all": [0], "decision_position": [5]}

    def capture(self, model, inputs, patch=None):
        controller = CompactSites(model, 6, [1, 2], self.groups(), patch)
        with controller.installed():
            output = model(*inputs)
        controller.validate()
        return controller, output

    def test_all_layer_compact_capture_is_exact_noop_and_input_is_aligned(self):
        model, inputs = ToyText(), toy_inputs()
        plain = model(*inputs)
        controller, observed = self.capture(model, inputs)
        self.assertTrue(torch.equal(plain, observed))
        self.assertEqual(len(controller.feature_views()), 289)
        self.assertTrue(torch.equal(controller.input16["source_t1"], controller.means[15]["source_t1"]))
        self.assertEqual(controller.donor16["layer"], 16)
        self.assertFalse(controller.states)
        self.assertFalse(any(layer._forward_hooks or layer._forward_pre_hooks for layer in model.layers))
        self.assertNotIn("_deepstack_process", model.__dict__)

    def test_identity_and_cross_condition_timing(self):
        model, inputs = ToyText(), toy_inputs()
        baseline, output = self.capture(model, inputs)
        spec = {"layer": 16, "location": "block_output", "recipient_positions": [1, 2], "donor": baseline.donor16}
        identity, actual = self.capture(model, inputs, spec)
        self.assertTrue(torch.equal(output, actual))
        assert_patch_timing(baseline, identity)
        donor_inputs = copy.deepcopy(inputs)
        donor_inputs[0][0, 1:3] += .4
        donor, _ = self.capture(model, donor_inputs)
        spec["donor"] = donor.donor16
        patched, actual = self.capture(model, inputs, spec)
        self.assertTrue(assert_patch_timing(baseline, patched)["upstream_and_same_block_text_exact"])
        self.assertTrue(torch.equal(baseline.means[16]["option_A_t1"], patched.means[16]["option_A_t1"]))
        self.assertFalse(torch.equal(baseline.means[17]["option_A_t1"], patched.means[17]["option_A_t1"]))
        self.assertFalse(torch.equal(output, actual))

    def test_failure_removes_every_hook_and_other_sites_are_rejected(self):
        model = ToyText()
        spec = {"layer": 20, "location": "block_output", "recipient_positions": [1, 2]}
        with self.assertRaises(ValueError):
            CompactSites(model, 6, [1, 2], self.groups(), spec)
        controller = CompactSites(model, 6, [1, 2], self.groups())
        with self.assertRaises(RuntimeError):
            with controller.installed():
                raise RuntimeError("toy forward failure")
        self.assertFalse(any(layer._forward_hooks or layer._forward_pre_hooks for layer in model.layers))
        self.assertNotIn("_deepstack_process", model.__dict__)

    def test_donor_order_requires_explicit_position_mapping(self):
        model, inputs = ToyText(), toy_inputs()
        baseline, _ = self.capture(model, inputs)
        spec = mapped_l16_patch(baseline.donor16, [2, 1], [1, 2], [1, 2])
        self.assertTrue(torch.equal(spec["donor"]["vectors"], baseline.donor16["vectors"][[1, 0]]))
        with self.assertRaises(ValueError):
            mapped_l16_patch(baseline.donor16, [1], [1], [1, 2])


if __name__ == "__main__":
    unittest.main()
