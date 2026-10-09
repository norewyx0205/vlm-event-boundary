"""Location-matched residual interventions and causal attention masks for Phase 3C."""

import inspect
from contextlib import ExitStack, contextmanager

import torch

try:
    from .phase3c_core import checked_positions, edge_budget
except ImportError:
    from phase3c_core import checked_positions, edge_budget


SITE_SEMANTICS = {
    "block_output": "decoder_block_output_before_any_deepstack_addition",
    "post_deepstack": "after_deepstack_addition_before_next_decoder_block",
}


def hidden_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise ValueError("Unsupported residual output structure.")


def replace_output(output, hidden):
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise ValueError("Unsupported residual output structure.")


def assert_hidden(hidden, length):
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != length:
        raise ValueError("Intervention requires one full, unpadded prompt without a KV cache.")


def transplant(hidden, positions, donor, layer, location):
    positions = checked_positions(positions, "recipient positions", hidden.shape[1])
    if donor.get("layer") != layer or donor.get("location") != location:
        raise ValueError("Donor capture semantics do not match the recipient intervention site.")
    checked_positions(donor["positions"], "donor positions")
    values = donor["vectors"]
    if (not torch.is_tensor(values) or values.ndim != 2 or
            values.shape != (len(positions), hidden.shape[-1]) or
            len(donor["positions"]) != len(positions) or values.dtype != hidden.dtype or
            not bool(torch.isfinite(values).all())):
        raise ValueError("Invalid/non-finite donor vectors, precision or one-to-one support.")
    result = hidden.clone()
    index = torch.tensor(positions, device=hidden.device, dtype=torch.long)
    values = values.to(hidden.device)
    result[0, index, :] = values
    if not torch.equal(result[0].index_select(0, index), values):
        raise RuntimeError("Residual replacement did not install the exact donor vectors.")
    return result


class ResidualSites:
    """Observe all block outputs and the actual post-addition DeepStack states."""

    def __init__(self, text_model, length, visual_positions, capture_positions=None, patch=None):
        self.text = text_model
        self.layers = text_model.layers
        if len(self.layers) != 36 or not callable(getattr(text_model, "_deepstack_process", None)):
            raise ValueError("Phase 3C requires the 36-block Qwen3-VL DeepStack interface.")
        self.length = length
        self.visual_positions = checked_positions(visual_positions, "visual positions", length)
        self.positions = (checked_positions(capture_positions, "capture union", length)
                          if capture_positions is not None else None)
        self.patch = patch
        if patch and (patch["location"] not in SITE_SEMANTICS or
                      type(patch["layer"]) is not int or not 0 <= patch["layer"] < 36 or
                      (patch["location"] == "post_deepstack" and patch["layer"] not in (0, 1, 2))):
            raise ValueError("Invalid patch location/layer.")
        self.states, self.visits, self.deepstack_layers = {}, [], []
        self.applied = 0

    def visit(self, hidden, layer, location):
        assert_hidden(hidden, self.length)
        if self.patch and (layer, location) == (self.patch["layer"], self.patch["location"]):
            hidden = transplant(hidden, self.patch["recipient_positions"], self.patch["donor"], layer, location)
            self.applied += 1
        if self.positions is not None:
            index = torch.tensor(self.positions, device=hidden.device, dtype=torch.long)
            self.states[(layer, location)] = {
                "layer": layer, "location": location, "positions": self.positions,
                "vectors": hidden[0].index_select(0, index).detach().cpu().clone(),
            }
        return hidden

    @contextmanager
    def installed(self):
        if self.visits or self.deepstack_layers:
            raise RuntimeError("Residual controller cannot be reused across forwards.")
        original = self.text._deepstack_process
        had_instance_override = "_deepstack_process" in self.text.__dict__
        prior_override = self.text.__dict__.get("_deepstack_process")

        def after_addition(hidden, visual_mask, embeddings):
            layer = self.visits[-1] if self.visits else None
            if layer != len(self.deepstack_layers) or layer not in (0, 1, 2):
                raise RuntimeError("Unexpected DeepStack injection order/location.")
            assert_hidden(hidden, self.length)
            if visual_mask.shape != (1, self.length) or visual_mask.dtype != torch.bool:
                raise ValueError("Unexpected DeepStack visual mask.")
            actual = visual_mask[0].nonzero().flatten().detach().cpu().tolist()
            if actual != self.visual_positions or embeddings.shape != (len(actual), hidden.shape[-1]):
                raise ValueError("DeepStack visual support differs from actual processor positions.")
            index = torch.tensor(actual, device=hidden.device, dtype=torch.long)
            expected = hidden[0].index_select(0, index) + embeddings.to(hidden.device, hidden.dtype)
            output = original(hidden, visual_mask, embeddings)
            if not torch.equal(output[0].index_select(0, index), expected):
                raise RuntimeError("Post-DeepStack state is not the observed addition result.")
            self.deepstack_layers.append(layer)
            return self.visit(output, layer, "post_deepstack")

        def output_hook(layer):
            def hook(_module, _args, output):
                if layer != len(self.visits):
                    raise RuntimeError("Decoder hooks were not called exactly once in order.")
                self.visits.append(layer)
                hidden = self.visit(hidden_output(output), layer, "block_output")
                return replace_output(output, hidden)
            return hook

        with ExitStack() as stack:
            for layer, module in enumerate(self.layers):
                stack.callback(module.register_forward_hook(output_hook(layer)).remove)
            self.text._deepstack_process = after_addition
            try:
                yield self
            finally:
                if had_instance_override:
                    self.text._deepstack_process = prior_override
                else:
                    del self.text._deepstack_process

    def validate(self):
        if self.visits != list(range(36)) or self.deepstack_layers != [0, 1, 2]:
            raise RuntimeError("Incomplete decoder/DeepStack hook execution.")
        if self.patch and self.applied != 1:
            raise RuntimeError("Planned patch did not execute exactly once.")
        if self.positions is not None and len(self.states) != 39:
            raise RuntimeError("Missing all-layer or post-DeepStack snapshots.")
        return {"block_output_sites": len(self.visits), "post_deepstack_layers": self.deepstack_layers,
                "patch_applied_count": self.applied, "site_semantics": SITE_SEMANTICS}


def selected_state(states, layer, location, positions):
    state = states[(layer, location)]
    lookup = {position: index for index, position in enumerate(state["positions"])}
    checked_positions(positions, "selected donor positions")
    if not set(positions) <= set(lookup):
        raise ValueError("Donor capture is missing planned positions.")
    return {"layer": layer, "location": location, "positions": positions,
            "vectors": state["vectors"][[lookup[position] for position in positions]].clone()}


def knockout_mask(mask, queries, keys, length, heads=32):
    expected = edge_budget(queries, keys, length)
    if (not torch.is_tensor(mask) or mask.ndim != 4 or mask.shape[0] != 1 or
            mask.shape[1] not in (1, heads) or mask.shape[2:] != (length, length) or
            not mask.is_floating_point() or bool(torch.isnan(mask).any()) or bool((mask > 0).any())):
        raise ValueError("Knockout requires an explicit additive full-prompt causal mask.")
    q = torch.tensor(queries, device=mask.device, dtype=torch.long)
    k = torch.tensor(keys, device=mask.device, dtype=torch.long)
    rows = mask[0, :, q, :]
    allowed = rows > torch.finfo(mask.dtype).min / 2
    chronological = torch.arange(length, device=mask.device)[None, :] <= q[:, None]
    if not torch.equal(allowed, chronological.unsqueeze(0).expand_as(allowed)):
        raise ValueError("Actual causal visibility differs from the audited edge-budget assumption.")
    counts = allowed[:, :, k].sum(-1)
    expected_counts = torch.tensor(expected["visible_causal_edges_by_query"], device=mask.device)
    if not torch.equal(counts, expected_counts.unsqueeze(0).expand_as(counts)):
        raise ValueError("Actual masked-edge budget differs from the CPU audit.")
    result = mask.clone()
    result[:, :, q[:, None], k[None, :]] = float("-inf")
    return result, expected


class AttentionKnockout:
    """Mask real attention inputs; reduce diagnostics to selected rows/edges only."""

    def __init__(self, layers, window, queries, keys, length, enabled=True, heads=32):
        if (not window or len(set(window)) != len(window) or
                any(type(layer) is not int or not 0 <= layer < len(layers) for layer in window)):
            raise ValueError("Invalid knockout layer window.")
        self.layers, self.window = layers, window
        self.queries = checked_positions(queries, "knockout queries", length)
        self.keys = checked_positions(keys, "knockout keys", length)
        self.length, self.enabled, self.heads = length, enabled, heads
        self.budget = edge_budget(queries, keys, length)
        self.diagnostics, self.pending = {}, set()

    @contextmanager
    def installed(self):
        if self.diagnostics or self.pending:
            raise RuntimeError("Knockout controller cannot be reused across forwards.")

        def before(layer):
            def hook(module, args, kwargs):
                if layer in self.pending:
                    raise RuntimeError("Duplicate attention call in a planned knockout layer.")
                if module.config._attn_implementation != "eager":
                    raise ValueError("Attention knockout requires eager attention.")
                bound = inspect.signature(module.forward).bind(*args, **kwargs)
                hidden = bound.arguments["hidden_states"]
                assert_hidden(hidden, self.length)
                if bound.arguments.get("past_key_values") is not None:
                    raise ValueError("Knockout forbids cached attention.")
                mask = bound.arguments.get("attention_mask")
                modified, budget = knockout_mask(mask, self.queries, self.keys, self.length, self.heads)
                if budget != self.budget:
                    raise RuntimeError("Knockout budget changed during execution.")
                self.pending.add(layer)
                if not self.enabled:
                    return None
                bound.arguments["attention_mask"] = modified
                return bound.args, bound.kwargs
            return hook

        def after(layer):
            def hook(_module, _args, output):
                if layer not in self.pending or not isinstance(output, tuple) or len(output) != 2:
                    raise RuntimeError("Unsupported eager attention output/call order.")
                weights = output[1]
                if weights is None or weights.shape != (1, self.heads, self.length, self.length):
                    raise ValueError("Actual attention weights are unavailable or misaligned.")
                q = torch.tensor(self.queries, device=weights.device, dtype=torch.long)
                k = torch.tensor(self.keys, device=weights.device, dtype=torch.long)
                rows = weights[0].index_select(1, q)
                blocked = rows.index_select(2, k)
                if not bool(torch.isfinite(rows).all()):
                    raise RuntimeError("Non-finite attention after knockout.")
                maximum = float(blocked.abs().max())
                sums = rows.float().sum(-1)
                if self.enabled and maximum != 0:
                    raise RuntimeError("Planned attention edges still carry probability mass.")
                if not torch.allclose(sums, torch.ones_like(sums), atol=0.005, rtol=0):
                    raise RuntimeError("Remaining attention was not renormalized.")
                self.diagnostics[layer] = {"max_blocked_probability": maximum,
                    "mean_selected_edge_mass_per_query_head": float(blocked.float().sum(-1).mean()),
                    "max_row_sum_error": float((sums - 1).abs().max()), "budget": self.budget}
            return hook

        with ExitStack() as stack:
            for layer in self.window:
                module = self.layers[layer].self_attn
                stack.callback(module.register_forward_pre_hook(before(layer), with_kwargs=True).remove)
                stack.callback(module.register_forward_hook(after(layer)).remove)
            yield self

    def validate(self):
        if set(self.diagnostics) != set(self.window) or self.pending != set(self.window):
            raise RuntimeError("Knockout did not execute at every requested layer.")
        return {"enabled": self.enabled, "orientation": "text_queries_to_visual_keys",
                "pre_softmax_mask": "negative_infinity", "all_heads": self.heads,
                "renormalized": True, "layers": {str(key): value for key, value in self.diagnostics.items()}}
