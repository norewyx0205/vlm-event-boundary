"""Section-scoped token groups and compact, timing-audited Phase 3D capture."""

import inspect
import re
from contextlib import ExitStack, contextmanager

import torch

try:
    from .activation_patching_core import _find_subsequences, _tokenizer_ids_and_offsets
    from .phase3b_core import _positions_for_span
    from .phase3c_core import checked_positions
    from .phase3c_interventions import ResidualSites, assert_hidden
except ImportError:
    from activation_patching_core import _find_subsequences, _tokenizer_ids_and_offsets
    from phase3b_core import _positions_for_span
    from phase3c_core import checked_positions
    from phase3c_interventions import ResidualSites, assert_hidden


def option_spans(row, input_ids, tokenizer, archived_groups):
    """Locate each identity separately inside each actual contextual option."""
    question = row.get("question") or "Which statement correctly describes the order of events?"
    core = f"{question}\n\nA: {row['option_A']}\nB: {row['option_B']}\n\nAnswer with only A or B."
    for prefix in ("\n\n", "\n", "", " "):
        ids, offsets = _tokenizer_ids_and_offsets(tokenizer, prefix+core)
        starts = _find_subsequences(input_ids, ids) if ids else []
        if len(starts) != 1:
            continue
        groups, start = {}, len(prefix)+len(question)+len("\n\nA: ")
        for option in ("A", "B"):
            text = row[f"option_{option}"]
            for obj in row["target_objects"]:
                matches = list(re.finditer(re.escape(obj["label"]), text, re.IGNORECASE))
                if len(matches) != 1:
                    raise ValueError("Each option must mention each literal target exactly once.")
                match = matches[0]
                groups[f"option_{option}_t{obj['id']}"] = sorted(_positions_for_span(
                    offsets, start+match.start(), start+match.end(), starts[0]))
            relations = list(re.finditer(r"\b(before|after)\b", text, re.IGNORECASE))
            if len(relations) != 1:
                raise ValueError("Each option needs exactly one relation word.")
            match = relations[0]
            groups[f"option_{option}_relation"] = sorted(_positions_for_span(
                offsets, start+match.start(), start+match.end(), starts[0]))
            start += len(text)+len("\nB: ")
        for identity in (1, 2):
            union = set(groups[f"option_A_t{identity}"]) | set(groups[f"option_B_t{identity}"])
            if union != set(archived_groups[f"option_target_{identity}_mentions"]):
                raise ValueError("Per-option mentions disagree with audited contextual token positions.")
        if (set(groups["option_A_relation"]) | set(groups["option_B_relation"]) !=
                set(archived_groups["option_temporal_relations"])):
            raise ValueError("Per-option relations disagree with the audited text group.")
        for name, positions in groups.items():
            checked_positions(positions, name, len(input_ids))
            if not set(positions) <= set(archived_groups["options_all"]):
                raise ValueError("An option subgroup escaped the actual option section.")
        return groups
    raise ValueError("Cannot locate the actual contextual option text; do not use standalone word IDs.")


def capture_groups(row, record, whole_positions, tokenizer):
    groups = option_spans(row, record["input_ids"], tokenizer, record["group_positions"])
    for name in ("query_all", "decision_position"):
        groups[name] = record["group_positions"][name]
    groups["whole_event2"] = whole_positions
    for target in (1, 2):
        values = record["group_positions"][f"video_t{target}_e2"]
        # Do not silently crop a different ROI into a seemingly complete source.
        if not set(values) <= set(whole_positions):
            raise ValueError("Literal target source group extends beyond the matched Event-2 support.")
        groups[f"source_t{target}"] = values
    for name, positions in groups.items():
        checked_positions(positions, name, len(record["input_ids"]))
    return groups


def mapped_l16_patch(donor_state, donor_positions, recipient_positions, recipient_whole_support):
    """Reorder observed vectors using explicit correspondence, never shape alone."""
    checked_positions(donor_positions, "explicit donor correspondence")
    checked_positions(recipient_positions, "explicit recipient correspondence")
    checked_positions(donor_state["positions"], "captured donor positions")
    checked_positions(recipient_whole_support, "complete recipient support")
    if (donor_state.get("layer") != 16 or donor_state.get("location") != "block_output" or
            len(donor_positions) != len(recipient_positions) or
            set(donor_positions) != set(donor_state["positions"]) or
            set(recipient_positions) != set(recipient_whole_support)):
        raise ValueError("Require complete one-to-one observed L16 donor/recipient support.")
    lookup = {position: index for index, position in enumerate(donor_state["positions"])}
    donor = {"layer": 16, "location": "block_output", "positions": donor_positions,
             "vectors": donor_state["vectors"][[lookup[position] for position in donor_positions]].clone()}
    return {"layer": 16, "location": "block_output", "recipient_positions": recipient_positions, "donor": donor}


class CompactSites(ResidualSites):
    """Pool planned groups on device; retain raw donor vectors only at L16."""

    def __init__(self, text_model, length, visual_positions, groups, patch=None):
        super().__init__(text_model, length, visual_positions, capture_positions=None, patch=patch)
        if patch and (patch["layer"] != 16 or patch["location"] != "block_output" or
                      set(patch["recipient_positions"]) != set(groups["whole_event2"])):
            raise ValueError("Phase 3D only patches the complete planned visual support at L16 output.")
        self.groups = {name: checked_positions(list(values), name, length) for name, values in groups.items()}
        required = {"source_t1", "source_t2", "whole_event2", "decision_position", "query_all"}
        required |= {f"option_{option}_t{target}" for option in ("A", "B") for target in (1, 2)}
        if not required <= set(self.groups):
            raise ValueError("Incomplete compact capture groups.")
        visual = set(visual_positions)
        for name, positions in self.groups.items():
            if name.startswith("source_") or name == "whole_event2":
                if not set(positions) <= visual:
                    raise ValueError("Visual source support includes nonvisual tokens.")
            elif set(positions) & visual:
                raise ValueError("Text readout overlaps the visual intervention support.")
        self.means, self.input16, self.donor16, self.text16 = {}, None, None, None
        self.previous15 = None
        self.input16_visits = 0
        self.union = sorted({p for positions in self.groups.values() for p in positions})

    def pool(self, hidden):
        return {name: hidden[0, values].detach().float().mean(0).cpu().clone()
                for name, values in self.groups.items()}

    def visit(self, hidden, layer, location):
        hidden = super().visit(hidden, layer, location)
        if location != "block_output":
            return hidden
        self.means[layer] = self.pool(hidden)
        if layer == 15:
            self.previous15 = hidden[0, self.union].detach().cpu().clone()
        if layer == 16:
            self.donor16 = {"layer": 16, "location": "block_output",
                            "positions": self.groups["whole_event2"],
                            "vectors": hidden[0, self.groups["whole_event2"]].detach().cpu().clone()}
            text_positions = sorted({p for name, positions in self.groups.items()
                                     if name.startswith("option_") for p in positions})
            self.text16 = hidden[0, text_positions].detach().cpu().clone()
        return hidden

    @contextmanager
    def installed(self):
        def before(module, args, kwargs):
            arguments = inspect.signature(module.forward).bind(*args, **kwargs).arguments
            hidden = arguments["hidden_states"]
            assert_hidden(hidden, self.length)
            self.input16_visits += 1
            if (self.input16_visits != 1 or self.previous15 is None or
                    not torch.equal(hidden[0, self.union].detach().cpu(), self.previous15)):
                raise RuntimeError("Actual L16 input differs from captured L15 output at planned positions.")
            self.input16 = self.pool(hidden)
        with ExitStack() as stack:
            stack.enter_context(super().installed())
            stack.callback(self.layers[16].register_forward_pre_hook(before, with_kwargs=True).remove)
            yield self

    def validate(self):
        result = super().validate()
        if set(self.means) != set(range(36)) or self.input16_visits != 1 or self.donor16 is None:
            raise RuntimeError("Incomplete compact block/input capture.")
        if any(not bool(torch.isfinite(value).all()) for groups in self.means.values() for value in groups.values()):
            raise RuntimeError("Nonfinite compact feature vectors.")
        return {**result, "L16_input_matches_L15_output_at_planned_positions": True,
                "block_output_mean_sites": 36, "raw_donor_layer": 16,
                "pooling": "per-option means equally averaged; canonical T1 minus T2"}

    def feature_views(self):
        self.validate()
        result = {}
        for layer, means in self.means.items():
            result[f"source_pair:L{layer}_output"] = means["source_t1"]-means["source_t2"]
            result[f"option_pair:L{layer}_output"] = (
                means["option_A_t1"]+means["option_B_t1"]-
                means["option_A_t2"]-means["option_B_t2"])/2
            result[f"decision:L{layer}_output"] = means["decision_position"]
            result[f"query:L{layer}_output"] = means["query_all"]
        result["source_pair:L16_input"] = self.input16["source_t1"]-self.input16["source_t2"]
        return result


def assert_patch_timing(baseline, patched):
    """Block-output replacement cannot retroactively alter same-block text."""
    baseline.validate()
    patched.validate()
    if baseline.groups != patched.groups or not patched.patch:
        raise ValueError("Compare the same recipient groups with a real planned patch.")
    for layer in range(17):
        for name in baseline.groups:
            if layer == 16 and (name.startswith("source_") or name == "whole_event2"):
                continue
            if not torch.equal(baseline.means[layer][name], patched.means[layer][name]):
                raise RuntimeError("Patch altered an upstream or already-computed same-block readout.")
    if any(not torch.equal(baseline.input16[name], patched.input16[name]) for name in baseline.groups):
        raise RuntimeError("L16 output patch changed the earlier L16 input.")
    if not torch.equal(baseline.text16, patched.text16):
        raise RuntimeError("Already-computed L16 option token vectors changed, even if means cancel.")
    return {"upstream_and_same_block_text_exact": True, "first_valid_downstream_block": 17}
