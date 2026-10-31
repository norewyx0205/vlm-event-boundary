"""Contextual option readouts and compact, read-only Phase 3E-B capture."""

import numpy as np
import torch

from .activation_patching_core import _find_subsequences, _tokenizer_ids_and_offsets
from .phase3b_core import _positions_for_span
from .phase3c_core import checked_positions
from .phase3c_interventions import assert_hidden
from .phase3d_core import statement
from .phase3d_features import option_spans
from .phase3e_preflight import AttentionSites, LAYERS

SCHEMA = "phase3e_option_feature_store_v1"
SITES = tuple(sorted(
    [f"L{layer}_block_output" for layer in LAYERS] +
    [f"L{layer}_input" for layer in (16, 17, 18)] +
    [f"L{layer}_{kind}" for layer in (17, 18)
     for kind in ("attention_contribution", "post_attention_residual")]))
READOUTS = ("reference_last", "reference_mean", "option_final")
FEATURE_NAMES = tuple(f"{kind}:{site}" for kind in READOUTS for site in SITES)
PRIMARY_NAMES = tuple(f"reference_last:L{layer}_block_output" for layer in (16, 17, 18))


def option_readouts(row, record, tokenizer):
    """Resolve full option spans and the actual reference token in this prompt."""
    ids, archived = record["input_ids"], record["group_positions"]
    mentions = option_spans(row, ids, tokenizer, archived)
    question = row.get("question") or "Which statement correctly describes the order of events?"
    core = f"{question}\n\nA: {row['option_A']}\nB: {row['option_B']}\n\nAnswer with only A or B."
    sections = None
    for prefix in ("\n\n", "\n", "", " "):
        contextual, offsets = _tokenizer_ids_and_offsets(tokenizer, prefix+core)
        starts = _find_subsequences(ids, contextual) if contextual else []
        if len(starts) != 1:
            continue
        start = len(prefix)+len(question)+len("\n\nA: ")
        sections = {}
        for option in ("A", "B"):
            text = row[f"option_{option}"]
            positions = sorted(_positions_for_span(offsets, start, start+len(text), starts[0]))
            sections[option] = checked_positions(positions, f"option_{option}", len(ids))
            start += len(text)+len("\nB: ")
        break
    if sections is None:
        raise ValueError("Cannot align the actual option section with the processor input.")
    return readouts_from_sections(row, record, mentions, sections)


def readouts_from_sections(row, record, mentions, sections):
    ids, archived = record["input_ids"], record["group_positions"]
    groups, audit = {}, {}
    for option in ("A", "B"):
        semantics = statement(row, row[f"option_{option}"])
        reference = mentions[f"option_{option}_t{semantics['reference_id']}"]
        subject = mentions[f"option_{option}_t{semantics['subject_id']}"]
        relation = mentions[f"option_{option}_relation"]
        span = sections[option]
        if (not set(span) <= set(archived["options_all"]) or
                not (max(subject) < min(relation) and max(relation) < min(reference)) or
                not set(reference+subject+relation) <= set(span)):
            raise ValueError("Option readout lacks the full causal statement context.")
        groups[f"option_{option}_reference_last"] = [max(reference)]
        groups[f"option_{option}_reference_mean"] = reference
        groups[f"option_{option}_option_final"] = [max(span)]
        groups[f"option_{option}_subject_mean"] = subject
        audit[option] = {"option_text": row[f"option_{option}"], **semantics,
            "full_option_positions": span, "reference_positions": reference,
            "reference_last_position": max(reference), "reference_last_token_id": ids[max(reference)],
            "reference_token_ids": [ids[p] for p in reference],
            "option_final_position": max(span), "subject_positions": subject,
            "relation_positions": relation}
    return groups, audit


def archived_readouts(row, mapping):
    """Recover option sections from previously audited, disjoint full-option spans."""
    positions = sorted(mapping["processor_record"]["group_positions"]["options_all"])
    runs = []
    for position in positions:
        if not runs or position != runs[-1][-1]+1:
            runs.append([])
        runs[-1].append(position)
    if len(runs) != 2:
        raise ValueError("Archived option spans cannot be split unambiguously into A and B.")
    return readouts_from_sections(row, mapping["processor_record"], mapping["capture_groups"],
                                 dict(zip(("A", "B"), runs)))


def audit_paired_readouts(rows, mappings):
    """Require identical contextual reference tokens at identical paired positions."""
    from .phase3e_paired_design import labels_and_pairs
    examples, pairs, boundaries = labels_and_pairs(rows)
    lookup = {e["example_id"]: e for e in examples}
    for pair in pairs+boundaries:
        left = lookup[pair.get("left_example_id", pair.get("low_example_id"))]
        right = lookup[pair.get("right_example_id", pair.get("temporal_example_id"))]
        a = mappings[left["eval_id"]]["option_readout_audit"][left["option"]]
        b = mappings[right["eval_id"]]["option_readout_audit"][right["option"]]
        if a != b:
            raise ValueError("Paired videos changed actual statement/reference token alignment.")
    return {"order_directions_checked": len(pairs), "boundary_pairs_checked": len(boundaries),
            "same_contextual_reference_tokens_and_positions": True}


class CompactOptionSites(AttentionSites):
    """Keep pooled vectors; raw timing checks cover reference-last/final only."""

    def __init__(self, text_model, length, groups):
        super().__init__(text_model, length, groups)
        names = [f"option_{o}_{kind}" for o in ("A", "B")
                 for kind in ("reference_last", "option_final")]
        if not set(names) <= set(groups):
            raise ValueError("Missing exact reference-last/option-final capture positions.")
        self.union = sorted({p for name in names for p in self.groups[name]})

    def take(self, name, hidden):
        assert_hidden(hidden, self.length)
        self.counts[name] += 1
        if self.counts[name] != 1:
            raise RuntimeError("Repeated capture visit; only one prefill is supported.")
        if name.startswith(("L17_", "L18_")):
            self.raw[name] = hidden[0, self.union].detach().cpu().clone()
        self.means[name] = {k: hidden[0, positions].detach().float().mean(0).cpu().clone()
                            for k, positions in self.groups.items()}
        if any(not bool(torch.isfinite(v).all()) for v in self.means[name].values()):
            raise RuntimeError("Nonfinite option/source readout.")

    def validate(self):
        result = super().validate()
        names = [f"option_{o}_{kind}" for o in ("A", "B")
                 for kind in ("reference_last", "option_final")]
        expected = sorted({p for name in names for p in self.groups[name]})
        raw_sites = {site for site in SITES if site.startswith(("L17_", "L18_"))}
        if self.union != expected or set(self.raw) != raw_sites:
            raise RuntimeError("Raw timing support differs from the declared exact option positions.")
        for site, values in self.means.items():
            if set(values) != set(self.groups):
                raise RuntimeError("Missing declared pooled readout.")
            dimensions = {tuple(value.shape) for value in values.values()}
            if len(dimensions) != 1 or len(next(iter(dimensions))) != 1:
                raise RuntimeError("Incompatible compact hidden dimensions.")
            if site in raw_sites:
                raw = self.raw[site]
                if raw.shape != (len(expected), next(iter(dimensions))[0]) or not bool(torch.isfinite(raw).all()):
                    raise RuntimeError("Invalid raw option timing vectors.")
                for name in names:
                    if len(self.groups[name]) != 1 or not torch.equal(
                            raw[expected.index(self.groups[name][0])].float(), values[name]):
                        raise RuntimeError("Exact option vector differs from its saved readout.")
        return {**result, "timing_raw_scope": "reference_last_and_option_final_only"}


def statement_features(payload, option, names=FEATURE_NAMES):
    values = []
    for name in names:
        kind, site = name.split(":")
        if kind not in READOUTS or site not in SITES:
            raise ValueError("Unknown declared option feature.")
        value = payload["means"][site][f"option_{option}_{kind}"]
        values.append(value.detach().cpu().float().numpy())
    result = np.stack(values)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("Expected finite [named feature, hidden dimension] vectors.")
    return result


def from_preflight_payload(payload, readouts):
    """CPU smoke export at L17/L18 from existing raw captures; never fit a probe."""
    position_lookup = {p: i for i, p in enumerate(payload["union_positions"])}
    means = {}
    for site, raw in payload["timing_raw"].items():
        means[site] = {}
        for name, positions in readouts.items():
            if not set(positions) <= set(position_lookup):
                raise ValueError("Old capture does not contain the newly requested readout.")
            selected = raw[[position_lookup[p] for p in positions]].float()
            means[site][name] = selected.mean(0)
    return {"means": means}
