"""CPU-only support, control-budget and selection rules for Phase 3C."""

import hashlib
import json
import math
import os
import struct
import tempfile
from collections import Counter
from pathlib import Path


SCHEMA = "phase3c_mechanistic_pilot_v1"
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
CONDITIONS = ("low_boundary", "temporal_boundary")
STRATA = {"primary_rescue": "rescue", "stable_both_correct_control": "stable"}
TEXT_GROUPS = ("options_all", "query_all")
ROI_GROUPS = ("video_t1_e2", "video_t2_e2")


def digest(payload):
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def input_ids_hash(ids):
    if not ids or any(type(value) is not int or value < 0 for value in ids):
        raise ValueError("Invalid actual prompt token IDs.")
    return hashlib.sha256(struct.pack("<" + "q" * len(ids), *ids)).hexdigest()


def boundary_outcomes(low, temporal, low_ko, temporal_ko):
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for value in (low, temporal, low_ko, temporal_ko)):
        raise ValueError("Boundary outcomes require finite margins in both conditions.")
    return {"delta_M_temporal": temporal_ko - temporal, "delta_M_low": low_ko - low,
            "advantage_base": temporal - low, "advantage_KO": temporal_ko - low_ko,
            "compression": (temporal - low) - (temporal_ko - low_ko)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write(path, payload, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            if jsonl:
                for row in payload:
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            else:
                json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def frozen_write(path, payload, jsonl=False):
    path = Path(path)
    if path.exists():
        prior = read_jsonl(path) if jsonl else read_json(path)
        if prior != payload:
            raise ValueError(f"Frozen Phase 3C artifact differs: {path}; use a new output root.")
    else:
        atomic_write(path, payload, jsonl)


def checked_positions(values, name, length=None, allow_empty=False):
    if not isinstance(values, list) or (not values and not allow_empty):
        raise ValueError(f"Missing/empty position list: {name}.")
    if any(type(value) is not int or value < 0 or (length is not None and value >= length)
           for value in values) or len(set(values)) != len(values):
        raise ValueError(f"Invalid or duplicate positions: {name}.")
    return values


def baseline_failures(row, index):
    failures = []
    decision = index.get("decision", {})
    margin = decision.get("margin")
    if not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin == 0:
        failures.append("baseline_margin_missing_nonfinite_or_zero")
    else:
        correct = decision.get("prediction") == row["correct_option"]
        if correct != (margin > 0):
            failures.append("prediction_margin_disagreement")
        if decision.get("is_correct") != correct:
            failures.append("correctness_metadata_disagreement")
        logits = (decision.get("correct_logit"), decision.get("incorrect_logit"))
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in logits):
            failures.append("missing_or_nonfinite_baseline_logits")
        elif not math.isclose(logits[0] - logits[1], margin, abs_tol=1e-6):
            failures.append("baseline_logit_margin_disagreement")
    if decision.get("prediction") not in ("A", "B"):
        failures.append("non_ab_first_token")
    if decision.get("correct_option") != row["correct_option"]:
        failures.append("correct_option_disagreement")
    if decision.get("prediction") != row.get("archived_prediction"):
        failures.append("archived_prediction_disagreement")
    parity = index.get("standard_parity", {})
    if parity.get("first_token_match") is not True or parity.get("logits_allclose") is not True:
        failures.append("missing_or_failed_standard_parity")
    if index.get("archived_input_parity", {}).get("matches") is not True:
        failures.append("missing_or_failed_archived_input_parity")
    if index.get("eval_id") != row["eval_id"]:
        failures.append("capture_eval_id_disagreement")
    return failures


def select_balanced(candidates, eligible_ids=None):
    """Choose ascending base IDs within fixed mover/behavior quotas, without effects."""
    eligible = set(eligible_ids) if eligible_ids is not None else None
    ordered = sorted(candidates, key=lambda row: (
        row["base_sample_id"], row["prompt_variant"] != "original", row["pair_id"],
    ))
    selected, used, counts = [], set(), Counter()
    quotas = {("rescue", 1): 4, ("rescue", 2): 4, ("stable", 1): 2, ("stable", 2): 2}
    for row in ordered:
        key = (row["stratum"], row["first_object_id"])
        if key not in quotas or row["base_sample_id"] in used:
            continue
        if eligible is not None and row["pair_id"] not in eligible:
            continue
        if counts[key] < quotas[key]:
            selected.append(row)
            used.add(row["base_sample_id"])
            counts[key] += 1
    missing = {f"{stratum}_target_{mover}_first": quota - counts[(stratum, mover)]
               for (stratum, mover), quota in quotas.items() if counts[(stratum, mover)] < quota}
    return selected, missing


def event_layout(row, video_metadata, visual_positions, prompt_length):
    positions = checked_positions(visual_positions, "visual_positions", prompt_length)
    if positions != sorted(positions):
        raise ValueError("Visual positions must follow processor sequence order.")
    grid = video_metadata["merged_video_grid_thw"]
    if len(grid) != 3 or any(type(value) is not int or value <= 0 for value in grid):
        raise ValueError("Invalid merged visual grid.")
    temporal, height, width = grid
    frames = video_metadata["source_frame_groups"]
    if len(positions) != temporal * height * width or len(frames) != temporal:
        raise ValueError("Actual visual positions/frame groups disagree with merged grid.")
    if video_metadata.get("visual_token_count") != len(positions):
        raise ValueError("Visual-token count metadata disagrees with actual positions.")
    timing = row["event_timing"]
    start, end = timing["second_event_start_frame"], timing["second_event_end_frame"]
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or end <= start:
        raise ValueError("Invalid Event-2 interval.")
    first_start, first_end = timing["first_event_start_frame"], timing["first_event_end_frame"]
    if first_start >= first_end or first_end > start:
        raise ValueError("Overlapping or invalid target-event intervals.")
    total = row["total_frames"]
    bins, lookup = [], {}
    for index, group in enumerate(frames):
        if not isinstance(group, list) or not group or any(
            type(frame) is not int or frame < 0 or frame >= total for frame in group
        ):
            raise ValueError("Invalid sampled-frame group.")
        # Match Phase 3B's inclusive Event-2 end and strict phase dominance.
        fraction = sum(start <= frame <= end for frame in group) / len(group)
        if fraction <= 0.5:
            continue
        progress = (sum(group) / len(group) - start) / (end - start)
        cells = {}
        for y in range(height):
            for x in range(width):
                position = positions[index * height * width + y * width + x]
                cells[(y, x)] = position
                lookup[position] = (index, y, x)
        bins.append({"temporal_index": index, "source_frames": group,
                     "event_fraction": fraction, "progress": progress, "cells": cells})
    if not bins:
        raise ValueError("No dominant Event-2 temporal bins.")
    return {"grid": grid, "bins": bins, "lookup": lookup, "prompt_length": prompt_length}


def event_bin_pairs(low_layout, temporal_layout, max_progress_error):
    if not 0 < max_progress_error <= 1:
        raise ValueError("Maximum progress error must be in (0, 1].")
    if low_layout["grid"][1:] != temporal_layout["grid"][1:]:
        raise ValueError("Merged spatial grids differ; no one-to-one spatial correspondence.")
    low, temporal = low_layout["bins"], temporal_layout["bins"]
    if len(low) != len(temporal):
        raise ValueError("Unequal Event-2 bin counts; complete primary mapping is unavailable.")
    pairs = []
    for source, target in zip(low, temporal):
        error = abs(source["progress"] - target["progress"])
        if error > max_progress_error:
            raise ValueError("Event-relative progress error exceeds the pre-specified bound.")
        pairs.append((source, target))
    return pairs


def reference_support(mapping, low_group, temporal_group, layouts):
    low = checked_positions(low_group, "low ROI", layouts[0]["prompt_length"])
    temporal = checked_positions(temporal_group, "temporal ROI", layouts[1]["prompt_length"])
    mapped_low = checked_positions(mapping["source_positions"], "mapped low ROI")
    mapped_temporal = checked_positions(mapping["target_positions"], "mapped temporal ROI")
    if len(mapped_low) != len(mapped_temporal):
        raise ValueError("Reference donor/recipient counts differ.")
    if not set(mapped_low) <= set(low) or not set(mapped_temporal) <= set(temporal):
        raise ValueError("Reference mapping contains positions outside the actual ROI.")
    if (mapping.get("source_token_count"), mapping.get("target_token_count"), mapping.get("mapped_token_count")) != (
            len(low), len(temporal), len(mapped_low)):
        raise ValueError("Reference support counts disagree with actual positions.")
    temporal_pairs = {item["source_temporal_index"]: item["target_temporal_index"]
                      for item in mapping["bin_pairs"]}
    if len(temporal_pairs) != len(mapping["bin_pairs"]) or len(set(temporal_pairs.values())) != len(temporal_pairs):
        raise ValueError("Reference temporal-bin correspondence is not one-to-one.")
    for left, right in zip(mapped_low, mapped_temporal):
        source, target = layouts[0]["lookup"][left], layouts[1]["lookup"][right]
        if source[1:] != target[1:] or temporal_pairs.get(source[0]) != target[0]:
            raise ValueError("Reference mapping does not preserve observed spatial cells.")
    return support_record(mapped_low, mapped_temporal, low, temporal, "single_roi_reference")


def support_record(low, temporal, low_support, temporal_support, scope):
    low, temporal = checked_positions(low, "mapped low"), checked_positions(temporal, "mapped temporal")
    low_support = checked_positions(low_support, "declared low support")
    temporal_support = checked_positions(temporal_support, "declared temporal support")
    if len(low) != len(temporal):
        raise ValueError("Support correspondence is not one-to-one.")
    if not set(low) <= set(low_support) or not set(temporal) <= set(temporal_support):
        raise ValueError("Mapped positions lie outside declared support.")
    return {
        "scope": scope, "method": "observed_donor_event_relative_replace",
        "low_positions": low, "temporal_positions": temporal,
        "mapped_token_count": len(low), "one_to_one": True,
        "low_support_count": len(low_support), "temporal_support_count": len(temporal_support),
        "unmatched_low_positions": sorted(set(low_support) - set(low)),
        "unmatched_temporal_positions": sorted(set(temporal_support) - set(temporal)),
        "directions": {
            "temporal_to_low": {"donor_condition": "temporal_boundary",
                "recipient_condition": "low_boundary", "recipient_coverage": len(low) / len(low_support)},
            "low_to_temporal": {"donor_condition": "low_boundary",
                "recipient_condition": "temporal_boundary", "recipient_coverage": len(temporal) / len(temporal_support)},
        },
    }


def expanded_supports(bin_pairs, groups):
    both_low, both_temporal, all_low, all_temporal = [], [], [], []
    roi = [{position for name in ROI_GROUPS for position in group[name]} for group in groups]
    for low, temporal in bin_pairs:
        for cell in sorted(low["cells"]):
            left, right = low["cells"][cell], temporal["cells"][cell]
            all_low.append(left)
            all_temporal.append(right)
            if left in roi[0] or right in roi[1]:
                both_low.append(left)
                both_temporal.append(right)
    return {
        "both_targets_event2": support_record(both_low, both_temporal, both_low, both_temporal,
            "cross_condition_roi_union_with_observed_grid_cells"),
        "whole_event2": support_record(all_low, all_temporal, all_low, all_temporal,
            "all_cells_in_dominant_event2_bins"),
    }


def edge_budget(queries, keys, prompt_length):
    queries = checked_positions(queries, "queries", prompt_length)
    keys = checked_positions(keys, "keys", prompt_length)
    visible = [sum(key <= query for key in keys) for query in queries]
    if not sum(visible):
        raise ValueError("Knockout contains no causally visible edges.")
    if any(count >= query + 1 for query, count in zip(queries, visible)):
        raise ValueError("Knockout would remove every permitted key for a query.")
    return {"query_count": len(queries), "key_count": len(keys),
            "visible_causal_edges_by_query": visible,
            "visible_causal_edges_per_head": sum(visible),
            "all_head_visible_causal_edges_per_layer": 32 * sum(visible),
            "already_masked_edges_per_head": len(queries) * len(keys) - sum(visible)}


def match_control(queries, target_keys, candidates, layout, seed_key):
    target = checked_positions(target_keys, "target keys", layout["prompt_length"])
    pool = checked_positions(candidates, "control candidates", layout["prompt_length"], allow_empty=True)
    if not (set(target) | set(pool)) <= set(layout["lookup"]):
        raise ValueError("Knockout/control keys must lie in audited Event-2 cells.")
    if set(pool) & set(target):
        raise ValueError("Control candidates overlap target keys.")
    target_bins, pool_bins = {}, {}
    for position in target:
        target_bins.setdefault(layout["lookup"][position][0], []).append(position)
    for position in pool:
        pool_bins.setdefault(layout["lookup"][position][0], []).append(position)
    chosen = []
    for index, keys in sorted(target_bins.items()):
        options = pool_bins.get(index, [])
        if len(options) < len(keys):
            return {"eligible": False, "reason": "insufficient_same_bin_control_keys",
                    "temporal_index": index, "needed": len(keys), "available": len(options)}
        options = sorted(options, key=lambda position: digest([seed_key, index, position]))
        chosen.extend(options[:len(keys)])
    budget = edge_budget(queries, target, layout["prompt_length"])
    actual = edge_budget(queries, chosen, layout["prompt_length"])
    if budget != actual:
        return {"eligible": False, "reason": "causal_edge_budget_mismatch",
                "target_budget": budget, "control_budget": actual}
    return {"eligible": True, "query_positions": queries, "target_key_positions": target,
            "control_key_positions": chosen, "target_budget": budget, "control_budget": actual,
            "event_scope": "event_2", "key_counts_by_temporal_bin": {
                str(index): len(keys) for index, keys in sorted(target_bins.items())},
            "baseline_blocked_attention_mass": None,
            "baseline_blocked_attention_mass_status": "requires_gpu_forward_diagnostic"}


def prepare_support_audit(candidate, mapping, records, max_progress_error=0.1, seed=42):
    layouts, groups = [], []
    for condition in CONDITIONS:
        record = records[condition]
        index = candidate["capture_indices"][condition]
        ids = record["input_ids"]
        if len(ids) != record["prompt_token_count"] or input_ids_hash(ids) != record["prompt_input_ids_sha256"]:
            raise ValueError("Actual token IDs disagree with the recorded prompt hash/length.")
        if record["attention_mask"] != [1] * len(ids):
            raise ValueError("This pilot requires a single unpadded processor input.")
        if record["visual_positions"] != [position for position, token in enumerate(ids) if token == 151656]:
            raise ValueError("Actual visual positions disagree with the pinned video placeholder IDs.")
        if record["sampled_frame_indices"] != [frame for group in record["video_metadata"]["source_frame_groups"] for frame in group]:
            raise ValueError("Frame groups are not backed by actual sampled-frame indices.")
        if record["group_positions"] != index["group_positions"]:
            raise ValueError("Live processor groups differ from archived capture groups.")
        metadata_key = "low_video_metadata" if condition == CONDITIONS[0] else "temporal_video_metadata"
        if record["video_metadata"] != mapping[metadata_key]:
            raise ValueError("Live processor video metadata differs from the archived mapping.")
        if record["prompt_input_ids_sha256"] != mapping["prompt_input_ids_sha256"]:
            raise ValueError("Actual prompt token IDs differ from the archived mapping.")
        layouts.append(event_layout(candidate["rows"][condition], record["video_metadata"],
            record["visual_positions"], record["prompt_token_count"]))
        groups.append(record["group_positions"])
        for name in ROI_GROUPS + ("video_distractors_e2",):
            if not set(groups[-1][name]) <= set(layouts[-1]["lookup"]):
                raise ValueError("An Event-2 ROI group lies outside its audited event bins.")
        background = set(record["background_event2_positions"])
        objects = {position for name in ROI_GROUPS + ("video_distractors_e2",) for position in groups[-1][name]}
        if background & objects:
            raise ValueError("Background control overlaps an annotated object ROI.")
    bins = event_bin_pairs(*layouts, max_progress_error)
    supports = {name: reference_support(mapping["groups"][name], groups[0][name], groups[1][name], layouts)
                for name in ROI_GROUPS}
    supports.update(expanded_supports(bins, groups))
    controls = {}
    for side, condition in enumerate(CONDITIONS):
        controls[condition] = {}
        layout, group, record = layouts[side], groups[side], records[condition]
        for text_group in TEXT_GROUPS:
            controls[condition][text_group] = {}
            for name, keys in (("target_1", group[ROI_GROUPS[0]]), ("target_2", group[ROI_GROUPS[1]]),
                               ("both_targets", sorted(set(group[ROI_GROUPS[0]] + group[ROI_GROUPS[1]])))):
                matched = {kind: match_control(group[text_group], keys, record[f"{kind}_event2_positions"],
                    layout, [seed, candidate["pair_id"], condition, text_group, name, kind])
                    for kind in ("background", "distractors")}
                if not matched["background"]["eligible"]:
                    raise ValueError("Primary background control cannot match the causal-edge budget.")
                controls[condition][text_group][name] = matched
    missing_capture = {}
    for side, condition in enumerate(CONDITIONS):
        old = set(candidate["capture_indices"][condition]["positions"])
        planned = set(supports["whole_event2"]["low_positions" if side == 0 else "temporal_positions"])
        missing_capture[condition] = {
            "new_visual_positions_to_capture": sorted(planned - old),
            "requires_post_deepstack_capture": True,
            "old_capture_semantics": "decoder_layer_residual_post_before_deepstack_addition",
        }
    return {"supports": supports, "knockout_controls": controls,
            "event_bin_pairs": [{
                "low_temporal_index": low["temporal_index"], "temporal_temporal_index": temporal["temporal_index"],
                "low_progress": low["progress"], "temporal_progress": temporal["progress"],
                "progress_error": abs(low["progress"] - temporal["progress"])}
                for low, temporal in bins],
            "capture_requirements": missing_capture,
            "primary_support_coverage": 1.0, "deepstack_decoder_injection_layers": [0, 1, 2]}
