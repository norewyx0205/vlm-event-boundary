"""Shared token grouping and event-relative mapping for Phase 3B."""

import math
import re
from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch

try:
    from .activation_patching_core import _find_subsequences, _tokenizer_ids_and_offsets
    from .probe_attention_roi import (
        first_video_metadata, object_position, source_frame_groups,
        token_descriptors, video_shape, visual_positions,
    )
    from .run_eval import (
        build_messages, process_video_inputs, processor_input_metadata,
    )
except ImportError:
    from activation_patching_core import _find_subsequences, _tokenizer_ids_and_offsets
    from probe_attention_roi import (
        first_video_metadata, object_position, source_frame_groups,
        token_descriptors, video_shape, visual_positions,
    )
    from run_eval import build_messages, process_video_inputs, processor_input_metadata


SCHEMA = "temporal_boundary_activation_patching_phase3b_v1"
EVENTS = ("event_1", "event_2")
VIDEO_GROUPS = tuple(
    f"video_{identity}_e{event}"
    for identity in ("t1", "t2", "distractors") for event in (1, 2)
)
TEXT_GROUPS = (
    "query_all", "options_all", "option_target_1_mentions",
    "option_target_2_mentions", "option_temporal_relations", "decision_position",
)
GROUPS = VIDEO_GROUPS + TEXT_GROUPS
PATCH_LAYERS = tuple(range(0, 36, 4))
MIN_ROI_OVERLAP = 0.10


def _positions_for_span(offsets, start, end, sequence_start):
    return {
        sequence_start + index
        for index, (left, right) in enumerate(offsets)
        if right > left and left < end and right > start
    }


def build_text_groups(row, input_ids, tokenizer):
    question = row.get("question") or "Which statement correctly describes the order of events?"
    option_a, option_b = row["option_A"], row["option_B"]
    core = f"{question}\n\nA: {option_a}\nB: {option_b}\n\nAnswer with only A or B."
    for prefix in ("\n\n", "\n", "", " "):
        token_ids, offsets = _tokenizer_ids_and_offsets(tokenizer, prefix + core)
        if not token_ids:
            continue
        starts = _find_subsequences(input_ids, token_ids)
        if len(starts) != 1:
            continue
        sequence_start = starts[0]
        q_start = len(prefix)
        q_end = q_start + len(question)
        a_start = q_end + len("\n\nA: ")
        a_end = a_start + len(option_a)
        b_start = a_end + len("\nB: ")
        b_end = b_start + len(option_b)
        query = _positions_for_span(offsets, q_start, q_end, sequence_start)
        options = _positions_for_span(offsets, a_start, a_end, sequence_start)
        options.update(_positions_for_span(offsets, b_start, b_end, sequence_start))
        if not query or not options or query & options:
            continue
        groups = {"query_all": query, "options_all": options}
        for target_id in (1, 2):
            target = next(item for item in row["target_objects"] if int(item["id"]) == target_id)
            label = target["label"]
            positions = set()
            for start, option in ((a_start, option_a), (b_start, option_b)):
                for match in re.finditer(re.escape(label), option, re.IGNORECASE):
                    positions.update(_positions_for_span(
                        offsets, start + match.start(), start + match.end(), sequence_start
                    ))
            groups[f"option_target_{target_id}_mentions"] = positions & options
        relations = set()
        for start, option in ((a_start, option_a), (b_start, option_b)):
            for match in re.finditer(r"\b(before|after)\b", option, re.IGNORECASE):
                relations.update(_positions_for_span(
                    offsets, start + match.start(), start + match.end(), sequence_start
                ))
        groups["option_temporal_relations"] = relations & options
        groups["decision_position"] = {len(input_ids) - 1}
        if all(groups[name] for name in TEXT_GROUPS):
            return {key: sorted(value) for key, value in groups.items()}
    raise RuntimeError(
        f"Cannot locate all section-scoped Phase 3B text groups for {row.get('eval_id')}."
    )


def _exclusive_roi(weights, row, descriptor, width, height):
    labels = ("target_1", "target_2", "distractors")
    active = [
        (label, float(weights.get(label, 0)))
        for label in labels if weights.get(label, 0) >= MIN_ROI_OVERLAP
    ]
    if not active:
        return None
    largest = max(weight for _, weight in active)
    tied = [label for label, weight in active if math.isclose(weight, largest, abs_tol=1e-9)]
    if len(tied) == 1:
        return tied[0]
    x = (descriptor["x_index"] + 0.5) * width / descriptor["merged_w"]
    y = (descriptor["y_index"] + 0.5) * height / descriptor["merged_h"]
    frame = descriptor["source_frame"]
    distances = []
    for label in tied:
        if label == "distractors":
            objects = row.get("distractors") or []
        else:
            target_id = int(label[-1])
            objects = [obj for obj in row["target_objects"] if int(obj["id"]) == target_id]
        centers = [object_position(obj, frame) for obj in objects]
        distances.append((min(
            (x - center[0]) ** 2 + (y - center[1]) ** 2
            for center in centers if center is not None
        ) if any(center is not None for center in centers) else float("inf"), label))
    distances.sort()
    return distances[0][1] if len(distances) == 1 or distances[0][0] < distances[1][0] else None


def build_video_groups(row, inputs, processor, video_path, video_kwargs, padding=8):
    visual, position_source = visual_positions(inputs, processor)
    shape = video_shape(video_path)
    grid = getattr(inputs, "video_grid_thw", None)
    if not visual or shape is None or grid is None or len(grid) != 1:
        raise RuntimeError("Video positions, shape, or grid are unavailable.")
    width, height, total_frames, _ = shape
    grid_t, grid_h, grid_w = [int(value) for value in grid[0].detach().cpu().tolist()]
    ratio = grid_t * grid_h * grid_w / len(visual)
    merge = round(math.sqrt(ratio))
    if merge < 1 or merge * merge * len(visual) != grid_t * grid_h * grid_w:
        raise RuntimeError("Cannot infer merged video token grid.")
    merged_h, merged_w = grid_h // merge, grid_w // merge
    frame_groups = source_frame_groups(first_video_metadata(video_kwargs), grid_t, total_frames)
    descriptors = token_descriptors(
        row, grid_t, merged_h, merged_w, width, height, frame_groups, padding, "overlap"
    )
    if len(descriptors) != len(visual):
        raise RuntimeError("Visual descriptor/token count mismatch.")
    groups = {name: [] for name in VIDEO_GROUPS}
    cells = {name: [] for name in VIDEO_GROUPS}
    ambiguous = 0
    for position, descriptor in zip(visual, descriptors):
        phases = descriptor["temporal_phase_weights"]
        ranked = sorted(phases.items(), key=lambda item: item[1], reverse=True)
        if not ranked or ranked[0][0] not in EVENTS or ranked[0][1] <= 0.5:
            continue
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            ambiguous += 1
            continue
        descriptor = {**descriptor, "merged_h": merged_h, "merged_w": merged_w}
        roi = _exclusive_roi(descriptor["spatial_roi_weights"], row, descriptor, width, height)
        if roi is None:
            ambiguous += 1
            continue
        event = ranked[0][0]
        identity = {"target_1": "t1", "target_2": "t2", "distractors": "distractors"}[roi]
        group = f"video_{identity}_e{event[-1]}"
        event_timing = row["event_timing"]
        start = event_timing["first_event_start_frame" if event == "event_1" else "second_event_start_frame"]
        end = event_timing["first_event_end_frame" if event == "event_1" else "second_event_end_frame"]
        progress = (float(np.mean(descriptor["source_frames"])) - start) / max(1, end - start)
        groups[group].append(position)
        cells[group].append({
            "position": position,
            "temporal_index": descriptor["temporal_index"],
            "progress": progress,
            "x": descriptor["x_index"], "y": descriptor["y_index"],
            "roi_weight": descriptor["spatial_roi_weights"].get(roi, 0.0),
        })
    metadata = {
        "visual_token_count": len(visual),
        "visual_position_source": position_source,
        "video_grid_thw": [grid_t, grid_h, grid_w],
        "merged_video_grid_thw": [grid_t, merged_h, merged_w],
        "source_frame_groups": frame_groups,
        "ambiguous_visual_cells": ambiguous,
        "minimum_primary_roi_overlap": MIN_ROI_OVERLAP,
        "event_phase_dominance_threshold": 0.5,
    }
    return groups, cells, metadata


def mover_roles(first_object_id):
    first, second = int(first_object_id), 3 - int(first_object_id)
    return {
        "first_mover_own_event": f"video_t{first}_e1",
        "first_mover_during_event_2": f"video_t{first}_e2",
        "second_mover_during_event_1": f"video_t{second}_e1",
        "second_mover_own_event": f"video_t{second}_e2",
    }


def prepare_example(row, processor, project_root, video_fps=None, video_num_frames=None,
                    video_max_pixels=None, padding=8, device=None, path_map=None):
    try:
        from .phase3b_paths import resolve_video_path
    except ImportError:
        from phase3b_paths import resolve_video_path
    video_path = resolve_video_path(row["video_path"], project_root, path_map)
    messages = build_messages(
        str(video_path), row["option_A"], row["option_B"], video_fps,
        video_num_frames, video_max_pixels, row.get("question"),
    )
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_video_inputs(messages)
    inputs = processor(
        text=[prompt], images=image_inputs, videos=video_inputs,
        **video_kwargs, padding=True, return_tensors="pt",
    )
    input_metadata = processor_input_metadata(inputs, video_inputs, video_kwargs)
    ids = inputs.input_ids[0].detach().cpu().tolist()
    text_groups = build_text_groups(row, ids, processor.tokenizer)
    video_groups, cells, video_metadata = build_video_groups(
        row, inputs, processor, video_path, video_kwargs, padding
    )
    if device is not None:
        inputs = inputs.to(device)
    return {
        "row": row, "inputs": inputs, "groups": {**video_groups, **text_groups},
        "cells": cells, "video_metadata": video_metadata, "input_metadata": input_metadata,
        "video_path": str(video_path),
    }


def _pair_bins(source_cells, target_cells):
    def bins(cells):
        grouped = defaultdict(list)
        for cell in cells:
            grouped[cell["temporal_index"]].append(cell)
        return sorted(grouped.values(), key=lambda items: np.mean([item["progress"] for item in items]))
    source, target = bins(source_cells), bins(target_cells)
    if not source or not target:
        return []
    if len(source) <= len(target):
        short, long, reverse = source, target, False
    else:
        short, long, reverse = target, source, True
    short_progress = [float(np.mean([item["progress"] for item in group])) for group in short]
    long_progress = [float(np.mean([item["progress"] for item in group])) for group in long]

    @lru_cache(None)
    def align(i, j):
        if i == len(short):
            return 0.0, ()
        if len(long) - j < len(short) - i:
            return float("inf"), ()
        paired_cost, paired_indices = align(i + 1, j + 1)
        paired = (abs(short_progress[i] - long_progress[j]) + paired_cost, (j,) + paired_indices)
        skipped = align(i, j + 1) if len(long) - j > len(short) - i else (float("inf"), ())
        return min((paired, skipped), key=lambda item: item[0])

    _, indices = align(0, 0)
    return [
        (long[index], short_bin) if reverse else (short_bin, long[index])
        for short_bin, index in zip(short, indices)
    ]


def map_group(source_cells, target_cells, min_coverage=0.5, max_progress_error=0.35):
    source_positions, target_positions = [], []
    bin_pairs = []
    rejected_bins = 0
    for source_bin, target_bin in _pair_bins(source_cells, target_cells):
        source_progress = float(np.mean([item["progress"] for item in source_bin]))
        target_progress = float(np.mean([item["progress"] for item in target_bin]))
        if abs(source_progress - target_progress) > max_progress_error:
            rejected_bins += 1
            continue
        source_xy = {(cell["x"], cell["y"]): cell for cell in source_bin}
        target_xy = {(cell["x"], cell["y"]): cell for cell in target_bin}
        for xy in sorted(source_xy.keys() & target_xy.keys()):
            source_positions.append(source_xy[xy]["position"])
            target_positions.append(target_xy[xy]["position"])
        bin_pairs.append({
            "source_temporal_index": source_bin[0]["temporal_index"],
            "target_temporal_index": target_bin[0]["temporal_index"],
            "source_progress": source_progress,
            "target_progress": target_progress,
            "progress_error": abs(source_progress - target_progress),
        })
    if len(source_positions) != len(set(source_positions)) or len(target_positions) != len(set(target_positions)):
        raise RuntimeError("Event-relative mapping is not one-to-one.")
    source_coverage = len(source_positions) / len(source_cells) if source_cells else 0.0
    target_coverage = len(target_positions) / len(target_cells) if target_cells else 0.0
    reason = None
    if not source_positions:
        reason = "no_matched_visual_cells"
    elif min(source_coverage, target_coverage) < min_coverage:
        reason = "coverage_below_threshold"
    return {
        "source_positions": source_positions, "target_positions": target_positions,
        "source_token_count": len(source_cells), "target_token_count": len(target_cells),
        "mapped_token_count": len(source_positions),
        "source_coverage": source_coverage, "target_coverage": target_coverage,
        "source_discard_fraction": 1 - source_coverage,
        "target_discard_fraction": 1 - target_coverage,
        "bin_pairs": bin_pairs, "rejected_temporal_bins": rejected_bins,
        "max_progress_error": max_progress_error,
        "eligible": reason is None, "failure_reason": reason,
    }


def build_pair_mappings(low, temporal, min_coverage=0.5, max_progress_error=0.35):
    return {
        group: {
            **map_group(low["cells"][group], temporal["cells"][group], min_coverage, max_progress_error),
            "mapping_scope": "aggregate_distractors" if group.startswith("video_distractors") else "identity_roi",
        }
        for group in VIDEO_GROUPS
    }
