import argparse
import inspect
import json
import math
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import transformers
from transformers.generation.utils import GenerationMixin

try:
    from .common import PROJECT_ROOT, read_jsonl
    from .make_roi_perturbation_dataset import (
        object_position,
        shape_mask,
        timed_distractor,
        visual_marker_window,
    )
    from .run_eval import (
        build_messages,
        configure_reproducibility,
        environment_metadata,
        load_model,
        parse_answer,
        process_video_inputs,
        processor_input_metadata,
    )
    from .visualize_attention_roi import write_visualizations
except ImportError:
    from common import PROJECT_ROOT, read_jsonl
    from make_roi_perturbation_dataset import (
        object_position,
        shape_mask,
        timed_distractor,
        visual_marker_window,
    )
    from run_eval import (
        build_messages,
        configure_reproducibility,
        environment_metadata,
        load_model,
        parse_answer,
        process_video_inputs,
        processor_input_metadata,
    )
    from visualize_attention_roi import write_visualizations


def object_label(obj):
    reference = obj.get("reference_label")
    if reference:
        return f"target_{obj['id']}_{reference}"
    return f"target_{obj['id']}_{obj.get('color', '')}_{obj.get('shape', '')}".strip("_")


def point_in_circle(x, y, center, radius):
    if center is None:
        return False
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2


def point_in_object(x, y, obj, center, padding):
    if center is None:
        return False
    radius = int(obj.get("radius") or 28)
    shape = obj.get("shape", "circle")
    if shape == "circle":
        return point_in_circle(x, y, center, radius + padding)
    if shape == "square":
        return (
            abs(x - center[0]) <= radius + padding
            and abs(y - center[1]) <= radius + padding
        )
    if shape == "triangle":
        cx, cy = center
        polygon = np.asarray(
            [[cx, cy - radius], [cx - radius, cy + radius], [cx + radius, cy + radius]],
            dtype=np.float32,
        )
        signed_distance = cv2.pointPolygonTest(polygon, (float(x), float(y)), True)
        return signed_distance >= -padding
    return point_in_circle(x, y, center, radius + padding)


def visual_token_ids(processor):
    tokenizer = processor.tokenizer
    candidates = [
        "<|video_pad|>",
        getattr(processor, "video_token", None),
    ]
    ids = set()
    for token in candidates:
        if not token:
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            ids.add(token_id)
    return ids


def visual_positions(inputs, processor):
    token_types = getattr(inputs, "mm_token_type_ids", None)
    if token_types is not None:
        positions = torch.nonzero(token_types[0] == 2, as_tuple=False).flatten().tolist()
        if positions:
            return positions, "mm_token_type_ids"
    token_ids = visual_token_ids(processor)
    input_ids = inputs.input_ids[0].detach().cpu().tolist()
    return [idx for idx, token_id in enumerate(input_ids) if token_id in token_ids], "token_id"


def video_shape(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    cap.release()
    return width, height, frames, fps


def first_video_metadata(video_kwargs):
    metadata = video_kwargs.get("video_metadata")
    if metadata is None:
        return None
    if isinstance(metadata, (list, tuple)):
        return metadata[0] if metadata else None
    return metadata


def metadata_value(metadata, key, default=None):
    if metadata is None:
        return default
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return getattr(metadata, key, default)


def source_frame_indices(video_metadata, grid_t, total_frames):
    sampled = metadata_value(video_metadata, "frames_indices")
    if sampled is not None:
        sampled = [int(value) for value in sampled]
    if sampled:
        groups = np.array_split(np.asarray(sampled, dtype=np.float32), grid_t)
        return [int(round(float(group.mean()))) for group in groups]
    if grid_t <= 1:
        return [max(0, total_frames // 2)]
    return [
        int(round(idx / (grid_t - 1) * max(0, total_frames - 1)))
        for idx in range(grid_t)
    ]


def infer_spatial_merge_size(grid_h, grid_w, visual_token_count, grid_t):
    if not visual_token_count or not grid_t:
        return 1
    raw_count = grid_t * grid_h * grid_w
    ratio = raw_count / visual_token_count
    merge_size = int(round(math.sqrt(ratio)))
    if (
        merge_size >= 1
        and merge_size * merge_size * visual_token_count == raw_count
        and grid_h % merge_size == 0
        and grid_w % merge_size == 0
    ):
        return merge_size
    return 1


def temporal_phase(row, frame_idx):
    timing = row.get("event_timing") or {}
    boundary = row.get("boundary_timing") or {}
    first_start = timing.get("first_event_start_frame")
    first_end = timing.get("first_event_end_frame")
    second_start = timing.get("second_event_start_frame")
    second_end = timing.get("second_event_end_frame")
    boundary_start = boundary.get("boundary_start_frame")
    boundary_end = boundary.get("boundary_end_frame")

    if first_start is not None and frame_idx < first_start:
        return "pre_event"
    if first_start is not None and first_end is not None and first_start <= frame_idx < first_end:
        return "event_1"
    if (
        boundary_start is not None
        and boundary_end is not None
        and boundary_end > boundary_start
        and boundary_start <= frame_idx < boundary_end
    ):
        return "boundary"
    if second_start is not None and second_end is not None and second_start <= frame_idx <= second_end:
        return "event_2"
    return "post_event"


def spatial_roi(row, x, y, frame_idx, roi_padding, width, height):
    marker_window = visual_marker_window(row, 0)
    if marker_window is not None and marker_window[0] <= frame_idx < marker_window[1]:
        if point_in_circle(x, y, (width / 2, height / 2), 48 + roi_padding):
            return "visual_marker"
        return "boundary_flash"

    targets = sorted(row.get("target_objects") or [], key=lambda item: item["id"])
    for target in targets:
        if point_in_object(x, y, target, object_position(target, frame_idx), roi_padding):
            return f"target_{target['id']}"

    for distractor in row.get("distractors") or []:
        distractor = timed_distractor(row, distractor)
        if point_in_object(
            x,
            y,
            distractor,
            object_position(distractor, frame_idx),
            roi_padding,
        ):
            return "distractors"
    return "background"


def spatial_roi_label_map(row, frame_idx, roi_padding, width, height):
    label_map = np.full((height, width), "background", dtype=object)
    marker_window = visual_marker_window(row, 0)
    if marker_window is not None and marker_window[0] <= frame_idx < marker_window[1]:
        label_map[:] = "boundary_flash"
        marker_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(
            marker_mask,
            (width // 2, height // 2),
            48 + roi_padding,
            255,
            -1,
        )
        label_map[marker_mask > 0] = "visual_marker"
        return label_map

    assigned = np.zeros((height, width), dtype=bool)
    targets = sorted(row.get("target_objects") or [], key=lambda item: item["id"])
    for target in targets:
        mask = shape_mask(
            (height, width, 3),
            target,
            object_position(target, frame_idx),
            roi_padding,
        ) > 0
        mask &= ~assigned
        label_map[mask] = f"target_{target['id']}"
        assigned |= mask

    distractor_mask = np.zeros((height, width), dtype=bool)
    for distractor in row.get("distractors") or []:
        distractor = timed_distractor(row, distractor)
        distractor_mask |= shape_mask(
            (height, width, 3),
            distractor,
            object_position(distractor, frame_idx),
            roi_padding,
        ) > 0
    distractor_mask &= ~assigned
    label_map[distractor_mask] = "distractors"
    return label_map


def cell_roi_weights(label_map, x_idx, y_idx, merged_w, merged_h, width, height):
    x0 = int(round(x_idx * width / merged_w))
    x1 = int(round((x_idx + 1) * width / merged_w))
    y0 = int(round(y_idx * height / merged_h))
    y1 = int(round((y_idx + 1) * height / merged_h))
    cell = label_map[y0:y1, x0:x1]
    labels, counts = np.unique(cell, return_counts=True)
    total = max(1, int(counts.sum()))
    return {str(label): int(count) / total for label, count in zip(labels, counts)}


def token_descriptors(
    row,
    grid_t,
    merged_h,
    merged_w,
    width,
    height,
    frame_indices,
    roi_padding,
    roi_assignment,
):
    descriptors = []
    for temporal_idx in range(grid_t):
        source_frame = frame_indices[temporal_idx]
        phase = temporal_phase(row, source_frame)
        label_map = (
            spatial_roi_label_map(row, source_frame, roi_padding, width, height)
            if roi_assignment == "overlap"
            else None
        )
        for y_idx in range(merged_h):
            y = (y_idx + 0.5) * height / merged_h
            for x_idx in range(merged_w):
                x = (x_idx + 0.5) * width / merged_w
                if label_map is not None:
                    weights = cell_roi_weights(
                        label_map,
                        x_idx,
                        y_idx,
                        merged_w,
                        merged_h,
                        width,
                        height,
                    )
                    dominant = max(weights, key=weights.get)
                else:
                    dominant = spatial_roi(
                        row,
                        x,
                        y,
                        source_frame,
                        roi_padding,
                        width,
                        height,
                    )
                    weights = {dominant: 1.0}
                descriptors.append({
                    "temporal_index": temporal_idx,
                    "source_frame": source_frame,
                    "x_index": x_idx,
                    "y_index": y_idx,
                    "spatial_roi": dominant,
                    "spatial_roi_weights": weights,
                    "temporal_phase": phase,
                })
    return descriptors


def attention_distribution(scores, assignments):
    total_attention = float(scores.sum())
    counts = defaultdict(float)
    masses = defaultdict(float)
    for score, assignment in zip(scores.tolist(), assignments):
        weights = assignment if isinstance(assignment, dict) else {assignment: 1.0}
        for label, weight in weights.items():
            counts[label] += float(weight)
            masses[label] += float(score) * float(weight)
    total_tokens = len(assignments)
    output = {}
    for label in sorted(counts):
        token_fraction = counts[label] / total_tokens if total_tokens else 0.0
        normalized_attention = masses[label] / total_attention if total_attention else 0.0
        output[label] = {
            "token_count": counts[label],
            "effective_token_count": counts[label],
            "token_fraction": token_fraction,
            "attention_mass": masses[label],
            "normalized_visual_attention": normalized_attention,
            "mean_attention_per_token": masses[label] / counts[label] if counts[label] else 0.0,
            "enrichment": normalized_attention / token_fraction if token_fraction else None,
        }
    return output


def reduce_attention(attention, head_reduction):
    if attention is None:
        raise RuntimeError(
            "The model returned no attention tensor. Use --attn_implementation eager."
        )
    values = attention[0, :, -1, :].float()
    if head_reduction == "max":
        values = values.max(dim=0).values
    else:
        values = values.mean(dim=0)
    return values.detach().cpu()


def select_layer_index(layer_index, layer_count):
    selected = layer_index if layer_index >= 0 else layer_count + layer_index
    if not 0 <= selected < layer_count:
        raise ValueError(f"Layer index {layer_index} is outside 0..{layer_count - 1}.")
    return selected


def aggregate_attention(
    row,
    inputs,
    processor,
    attentions,
    video_path,
    video_kwargs,
    roi_padding,
    roi_assignment,
    roi_padding_sensitivity,
    layer_index,
    head_reduction,
):
    positions, position_source = visual_positions(inputs, processor)
    shape = video_shape(video_path)
    grid = getattr(inputs, "video_grid_thw", None)
    if not positions:
        raise RuntimeError("No video-token positions were found in the processor output.")
    if shape is None or grid is None:
        raise RuntimeError("Video shape or video_grid_thw is unavailable.")
    if not attentions or attentions[0] is None:
        raise RuntimeError("No decoder attentions were returned; use eager attention.")

    width, height, total_frames, source_fps = shape
    grid_t, grid_h, grid_w = [int(value) for value in grid[0].detach().cpu().tolist()]
    merge_size = infer_spatial_merge_size(grid_h, grid_w, len(positions), grid_t)
    merged_h, merged_w = grid_h // merge_size, grid_w // merge_size
    expected_tokens = grid_t * merged_h * merged_w
    if expected_tokens != len(positions):
        raise RuntimeError(
            f"Merged grid expects {expected_tokens} video tokens but found {len(positions)}. "
            f"grid={grid_t, grid_h, grid_w}, merge_size={merge_size}."
        )

    frame_indices = source_frame_indices(first_video_metadata(video_kwargs), grid_t, total_frames)
    descriptors = token_descriptors(
        row,
        grid_t,
        merged_h,
        merged_w,
        width,
        height,
        frame_indices,
        roi_padding,
        roi_assignment,
    )
    spatial_assignments = [item["spatial_roi_weights"] for item in descriptors]
    temporal_labels = [item["temporal_phase"] for item in descriptors]
    selected_layer = select_layer_index(layer_index, len(attentions))

    layer_profiles = []
    selected_scores = None
    selected_all_scores = None
    for idx, attention in enumerate(attentions):
        all_scores = reduce_attention(attention, head_reduction)
        if max(positions) >= len(all_scores):
            raise RuntimeError(
                f"Attention key length {len(all_scores)} does not cover video position {max(positions)}."
            )
        visual_scores = all_scores[positions]
        layer_profiles.append({
            "layer": idx,
            "visual_attention_fraction": float(visual_scores.sum() / max(float(all_scores.sum()), 1e-12)),
            "spatial_roi": attention_distribution(visual_scores, spatial_assignments),
            "temporal_phase": attention_distribution(visual_scores, temporal_labels),
        })
        if idx == selected_layer:
            selected_scores = visual_scores
            selected_all_scores = all_scores

    visual_total = max(float(selected_scores.sum()), 1e-12)
    attention_map = (selected_scores / visual_total).reshape(grid_t, merged_h, merged_w)
    temporal_attention = attention_map.sum(dim=(1, 2)).tolist()
    temporal_phases = [temporal_phase(row, frame_idx) for frame_idx in frame_indices]
    targets = sorted(row.get("target_objects") or [], key=lambda item: item["id"])
    padding_profiles = {}
    for padding in roi_padding_sensitivity:
        padding_descriptors = token_descriptors(
            row,
            grid_t,
            merged_h,
            merged_w,
            width,
            height,
            frame_indices,
            padding,
            roi_assignment,
        )
        padding_profiles[str(padding)] = attention_distribution(
            selected_scores,
            [item["spatial_roi_weights"] for item in padding_descriptors],
        )

    return {
        "video_token_position_source": position_source,
        "visual_token_count": len(positions),
        "raw_video_grid_thw": [grid_t, grid_h, grid_w],
        "spatial_merge_size": merge_size,
        "merged_video_grid_thw": [grid_t, merged_h, merged_w],
        "source_video_shape": [total_frames, height, width],
        "source_fps": source_fps,
        "source_frame_indices": frame_indices,
        "selected_layer": selected_layer,
        "head_reduction": head_reduction,
        "roi_assignment_method": roi_assignment,
        "roi_padding": roi_padding,
        "roi_padding_sensitivity": padding_profiles,
        "selected_layer_visual_attention_fraction": float(
            selected_scores.sum() / max(float(selected_all_scores.sum()), 1e-12)
        ),
        "spatial_roi_attention": attention_distribution(selected_scores, spatial_assignments),
        "temporal_phase_attention": attention_distribution(selected_scores, temporal_labels),
        "temporal_attention": temporal_attention,
        "temporal_phases": temporal_phases,
        "attention_map": attention_map.tolist(),
        "layer_roi_profiles": layer_profiles,
        "target_labels": {
            f"target_{target['id']}": object_label(target)
            for target in targets
        },
    }


def model_forward(model, kwargs):
    try:
        return model(**kwargs, logits_to_keep=1)
    except TypeError as exc:
        if "logits_to_keep" not in str(exc):
            raise
        return model(**kwargs)


def eos_token_ids(model):
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        return set()
    if isinstance(eos, (list, tuple, set)):
        return {int(value) for value in eos}
    return {int(eos)}


def generation_cache_api():
    parameters = inspect.signature(
        GenerationMixin.prepare_inputs_for_generation
    ).parameters
    if "next_sequence_length" in parameters:
        return "next_sequence_length"
    if "cache_position" in parameters:
        return "cache_position"
    raise RuntimeError(
        "Unsupported Transformers generation API: neither next_sequence_length "
        "nor cache_position is available."
    )


def cache_sequence_length(past_key_values):
    if past_key_values is None:
        return 0
    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    if callable(get_seq_length):
        return int(get_seq_length())
    try:
        return int(past_key_values[0][0].shape[-2])
    except (IndexError, TypeError, AttributeError):
        return None


def prepare_prefill_position_ids(model, inputs, cache_api):
    if cache_api != "next_sequence_length":
        return None
    prepare_positions = getattr(model, "_prepare_position_ids_for_generation", None)
    if not callable(prepare_positions):
        raise RuntimeError(
            "Transformers 5.x attention probing requires the model's generation "
            "position-id helper, but it is unavailable."
        )
    return prepare_positions(inputs.input_ids, dict(inputs))


def extend_position_ids(position_ids, generated_length):
    if position_ids is None:
        return None
    offset_shape = [1] * (position_ids.ndim - 1) + [generated_length]
    offsets = torch.arange(
        1,
        generated_length + 1,
        dtype=position_ids.dtype,
        device=position_ids.device,
    ).view(*offset_shape)
    return torch.cat([position_ids, position_ids[..., -1:] + offsets], dim=-1)


def standard_first_token(model, inputs):
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
    prompt_length = int(inputs.input_ids.shape[1])
    token_id = int(generated.sequences[0, prompt_length])
    if not generated.scores:
        raise RuntimeError("Standard generation did not return first-token scores for parity validation.")
    scores = generated.scores[0][0].float().detach().cpu()
    return token_id, scores


def validate_single_token_query(prepared, cache_length, expected_cache_length):
    query = prepared.get("input_ids")
    query_embeds = prepared.get("inputs_embeds")
    query_length = (
        int(query.shape[1])
        if query is not None
        else int(query_embeds.shape[1])
        if query_embeds is not None
        else None
    )
    if query_length != 1:
        raise RuntimeError(
            "Cached attention probe did not reduce the decoder query to one token; "
            f"got query_length={query_length}. Refusing a full QxK attention allocation."
        )
    for sequence_key in ("position_ids", "mm_token_type_ids"):
        value = prepared.get(sequence_key)
        if value is not None and value.shape[-1] != query_length:
            prepared[sequence_key] = value[..., -query_length:].clone(
                memory_format=torch.contiguous_format
            )
    if cache_length is not None and cache_length != expected_cache_length:
        raise RuntimeError(
            "Cached attention probe found an unexpected KV-cache length before "
            f"the decoder step: cache={cache_length}, expected={expected_cache_length}."
        )
    prepared_mask = prepared.get("attention_mask")
    expected_mask_length = expected_cache_length + query_length
    if prepared_mask is not None and prepared_mask.shape[-1] != expected_mask_length:
        raise RuntimeError(
            "Cached attention probe found an incompatible attention mask: "
            f"mask={prepared_mask.shape[-1]}, expected={expected_mask_length}."
        )
    return query_length


def greedy_generate_with_decision_attention(
    model,
    processor,
    inputs,
    max_new_tokens,
    verify_standard_generation=True,
    require_standard_logits_match=True,
    parity_rtol=1e-3,
    parity_atol=1e-3,
):
    prompt_length = int(inputs.input_ids.shape[1])
    if prompt_length < 2:
        raise RuntimeError("Decision-position probing requires at least two prompt tokens.")
    prefix_length = prompt_length - 1
    cache_api = generation_cache_api()
    full_position_ids = prepare_prefill_position_ids(model, inputs, cache_api)
    standard_token_id = None
    standard_scores = None
    if verify_standard_generation:
        standard_token_id, standard_scores = standard_first_token(model, inputs)

    prefill_kwargs = dict(inputs)
    prefill_kwargs["input_ids"] = inputs.input_ids[:, :prefix_length]
    if getattr(inputs, "attention_mask", None) is not None:
        prefill_kwargs["attention_mask"] = inputs.attention_mask[:, :prefix_length]
    if getattr(inputs, "mm_token_type_ids", None) is not None:
        prefill_kwargs["mm_token_type_ids"] = inputs.mm_token_type_ids[:, :prefix_length]
    prefill_kwargs.update({
        "use_cache": True,
        "output_attentions": False,
        "return_dict": True,
    })
    if full_position_ids is not None:
        prefill_kwargs["position_ids"] = full_position_ids[..., :prefix_length]
    if cache_api == "cache_position":
        prefill_kwargs["cache_position"] = torch.arange(
            prefix_length,
            dtype=torch.long,
            device=inputs.input_ids.device,
        )
    with torch.inference_mode():
        prefill = model_forward(model, prefill_kwargs)

    past_key_values = prefill.past_key_values
    full_input_ids = inputs.input_ids
    full_attention_mask = getattr(inputs, "attention_mask", None)
    if full_attention_mask is None:
        full_attention_mask = torch.ones_like(full_input_ids)
    full_mm_types = getattr(inputs, "mm_token_type_ids", None)
    static_kwargs = {
        key: value
        for key, value in inputs.items()
        if key not in {"input_ids", "attention_mask", "position_ids", "mm_token_type_ids"}
    }
    decision_kwargs = {
        "past_key_values": past_key_values,
        "attention_mask": full_attention_mask,
        "use_cache": True,
        "is_first_iteration": False,
        **static_kwargs,
    }
    if full_position_ids is not None:
        decision_kwargs["position_ids"] = full_position_ids
    if full_mm_types is not None:
        decision_kwargs["mm_token_type_ids"] = full_mm_types
    if cache_api == "next_sequence_length":
        decision_kwargs["next_sequence_length"] = 1
    else:
        decision_kwargs["cache_position"] = torch.tensor(
            [prefix_length], dtype=torch.long, device=full_input_ids.device
        )
    prepared = model.prepare_inputs_for_generation(full_input_ids, **decision_kwargs)
    validate_single_token_query(
        prepared,
        cache_sequence_length(past_key_values),
        prefix_length,
    )
    prepared.update({
        "use_cache": True,
        "output_attentions": True,
        "return_dict": True,
    })
    with torch.inference_mode():
        decision_output = model_forward(model, prepared)
    if not decision_output.attentions or decision_output.attentions[0] is None:
        raise RuntimeError(
            "Decoder attentions are unavailable. Run the probe with --attn_implementation eager."
        )

    decision_scores = decision_output.logits[:, -1, :]
    next_token = decision_scores.argmax(dim=-1, keepdim=True)
    decision_token_id = int(next_token[0, 0])
    first_token_match = (
        decision_token_id == standard_token_id
        if standard_token_id is not None
        else None
    )
    if first_token_match is False:
        raise RuntimeError(
            "Decision-position cache split changed the standard greedy first token: "
            f"decision={decision_token_id}, standard={standard_token_id}."
        )

    logits_max_abs_diff = None
    logits_allclose = None
    if standard_scores is not None:
        decision_cpu = decision_scores[0].float().detach().cpu()
        logits_max_abs_diff = float((decision_cpu - standard_scores).abs().max())
        logits_allclose = bool(
            torch.allclose(
                decision_cpu,
                standard_scores,
                rtol=parity_rtol,
                atol=parity_atol,
            )
        )
        if require_standard_logits_match and not logits_allclose:
            raise RuntimeError(
                "Decision-position cache split changed the standard first-token logits: "
                f"max_abs_diff={logits_max_abs_diff:.6g}, rtol={parity_rtol}, atol={parity_atol}."
            )

    past_key_values = decision_output.past_key_values
    generated = []
    eos_ids = eos_token_ids(model)
    for generated_index in range(max_new_tokens):
        token_id = int(next_token[0, 0])
        generated.append(next_token)
        if token_id in eos_ids or generated_index + 1 >= max_new_tokens:
            break

        full_input_ids = torch.cat([full_input_ids, next_token], dim=-1)
        full_attention_mask = torch.cat(
            [
                full_attention_mask,
                torch.ones(
                    (full_attention_mask.shape[0], 1),
                    dtype=full_attention_mask.dtype,
                    device=full_attention_mask.device,
                ),
            ],
            dim=-1,
        )
        generation_kwargs = dict(static_kwargs)
        step_position_ids = extend_position_ids(full_position_ids, len(generated))
        if step_position_ids is not None:
            generation_kwargs["position_ids"] = step_position_ids
        if full_mm_types is not None:
            generation_kwargs["mm_token_type_ids"] = torch.cat(
                [
                    full_mm_types,
                    torch.zeros(
                        (full_mm_types.shape[0], len(generated)),
                        dtype=full_mm_types.dtype,
                        device=full_mm_types.device,
                    ),
                ],
                dim=-1,
            )
        prepare_kwargs = {
            "past_key_values": past_key_values,
            "attention_mask": full_attention_mask,
            "use_cache": True,
            "is_first_iteration": False,
            **generation_kwargs,
        }
        if cache_api == "next_sequence_length":
            prepare_kwargs["next_sequence_length"] = 1
        else:
            prepare_kwargs["cache_position"] = torch.tensor(
                [prompt_length + len(generated) - 1],
                dtype=torch.long,
                device=full_input_ids.device,
            )
        prepared = model.prepare_inputs_for_generation(full_input_ids, **prepare_kwargs)
        validate_single_token_query(
            prepared,
            cache_sequence_length(past_key_values),
            prompt_length + len(generated) - 1,
        )
        prepared.update({
            "use_cache": True,
            "output_attentions": False,
            "return_dict": True,
        })
        with torch.inference_mode():
            step_output = model_forward(model, prepared)
        past_key_values = step_output.past_key_values
        next_token = step_output.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    generated_ids = torch.cat(generated, dim=-1) if generated else torch.empty((1, 0), dtype=torch.long)
    raw_response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    predicted_token_text = processor.batch_decode(
        generated[0],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    query_token_id = int(inputs.input_ids[0, -1])
    query_token_text = processor.batch_decode(
        inputs.input_ids[:, -1:],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    return raw_response, decision_output.attentions, {
        "attention_semantics": "prompt_final_position_decision_query",
        "query_token_id": query_token_id,
        "query_token_text": query_token_text,
        "predicted_first_token_id": decision_token_id,
        "predicted_first_token_text": predicted_token_text,
        "standard_first_token_id": standard_token_id,
        "standard_first_token_match": first_token_match,
        "standard_logits_max_abs_diff": logits_max_abs_diff,
        "standard_logits_allclose": logits_allclose,
        "standard_logits_parity_rtol": parity_rtol,
        "standard_logits_parity_atol": parity_atol,
        "generated_token_ids": [int(token[0, 0]) for token in generated],
        "cache_api": cache_api,
        "prompt_length": prompt_length,
        "prefix_cache_length": prefix_length,
        "prefill_position_ids_shape": (
            list(full_position_ids.shape) if full_position_ids is not None else None
        ),
    }


def probe_row(model, processor, row, args):
    video_path = PROJECT_ROOT / row["video_path"]
    if not video_path.exists():
        video_path = Path(row["video_path"])
    messages = build_messages(
        str(video_path),
        row["option_A"],
        row["option_B"],
        args.video_fps,
        args.video_num_frames,
        args.video_max_pixels,
        row.get("question"),
    )
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_video_inputs(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        **video_kwargs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    input_metadata = processor_input_metadata(inputs, video_inputs, video_kwargs)
    raw_response, decision_attentions, query_metadata = greedy_generate_with_decision_attention(
        model,
        processor,
        inputs,
        args.max_new_tokens,
        verify_standard_generation=args.verify_standard_generation,
        require_standard_logits_match=args.require_standard_logits_match,
        parity_rtol=args.parity_rtol,
        parity_atol=args.parity_atol,
    )
    prediction = parse_answer(raw_response)
    archived_prediction = row.get("archived_prediction")
    prediction_match = (
        prediction == archived_prediction
        if archived_prediction not in (None, "")
        else None
    )
    if args.require_archived_prediction_match and prediction_match is False:
        raise RuntimeError(
            "Attention probe prediction differs from the archived main evaluation: "
            f"probe={prediction}, archived={archived_prediction}, eval_id={row.get('eval_id')}."
        )
    result = aggregate_attention(
        row,
        inputs,
        processor,
        decision_attentions,
        video_path,
        video_kwargs,
        args.roi_padding,
        args.roi_assignment,
        args.roi_padding_sensitivity_values,
        args.visualization_layer,
        args.head_reduction,
    )
    result.update({
        "eval_id": row.get("eval_id"),
        "video_id": row.get("video_id"),
        "video_path": row.get("video_path"),
        "dataset_version": row.get("dataset_version"),
        "difficulty_level": row.get("difficulty_level"),
        "condition": row.get("condition"),
        "feature_variant": row.get("feature_variant"),
        "size_scene_variant": row.get("size_scene_variant"),
        "base_sample_id": row.get("base_sample_id"),
        "prompt_variant": row.get("prompt_variant"),
        "question": row.get("question"),
        "option_A": row.get("option_A"),
        "option_B": row.get("option_B"),
        "correct_option": row.get("correct_option"),
        "target_objects": row.get("target_objects") or [],
        "distractors": row.get("distractors") or [],
        "event_timing": row.get("event_timing") or {},
        "boundary_timing": row.get("boundary_timing") or {},
        "prediction": prediction,
        "is_correct": prediction == row.get("correct_option"),
        "archived_prediction": archived_prediction,
        "archived_is_correct": row.get("archived_is_correct"),
        "prediction_match": prediction_match,
        "attention_case_label": row.get("attention_case_label"),
        "raw_response": raw_response,
        "attention_semantics": query_metadata["attention_semantics"],
        "decision_query": query_metadata,
        "input_metadata": input_metadata,
    })
    del inputs, decision_attentions
    if args.empty_cache_each_sample and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def parse_csv_filter(value):
    return {part.strip() for part in value.split(",") if part.strip()} if value else None


def parse_int_csv(value):
    return sorted({int(part.strip()) for part in value.split(",") if part.strip()})


def filter_rows(rows, conditions, prompt_variants, base_sample_ids):
    if conditions:
        rows = [row for row in rows if row.get("condition") in conditions]
    if prompt_variants:
        rows = [row for row in rows if row.get("prompt_variant") in prompt_variants]
    if base_sample_ids:
        rows = [row for row in rows if int(row.get("base_sample_id")) in base_sample_ids]
    return rows


def select_probe_rows(rows, max_samples):
    if any(row.get("attention_case_label") for row in rows):
        selected = []
        grouped_cases = defaultdict(list)
        for row in rows:
            key = row.get("pairing_id") or row.get("video_id") or row.get("eval_id")
            grouped_cases[key].append(row)
        for group in grouped_cases.values():
            if len(selected) + len(group) > max_samples:
                break
            selected.extend(group)
        return selected
    grouped = defaultdict(list)
    for row in rows:
        key = row.get("pairing_id") or row.get("video_id") or row.get("eval_id")
        grouped[key].append(row)
    condition_order = (
        "low_boundary",
        "temporal_boundary",
        "visual_boundary",
        "audio_boundary",
    )
    by_condition = defaultdict(list)
    for group in grouped.values():
        group.sort(key=lambda row: (row.get("prompt_variant") != "original", row.get("eval_id")))
        condition = group[0].get("condition") or "other"
        by_condition[condition].append(group)
    for groups in by_condition.values():
        groups.sort(key=lambda group: (str(group[0].get("base_sample_id")), group[0].get("eval_id")))

    selected = []
    condition_names = [name for name in condition_order if name in by_condition]
    condition_names.extend(sorted(name for name in by_condition if name not in condition_names))
    round_index = 0
    while len(selected) < max_samples:
        added = False
        for condition in condition_names:
            groups = by_condition[condition]
            if round_index >= len(groups):
                continue
            group = groups[round_index]
            if len(selected) + len(group) > max_samples:
                continue
            selected.extend(group)
            added = True
            if len(selected) >= max_samples:
                break
        if not added:
            break
        round_index += 1
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--visualization_dir", default=None)
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--deterministic_warn_only", action="store_true")
    parser.add_argument("--attn_implementation", default="eager", choices=["eager"])
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--prompt_variants", default=None)
    parser.add_argument("--base_sample_ids", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument(
        "--verify_standard_generation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the split-cache decision query to match standard greedy first-token generation.",
    )
    parser.add_argument(
        "--require_standard_logits_match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail when split-cache and standard first-token logits exceed parity tolerances.",
    )
    parser.add_argument("--parity_rtol", type=float, default=1e-3)
    parser.add_argument("--parity_atol", type=float, default=1e-3)
    parser.add_argument(
        "--require_archived_prediction_match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail when rows carrying archived_prediction disagree with the probe prediction.",
    )
    parser.add_argument("--video_fps", type=float, default=None)
    parser.add_argument("--video_num_frames", type=int, default=None)
    parser.add_argument("--video_max_pixels", type=int, default=None)
    parser.add_argument("--roi_padding", type=int, default=8)
    parser.add_argument(
        "--roi_assignment",
        choices=["overlap", "center"],
        default="overlap",
        help="Assign merged video cells by fractional ROI overlap or legacy center points.",
    )
    parser.add_argument(
        "--roi_padding_sensitivity",
        default="0,4,8,12",
        help="Comma-separated padding values saved as selected-layer sensitivity profiles.",
    )
    parser.add_argument("--visualization_layer", type=int, default=-1)
    parser.add_argument("--head_reduction", choices=["mean", "max"], default="mean")
    parser.add_argument("--max_overlay_frames", type=int, default=6)
    parser.add_argument("--heatmap_alpha", type=float, default=0.45)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--empty_cache_each_sample", action="store_true")
    args = parser.parse_args()
    if args.video_fps is not None and args.video_num_frames is not None:
        parser.error("Use only one temporal sampling control: --video_fps or --video_num_frames.")
    if args.max_samples <= 0 or args.max_new_tokens <= 0:
        parser.error("--max_samples and --max_new_tokens must be positive.")
    if args.roi_padding < 0:
        parser.error("--roi_padding must be non-negative.")
    if args.parity_rtol < 0 or args.parity_atol < 0:
        parser.error("Parity tolerances must be non-negative.")
    try:
        args.roi_padding_sensitivity_values = parse_int_csv(args.roi_padding_sensitivity)
    except ValueError:
        parser.error("--roi_padding_sensitivity must contain comma-separated integers.")
    if any(value < 0 for value in args.roi_padding_sensitivity_values):
        parser.error("ROI padding sensitivity values must be non-negative.")
    if args.roi_padding not in args.roi_padding_sensitivity_values:
        args.roi_padding_sensitivity_values.append(args.roi_padding)
        args.roi_padding_sensitivity_values.sort()
    if not 0.0 <= args.heatmap_alpha <= 1.0:
        parser.error("--heatmap_alpha must be between 0 and 1.")

    conditions = parse_csv_filter(args.conditions)
    prompt_variants = parse_csv_filter(args.prompt_variants)
    base_sample_ids = (
        {int(value) for value in parse_csv_filter(args.base_sample_ids)}
        if args.base_sample_ids
        else None
    )
    rows = select_probe_rows(filter_rows(
        read_jsonl(args.annotation_path),
        conditions,
        prompt_variants,
        base_sample_ids,
    ), args.max_samples)
    if not rows:
        raise ValueError("No complete mirrored attention cases matched the requested filters/limit.")
    configure_reproducibility(
        args.seed,
        deterministic=args.deterministic,
        deterministic_warn_only=args.deterministic_warn_only,
    )
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    gpu_memory_gb = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
        if torch.cuda.is_available()
        else 0.0
    )
    print(
        f"Attention probe runtime: transformers={transformers.__version__}, "
        f"torch={torch.__version__}, device={gpu_name}, memory={gpu_memory_gb:.1f} GB, "
        f"cache_api={generation_cache_api()}, rows={len(rows)}",
        flush=True,
    )
    print(
        f"Loading {args.model_name} with attention implementation "
        f"{args.attn_implementation}...",
        flush=True,
    )
    model, processor = load_model(
        args.model_name,
        model_revision=args.model_revision,
        attn_implementation=args.attn_implementation,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for idx, row in enumerate(rows, start=1):
        print(f"Attention probe {idx}/{len(rows)}: {row.get('eval_id')}", flush=True)
        try:
            outputs.append(probe_row(model, processor, row, args))
            output_path.write_text(
                json.dumps(outputs, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            print(
                f"Attention probe failed at {row.get('eval_id')}. Full traceback:",
                flush=True,
            )
            traceback.print_exc()
            raise

    output_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    archived_matches = [row.get("prediction_match") for row in outputs if row.get("prediction_match") is not None]
    logit_differences = [
        (row.get("decision_query") or {}).get("standard_logits_max_abs_diff")
        for row in outputs
    ]
    logit_differences = [value for value in logit_differences if value is not None]
    standard_token_matches = [
        (row.get("decision_query") or {}).get("standard_first_token_match")
        for row in outputs
    ]
    standard_token_matches = [value for value in standard_token_matches if value is not None]
    standard_logits_matches = [
        (row.get("decision_query") or {}).get("standard_logits_allclose")
        for row in outputs
    ]
    standard_logits_matches = [value for value in standard_logits_matches if value is not None]
    parity_summary = {
        "attention_semantics": "prompt_final_position_attention_predicting_first_answer_token",
        "rows": len(outputs),
        "standard_first_token_rows": len(standard_token_matches),
        "standard_first_token_matches": sum(int(value) for value in standard_token_matches),
        "standard_logits_rows": len(standard_logits_matches),
        "standard_logits_allclose": sum(int(value) for value in standard_logits_matches),
        "maximum_standard_logits_absolute_difference": max(logit_differences, default=None),
        "archived_prediction_rows": len(archived_matches),
        "archived_prediction_matches": sum(int(value) for value in archived_matches),
        "case_labels": dict(
            Counter(row.get("attention_case_label") or "unlabelled" for row in outputs)
        ),
    }
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    summary_path.write_text(
        json.dumps(parity_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config_path = output_path.with_name(f"{output_path.stem}_config.json")
    config_payload = {
        **vars(args),
        "resolved_attention_implementation": getattr(
            model.config,
            "_attn_implementation",
            args.attn_implementation,
        ),
        "generation_cache_api": generation_cache_api(),
        "attention_cache_strategy": "prefix_prefill_then_prompt_final_decision_query",
        "attention_semantics": "prompt_final_position_attention_predicting_first_answer_token",
        "selected_eval_ids": [row.get("eval_id") for row in rows],
        "environment": environment_metadata(model),
    }
    config_path.write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote attention ROI probe results to {output_path}")
    print(f"Wrote attention parity summary to {summary_path}")

    if args.plots:
        visualization_dir = (
            Path(args.visualization_dir)
            if args.visualization_dir
            else output_path.parent / f"{output_path.stem}_figures"
        )
        written = write_visualizations(
            outputs,
            visualization_dir,
            project_root=PROJECT_ROOT,
            max_frames=args.max_overlay_frames,
            alpha=args.heatmap_alpha,
        )
        print(f"Wrote {len(written)} attention visualization files to {visualization_dir}")


if __name__ == "__main__":
    main()
