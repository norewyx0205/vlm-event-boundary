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
    from .make_roi_perturbation_dataset import object_position, timed_distractor, visual_marker_window
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
    from make_roi_perturbation_dataset import object_position, timed_distractor, visual_marker_window
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


def token_descriptors(row, grid_t, merged_h, merged_w, width, height, frame_indices, roi_padding):
    descriptors = []
    for temporal_idx in range(grid_t):
        source_frame = frame_indices[temporal_idx]
        phase = temporal_phase(row, source_frame)
        for y_idx in range(merged_h):
            y = (y_idx + 0.5) * height / merged_h
            for x_idx in range(merged_w):
                x = (x_idx + 0.5) * width / merged_w
                descriptors.append({
                    "temporal_index": temporal_idx,
                    "source_frame": source_frame,
                    "x_index": x_idx,
                    "y_index": y_idx,
                    "spatial_roi": spatial_roi(
                        row,
                        x,
                        y,
                        source_frame,
                        roi_padding,
                        width,
                        height,
                    ),
                    "temporal_phase": phase,
                })
    return descriptors


def attention_distribution(scores, labels):
    total_attention = float(scores.sum())
    counts = Counter(labels)
    masses = defaultdict(float)
    for score, label in zip(scores.tolist(), labels):
        masses[label] += float(score)
    total_tokens = len(labels)
    output = {}
    for label in sorted(counts):
        token_fraction = counts[label] / total_tokens if total_tokens else 0.0
        normalized_attention = masses[label] / total_attention if total_attention else 0.0
        output[label] = {
            "token_count": counts[label],
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
    )
    spatial_labels = [item["spatial_roi"] for item in descriptors]
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
            "spatial_roi": attention_distribution(visual_scores, spatial_labels),
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
        "selected_layer_visual_attention_fraction": float(
            selected_scores.sum() / max(float(selected_all_scores.sum()), 1e-12)
        ),
        "spatial_roi_attention": attention_distribution(selected_scores, spatial_labels),
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


def greedy_generate_with_answer_attention(model, processor, inputs, max_new_tokens):
    prefill_kwargs = dict(inputs)
    prompt_length = int(inputs.input_ids.shape[1])
    cache_api = generation_cache_api()
    prefill_position_ids = prepare_prefill_position_ids(model, inputs, cache_api)
    prefill_kwargs.update({
        "use_cache": True,
        "output_attentions": False,
        "return_dict": True,
    })
    if prefill_position_ids is not None:
        prefill_kwargs["position_ids"] = prefill_position_ids
    if cache_api == "cache_position":
        prefill_kwargs["cache_position"] = torch.arange(
            prompt_length,
            dtype=torch.long,
            device=inputs.input_ids.device,
        )
    with torch.inference_mode():
        prefill = model_forward(model, prefill_kwargs)

    past_key_values = prefill.past_key_values
    next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    full_input_ids = inputs.input_ids
    base_attention_mask = getattr(inputs, "attention_mask", None)
    if base_attention_mask is None:
        base_attention_mask = torch.ones_like(full_input_ids)
    base_mm_types = getattr(inputs, "mm_token_type_ids", None)
    static_kwargs = {
        key: value
        for key, value in inputs.items()
        if key not in {"input_ids", "attention_mask", "position_ids", "mm_token_type_ids"}
    }
    generated = []
    attention_steps = []
    eos_ids = eos_token_ids(model)

    for _ in range(max_new_tokens):
        token_id = int(next_token[0, 0])
        generated.append(next_token)
        if token_id in eos_ids:
            break

        full_input_ids = torch.cat([full_input_ids, next_token], dim=-1)
        attention_mask = torch.cat(
            [
                base_attention_mask,
                torch.ones(
                    (base_attention_mask.shape[0], len(generated)),
                    dtype=base_attention_mask.dtype,
                    device=base_attention_mask.device,
                ),
            ],
            dim=-1,
        )
        generation_kwargs = dict(static_kwargs)
        step_position_ids = extend_position_ids(prefill_position_ids, len(generated))
        if step_position_ids is not None:
            generation_kwargs["position_ids"] = step_position_ids
        if base_mm_types is not None:
            generation_kwargs["mm_token_type_ids"] = torch.cat(
                [
                    base_mm_types,
                    torch.zeros(
                        (base_mm_types.shape[0], len(generated)),
                        dtype=base_mm_types.dtype,
                        device=base_mm_types.device,
                    ),
                ],
                dim=-1,
            )
        prepare_kwargs = {
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
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
        prepared = model.prepare_inputs_for_generation(
            full_input_ids,
            **prepare_kwargs,
        )
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
        cache_length = cache_sequence_length(past_key_values)
        expected_cache_length = prompt_length + len(generated) - 1
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
        prepared.update({
            "use_cache": True,
            "output_attentions": True,
            "return_dict": True,
        })
        with torch.inference_mode():
            step_output = model_forward(model, prepared)
        if not step_output.attentions or step_output.attentions[0] is None:
            raise RuntimeError(
                "Decoder attentions are unavailable. Run the probe with --attn_implementation eager."
            )
        attention_steps.append(step_output.attentions)
        past_key_values = step_output.past_key_values
        next_token = step_output.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    generated_ids = torch.cat(generated, dim=-1) if generated else torch.empty((1, 0), dtype=torch.long)
    raw_response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    selected_step = None
    for idx in range(min(len(generated), len(attention_steps))):
        prefix_ids = torch.cat(generated[: idx + 1], dim=-1)
        prefix = processor.batch_decode(
            prefix_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        if parse_answer(prefix) != "UNKNOWN":
            selected_step = idx
            break
    if selected_step is None:
        selected_step = 0 if attention_steps else None
    if selected_step is None:
        raise RuntimeError("No generated non-EOS token was available for attention probing.")
    selected_token_id = int(generated[selected_step][0, 0])
    selected_token_text = processor.batch_decode(
        generated[selected_step],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    return raw_response, attention_steps[selected_step], {
        "selected_generation_step": selected_step,
        "selected_token_id": selected_token_id,
        "selected_token_text": selected_token_text,
        "generated_token_ids": [int(token[0, 0]) for token in generated],
        "cache_api": cache_api,
        "prompt_length": prompt_length,
        "prefill_position_ids_shape": (
            list(prefill_position_ids.shape) if prefill_position_ids is not None else None
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
    raw_response, answer_attentions, query_metadata = greedy_generate_with_answer_attention(
        model,
        processor,
        inputs,
        args.max_new_tokens,
    )
    result = aggregate_attention(
        row,
        inputs,
        processor,
        answer_attentions,
        video_path,
        video_kwargs,
        args.roi_padding,
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
        "prediction": parse_answer(raw_response),
        "is_correct": parse_answer(raw_response) == row.get("correct_option"),
        "raw_response": raw_response,
        "answer_query": query_metadata,
        "input_metadata": input_metadata,
    })
    del inputs, answer_attentions
    if args.empty_cache_each_sample and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def parse_csv_filter(value):
    return {part.strip() for part in value.split(",") if part.strip()} if value else None


def filter_rows(rows, conditions, prompt_variants, base_sample_ids):
    if conditions:
        rows = [row for row in rows if row.get("condition") in conditions]
    if prompt_variants:
        rows = [row for row in rows if row.get("prompt_variant") in prompt_variants]
    if base_sample_ids:
        rows = [row for row in rows if int(row.get("base_sample_id")) in base_sample_ids]
    return rows


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
    parser.add_argument("--video_fps", type=float, default=None)
    parser.add_argument("--video_num_frames", type=int, default=None)
    parser.add_argument("--video_max_pixels", type=int, default=None)
    parser.add_argument("--roi_padding", type=int, default=8)
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
    if not 0.0 <= args.heatmap_alpha <= 1.0:
        parser.error("--heatmap_alpha must be between 0 and 1.")

    conditions = parse_csv_filter(args.conditions)
    prompt_variants = parse_csv_filter(args.prompt_variants)
    base_sample_ids = (
        {int(value) for value in parse_csv_filter(args.base_sample_ids)}
        if args.base_sample_ids
        else None
    )
    rows = filter_rows(
        read_jsonl(args.annotation_path),
        conditions,
        prompt_variants,
        base_sample_ids,
    )[: args.max_samples]
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
    config_path = output_path.with_name(f"{output_path.stem}_config.json")
    config_payload = {
        **vars(args),
        "resolved_attention_implementation": getattr(
            model.config,
            "_attn_implementation",
            args.attn_implementation,
        ),
        "generation_cache_api": generation_cache_api(),
        "attention_cache_strategy": "prefill_mrope_then_single_token_decode",
        "selected_eval_ids": [row.get("eval_id") for row in rows],
        "environment": environment_metadata(model),
    }
    config_path.write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote attention ROI probe results to {output_path}")

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
