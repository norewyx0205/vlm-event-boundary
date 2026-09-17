import json
import math
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

try:
    from .probe_attention_roi import (
        first_video_metadata,
        model_forward,
        source_frame_groups,
        standard_first_token,
        token_descriptors,
        video_shape,
        visual_positions,
    )
    from .run_eval import (
        build_messages,
        parse_answer,
        process_video_inputs,
        processor_input_metadata,
    )
except ImportError:
    from probe_attention_roi import (
        first_video_metadata,
        model_forward,
        source_frame_groups,
        standard_first_token,
        token_descriptors,
        video_shape,
        visual_positions,
    )
    from run_eval import (
        build_messages,
        parse_answer,
        process_video_inputs,
        processor_input_metadata,
    )


RESIDUAL_STREAM_LOCATION = "decoder_layer_residual_post"
POOLING_METHOD = "mean_over_token_group"
DEFAULT_TOKEN_GROUPS = (
    "visual_all",
    "roi_target_1",
    "roi_target_2",
    "roi_targets",
    "roi_distractors",
    "phase_event_1",
    "phase_gap",
    "phase_event_2",
    "text_target_1_mentions",
    "text_target_2_mentions",
    "text_temporal_relations",
    "text_options",
    "decision_position",
)

REQUIRED_TEXT_TOKEN_GROUPS = (
    "text_target_1_mentions",
    "text_target_2_mentions",
    "text_temporal_relations",
    "text_options",
)

PILOT_EXCLUDED_TOKEN_GROUPS = {
    "phase_gap": (
        "Excluded from the first causal pilot because low_boundary has no "
        "semantically matched inter-event gap. An absolute-time or event-relative "
        "control window must be defined before this group can be compared."
    ),
}


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def pair_id(row):
    return (
        f"l5_full_base_{int(row['base_sample_id']):03d}_"
        f"{row['prompt_variant']}"
    )


def group_manifest_rows(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key = pair_id(row)
        condition = row.get("condition")
        if condition not in {"low_boundary", "temporal_boundary"}:
            continue
        if condition in grouped[key]:
            raise ValueError(f"Duplicate {condition} row for matched pair {key}.")
        grouped[key][condition] = row
    incomplete = {
        key: sorted({"low_boundary", "temporal_boundary"} - set(value))
        for key, value in grouped.items()
        if set(value) != {"low_boundary", "temporal_boundary"}
    }
    if incomplete:
        details = "; ".join(f"{key}: missing {missing}" for key, missing in incomplete.items())
        raise ValueError(f"Incomplete low/temporal matched pairs: {details}")
    return dict(sorted(grouped.items()))


def locate_decoder_layers(model):
    candidates = (
        ("model", "language_model", "layers"),
        ("model", "model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
    )
    for path in candidates:
        current = model
        for part in path:
            current = getattr(current, part, None)
            if current is None:
                break
        if isinstance(current, torch.nn.ModuleList) and len(current):
            return current, ".".join(path)

    fallbacks = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or not len(module):
            continue
        lowered = name.lower()
        if name.endswith("layers") and ("language" in lowered or "model" in lowered):
            fallbacks.append((len(module), name, module))
    if not fallbacks:
        raise RuntimeError("Could not locate the language-model decoder layer ModuleList.")
    _, name, layers = max(fallbacks, key=lambda item: item[0])
    return layers, name


def hidden_from_layer_output(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported decoder layer output type: {type(output).__name__}")


def replace_layer_hidden(output, hidden):
    if torch.is_tensor(output):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError(f"Unsupported decoder layer output type: {type(output).__name__}")


def _find_subsequences(sequence, pattern):
    if not pattern or len(pattern) > len(sequence):
        return []
    starts = []
    width = len(pattern)
    for index in range(len(sequence) - width + 1):
        if sequence[index:index + width] == pattern:
            starts.append(index)
    return starts


def _phrase_positions(tokenizer, input_ids, phrases):
    positions = set()
    for phrase in phrases:
        if not phrase:
            continue
        variants = {str(phrase), str(phrase).capitalize()}
        if str(phrase).lower().startswith("the "):
            variants.add(str(phrase)[4:])
            variants.add(str(phrase)[4:].capitalize())
        for variant in variants:
            for contextual_variant in (variant, f" {variant}", f"\n{variant}"):
                token_ids = tokenizer.encode(
                    contextual_variant, add_special_tokens=False
                )
                for start in _find_subsequences(input_ids, token_ids):
                    positions.update(range(start, start + len(token_ids)))
    return sorted(positions)


def _tokenizer_ids_and_offsets(tokenizer, text):
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (TypeError, NotImplementedError, ValueError):
        return None, None
    token_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    if offsets and offsets and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    if token_ids is None or offsets is None or len(token_ids) != len(offsets):
        return None, None
    return list(token_ids), [tuple(item) for item in offsets]


def _option_positions(tokenizer, input_ids, option_a, option_b):
    """Locate option sentence tokens in their rendered A:/B: prompt context."""
    if not option_a or not option_b:
        return []
    core = (
        f"A: {option_a}\n"
        f"B: {option_b}\n\n"
        "Answer with only A or B."
    )
    for prefix in ("\n\n", "\n", "", " "):
        contextual = f"{prefix}{core}"
        token_ids, offsets = _tokenizer_ids_and_offsets(tokenizer, contextual)
        if not token_ids:
            continue
        starts = _find_subsequences(input_ids, token_ids)
        if not starts:
            continue
        option_a_start = len(prefix) + len("A: ")
        option_a_end = option_a_start + len(str(option_a))
        option_b_start = option_a_end + len("\nB: ")
        option_b_end = option_b_start + len(str(option_b))
        spans = ((option_a_start, option_a_end), (option_b_start, option_b_end))
        positions = set()
        for sequence_start in starts:
            for local_index, (start, end) in enumerate(offsets):
                if end <= start:
                    continue
                if any(
                    start < span_end and end > span_start
                    for span_start, span_end in spans
                ):
                    positions.add(sequence_start + local_index)
        if positions:
            return sorted(positions)

    # Offset mappings are unavailable for some slow tokenizers. In that case,
    # match each complete labelled line and retain it as a conservative span.
    positions = set()
    for label, option in (("A", option_a), ("B", option_b)):
        for contextual in (
            f"\n{label}: {option}\n",
            f"{label}: {option}\n",
            f"\n{label}: {option}",
            f"{label}: {option}",
        ):
            token_ids = tokenizer.encode(contextual, add_special_tokens=False)
            for start in _find_subsequences(input_ids, token_ids):
                positions.update(range(start, start + len(token_ids)))
    return sorted(positions)


def validate_required_text_groups(groups, row):
    missing = [name for name in REQUIRED_TEXT_TOKEN_GROUPS if not groups.get(name)]
    if missing:
        raise RuntimeError(
            "Failed to locate required Phase 3 text token group(s) "
            f"{missing} for {row.get('eval_id', 'unknown eval')}. This usually "
            "indicates that prompt-context token spans no longer match the active "
            "chat template/tokenizer; refusing to continue with silent empty groups."
        )


def _target_phrases(target):
    phrases = [target.get("label"), target.get("reference_label")]
    color = target.get("color")
    shape = target.get("shape")
    size_label = target.get("size_label") or target.get("target_size_label")
    if color and shape:
        phrases.append(f"the {color} {shape}")
    if size_label and shape:
        phrases.extend((f"the {size_label} {shape}", f"{size_label} {shape}"))
    return [phrase for phrase in phrases if phrase]


def build_token_groups(
    row,
    inputs,
    processor,
    video_path,
    video_kwargs,
    roi_padding=8,
    roi_assignment="overlap",
):
    visual, visual_position_source = visual_positions(inputs, processor)
    shape = video_shape(video_path)
    grid = getattr(inputs, "video_grid_thw", None)
    if not visual or shape is None or grid is None:
        raise RuntimeError("Cannot build Phase 3 token groups without video positions/grid metadata.")
    width, height, total_frames, _ = shape
    grid_t, grid_h, grid_w = [int(value) for value in grid[0].detach().cpu().tolist()]
    raw_count = grid_t * grid_h * grid_w
    ratio = raw_count / len(visual)
    merge_size = int(round(math.sqrt(ratio)))
    if merge_size < 1 or merge_size * merge_size * len(visual) != raw_count:
        raise RuntimeError(
            f"Cannot infer merged video grid for {len(visual)} tokens and grid "
            f"{grid_t, grid_h, grid_w}."
        )
    merged_h, merged_w = grid_h // merge_size, grid_w // merge_size
    frame_groups = source_frame_groups(
        first_video_metadata(video_kwargs), grid_t, total_frames
    )
    descriptors = token_descriptors(
        row,
        grid_t,
        merged_h,
        merged_w,
        width,
        height,
        frame_groups,
        roi_padding,
        roi_assignment,
    )
    if len(descriptors) != len(visual):
        raise RuntimeError(
            f"Descriptor/token mismatch: descriptors={len(descriptors)}, visual={len(visual)}."
        )

    groups = defaultdict(set)
    groups["visual_all"].update(visual)
    for position, descriptor in zip(visual, descriptors):
        spatial = descriptor["spatial_roi_weights"]
        phases = descriptor["temporal_phase_weights"]
        if spatial.get("target_1", 0.0) > 0:
            groups["roi_target_1"].add(position)
            groups["roi_targets"].add(position)
        if spatial.get("target_2", 0.0) > 0:
            groups["roi_target_2"].add(position)
            groups["roi_targets"].add(position)
        if spatial.get("distractors", 0.0) > 0:
            groups["roi_distractors"].add(position)
        if phases.get("event_1", 0.0) > 0:
            groups["phase_event_1"].add(position)
        if phases.get("boundary", 0.0) > 0:
            groups["phase_gap"].add(position)
        if phases.get("event_2", 0.0) > 0:
            groups["phase_event_2"].add(position)

    ids = inputs.input_ids[0].detach().cpu().tolist()
    targets = sorted(row.get("target_objects") or [], key=lambda item: item.get("id", 0))
    for target in targets:
        target_id = int(target.get("id", 0))
        if target_id in (1, 2):
            groups[f"text_target_{target_id}_mentions"].update(
                _phrase_positions(processor.tokenizer, ids, _target_phrases(target))
            )
    groups["text_temporal_relations"].update(
        _phrase_positions(processor.tokenizer, ids, ["before", "after"])
    )
    groups["text_options"].update(
        _option_positions(
            processor.tokenizer,
            ids,
            row.get("option_A"),
            row.get("option_B"),
        )
    )
    groups["decision_position"].add(len(ids) - 1)

    output = {
        name: sorted(groups.get(name, set()))
        for name in DEFAULT_TOKEN_GROUPS
    }
    validate_required_text_groups(output, row)
    metadata = {
        "visual_token_position_source": visual_position_source,
        "visual_token_count": len(visual),
        "raw_video_grid_thw": [grid_t, grid_h, grid_w],
        "merged_video_grid_thw": [grid_t, merged_h, merged_w],
        "spatial_merge_size": merge_size,
        "source_frame_groups": frame_groups,
        "token_group_counts": {name: len(value) for name, value in output.items()},
        "text_option_span_locator": (
            "contextual_labeled_option_block_offsets_with_labelled_line_fallback_v1"
        ),
        "required_text_token_groups": list(REQUIRED_TEXT_TOKEN_GROUPS),
        "roi_padding": roi_padding,
        "roi_assignment": roi_assignment,
    }
    return output, metadata


def resolve_video_path(row, project_root):
    candidate = Path(project_root) / row["video_path"]
    if candidate.exists():
        return candidate
    candidate = Path(row["video_path"])
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Video is missing: {row['video_path']}")


def prepare_example(model, processor, row, args):
    video_path = resolve_video_path(row, args.project_root)
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
    groups, group_metadata = build_token_groups(
        row,
        inputs,
        processor,
        video_path,
        video_kwargs,
        roi_padding=args.roi_padding,
        roi_assignment=args.roi_assignment,
    )
    return {
        "row": row,
        "video_path": video_path,
        "inputs": inputs,
        "groups": groups,
        "group_metadata": group_metadata,
        "input_metadata": processor_input_metadata(inputs, video_inputs, video_kwargs),
    }


def answer_token_ids(processor):
    token_ids = {}
    for label in ("A", "B"):
        encoded = processor.tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(
                f"Expected {label!r} to encode as one first-answer token, got {encoded}."
            )
        token_ids[label] = int(encoded[0])
    if token_ids["A"] == token_ids["B"]:
        raise RuntimeError("A and B resolved to the same tokenizer id.")
    return token_ids


def decision_from_logits(logits, processor, correct_option):
    logits = logits.float().detach().cpu()
    token_ids = answer_token_ids(processor)
    incorrect_option = "B" if correct_option == "A" else "A"
    correct_logit = float(logits[token_ids[correct_option]])
    incorrect_logit = float(logits[token_ids[incorrect_option]])
    first_id = int(logits.argmax())
    first_text = processor.batch_decode(
        torch.tensor([[first_id]]),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    prediction = parse_answer(first_text)
    return {
        "prediction": prediction,
        "predicted_first_token_id": first_id,
        "predicted_first_token_text": first_text,
        "correct_option": correct_option,
        "incorrect_option": incorrect_option,
        "correct_token_id": token_ids[correct_option],
        "incorrect_token_id": token_ids[incorrect_option],
        "correct_logit": correct_logit,
        "incorrect_logit": incorrect_logit,
        "margin": correct_logit - incorrect_logit,
        "is_correct": prediction == correct_option,
    }


def _capture_hook(storage, layer_index, groups, max_tokenwise_vectors):
    def hook(_module, _inputs, output):
        hidden = hidden_from_layer_output(output)
        for name, positions in groups.items():
            if not positions:
                storage[layer_index][name] = {
                    "count": 0,
                    "positions": [],
                    "mean": None,
                    "values": None,
                }
                continue
            index = torch.as_tensor(positions, device=hidden.device, dtype=torch.long)
            values = hidden[0].index_select(0, index)
            storage[layer_index][name] = {
                "count": len(positions),
                "positions": list(positions),
                "mean": values.float().mean(dim=0).detach().cpu(),
                "values": (
                    values.float().detach().cpu()
                    if len(positions) <= max_tokenwise_vectors
                    else None
                ),
            }
        return output
    return hook


def run_baseline_capture(
    model,
    processor,
    prepared,
    layer_indices=None,
    groups_by_layer=None,
    max_tokenwise_vectors=256,
    verify_standard=True,
):
    layers, layer_path = locate_decoder_layers(model)
    if layer_indices is None:
        layer_indices = list(range(len(layers)))
    storage = defaultdict(dict)
    with ExitStack() as stack:
        for layer_index in layer_indices:
            groups = (
                groups_by_layer.get(layer_index, {})
                if groups_by_layer is not None
                else prepared["groups"]
            )
            handle = layers[layer_index].register_forward_hook(
                _capture_hook(storage, layer_index, groups, max_tokenwise_vectors)
            )
            stack.callback(handle.remove)
        kwargs = dict(prepared["inputs"])
        kwargs.update({"use_cache": False, "return_dict": True})
        with torch.inference_mode():
            output = model_forward(model, kwargs)
    logits = output.logits[0, -1, :]
    decision = decision_from_logits(
        logits, processor, prepared["row"]["correct_option"]
    )
    standard = None
    if verify_standard:
        standard_id, standard_scores = standard_first_token(model, prepared["inputs"])
        finite = torch.isfinite(standard_scores) & torch.isfinite(logits.float().detach().cpu())
        differences = (
            logits.float().detach().cpu()[finite] - standard_scores[finite]
        ).abs()
        standard = {
            "standard_first_token_id": standard_id,
            "first_token_match": standard_id == decision["predicted_first_token_id"],
            "logits_max_abs_diff": float(differences.max()) if differences.numel() else None,
            "logits_mean_abs_diff": float(differences.mean()) if differences.numel() else None,
        }
        if not standard["first_token_match"]:
            raise RuntimeError(
                "Full-prompt hidden-state extraction changed standard first-token semantics: "
                f"forward={decision['predicted_first_token_id']}, standard={standard_id}."
            )
    return {
        "captures": dict(storage),
        "decision": decision,
        "logits": logits.float().detach().cpu(),
        "layer_count": len(layers),
        "decoder_layer_path": layer_path,
        "standard_parity": standard,
    }


def cosine_distance(left, right, eps=1e-12):
    denominator = max(float(left.norm() * right.norm()), eps)
    similarity = float(torch.dot(left.float(), right.float()) / denominator)
    return 1.0 - similarity


def relative_l2(low, temporal, eps=1e-12):
    return float((temporal.float() - low.float()).norm() / max(float(low.float().norm()), eps))


def capture_alignment(low_capture, temporal_capture):
    low_positions = low_capture.get("positions")
    temporal_positions = temporal_capture.get("positions")
    positions_available = low_positions is not None and temporal_positions is not None
    positions_identical = bool(
        positions_available and list(low_positions) == list(temporal_positions)
    )
    low_values = low_capture.get("values")
    temporal_values = temporal_capture.get("values")
    value_shapes_identical = bool(
        low_values is not None
        and temporal_values is not None
        and low_values.shape == temporal_values.shape
    )
    tokenwise_eligible = bool(
        positions_identical and value_shapes_identical and low_values.shape[0] > 0
    )
    positionwise_alignment_eligible = bool(
        positions_identical and len(low_positions) > 0
    )
    if not positions_available:
        position_alignment = "positions_unavailable"
    elif positions_identical:
        position_alignment = "identical_sequence_positions"
    else:
        position_alignment = "different_sequence_positions"
    return {
        "position_alignment": position_alignment,
        "positions_identical": positions_identical,
        "event_relative_mapping_used": False,
        "event_relative_mapping": None,
        "value_shapes_identical": value_shapes_identical,
        "positionwise_alignment_eligible": positionwise_alignment_eligible,
        "tokenwise_metrics_eligible": tokenwise_eligible,
        "tokenwise_alignment_assumption": (
            "identical_sequence_positions"
            if tokenwise_eligible
            else "not_computed_without_explicit_token_correspondence"
        ),
        "patch_method_eligibility": (
            "positionwise_replace"
            if positionwise_alignment_eligible
            else "pooled_mean_delta"
        ),
    }


def divergence_metrics(low_capture, temporal_capture, eps=1e-12):
    alignment = capture_alignment(low_capture, temporal_capture)
    low_mean = low_capture.get("mean")
    temporal_mean = temporal_capture.get("mean")
    if low_mean is None or temporal_mean is None:
        return {
            "status": "missing_token_group",
            "cosine_distance": None,
            "relative_l2": None,
            "tokenwise_cosine_mean": None,
            "tokenwise_cosine_median": None,
            "tokenwise_relative_l2_mean": None,
            **alignment,
        }
    result = {
        "status": "ok",
        "cosine_distance": cosine_distance(low_mean, temporal_mean, eps),
        "relative_l2": relative_l2(low_mean, temporal_mean, eps),
        "tokenwise_cosine_mean": None,
        "tokenwise_cosine_median": None,
        "tokenwise_relative_l2_mean": None,
        **alignment,
    }
    low_values = low_capture.get("values")
    temporal_values = temporal_capture.get("values")
    if alignment["tokenwise_metrics_eligible"]:
        similarities = torch.nn.functional.cosine_similarity(
            low_values.float(), temporal_values.float(), dim=-1, eps=eps
        )
        denominators = low_values.float().norm(dim=-1).clamp_min(eps)
        relative = (temporal_values.float() - low_values.float()).norm(dim=-1) / denominators
        distances = 1.0 - similarities
        result.update({
            "tokenwise_cosine_mean": float(distances.mean()),
            "tokenwise_cosine_median": float(distances.median()),
            "tokenwise_relative_l2_mean": float(relative.mean()),
        })
    return result


def selected_capture_groups(candidates, prepared):
    by_layer = defaultdict(dict)
    for candidate in candidates:
        layer = int(candidate["layer"])
        group = candidate["token_group"]
        by_layer[layer][group] = prepared["groups"].get(group, [])
    return dict(by_layer)


def patch_method(source_capture, target_capture):
    alignment = capture_alignment(source_capture, target_capture)
    if alignment["positionwise_alignment_eligible"]:
        source_values = source_capture.get("values")
        target_values = target_capture.get("values")
        if (
            source_values is not None
            and target_values is not None
            and source_values.shape == target_values.shape
        ):
            return "positionwise_replace"
    return "pooled_mean_delta"


def run_patched_forward(
    model,
    processor,
    prepared_target,
    layer_index,
    token_group,
    source_capture,
    target_capture,
):
    positions = prepared_target["groups"].get(token_group, [])
    if not positions:
        raise RuntimeError(f"Target token group {token_group!r} is empty.")
    if source_capture.get("mean") is None or target_capture.get("mean") is None:
        raise RuntimeError(f"Source/target capture is unavailable for {token_group!r}.")
    method = patch_method(source_capture, target_capture)
    layers, _ = locate_decoder_layers(model)

    def patch_hook(_module, _inputs, output):
        hidden = hidden_from_layer_output(output)
        patched = hidden.clone()
        index = torch.as_tensor(positions, device=hidden.device, dtype=torch.long)
        if method == "positionwise_replace":
            values = source_capture["values"].to(device=hidden.device, dtype=hidden.dtype)
            patched[0].index_copy_(0, index, values)
        else:
            delta = (
                source_capture["mean"] - target_capture["mean"]
            ).to(device=hidden.device, dtype=hidden.dtype)
            patched[0, index, :] = patched[0, index, :] + delta
        return replace_layer_hidden(output, patched)

    handle = layers[layer_index].register_forward_hook(patch_hook)
    try:
        kwargs = dict(prepared_target["inputs"])
        kwargs.update({"use_cache": False, "return_dict": True})
        with torch.inference_mode():
            output = model_forward(model, kwargs)
    finally:
        handle.remove()
    logits = output.logits[0, -1, :]
    return {
        "patch_method": method,
        "decision": decision_from_logits(
            logits, processor, prepared_target["row"]["correct_option"]
        ),
        "logits": logits.float().detach().cpu(),
    }


def validate_archived_prediction(prepared, decision):
    archived = prepared["row"].get("archived_prediction")
    if archived not in (None, "") and decision["prediction"] != archived:
        raise RuntimeError(
            "Phase 3 baseline differs from archived evaluation: "
            f"eval_id={prepared['row'].get('eval_id')}, "
            f"current={decision['prediction']}, archived={archived}."
        )
    return archived in (None, "") or decision["prediction"] == archived


def validate_archived_input_metadata(prepared):
    archived = prepared["row"].get("archived_input_metadata")
    if not archived:
        return {
            "available": False,
            "matches": None,
            "checked_fields": [],
            "mismatches": {},
        }
    current = prepared["input_metadata"]
    checked = (
        "video_grid_thw",
        "visual_token_count_from_grid_thw",
        "video_token_count_from_mm_token_type_ids",
        "video_inputs",
        "pixel_values_videos",
        "input_ids",
    )
    mismatches = {
        key: {"archived": archived.get(key), "current": current.get(key)}
        for key in checked
        if archived.get(key) != current.get(key)
    }
    result = {
        "available": True,
        "matches": not mismatches,
        "checked_fields": list(checked),
        "mismatches": mismatches,
    }
    if mismatches:
        raise RuntimeError(
            "Phase 3 processor inputs differ from the archived behavioural run: "
            + ", ".join(sorted(mismatches))
        )
    return result
