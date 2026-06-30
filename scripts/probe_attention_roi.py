import argparse
import json
from pathlib import Path

import cv2
import torch

try:
    from .common import PROJECT_ROOT, read_jsonl
    from .run_eval import build_messages, load_model, parse_answer, process_video_inputs
except ImportError:
    from common import PROJECT_ROOT, read_jsonl
    from run_eval import build_messages, load_model, parse_answer, process_video_inputs


def object_label(obj):
    if obj.get("reference_label") in {"smallest", "largest"}:
        return f"target_{obj['id']}_{obj['reference_label']}"
    return f"target_{obj['id']}_{obj.get('color', '')}_{obj.get('shape', '')}".strip("_")


def lerp(a, b, t):
    return a + (b - a) * t


def position_at(obj, frame_idx):
    p0 = obj.get("from")
    p1 = obj.get("to")
    start = obj.get("start_frame", 0)
    end = obj.get("end_frame", 0)
    if p0 is None or p1 is None:
        return None
    if end <= start:
        return tuple(p0)
    if frame_idx <= start:
        return tuple(p0)
    if frame_idx >= end:
        return tuple(p1)
    progress = (frame_idx - start) / max(1, end - start)
    return (lerp(p0[0], p1[0], progress), lerp(p0[1], p1[1], progress))


def point_in_circle(x, y, center, radius):
    if center is None:
        return False
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2


def visual_token_ids(processor):
    tokenizer = processor.tokenizer
    candidates = [
        "<|video_pad|>",
        "<|image_pad|>",
        getattr(processor, "video_token", None),
        getattr(processor, "image_token", None),
    ]
    ids = set()
    for token in candidates:
        if not token:
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            ids.add(token_id)
    return ids


def video_shape(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, frames


def roi_for_token(row, token_idx, grid_t, grid_h, grid_w, width, height, total_frames):
    per_frame = grid_h * grid_w
    t = token_idx // per_frame
    spatial = token_idx % per_frame
    y_idx = spatial // grid_w
    x_idx = spatial % grid_w
    x = (x_idx + 0.5) * width / grid_w
    y = (y_idx + 0.5) * height / grid_h
    frame_idx = round(t / max(1, grid_t - 1) * max(0, total_frames - 1))

    for target in row.get("target_objects", []):
        radius = int(target.get("radius") or 28) + 8
        if point_in_circle(x, y, position_at(target, frame_idx), radius):
            return object_label(target)

    for distractor in row.get("distractors", []):
        radius = int(distractor.get("radius") or 23) + 8
        p0 = distractor.get("from")
        p1 = distractor.get("to")
        if point_in_circle(x, y, p0, radius) or point_in_circle(x, y, p1, radius):
            return "distractor"

    timing = row.get("boundary_timing") or {}
    start = timing.get("boundary_start_frame")
    end = timing.get("boundary_end_frame")
    if start is not None and end is not None and start <= frame_idx <= end:
        if point_in_circle(x, y, (width / 2, height / 2), 72):
            return "boundary_region"

    return "background"


def aggregate_attention(row, inputs, processor, attentions, video_path):
    token_ids = visual_token_ids(processor)
    input_ids = inputs.input_ids[0].detach().cpu().tolist()
    visual_positions = [idx for idx, token_id in enumerate(input_ids) if token_id in token_ids]

    if not visual_positions:
        return {
            "warning": "No visual placeholder tokens found in input_ids; check tokenizer special tokens.",
            "roi_attention": {},
        }

    final_step_attn = attentions[-1][-1]
    if isinstance(final_step_attn, tuple):
        final_step_attn = final_step_attn[-1]
    # Shape is usually [batch, heads, query_len, key_len] for the generated token.
    scores = final_step_attn[0, :, -1, :].float().mean(dim=0).detach().cpu()

    shape = video_shape(video_path)
    grid = getattr(inputs, "video_grid_thw", None)
    roi_totals = {}
    if shape and grid is not None:
        width, height, total_frames = shape
        grid_t, grid_h, grid_w = [int(value) for value in grid[0].detach().cpu().tolist()]
        expected = grid_t * grid_h * grid_w
        if expected == len(visual_positions):
            for local_idx, input_idx in enumerate(visual_positions):
                roi = roi_for_token(row, local_idx, grid_t, grid_h, grid_w, width, height, total_frames)
                roi_totals[roi] = roi_totals.get(roi, 0.0) + float(scores[input_idx])
        else:
            roi_totals["visual_tokens_total"] = float(scores[visual_positions].sum())
            roi_totals["grid_token_mismatch"] = float(expected - len(visual_positions))
    else:
        roi_totals["visual_tokens_total"] = float(scores[visual_positions].sum())

    total = sum(value for key, value in roi_totals.items() if not key.endswith("mismatch"))
    normalized = {
        key: value / total if total else None
        for key, value in roi_totals.items()
        if not key.endswith("mismatch")
    }
    return {
        "visual_token_count": len(visual_positions),
        "roi_attention": roi_totals,
        "roi_attention_normalized": normalized,
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
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            output_attentions=True,
            return_dict_in_generate=True,
        )
    generated_ids = output.sequences[:, inputs.input_ids.shape[1]:]
    raw_response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    result = aggregate_attention(row, inputs, processor, output.attentions, video_path)
    result.update({
        "eval_id": row.get("eval_id"),
        "video_id": row.get("video_id"),
        "condition": row.get("condition"),
        "feature_variant": row.get("feature_variant"),
        "size_scene_variant": row.get("size_scene_variant"),
        "correct_option": row.get("correct_option"),
        "prediction": parse_answer(raw_response),
        "raw_response": raw_response,
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--video_fps", type=float, default=None)
    parser.add_argument("--video_max_pixels", type=int, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.annotation_path)
    rows = rows[:args.max_samples]
    model, processor = load_model(
        args.model_name,
        model_revision=args.model_revision,
        attn_implementation=args.attn_implementation,
    )
    outputs = [probe_row(model, processor, row, args) for row in rows]
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote attention ROI probe results to {output_path}")


if __name__ == "__main__":
    main()
