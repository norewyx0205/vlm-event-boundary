import argparse
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

try:
    from .common import PROJECT_ROOT, read_jsonl, write_jsonl
    from .generate_ladder_dataset import DRAW_COLORS, draw_shape, moving_distractor_window
except ImportError:
    from common import PROJECT_ROOT, read_jsonl, write_jsonl
    from generate_ladder_dataset import DRAW_COLORS, draw_shape, moving_distractor_window


DEFAULT_PERTURBATIONS = (
    "original,mask_target_1,mask_target_2,mask_distractors,remove_visual_marker"
)
PERTURBATION_ALIASES = {"mask_boundary": "remove_visual_marker"}
VALID_PERTURBATIONS = {
    "original",
    "mask_target_1",
    "mask_target_2",
    "mask_targets",
    "mask_distractors",
    "remove_visual_marker",
}


def lerp(a, b, t):
    return a + (b - a) * t


def object_position(obj, frame_idx):
    start = obj.get("start_frame")
    end = obj.get("end_frame")
    p0 = obj.get("from")
    p1 = obj.get("to")
    if p0 is None or p1 is None:
        return None
    if start is None or end is None or start == end:
        return tuple(p0)
    if frame_idx <= start:
        return tuple(p0)
    if frame_idx >= end:
        return tuple(p1)
    t = (frame_idx - start) / max(1, end - start)
    return (lerp(p0[0], p1[0], t), lerp(p0[1], p1[1], t))


def row_timing(row):
    return {
        **(row.get("event_timing") or {}),
        **(row.get("boundary_timing") or {}),
        "total_frames": int(row.get("total_frames") or 0),
    }


def timed_distractor(row, distractor):
    timed = dict(distractor)
    if timed.get("start_frame") is not None and timed.get("end_frame") is not None:
        return timed
    if distractor.get("motion_kind") != "unrelated_motion":
        timed["to"] = timed.get("from")
        timed["start_frame"] = 0
        timed["end_frame"] = max(0, int(row.get("total_frames") or 1) - 1)
        return timed

    start, end = moving_distractor_window(distractor, row_timing(row))
    timed["start_frame"] = int(start)
    timed["end_frame"] = int(end)
    return timed


def motion_window(obj):
    start = obj.get("start_frame")
    end = obj.get("end_frame")
    if start is None or end is None:
        return None
    return int(start), int(end)


def frame_in_scope(obj, frame_idx, mask_scope):
    if mask_scope == "all_frames" or obj.get("motion_kind") == "static":
        return True
    window = motion_window(obj)
    return window is None or window[0] <= frame_idx <= window[1]


def shape_mask(frame_shape, obj, center, padding):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    if center is None:
        return mask
    x, y = int(round(center[0])), int(round(center[1]))
    radius = int(obj.get("radius") or 28)
    shape = obj.get("shape", "circle")
    if shape == "circle":
        cv2.circle(mask, (x, y), radius, 255, -1)
    elif shape == "square":
        cv2.rectangle(mask, (x - radius, y - radius), (x + radius, y + radius), 255, -1)
    elif shape == "triangle":
        points = np.array(
            [[x, y - radius], [x - radius, y + radius], [x + radius, y + radius]],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points], 255)
    else:
        cv2.circle(mask, (x, y), radius, 255, -1)

    if padding > 0:
        kernel_size = padding * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel)
    return mask


def trajectory_mask(frame_shape, obj, padding):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    p0 = obj.get("from")
    p1 = obj.get("to")
    if p0 is None:
        return mask
    radius = int(obj.get("radius") or 28)
    if obj.get("shape") in {"square", "triangle"}:
        radius = int(math.ceil(radius * math.sqrt(2)))
    radius += padding
    start = (int(round(p0[0])), int(round(p0[1])))
    end = start if p1 is None else (int(round(p1[0])), int(round(p1[1])))
    cv2.line(mask, start, end, 255, max(1, radius * 2))
    cv2.circle(mask, start, radius, 255, -1)
    cv2.circle(mask, end, radius, 255, -1)
    return mask


def apply_object_mask(frame, obj, frame_idx, bg_color, padding, mask_mode, mask_scope):
    if not frame_in_scope(obj, frame_idx, mask_scope):
        return 0
    if mask_mode == "trajectory":
        mask = trajectory_mask(frame.shape, obj, padding)
    else:
        mask = shape_mask(frame.shape, obj, object_position(obj, frame_idx), padding)
    changed = int(np.count_nonzero(mask))
    frame[mask > 0] = bg_color
    return changed


def estimate_background(frame, fixed_value=None, sample_width=16):
    if fixed_value is not None:
        return tuple([int(fixed_value)] * 3)
    height, width = frame.shape[:2]
    size = max(1, min(sample_width, height // 4, width // 4))
    corners = np.concatenate(
        [
            frame[:size, :size].reshape(-1, 3),
            frame[:size, width - size :].reshape(-1, 3),
            frame[height - size :, :size].reshape(-1, 3),
            frame[height - size :, width - size :].reshape(-1, 3),
        ],
        axis=0,
    )
    # Match the decoded background exactly; lossy video codecs can shift the
    # nominally neutral source value by a few units in individual channels.
    return tuple(int(round(float(value))) for value in np.median(corners, axis=0))


def draw_annotated_object(frame, obj, frame_idx):
    center = object_position(obj, frame_idx)
    if center is None:
        return
    color_name = obj.get("color", "black")
    if color_name not in DRAW_COLORS:
        raise ValueError(f"Unknown annotation color: {color_name}")
    draw_shape(
        frame,
        obj.get("shape", "circle"),
        color_name,
        center[0],
        center[1],
        size=int(obj.get("radius") or 28),
    )


def reconstruct_scene(frame, row, frame_idx, bg_color, targets, distractors):
    reconstructed = np.empty_like(frame)
    reconstructed[:] = bg_color
    for distractor in distractors:
        draw_annotated_object(reconstructed, distractor, frame_idx)
    for target in targets:
        draw_annotated_object(reconstructed, target, frame_idx)
    changed = int(np.count_nonzero(np.any(frame != reconstructed, axis=2)))
    frame[:] = reconstructed
    return changed


def visual_marker_window(row, padding_frames):
    timing = row.get("boundary_timing") or {}
    if timing.get("visual_marker") in (None, "", "none"):
        return None
    start = timing.get("boundary_start_frame")
    end = timing.get("boundary_end_frame")
    if start is None or end is None:
        return None
    total_frames = int(row.get("total_frames") or end + 1)
    return max(0, int(start) - padding_frames), min(total_frames, int(end) + padding_frames)


def perturbation_applicable(row, perturbation_type):
    targets = row.get("target_objects") or []
    if perturbation_type == "original":
        return True
    if perturbation_type == "mask_target_1":
        return len(targets) >= 1
    if perturbation_type == "mask_target_2":
        return len(targets) >= 2
    if perturbation_type == "mask_targets":
        return bool(targets)
    if perturbation_type == "mask_distractors":
        return bool(row.get("distractors"))
    if perturbation_type == "remove_visual_marker":
        return visual_marker_window(row, 0) is not None
    return False


def mux_source_audio(video_only_path, source_path, output_path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_only_path),
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def perturb_video(source_path, output_path, row, perturbation_type, args):
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or row.get("fps") or 15
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_only_path = output_path.with_name(f"{output_path.stem}_video_only{output_path.suffix}")
    writer = cv2.VideoWriter(
        str(video_only_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video writer: {video_only_path}")

    targets = sorted(row.get("target_objects") or [], key=lambda item: item["id"])
    distractors = [timed_distractor(row, item) for item in row.get("distractors") or []]
    marker_window = visual_marker_window(row, args.boundary_padding_frames)
    affected_frames = 0
    affected_pixels = 0
    frame_idx = 0
    bg_color = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if bg_color is None:
            bg_color = estimate_background(frame, args.background_value)

        frame_changed_pixels = 0
        marker_is_visible = (
            marker_window is not None
            and marker_window[0] <= frame_idx < marker_window[1]
        )
        protect_marker = args.protect_visual_marker and marker_is_visible
        if perturbation_type == "mask_target_1" and targets and not protect_marker:
            frame_changed_pixels += apply_object_mask(
                frame,
                targets[0],
                frame_idx,
                bg_color,
                args.mask_padding,
                args.mask_mode,
                args.mask_scope,
            )
        elif perturbation_type == "mask_target_2" and len(targets) > 1 and not protect_marker:
            frame_changed_pixels += apply_object_mask(
                frame,
                targets[1],
                frame_idx,
                bg_color,
                args.mask_padding,
                args.mask_mode,
                args.mask_scope,
            )
        elif perturbation_type == "mask_targets" and not protect_marker:
            for target in targets:
                frame_changed_pixels += apply_object_mask(
                    frame,
                    target,
                    frame_idx,
                    bg_color,
                    args.mask_padding,
                    args.mask_mode,
                    args.mask_scope,
                )
        elif perturbation_type == "mask_distractors" and not protect_marker:
            for distractor in distractors:
                frame_changed_pixels += apply_object_mask(
                    frame,
                    distractor,
                    frame_idx,
                    bg_color,
                    args.mask_padding,
                    args.mask_mode,
                    args.mask_scope,
                )
        elif (
            perturbation_type == "remove_visual_marker"
            and marker_window is not None
            and marker_window[0] <= frame_idx < marker_window[1]
        ):
            frame_changed_pixels += reconstruct_scene(
                frame,
                row,
                frame_idx,
                bg_color,
                targets,
                distractors,
            )

        if frame_changed_pixels:
            affected_frames += 1
            affected_pixels += frame_changed_pixels
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    if args.preserve_audio:
        mux_source_audio(video_only_path, source_path, output_path)
        video_only_path.unlink(missing_ok=True)
    else:
        video_only_path.replace(output_path)

    total_pixels = max(1, frame_idx * width * height)
    return {
        "source_frames": frame_idx,
        "affected_frames": affected_frames,
        "affected_frame_fraction": affected_frames / max(1, frame_idx),
        "affected_pixels": affected_pixels,
        "affected_pixel_fraction": affected_pixels / total_pixels,
        "background_color_bgr": list(bg_color or ()),
        "audio_preserved": bool(args.preserve_audio),
    }


def relative_or_absolute(path):
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def paired_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("pairing_id") or row.get("video_id")].append(row)
    for items in grouped.values():
        original = next((row for row in items if row.get("prompt_variant") == "original"), items[0])
        yield original, items


def filter_rows(rows, conditions, base_sample_ids, max_base_samples):
    if conditions:
        rows = [row for row in rows if row.get("condition") in conditions]
    if base_sample_ids:
        rows = [row for row in rows if int(row.get("base_sample_id")) in base_sample_ids]
    elif max_base_samples is not None:
        selected = []
        for row in rows:
            base_id = row.get("base_sample_id")
            if base_id not in selected:
                selected.append(base_id)
            if len(selected) >= max_base_samples:
                break
        rows = [row for row in rows if row.get("base_sample_id") in set(selected)]
    return rows


def parse_csv_values(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def canonical_perturbations(text):
    perturbations = []
    for value in parse_csv_values(text):
        canonical = PERTURBATION_ALIASES.get(value, value)
        if canonical not in VALID_PERTURBATIONS:
            raise ValueError(
                f"Unknown perturbation {value!r}; choose from {sorted(VALID_PERTURBATIONS)}"
            )
        if canonical not in perturbations:
            perturbations.append(canonical)
    return perturbations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--perturbations",
        default=DEFAULT_PERTURBATIONS,
        help="Comma-separated perturbations. mask_boundary is accepted as a legacy alias.",
    )
    parser.add_argument("--conditions", default=None, help="Optional comma-separated condition filter.")
    parser.add_argument("--base_sample_ids", default=None, help="Optional comma-separated base IDs.")
    parser.add_argument("--max_base_samples", type=int, default=None)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--mask_padding", type=int, default=6)
    parser.add_argument("--mask_mode", choices=["dynamic", "trajectory"], default="dynamic")
    parser.add_argument("--mask_scope", choices=["all_frames", "motion_window"], default="all_frames")
    parser.add_argument("--boundary_padding_frames", type=int, default=0)
    parser.add_argument("--background_value", type=int, default=None)
    parser.add_argument("--protect_visual_marker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preserve_audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep_inapplicable", action="store_true")
    args = parser.parse_args()
    if args.max_base_samples is not None and args.max_videos is not None:
        parser.error("Use only one limit: --max_base_samples or --max_videos.")
    if args.mask_padding < 0 or args.boundary_padding_frames < 0:
        parser.error("Mask padding values must be non-negative.")
    if args.background_value is not None and not 0 <= args.background_value <= 255:
        parser.error("--background_value must be between 0 and 255.")

    annotation_path = Path(args.annotation_path)
    output_root = Path(args.output_root)
    video_root = output_root / "videos"
    perturbations = canonical_perturbations(args.perturbations)
    conditions = parse_csv_values(args.conditions) if args.conditions else None
    base_sample_ids = (
        {int(value) for value in parse_csv_values(args.base_sample_ids)}
        if args.base_sample_ids
        else None
    )

    rows = filter_rows(
        read_jsonl(annotation_path),
        conditions,
        base_sample_ids,
        args.max_base_samples,
    )
    output_rows = []
    intervention_stats = []
    processed = 0
    skipped_inapplicable = defaultdict(int)
    parameters = {
        "mask_padding": args.mask_padding,
        "mask_mode": args.mask_mode,
        "mask_scope": args.mask_scope,
        "boundary_padding_frames": args.boundary_padding_frames,
        "background_value": args.background_value,
        "protect_visual_marker": args.protect_visual_marker,
        "preserve_audio": args.preserve_audio,
    }

    for source_row, eval_rows in paired_rows(rows):
        if args.max_videos is not None and processed >= args.max_videos:
            break
        source_video = PROJECT_ROOT / source_row["video_path"]
        if not source_video.exists():
            source_video = Path(source_row["video_path"])
        if not source_video.exists():
            raise FileNotFoundError(f"Source video not found: {source_video}")

        for perturbation_type in perturbations:
            applicable = perturbation_applicable(source_row, perturbation_type)
            if not applicable and not args.keep_inapplicable:
                skipped_inapplicable[perturbation_type] += 1
                continue

            stats = {
                "source_frames": int(source_row.get("total_frames") or 0),
                "affected_frames": 0,
                "affected_frame_fraction": 0.0,
                "affected_pixels": 0,
                "affected_pixel_fraction": 0.0,
                "background_color_bgr": [],
                "audio_preserved": True,
            }
            if perturbation_type == "original":
                output_video = source_video
            else:
                output_video = video_root / perturbation_type / source_row["video_id"]
                stats = perturb_video(
                    source_video,
                    output_video,
                    source_row,
                    perturbation_type,
                    args,
                )

            stat_row = {
                "source_video_id": source_row["video_id"],
                "condition": source_row.get("condition"),
                "base_sample_id": source_row.get("base_sample_id"),
                "perturbation_type": perturbation_type,
                "perturbation_applicable": applicable,
                **stats,
            }
            intervention_stats.append(stat_row)
            for row in eval_rows:
                output_row = dict(row)
                output_row["eval_id"] = f"{row['eval_id']}_roi_{perturbation_type}"
                output_row["source_video_id"] = source_row["video_id"]
                output_row["source_video_path"] = source_row["video_path"]
                output_row["video_path"] = relative_or_absolute(output_video)
                output_row["video_id"] = output_video.name
                output_row["perturbation_type"] = perturbation_type
                output_row["perturbation_target"] = perturbation_type.replace("mask_", "")
                output_row["perturbation_applicable"] = applicable
                output_row["perturbation_parameters"] = parameters
                output_row["perturbation_stats"] = stats
                output_rows.append(output_row)
        processed += 1

    output_annotation = output_root / "annotations.jsonl"
    write_jsonl(output_annotation, output_rows)
    write_jsonl(output_root / "perturbation_stats.jsonl", intervention_stats)
    manifest = {
        "source_annotation_path": str(annotation_path),
        "output_annotation_path": str(output_annotation),
        "perturbations": perturbations,
        "parameters": parameters,
        "condition_filter": sorted(conditions) if conditions else None,
        "base_sample_id_filter": sorted(base_sample_ids) if base_sample_ids else None,
        "source_videos": processed,
        "eval_rows": len(output_rows),
        "skipped_inapplicable": dict(skipped_inapplicable),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(output_rows)} perturbation eval rows to {output_annotation}")
    print(f"Wrote intervention statistics to {output_root / 'perturbation_stats.jsonl'}")


if __name__ == "__main__":
    main()
