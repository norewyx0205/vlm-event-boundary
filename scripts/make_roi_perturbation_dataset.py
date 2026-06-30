import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

try:
    from .common import PROJECT_ROOT, read_jsonl, write_jsonl
except ImportError:
    from common import PROJECT_ROOT, read_jsonl, write_jsonl


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


def mask_circle(frame, center, radius, color):
    if center is None:
        return
    x, y = int(round(center[0])), int(round(center[1]))
    cv2.circle(frame, (x, y), int(radius), color, -1)


def target_radius(obj):
    return int(obj.get("radius") or 32) + 10


def distractor_radius(obj):
    return int(obj.get("radius") or 28) + 10


def mask_target(frame, obj, frame_idx, bg_color):
    mask_circle(frame, object_position(obj, frame_idx), target_radius(obj), bg_color)


def mask_distractor(frame, obj, bg_color):
    p0 = obj.get("from")
    p1 = obj.get("to")
    radius = distractor_radius(obj)
    if p0:
        mask_circle(frame, p0, radius, bg_color)
    if p1:
        mask_circle(frame, p1, radius, bg_color)
    if p0 and p1 and p0 != p1:
        cv2.line(
            frame,
            (int(p0[0]), int(p0[1])),
            (int(p1[0]), int(p1[1])),
            bg_color,
            radius * 2,
        )


def mask_boundary(frame, row, frame_idx, bg_color):
    timing = row.get("boundary_timing") or {}
    start = timing.get("boundary_start_frame")
    end = timing.get("boundary_end_frame")
    if start is None or end is None or not (start <= frame_idx <= end):
        return
    cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), 64, bg_color, -1)


def perturb_video(source_path, output_path, row, perturbation_type):
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or row.get("fps") or 15
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    targets = sorted(row.get("target_objects") or [], key=lambda item: item["id"])
    distractors = row.get("distractors") or []

    frame_idx = 0
    bg_color = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if bg_color is None:
            bg_color = tuple(int(value) for value in frame[0, 0].tolist())

        if perturbation_type == "mask_target_1" and targets:
            mask_target(frame, targets[0], frame_idx, bg_color)
        elif perturbation_type == "mask_target_2" and len(targets) > 1:
            mask_target(frame, targets[1], frame_idx, bg_color)
        elif perturbation_type == "mask_targets":
            for target in targets:
                mask_target(frame, target, frame_idx, bg_color)
        elif perturbation_type == "mask_distractors":
            for distractor in distractors:
                mask_distractor(frame, distractor, bg_color)
        elif perturbation_type == "mask_boundary":
            mask_boundary(frame, row, frame_idx, bg_color)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--perturbations",
        default="mask_target_1,mask_target_2,mask_distractors,mask_boundary",
        help="Comma-separated perturbations.",
    )
    parser.add_argument("--max_videos", type=int, default=None)
    args = parser.parse_args()

    annotation_path = Path(args.annotation_path)
    output_root = Path(args.output_root)
    video_root = output_root / "videos"
    perturbations = [part.strip() for part in args.perturbations.split(",") if part.strip()]

    rows = read_jsonl(annotation_path)
    output_rows = []
    processed = 0
    for original, eval_rows in paired_rows(rows):
        if args.max_videos is not None and processed >= args.max_videos:
            break
        source_video = PROJECT_ROOT / original["video_path"]
        if not source_video.exists():
            source_video = Path(original["video_path"])
        for perturbation_type in perturbations:
            output_video = video_root / perturbation_type / original["video_id"]
            perturb_video(source_video, output_video, original, perturbation_type)
            for row in eval_rows:
                output_row = dict(row)
                output_row["eval_id"] = f"{row['eval_id']}_{perturbation_type}"
                output_row["video_path"] = relative_or_absolute(output_video)
                output_row["video_id"] = output_video.name
                output_row["perturbation_type"] = perturbation_type
                output_row["perturbation_target"] = perturbation_type.replace("mask_", "")
                output_rows.append(output_row)
        processed += 1

    output_annotation = output_root / "annotations.jsonl"
    write_jsonl(output_annotation, output_rows)
    manifest = {
        "source_annotation_path": str(annotation_path),
        "output_annotation_path": str(output_annotation),
        "perturbations": perturbations,
        "source_videos": processed,
        "eval_rows": len(output_rows),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(output_rows)} perturbation eval rows to {output_annotation}")


if __name__ == "__main__":
    main()
