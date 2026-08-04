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
    "original,reencode_control,mask_target_1,mask_target_2,mask_distractors,"
    "mask_background_control,remove_visual_marker,gap_removed,gap_shortened,gap_shifted"
)
PERTURBATION_ALIASES = {"mask_boundary": "remove_visual_marker"}
VALID_PERTURBATIONS = {
    "original",
    "reencode_control",
    "mask_target_1",
    "mask_target_2",
    "mask_targets",
    "mask_distractors",
    "mask_background_control",
    "remove_visual_marker",
    "gap_removed",
    "gap_shortened",
    "gap_shifted",
}
TEMPORAL_PERTURBATIONS = {"gap_removed", "gap_shortened", "gap_shifted"}


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


def object_mask(frame_shape, obj, frame_idx, padding, mask_mode, mask_scope):
    if not frame_in_scope(obj, frame_idx, mask_scope):
        return np.zeros(frame_shape[:2], dtype=np.uint8)
    if mask_mode == "trajectory":
        return trajectory_mask(frame_shape, obj, padding)
    return shape_mask(frame_shape, obj, object_position(obj, frame_idx), padding)


def union_object_masks(frame_shape, objects, frame_idx, padding, mask_mode, mask_scope):
    union = np.zeros(frame_shape[:2], dtype=np.uint8)
    for obj in objects:
        union = cv2.bitwise_or(
            union,
            object_mask(frame_shape, obj, frame_idx, padding, mask_mode, mask_scope),
        )
    return union


def object_bounding_radius(obj, padding=0):
    radius = int(obj.get("radius") or 28)
    if obj.get("shape") in {"square", "triangle"}:
        radius = int(math.ceil(radius * math.sqrt(2)))
    return radius + padding


def sham_overlap_frames(candidate, occupied_objects, total_frames, padding, clearance):
    candidate_radius = object_bounding_radius(candidate, padding)
    overlaps = 0
    for frame_idx in range(total_frames):
        candidate_center = object_position(candidate, frame_idx)
        for obj in occupied_objects:
            center = object_position(obj, frame_idx)
            if candidate_center is None or center is None:
                continue
            minimum_distance = (
                candidate_radius
                + object_bounding_radius(obj, padding)
                + clearance
            )
            if (
                (candidate_center[0] - center[0]) ** 2
                + (candidate_center[1] - center[1]) ** 2
                < minimum_distance**2
            ):
                overlaps += 1
                break
    return overlaps


def place_background_sham(
    reference,
    occupied_objects,
    total_frames,
    width,
    height,
    padding,
    clearance,
    label,
):
    reference = dict(reference)
    p0 = reference.get("from")
    p1 = reference.get("to")
    if p0 is None or p1 is None:
        raise ValueError("Background sham reference must have an annotated trajectory.")
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    margin = int(reference.get("radius") or 28) + padding + clearance
    step = max(16, margin // 2)
    candidates = []
    for y in range(margin, height - margin + 1, step):
        for x in range(margin, width - margin + 1, step):
            end_x, end_y = x + dx, y + dy
            if not (margin <= end_x <= width - margin and margin <= end_y <= height - margin):
                continue
            candidate = dict(reference)
            candidate["from"] = [x, y]
            candidate["to"] = [end_x, end_y]
            overlap = sham_overlap_frames(
                candidate,
                occupied_objects,
                total_frames,
                padding,
                clearance,
            )
            candidates.append((overlap, y, x, candidate))
    if not candidates:
        raise RuntimeError("Could not place a background sham trajectory inside the frame.")
    overlap, _, _, candidate = min(candidates, key=lambda item: item[:3])
    if overlap:
        raise RuntimeError(
            "Could not place a non-overlapping background sham trajectory; "
            f"minimum overlap was {overlap} frames."
        )
    candidate["id"] = f"background_control_{label}"
    candidate["reference_source"] = label
    return candidate


def background_sham_objects(row, width, height, reference_name, padding, clearance):
    targets = sorted(row.get("target_objects") or [], key=lambda item: item["id"])
    distractors = [timed_distractor(row, item) for item in row.get("distractors") or []]
    occupied_objects = targets + distractors
    total_frames = int(row.get("total_frames") or 1)

    if reference_name == "distractors":
        references = sorted(
            [(item, f"distractor_{item.get('id')}") for item in distractors],
            key=lambda pair: -(int(pair[0].get("radius") or 28)),
        )
    else:
        reference_index = 1 if reference_name == "target_2" and len(targets) > 1 else 0
        references = [(targets[reference_index], reference_name)]

    shams = []
    for reference, label in references:
        sham = place_background_sham(
            reference,
            occupied_objects,
            total_frames,
            width,
            height,
            padding,
            clearance,
            label,
        )
        shams.append(sham)
        occupied_objects.append(sham)
    return shams


def translate_mask(mask, offset):
    dx, dy = offset
    matrix = np.asarray([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(
        mask,
        matrix,
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def choose_background_sham_offset(
    frame_shape,
    row,
    reference_objects,
    occupied_objects,
    padding,
    mask_mode,
    mask_scope,
    clearance,
):
    height, width = frame_shape[:2]
    offsets = [
        (dx, dy)
        for dy in (-192, -128, -64, 0, 64, 128, 192)
        for dx in (-192, -128, -64, 0, 64, 128, 192)
        if (dx, dy) != (0, 0)
    ]
    total_frames = int(row.get("total_frames") or 1)
    sampled_frames = sorted(set(range(0, total_frames, max(1, total_frames // 18))) | {total_frames - 1})
    def build_frame_masks(frame_indices):
        masks = []
        centroids = []
        for frame_idx in frame_indices:
            reference_mask = union_object_masks(
                frame_shape,
                reference_objects,
                frame_idx,
                padding,
                mask_mode,
                mask_scope,
            )
            occupied = union_object_masks(
                frame_shape,
                occupied_objects,
                frame_idx,
                padding + clearance,
                "dynamic",
                "all_frames",
            )
            masks.append((reference_mask, occupied))
            centroids.append(mask_centroid(reference_mask))
        return masks, centroid_path_length(centroids)

    def score_offset(offset, frame_masks, reference_path_length):
        sham_centroids = []
        repair_pixels = 0
        for reference_mask, occupied in frame_masks:
            shifted = translate_mask(reference_mask, offset)
            retained = np.where((shifted > 0) & (occupied == 0), 255, 0).astype(np.uint8)
            repair_pixels += int(np.count_nonzero(reference_mask)) - int(np.count_nonzero(retained))
            sham = area_matched_background_mask(reference_mask, occupied, offset)
            sham_centroids.append(mask_centroid(sham))
        sham_path_length = centroid_path_length(sham_centroids)
        relative_path_difference = abs(sham_path_length - reference_path_length) / max(
            reference_path_length, 1.0
        )
        return (
            relative_path_difference,
            repair_pixels,
            abs(offset[0]) + abs(offset[1]),
            offset,
        )

    coarse_masks, coarse_reference_path = build_frame_masks(sampled_frames)
    coarse_scores = sorted(
        score_offset(offset, coarse_masks, coarse_reference_path) for offset in offsets
    )

    # Re-rank a small candidate set on denser temporal sampling. Short reference
    # trajectories are especially sensitive to sparse-frame approximation.
    dense_frames = sorted(
        set(range(0, total_frames, max(1, total_frames // 54))) | {total_frames - 1}
    )
    dense_masks, dense_reference_path = build_frame_masks(dense_frames)
    dense_scores = [
        score_offset(score[3], dense_masks, dense_reference_path)
        for score in coarse_scores[:6]
    ]
    return min(dense_scores)[3]


def area_matched_background_mask(reference_mask, occupied_mask, offset):
    desired_area = int(np.count_nonzero(reference_mask))
    if desired_area == 0:
        return np.zeros_like(reference_mask)
    reference_center = mask_centroid(reference_mask)
    if reference_center is None:
        return np.zeros_like(reference_mask)
    center_x = reference_center[0] + offset[0]
    center_y = reference_center[1] + offset[1]
    yy, xx = np.ogrid[: reference_mask.shape[0], : reference_mask.shape[1]]
    distance = (xx - center_x) ** 2 + (yy - center_y) ** 2
    available = occupied_mask == 0
    candidates = np.flatnonzero(available)
    if len(candidates) < desired_area:
        raise RuntimeError(
            f"Background sham needs {desired_area} pixels but only {len(candidates)} are free."
        )
    candidate_distances = distance.reshape(-1)[candidates]
    selected = candidates[
        np.argpartition(candidate_distances, desired_area - 1)[:desired_area]
    ]
    sham = np.zeros_like(reference_mask)
    sham.reshape(-1)[selected] = 255
    return sham


def mask_centroid(mask):
    moments = cv2.moments(mask, binaryImage=True)
    if not moments["m00"]:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def centroid_path_length(centroids):
    points = [point for point in centroids if point is not None]
    return sum(
        math.hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(points, points[1:])
    )


def shift_value_after_gap(value, gap_end, amount):
    if value is None:
        return None
    return int(value) - amount if int(value) >= gap_end else int(value)


def temporal_intervention(row, perturbation_type, total_frames, fps, shortened_sec):
    event = dict(row.get("event_timing") or {})
    boundary = dict(row.get("boundary_timing") or {})
    gap_start = int(boundary.get("boundary_start_frame") or 0)
    gap_end = int(boundary.get("boundary_end_frame") or gap_start)
    gap_frames = gap_end - gap_start
    if gap_frames <= 0:
        raise ValueError(f"{perturbation_type} requires a positive temporal gap.")
    if int(row.get("total_frames") or total_frames) != total_frames:
        raise ValueError("Decoded frame count differs from annotation total_frames.")
    if boundary.get("audio_marker") not in (None, "", "none"):
        raise ValueError("Temporal-gap interventions do not support an audio marker in the moved interval.")

    timed_objects = [timed_distractor(row, item) for item in row.get("distractors") or []]
    for obj in list(row.get("target_objects") or []) + timed_objects:
        window = motion_window(obj)
        if window and max(window[0], gap_start) < min(window[1], gap_end):
            raise ValueError(
                f"{perturbation_type} would remove active motion for object {obj.get('id')}; "
                "use stimuli with a stationary inter-event gap."
            )

    first_start = int(event["first_event_start_frame"])
    if perturbation_type == "gap_shifted":
        plan = (
            list(range(first_start))
            + [first_start] * gap_frames
            + list(range(first_start, gap_start))
            + list(range(gap_end, total_frames))
        )
        event["first_event_start_frame"] += gap_frames
        event["first_event_end_frame"] += gap_frames
        boundary.update({
            "boundary_start_frame": first_start,
            "boundary_end_frame": first_start + gap_frames,
            "gap_frames": gap_frames,
            "temporal_gap_location": "before_first_event",
        })
        target_updates = []
        for target in row.get("target_objects") or []:
            updated = dict(target)
            if int(updated.get("start_frame") or 0) < gap_start:
                updated["start_frame"] = int(updated["start_frame"]) + gap_frames
                updated["end_frame"] = int(updated["end_frame"]) + gap_frames
            target_updates.append(updated)
        moved_frames = 0
    else:
        retained_gap = 0
        if perturbation_type == "gap_shortened":
            retained_gap = min(gap_frames - 1, max(1, int(round(shortened_sec * fps))))
        moved_frames = gap_frames - retained_gap
        plan = (
            list(range(gap_start + retained_gap))
            + list(range(gap_end, total_frames))
            + [total_frames - 1] * moved_frames
        )
        event["second_event_start_frame"] -= moved_frames
        event["second_event_end_frame"] -= moved_frames
        for key in ("unrelated_event_start_frame", "unrelated_event_end_frame"):
            event[key] = shift_value_after_gap(event.get(key), gap_end, moved_frames)
        boundary.update({
            "boundary_start_frame": gap_start,
            "boundary_end_frame": gap_start + retained_gap,
            "gap_frames": retained_gap,
            "temporal_gap_location": "between_events" if retained_gap else "removed",
        })
        target_updates = []
        for target in row.get("target_objects") or []:
            updated = dict(target)
            updated["start_frame"] = shift_value_after_gap(
                updated.get("start_frame"), gap_end, moved_frames
            )
            updated["end_frame"] = shift_value_after_gap(
                updated.get("end_frame"), gap_end, moved_frames
            )
            target_updates.append(updated)

    if len(plan) != total_frames:
        raise RuntimeError(
            f"Temporal intervention produced {len(plan)} frames; expected {total_frames}."
        )
    updates = {
        "event_timing": event,
        "boundary_timing": boundary,
        "target_objects": target_updates,
        "temporal_intervention": {
            "type": perturbation_type,
            "original_gap_frames": gap_frames,
            "realized_gap_frames": int(boundary["gap_frames"]),
            "moved_frames": moved_frames,
            "total_frames_preserved": True,
            "frame_map": "stored_in_perturbation_stats",
        },
    }
    return plan, updates


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
    if perturbation_type in {"original", "reencode_control"}:
        return True
    if perturbation_type == "mask_target_1":
        return len(targets) >= 1
    if perturbation_type == "mask_target_2":
        return len(targets) >= 2
    if perturbation_type == "mask_targets":
        return bool(targets)
    if perturbation_type == "mask_distractors":
        return bool(row.get("distractors"))
    if perturbation_type == "mask_background_control":
        return bool(targets)
    if perturbation_type == "remove_visual_marker":
        return visual_marker_window(row, 0) is not None
    if perturbation_type in TEMPORAL_PERTURBATIONS:
        timing = row.get("boundary_timing") or {}
        return (
            row.get("condition") == "temporal_boundary"
            and int(timing.get("gap_frames") or 0) > 0
        )
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


def decoded_video_difference(source_path, comparison_path):
    source = cv2.VideoCapture(str(source_path))
    comparison = cv2.VideoCapture(str(comparison_path))
    if not source.isOpened() or not comparison.isOpened():
        source.release()
        comparison.release()
        raise RuntimeError("Could not decode source/re-encoded videos for codec metrics.")
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    squared_error = 0.0
    absolute_error = 0.0
    channel_values = 0
    changed_pixels = 0
    frames = 0
    while True:
        source_ok, source_frame = source.read()
        comparison_ok, comparison_frame = comparison.read()
        if not source_ok or not comparison_ok:
            break
        difference = source_frame.astype(np.float32) - comparison_frame.astype(np.float32)
        squared_error += float(np.square(difference).sum())
        absolute_error += float(np.abs(difference).sum())
        channel_values += int(difference.size)
        changed_pixels += int(np.count_nonzero(np.any(difference != 0, axis=2)))
        frames += 1
    source.release()
    comparison.release()
    mse = squared_error / max(1, channel_values)
    return {
        "compared_frames": frames,
        "mean_absolute_channel_error": absolute_error / max(1, channel_values),
        "mean_squared_channel_error": mse,
        "psnr_db": 10 * math.log10((255.0**2) / mse) if mse else None,
        "decoded_changed_pixel_fraction": changed_pixels
        / max(1, frames * width * height),
    }


def perturb_video(source_path, output_path, row, perturbation_type, args):
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or row.get("fps") or 15
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    temporal_updates = {}
    frame_plan = None
    source_frames = None
    if perturbation_type in TEMPORAL_PERTURBATIONS:
        source_frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            source_frames.append(frame)
        cap.release()
        decoded_frame_count = len(source_frames)
        frame_plan, temporal_updates = temporal_intervention(
            row,
            perturbation_type,
            decoded_frame_count,
            fps,
            args.gap_shortened_sec,
        )
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

    effective_row = {**row, **temporal_updates}
    targets = sorted(effective_row.get("target_objects") or [], key=lambda item: item["id"])
    distractors = [timed_distractor(row, item) for item in row.get("distractors") or []]
    marker_window = visual_marker_window(row, args.boundary_padding_frames)
    sham_objects = []
    sham_offset = None
    if perturbation_type == "mask_background_control":
        if args.sham_reference == "distractors":
            sham_offset = choose_background_sham_offset(
                (height, width, 3),
                row,
                distractors,
                targets + distractors,
                args.mask_padding,
                args.mask_mode,
                args.mask_scope,
                args.sham_clearance,
            )
        else:
            sham_objects = background_sham_objects(
                row,
                width,
                height,
                args.sham_reference,
                args.mask_padding,
                args.sham_clearance,
            )
    mask_assignment_frames = 0
    masked_pixel_assignments = 0
    changed_frames = 0
    changed_pixels = 0
    sham_reference_centroids = []
    sham_centroids = []
    sham_area_matches = []
    frame_idx = 0
    bg_color = None

    while True:
        if frame_plan is not None:
            if frame_idx >= len(frame_plan):
                break
            source_idx = frame_plan[frame_idx]
            frame = source_frames[source_idx].copy()
            comparison_frame = source_frames[frame_idx]
        else:
            ok, frame = cap.read()
            if not ok:
                break
            source_idx = frame_idx
            comparison_frame = frame.copy()
        if bg_color is None:
            bg_color = estimate_background(frame, args.background_value)

        assignment_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        marker_is_visible = (
            marker_window is not None
            and marker_window[0] <= frame_idx < marker_window[1]
        )
        protect_marker = args.protect_visual_marker and marker_is_visible
        if perturbation_type == "mask_target_1" and targets and not protect_marker:
            assignment_mask = union_object_masks(
                frame.shape,
                [targets[0]],
                frame_idx,
                args.mask_padding,
                args.mask_mode,
                args.mask_scope,
            )
        elif perturbation_type == "mask_target_2" and len(targets) > 1 and not protect_marker:
            assignment_mask = union_object_masks(
                frame.shape,
                [targets[1]],
                frame_idx,
                args.mask_padding,
                args.mask_mode,
                args.mask_scope,
            )
        elif perturbation_type == "mask_targets" and not protect_marker:
            assignment_mask = union_object_masks(
                frame.shape,
                targets,
                frame_idx,
                args.mask_padding,
                args.mask_mode,
                args.mask_scope,
            )
        elif perturbation_type == "mask_distractors" and not protect_marker:
            assignment_mask = union_object_masks(
                frame.shape,
                distractors,
                frame_idx,
                args.mask_padding,
                args.mask_mode,
                args.mask_scope,
            )
        elif perturbation_type == "mask_background_control" and not protect_marker:
            if args.sham_reference == "distractors":
                reference_mask = union_object_masks(
                    frame.shape,
                    distractors,
                    frame_idx,
                    args.mask_padding,
                    args.mask_mode,
                    args.mask_scope,
                )
                occupied_mask = union_object_masks(
                    frame.shape,
                    targets + distractors,
                    frame_idx,
                    args.mask_padding + args.sham_clearance,
                    "dynamic",
                    "all_frames",
                )
                assignment_mask = area_matched_background_mask(
                    reference_mask,
                    occupied_mask,
                    sham_offset,
                )
                sham_reference_centroids.append(mask_centroid(reference_mask))
                sham_centroids.append(mask_centroid(assignment_mask))
                sham_area_matches.append(
                    int(np.count_nonzero(reference_mask))
                    == int(np.count_nonzero(assignment_mask))
                )
            else:
                assignment_mask = union_object_masks(
                    frame.shape,
                    sham_objects,
                    frame_idx,
                    args.mask_padding,
                    args.mask_mode,
                    args.mask_scope,
                )
        elif (
            perturbation_type == "remove_visual_marker"
            and marker_window is not None
            and marker_window[0] <= frame_idx < marker_window[1]
        ):
            reconstruct_scene(
                frame,
                row,
                frame_idx,
                bg_color,
                targets,
                distractors,
            )

        if np.any(assignment_mask):
            mask_assignment_frames += 1
            masked_pixel_assignments += int(np.count_nonzero(assignment_mask))
            frame[assignment_mask > 0] = bg_color
        frame_changed_pixels = int(np.count_nonzero(np.any(frame != comparison_frame, axis=2)))
        if frame_changed_pixels:
            changed_frames += 1
            changed_pixels += frame_changed_pixels
        writer.write(frame)
        frame_idx += 1

    if cap.isOpened():
        cap.release()
    writer.release()
    if args.preserve_audio:
        mux_source_audio(video_only_path, source_path, output_path)
        video_only_path.unlink(missing_ok=True)
    else:
        video_only_path.replace(output_path)

    sham_reference_path_length = centroid_path_length(sham_reference_centroids)
    sham_path_length = centroid_path_length(sham_centroids)
    sham_path_relative_error = None
    if sham_reference_centroids:
        sham_path_relative_error = abs(sham_path_length - sham_reference_path_length) / max(
            sham_reference_path_length, 1.0
        )
        if sham_path_relative_error > args.sham_max_path_relative_error:
            output_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Background sham trajectory mismatch: "
                f"relative path-length error {sham_path_relative_error:.3f} exceeds "
                f"--sham_max_path_relative_error={args.sham_max_path_relative_error:.3f}."
            )

    total_pixels = max(1, frame_idx * width * height)
    stats = {
        "source_frames": frame_idx,
        "output_frames": frame_idx,
        "mask_assignment_frames": mask_assignment_frames,
        "mask_assignment_frame_fraction": mask_assignment_frames / max(1, frame_idx),
        "masked_pixel_assignments": masked_pixel_assignments,
        "masked_pixel_assignment_fraction": masked_pixel_assignments / total_pixels,
        "changed_frames": changed_frames,
        "changed_frame_fraction": changed_frames / max(1, frame_idx),
        "changed_pixels": changed_pixels,
        "changed_pixel_fraction": changed_pixels / total_pixels,
        "affected_frames": changed_frames,
        "affected_frame_fraction": changed_frames / max(1, frame_idx),
        "affected_pixels": changed_pixels,
        "affected_pixel_fraction": changed_pixels / total_pixels,
        "background_color_bgr": list(bg_color or ()),
        "audio_preserved": bool(args.preserve_audio),
        "codec_pipeline_applied": True,
        "source_frame_indices": frame_plan,
        "background_sham_objects": sham_objects,
        "background_sham_offset": list(sham_offset) if sham_offset is not None else None,
        "background_sham_area_match_rate": (
            sum(int(value) for value in sham_area_matches) / len(sham_area_matches)
            if sham_area_matches
            else None
        ),
        "background_sham_reference_centroid_path_length": sham_reference_path_length,
        "background_sham_centroid_path_length": sham_path_length,
        "background_sham_path_length_relative_error": sham_path_relative_error,
        "background_sham_path_match_within_tolerance": (
            sham_path_relative_error <= args.sham_max_path_relative_error
            if sham_path_relative_error is not None
            else None
        ),
    }
    if perturbation_type == "reencode_control" and args.measure_codec_effect:
        stats["codec_effect"] = decoded_video_difference(source_path, output_path)
    return stats, temporal_updates


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
    parser.add_argument(
        "--sham_reference",
        choices=["distractors", "target_1", "target_2"],
        default="distractors",
    )
    parser.add_argument("--sham_clearance", type=int, default=4)
    parser.add_argument("--sham_max_path_relative_error", type=float, default=0.10)
    parser.add_argument("--gap_shortened_sec", type=float, default=1.0)
    parser.add_argument("--boundary_padding_frames", type=int, default=0)
    parser.add_argument("--background_value", type=int, default=None)
    parser.add_argument("--protect_visual_marker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preserve_audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--measure_codec_effect", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep_inapplicable", action="store_true")
    args = parser.parse_args()
    if args.max_base_samples is not None and args.max_videos is not None:
        parser.error("Use only one limit: --max_base_samples or --max_videos.")
    if args.mask_padding < 0 or args.boundary_padding_frames < 0 or args.sham_clearance < 0:
        parser.error("Mask padding values must be non-negative.")
    if args.gap_shortened_sec <= 0:
        parser.error("--gap_shortened_sec must be positive.")
    if args.sham_max_path_relative_error < 0:
        parser.error("--sham_max_path_relative_error must be non-negative.")
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
        "sham_reference": args.sham_reference,
        "sham_clearance": args.sham_clearance,
        "sham_max_path_relative_error": args.sham_max_path_relative_error,
        "gap_shortened_sec": args.gap_shortened_sec,
        "boundary_padding_frames": args.boundary_padding_frames,
        "background_value": args.background_value,
        "protect_visual_marker": args.protect_visual_marker,
        "preserve_audio": args.preserve_audio,
        "measure_codec_effect": args.measure_codec_effect,
    }

    for source_row, eval_rows in paired_rows(rows):
        if args.max_videos is not None and processed >= args.max_videos:
            break
        print(
            f"ROI source {processed + 1}: video_id={source_row.get('video_id')}, "
            f"base_sample_id={source_row.get('base_sample_id')}, "
            f"condition={source_row.get('condition')}",
            flush=True,
        )
        source_video = PROJECT_ROOT / source_row["video_path"]
        if not source_video.exists():
            source_video = Path(source_row["video_path"])
        if not source_video.exists():
            raise FileNotFoundError(f"Source video not found: {source_video}")

        for perturbation_type in perturbations:
            applicable = perturbation_applicable(source_row, perturbation_type)
            if (
                perturbation_type == "mask_background_control"
                and args.sham_reference == "distractors"
            ):
                applicable = bool(source_row.get("distractors"))
            if not applicable and not args.keep_inapplicable:
                skipped_inapplicable[perturbation_type] += 1
                continue

            stats = {
                "source_frames": int(source_row.get("total_frames") or 0),
                "output_frames": int(source_row.get("total_frames") or 0),
                "mask_assignment_frames": 0,
                "mask_assignment_frame_fraction": 0.0,
                "masked_pixel_assignments": 0,
                "masked_pixel_assignment_fraction": 0.0,
                "changed_frames": 0,
                "changed_frame_fraction": 0.0,
                "changed_pixels": 0,
                "changed_pixel_fraction": 0.0,
                "affected_frames": 0,
                "affected_frame_fraction": 0.0,
                "affected_pixels": 0,
                "affected_pixel_fraction": 0.0,
                "background_color_bgr": [],
                "audio_preserved": True,
                "codec_pipeline_applied": False,
                "source_frame_indices": None,
                "background_sham_objects": [],
                "background_sham_offset": None,
                "background_sham_area_match_rate": None,
                "background_sham_reference_centroid_path_length": 0.0,
                "background_sham_centroid_path_length": 0.0,
                "background_sham_path_length_relative_error": None,
                "background_sham_path_match_within_tolerance": None,
            }
            metadata_updates = {}
            if perturbation_type == "original":
                output_video = source_video
            else:
                output_video = video_root / perturbation_type / source_row["video_id"]
                try:
                    stats, metadata_updates = perturb_video(
                        source_video,
                        output_video,
                        source_row,
                        perturbation_type,
                        args,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "ROI perturbation failed for "
                        f"video_id={source_row.get('video_id')}, "
                        f"base_sample_id={source_row.get('base_sample_id')}, "
                        f"condition={source_row.get('condition')}, "
                        f"perturbation_type={perturbation_type}: {exc}"
                    ) from exc

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
                output_row.update(metadata_updates)
                output_row["eval_id"] = f"{row['eval_id']}_roi_{perturbation_type}"
                output_row["source_eval_id"] = row["eval_id"]
                output_row["source_pairing_id"] = row.get("pairing_id")
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
