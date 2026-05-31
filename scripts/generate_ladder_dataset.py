import argparse
import json
import random
import subprocess
import wave
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


W, H = 512, 512
CONDITIONS = ["low_boundary", "temporal_boundary", "visual_boundary", "audio_boundary"]
COLORS = {
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "green": (0, 180, 0),
    "yellow": (0, 220, 220),
    "purple": (180, 0, 180),
    "orange": (0, 140, 255),
}
SHAPES = ["circle", "square", "triangle"]
DIRECTIONS = ["right", "left", "down", "up"]

LEVELS = [
    {
        "difficulty_level": 1,
        "difficulty_name": "level_1_simple",
        "randomized_targets": False,
        "static_distractors": False,
        "moving_distractors": False,
        "include_unrelated_later_motion": False,
    },
    {
        "difficulty_level": 2,
        "difficulty_name": "level_2_randomized",
        "randomized_targets": True,
        "static_distractors": False,
        "moving_distractors": False,
        "include_unrelated_later_motion": False,
    },
    {
        "difficulty_level": 3,
        "difficulty_name": "level_3_static_distractors",
        "randomized_targets": True,
        "static_distractors": True,
        "moving_distractors": False,
        "include_unrelated_later_motion": False,
    },
    {
        "difficulty_level": 4,
        "difficulty_name": "level_4_moving_distractors",
        "randomized_targets": True,
        "static_distractors": True,
        "moving_distractors": True,
        "include_unrelated_later_motion": True,
    },
]


def draw_shape(frame, shape, color_name, x, y, size=28):
    color = COLORS[color_name]
    x, y = int(x), int(y)

    if shape == "circle":
        cv2.circle(frame, (x, y), size, color, -1)
    elif shape == "square":
        cv2.rectangle(frame, (x - size, y - size), (x + size, y + size), color, -1)
    elif shape == "triangle":
        pts = np.array([[x, y - size], [x - size, y + size], [x + size, y + size]])
        cv2.fillPoly(frame, [pts], color)
    else:
        raise ValueError(f"Unknown shape: {shape}")


def lerp(a, b, t):
    return a + (b - a) * t


def object_position(motion, frame_idx):
    if frame_idx < motion["start_frame"]:
        return motion["from"]
    if frame_idx > motion["end_frame"]:
        return motion["to"]

    progress = (frame_idx - motion["start_frame"]) / max(1, motion["end_frame"] - motion["start_frame"])
    return (
        lerp(motion["from"][0], motion["to"][0], progress),
        lerp(motion["from"][1], motion["to"][1], progress),
    )


def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def far_enough(point, existing, min_dist=85):
    return all(distance(point, other) >= min_dist for other in existing)


def sample_point(existing, margin=70, min_dist=85):
    for _ in range(250):
        point = (random.randint(margin, W - margin), random.randint(margin, H - margin))
        if far_enough(point, existing, min_dist):
            existing.append(point)
            return point
    raise RuntimeError("Could not sample a non-overlapping point.")


def sample_path(existing, randomized=True):
    if not randomized:
        raise ValueError("sample_path(randomized=False) should not be used.")

    margin = 70
    distance_px = random.randint(105, 165)

    for _ in range(250):
        direction = random.choice(DIRECTIONS)

        if direction == "right":
            start = (random.randint(margin, W - margin - distance_px), random.randint(margin, H - margin))
            end = (start[0] + distance_px, start[1])
        elif direction == "left":
            start = (random.randint(margin + distance_px, W - margin), random.randint(margin, H - margin))
            end = (start[0] - distance_px, start[1])
        elif direction == "down":
            start = (random.randint(margin, W - margin), random.randint(margin, H - margin - distance_px))
            end = (start[0], start[1] + distance_px)
        else:
            start = (random.randint(margin, W - margin), random.randint(margin + distance_px, H - margin))
            end = (start[0], start[1] - distance_px)

        if far_enough(start, existing) and far_enough(end, existing):
            existing.extend([start, end])
            return {"direction": direction, "from": start, "to": end}

    raise RuntimeError("Could not sample a non-overlapping path.")


def fixed_target_paths():
    return (
        {"direction": "right", "from": (120, 180), "to": (260, 180)},
        {"direction": "left", "from": (390, 330), "to": (250, 330)},
    )


def object_text(obj):
    return f"the {obj['color']} {obj['shape']}"


def event_text(obj):
    return f"The {obj['color']} {obj['shape']} moved {obj['path']['direction']}."


def make_motion(path, start_frame, end_frame):
    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "from": path["from"],
        "to": path["to"],
    }


def make_static_motion(point):
    return {"start_frame": 0, "end_frame": 0, "from": point, "to": point}


def get_level_duration(level, durations):
    return durations[level["difficulty_level"] - 1]


def get_timing(condition, fps, duration_sec, event_duration, temporal_gap, visual_marker, include_unrelated):
    total_frames = fps * duration_sec
    start_hold = fps if duration_sec <= 10 else 2 * fps
    first_start = start_hold
    first_end = first_start + event_duration

    if condition == "low_boundary":
        boundary_start = first_end
        boundary_end = first_end
        second_start = first_end
        gap_frames = 0
        visual_marker_name = "none"
        audio_marker = "none"
    elif condition == "temporal_boundary":
        boundary_start = first_end
        boundary_end = first_end + temporal_gap
        second_start = boundary_end
        gap_frames = temporal_gap
        visual_marker_name = "none"
        audio_marker = "none"
    elif condition == "visual_boundary":
        boundary_start = first_end
        boundary_end = first_end + visual_marker
        second_start = boundary_end
        gap_frames = 0
        visual_marker_name = "white_flash_black_dot"
        audio_marker = "none"
    elif condition == "audio_boundary":
        boundary_start = first_end
        boundary_end = first_end + visual_marker
        second_start = boundary_end
        gap_frames = 0
        visual_marker_name = "none"
        audio_marker = "beep"
    else:
        raise ValueError(f"Unknown condition: {condition}")

    second_end = second_start + event_duration
    unrelated_start = int(total_frames * 0.55) if include_unrelated else None
    unrelated_end = unrelated_start + event_duration if include_unrelated else None

    if max(second_end, unrelated_end or 0) >= total_frames:
        raise ValueError("Timing exceeds video length.")

    return {
        "total_frames": total_frames,
        "first_event_start_frame": first_start,
        "first_event_end_frame": first_end,
        "second_event_start_frame": second_start,
        "second_event_end_frame": second_end,
        "boundary_start_frame": boundary_start,
        "boundary_end_frame": boundary_end,
        "gap_frames": gap_frames,
        "visual_marker": visual_marker_name,
        "audio_marker": audio_marker,
        "unrelated_event_start_frame": unrelated_start,
        "unrelated_event_end_frame": unrelated_end,
    }


def add_beep_to_video(video_path, output_path, duration_sec, beep_time, beep_duration, freq=880, audio_fps=44100):
    audio_path = output_path.with_suffix(".wav")
    total_audio_samples = int(duration_sec * audio_fps)
    audio = np.zeros(total_audio_samples, dtype=np.float32)
    start = int(beep_time * audio_fps)
    end = min(start + int(beep_duration * audio_fps), total_audio_samples)
    t = np.arange(end - start) / audio_fps
    audio[start:end] = 0.35 * np.sin(2 * np.pi * freq * t)
    audio_int16 = np.int16(audio * 32767)

    with wave.open(str(audio_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(audio_fps)
        wf.writeframes(audio_int16.tobytes())

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    audio_path.unlink(missing_ok=True)


def render_video(video_path, condition, fps, duration_sec, objects, distractors, timing):
    total_frames = timing["total_frames"]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (W, H))

    for frame_idx in range(total_frames):
        frame = np.ones((H, W, 3), dtype=np.uint8) * 245

        for distractor in distractors:
            x, y = object_position(distractor["motion"], frame_idx)
            draw_shape(frame, distractor["shape"], distractor["color"], x, y, size=23)

        for obj in objects:
            x, y = object_position(obj["motion"], frame_idx)
            draw_shape(frame, obj["shape"], obj["color"], x, y)

        if (
            condition == "visual_boundary"
            and timing["boundary_start_frame"] <= frame_idx < timing["boundary_end_frame"]
        ):
            frame[:] = 255
            cv2.circle(frame, (W // 2, H // 2), 48, (0, 0, 0), -1)

        writer.write(frame)

    writer.release()


def balanced_binary_sequence(first_value, second_value, count):
    values = [first_value, second_value] * (count // 2)
    if len(values) < count:
        values.append(first_value)
    random.shuffle(values)
    return values


def make_target_identity_specs(samples_per_level):
    specs = []

    for base_id in range(1, samples_per_level + 1):
        colors = random.sample(list(COLORS.keys()), 2)
        shapes = random.sample(SHAPES, 2)
        specs.append({
            "base_sample_id": base_id,
            "object_1": {
                "id": 1,
                "shape": shapes[0],
                "color": colors[0],
            },
            "object_2": {
                "id": 2,
                "shape": shapes[1],
                "color": colors[1],
            },
        })

    return specs


def make_target_objects(level, first_object_id, identity_spec):
    existing = []

    if level["randomized_targets"]:
        path_1 = sample_path(existing)
        path_2 = sample_path(existing)
    else:
        path_1, path_2 = fixed_target_paths()
        first_object_id = 1

    object_1 = {
        **identity_spec["object_1"],
        "path": path_1,
    }
    object_2 = {
        **identity_spec["object_2"],
        "path": path_2,
    }
    return object_1, object_2, first_object_id, existing


def make_distractors(level, existing, target_colors, static_distractors, moving_distractors):
    distractors = []
    distractor_colors = [color for color in COLORS if color not in target_colors] or list(COLORS)

    static_count = random.randint(1, static_distractors) if level["static_distractors"] and static_distractors else 0
    moving_count = random.randint(1, moving_distractors) if level["moving_distractors"] and moving_distractors else 0

    for idx in range(static_count):
        point = sample_point(existing)
        distractors.append({
            "id": len(distractors) + 1,
            "shape": random.choice(SHAPES),
            "color": random.choice(distractor_colors),
            "motion_kind": "static",
            "path": {"direction": "none", "from": point, "to": point},
        })

    for idx in range(moving_count):
        distractors.append({
            "id": len(distractors) + 1,
            "shape": random.choice(SHAPES),
            "color": random.choice(distractor_colors),
            "motion_kind": "unrelated_motion",
            "path": sample_path(existing),
        })

    return distractors


def timed_distractors(distractors, timing):
    timed = []
    unrelated_start = timing["unrelated_event_start_frame"]
    unrelated_end = timing["unrelated_event_end_frame"]

    for distractor in distractors:
        if distractor["motion_kind"] == "unrelated_motion" and unrelated_end is not None:
            motion = make_motion(distractor["path"], unrelated_start, unrelated_end)
        else:
            motion = make_static_motion(distractor["path"]["from"])

        timed.append({**distractor, "motion": motion})

    return timed


def make_sentence_pair(first_obj, second_obj, correct_relation):
    incorrect_relation = "after" if correct_relation == "before" else "before"

    if correct_relation == "before":
        subject = first_obj
        other = second_obj
    else:
        subject = second_obj
        other = first_obj

    correct_sentence = f"{object_text(subject).capitalize()} moves {correct_relation} {object_text(other)}."
    incorrect_sentence = f"{object_text(subject).capitalize()} moves {incorrect_relation} {object_text(other)}."
    return correct_relation, incorrect_relation, correct_sentence, incorrect_sentence


def make_eval_rows(video_annotation):
    base_eval_id = video_annotation["eval_id_base"]
    correct_sentence = video_annotation["correct_sentence"]
    incorrect_sentence = video_annotation["incorrect_sentence"]

    original = dict(video_annotation)
    original.update({
        "eval_id": f"{base_eval_id}_original",
        "prompt_variant": "original",
        "option_A": correct_sentence,
        "option_B": incorrect_sentence,
        "correct_option": "A",
    })

    swapped = dict(video_annotation)
    swapped.update({
        "eval_id": f"{base_eval_id}_swapped",
        "prompt_variant": "swapped",
        "option_A": incorrect_sentence,
        "option_B": correct_sentence,
        "correct_option": "B",
    })

    original.pop("eval_id_base", None)
    swapped.pop("eval_id_base", None)
    return [original, swapped]


def serialize_obj(obj, start_frame, end_frame):
    return {
        "id": obj["id"],
        "shape": obj["shape"],
        "color": obj["color"],
        "label": object_text(obj),
        "direction": obj["path"]["direction"],
        "from": obj["path"]["from"],
        "to": obj["path"]["to"],
        "start_frame": start_frame,
        "end_frame": end_frame,
    }


def generate_level(level, args, durations, target_identity_specs):
    level_dir = Path(args.output_root) / level["difficulty_name"]
    video_dir = level_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = level_dir / "annotations.jsonl"

    for stale in video_dir.glob("*.mp4"):
        stale.unlink()

    duration_sec = get_level_duration(level, durations)
    event_duration = int(args.event_duration_sec * args.fps)
    temporal_gap = int(args.temporal_gap_sec * args.fps)
    visual_marker = int(args.visual_marker_sec * args.fps)
    all_rows = []
    first_object_ids = balanced_binary_sequence(1, 2, args.samples_per_level)
    correct_relations = balanced_binary_sequence("before", "after", args.samples_per_level)

    for base_id in range(1, args.samples_per_level + 1):
        first_object_id = first_object_ids[base_id - 1] if level["randomized_targets"] else 1
        identity_spec = target_identity_specs[base_id - 1]
        object_1, object_2, first_object_id, existing = make_target_objects(level, first_object_id, identity_spec)
        second_object_id = 2 if first_object_id == 1 else 1
        first_obj = object_1 if first_object_id == 1 else object_2
        second_obj = object_2 if first_object_id == 1 else object_1
        correct_relation = correct_relations[base_id - 1]
        correct_relation, incorrect_relation, correct_sentence, incorrect_sentence = make_sentence_pair(
            first_obj,
            second_obj,
            correct_relation,
        )
        distractors = make_distractors(
            level,
            existing,
            [object_1["color"], object_2["color"]],
            args.static_distractors,
            args.moving_distractors,
        )

        for condition in CONDITIONS:
            include_unrelated = level["include_unrelated_later_motion"] and not args.disable_unrelated_later_motion
            timing = get_timing(
                condition,
                args.fps,
                duration_sec,
                event_duration,
                temporal_gap,
                visual_marker,
                include_unrelated,
            )

            if first_object_id == 1:
                obj_1_start, obj_1_end = timing["first_event_start_frame"], timing["first_event_end_frame"]
                obj_2_start, obj_2_end = timing["second_event_start_frame"], timing["second_event_end_frame"]
            else:
                obj_1_start, obj_1_end = timing["second_event_start_frame"], timing["second_event_end_frame"]
                obj_2_start, obj_2_end = timing["first_event_start_frame"], timing["first_event_end_frame"]

            objects = [
                {**object_1, "motion": make_motion(object_1["path"], obj_1_start, obj_1_end)},
                {**object_2, "motion": make_motion(object_2["path"], obj_2_start, obj_2_end)},
            ]
            timed_dist = timed_distractors(distractors, timing)
            stem = f"level_{level['difficulty_level']}_sample_{base_id:03d}_{condition}"
            final_video = video_dir / f"{stem}.mp4"

            if condition == "audio_boundary":
                silent_video = video_dir / f"{stem}_silent.mp4"
                render_video(silent_video, condition, args.fps, duration_sec, objects, timed_dist, timing)
                add_beep_to_video(
                    silent_video,
                    final_video,
                    duration_sec,
                    timing["boundary_start_frame"] / args.fps,
                    args.audio_beep_duration_sec,
                )
                silent_video.unlink(missing_ok=True)
            else:
                render_video(final_video, condition, args.fps, duration_sec, objects, timed_dist, timing)

            video_annotation = {
                "dataset_version": args.dataset_version,
                "difficulty_level": level["difficulty_level"],
                "difficulty_name": level["difficulty_name"],
                "condition": condition,
                "boundary_type": condition.replace("_boundary", ""),
                "base_sample_id": base_id,
                "video_id": final_video.name,
                "video_path": str(final_video),
                "eval_id_base": stem,
                "fps": args.fps,
                "duration_sec": duration_sec,
                "total_frames": timing["total_frames"],
                "target_objects": [
                    serialize_obj(object_1, obj_1_start, obj_1_end),
                    serialize_obj(object_2, obj_2_start, obj_2_end),
                ],
                "first_object_id": first_object_id,
                "second_object_id": second_object_id,
                "event_1": event_text(first_obj),
                "event_2": event_text(second_obj),
                "distractors": [
                    {
                        "id": d["id"],
                        "shape": d["shape"],
                        "color": d["color"],
                        "label": f"the {d['color']} {d['shape']}",
                        "motion_kind": d["motion_kind"],
                        "direction": d["path"]["direction"],
                        "from": d["path"]["from"],
                        "to": d["path"]["to"],
                    }
                    for d in distractors
                ],
                "distractor_count": len(distractors),
                "static_distractor_count": sum(1 for d in distractors if d["motion_kind"] == "static"),
                "moving_distractor_count": sum(1 for d in distractors if d["motion_kind"] == "unrelated_motion"),
                "event_timing": {
                    "first_event_start_frame": timing["first_event_start_frame"],
                    "first_event_end_frame": timing["first_event_end_frame"],
                    "second_event_start_frame": timing["second_event_start_frame"],
                    "second_event_end_frame": timing["second_event_end_frame"],
                    "unrelated_event_start_frame": timing["unrelated_event_start_frame"],
                    "unrelated_event_end_frame": timing["unrelated_event_end_frame"],
                },
                "boundary_timing": {
                    "boundary_start_frame": timing["boundary_start_frame"],
                    "boundary_end_frame": timing["boundary_end_frame"],
                    "gap_frames": timing["gap_frames"],
                    "visual_marker": timing["visual_marker"],
                    "audio_marker": timing["audio_marker"],
                },
                "correct_relation": correct_relation,
                "incorrect_relation": incorrect_relation,
                "correct_sentence": correct_sentence,
                "incorrect_sentence": incorrect_sentence,
            }

            all_rows.extend(make_eval_rows(video_annotation))

    with open(annotation_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{level['difficulty_name']}: wrote {len(all_rows)} eval rows to {annotation_path}")


def write_data_readme(args, durations):
    output_root = Path(args.output_root)
    readme = output_root.parent / "README.md"
    lines = [
        "# Ladder Dataset",
        "",
        f"Dataset version: `{args.dataset_version}`",
        "",
        "Each level contains unique videos plus mirrored evaluation rows in `annotations.jsonl`.",
        "",
        "| Level | Name | Duration | Description |",
        "| --- | --- | ---: | --- |",
    ]
    descriptions = {
        1: "simple two-target baseline with no distractors",
        2: "randomized target positions, directions, and order",
        3: "randomized targets plus static distractors",
        4: "randomized targets plus static/moving distractors and later unrelated motion",
    }
    for level, duration in zip(LEVELS, durations):
        lines.append(
            f"| {level['difficulty_level']} | `{level['difficulty_name']}` | {duration}s | {descriptions[level['difficulty_level']]} |"
        )
    lines.extend([
        "",
        "For a fixed `base_sample_id`, the two target objects keep the same color and shape across all difficulty levels and boundary conditions. This controls for possible color/shape response bias.",
        "",
        "Every video is evaluated twice:",
        "",
        "- `prompt_variant=original`: correct sentence in option A",
        "- `prompt_variant=swapped`: correct sentence in option B",
        "",
    ])
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text("\n".join(lines), encoding="utf-8")


def parse_durations(value):
    durations = [int(part.strip()) for part in value.split(",")]
    if len(durations) != 4:
        raise argparse.ArgumentTypeError("--level_durations must contain exactly four comma-separated integers.")
    return durations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_version", default="ladder_v1")
    parser.add_argument("--samples_per_level", type=int, default=30)
    parser.add_argument("--output_root", default="data/ladder_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--level_durations", type=parse_durations, default=parse_durations("10,12,14,20"))
    parser.add_argument("--event_duration_sec", type=float, default=2.0)
    parser.add_argument("--temporal_gap_sec", type=float, default=3.0)
    parser.add_argument("--visual_marker_sec", type=float, default=1.0)
    parser.add_argument("--audio_beep_duration_sec", type=float, default=0.35)
    parser.add_argument("--static_distractors", type=int, default=2)
    parser.add_argument("--moving_distractors", type=int, default=1)
    parser.add_argument(
        "--disable_unrelated_later_motion",
        action="store_true",
        help="Disable the later unrelated motion event in level 4.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    target_identity_specs = make_target_identity_specs(args.samples_per_level)

    for level in LEVELS:
        generate_level(level, args, args.level_durations, target_identity_specs)

    write_data_readme(args, args.level_durations)


if __name__ == "__main__":
    main()
