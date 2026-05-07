import cv2
import json
import random
import wave
import subprocess
import imageio_ffmpeg
import numpy as np
from pathlib import Path


# =========================
# Config
# =========================

OUT_DIR = Path("synthetic_boundary_videos")
VIDEO_DIR = OUT_DIR / "videos"
ANNOTATION_PATH = OUT_DIR / "annotations.jsonl"

VIDEO_DIR.mkdir(parents=True, exist_ok=True)

W, H = 512, 512
FPS = 15
DURATION_SEC = 20
TOTAL_FRAMES = FPS * DURATION_SEC  # 300 frames

EVENT_DURATION = 30  # 2 seconds
START_HOLD = 30      # 2 seconds

TEMPORAL_GAP_FRAMES = 45  # 3 seconds
VISUAL_MARKER_FRAMES = 15 # 1 second
AUDIO_MARKER_DURATION = 0.35

SAMPLES_PER_CONDITION = 30
DISTRACTOR_COUNT_RANGE = (1, 2)
UNRELATED_EVENT_START = 165
UNRELATED_EVENT_DURATION = 30

CONDITIONS = [
    "low_boundary",
    "temporal_boundary",
    "visual_boundary",
    "audio_boundary",
]

COLORS = {
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "green": (0, 180, 0),
    "yellow": (0, 220, 220),
    "purple": (180, 0, 180),
    "orange": (0, 140, 255),
}

SHAPES = ["circle", "square", "triangle"]
DIRECTIONS = {
    "right": (1, 0),
    "left": (-1, 0),
    "down": (0, 1),
    "up": (0, -1),
}


# =========================
# Drawing utilities
# =========================

def draw_shape(frame, shape, color_name, x, y, size=28):
    color = COLORS[color_name]

    if shape == "circle":
        cv2.circle(frame, (int(x), int(y)), size, color, -1)

    elif shape == "square":
        cv2.rectangle(
            frame,
            (int(x - size), int(y - size)),
            (int(x + size), int(y + size)),
            color,
            -1,
        )

    elif shape == "triangle":
        pts = np.array([
            [int(x), int(y - size)],
            [int(x - size), int(y + size)],
            [int(x + size), int(y + size)],
        ])
        cv2.fillPoly(frame, [pts], color)

    else:
        raise ValueError(f"Unknown shape: {shape}")


def lerp(a, b, t):
    return a + (b - a) * t


def object_position(event, frame_idx):
    if frame_idx < event["start"]:
        return event["from"]
    if frame_idx > event["end"]:
        return event["to"]

    progress = (frame_idx - event["start"]) / max(1, event["end"] - event["start"])
    x = lerp(event["from"][0], event["to"][0], progress)
    y = lerp(event["from"][1], event["to"][1], progress)
    return x, y


def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def is_far_enough(point, existing_points, min_dist=85):
    return all(distance(point, other) >= min_dist for other in existing_points)


def sample_motion(existing_points, min_dist=85):
    margin = 70
    distance_px = random.randint(105, 165)

    for _ in range(250):
        direction = random.choice(list(DIRECTIONS.keys()))

        if direction == "right":
            start = (
                random.randint(margin, W - margin - distance_px),
                random.randint(margin, H - margin),
            )
            end = (start[0] + distance_px, start[1])
        elif direction == "left":
            start = (
                random.randint(margin + distance_px, W - margin),
                random.randint(margin, H - margin),
            )
            end = (start[0] - distance_px, start[1])
        elif direction == "down":
            start = (
                random.randint(margin, W - margin),
                random.randint(margin, H - margin - distance_px),
            )
            end = (start[0], start[1] + distance_px)
        else:
            start = (
                random.randint(margin, W - margin),
                random.randint(margin + distance_px, H - margin),
            )
            end = (start[0], start[1] - distance_px)

        if is_far_enough(start, existing_points, min_dist) and is_far_enough(end, existing_points, min_dist):
            existing_points.extend([start, end])
            return {
                "direction": direction,
                "from": start,
                "to": end,
            }

    raise RuntimeError("Could not sample a non-overlapping motion path.")


def sample_static_point(existing_points, min_dist=85):
    margin = 70

    for _ in range(250):
        point = (
            random.randint(margin, W - margin),
            random.randint(margin, H - margin),
        )

        if is_far_enough(point, existing_points, min_dist):
            existing_points.append(point)
            return point

    raise RuntimeError("Could not sample a non-overlapping static point.")


# =========================
# Audio utility
# =========================

def add_beep_to_video(video_path, output_path, beep_time, duration=0.35, freq=880, audio_fps=44100):
    video_path = Path(video_path)
    output_path = Path(output_path)

    audio_path = output_path.with_suffix(".wav")

    total_audio_samples = int(DURATION_SEC * audio_fps)
    audio = np.zeros(total_audio_samples, dtype=np.float32)

    start = int(beep_time * audio_fps)
    end = min(start + int(duration * audio_fps), total_audio_samples)

    t = np.arange(end - start) / audio_fps
    audio[start:end] = 0.35 * np.sin(2 * np.pi * freq * t)

    # float32 [-1,1] -> int16
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


# =========================
# Video generation
# =========================

def make_motion(path, start, end):
    return {
        "start": start,
        "end": end,
        "from": path["from"],
        "to": path["to"],
    }


def make_static_motion(point):
    return {
        "start": 0,
        "end": 0,
        "from": point,
        "to": point,
    }


def make_silent_video(video_path, condition, spec):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (W, H))

    boundary_start = spec["boundary_start_frame"]
    boundary_end = spec["boundary_end_frame"]

    for f in range(TOTAL_FRAMES):
        frame = np.ones((H, W, 3), dtype=np.uint8) * 245

        for distractor in spec["distractors"]:
            x, y = object_position(distractor["motion"], f)
            draw_shape(frame, distractor["shape"], distractor["color"], x, y, size=23)

        x1, y1 = object_position(spec["object_1_motion"], f)
        x2, y2 = object_position(spec["object_2_motion"], f)

        draw_shape(frame, spec["shape_1"], spec["color_1"], x1, y1)
        draw_shape(frame, spec["shape_2"], spec["color_2"], x2, y2)

        # Visual boundary marker: white flash + black central marker
        if condition == "visual_boundary" and boundary_start <= f < boundary_end:
            frame[:] = 255
            cv2.circle(frame, (W // 2, H // 2), 48, (0, 0, 0), -1)

        writer.write(frame)

    writer.release()


def get_timing(condition):
    """
    All videos are 20 seconds / 300 frames.

    The same base scene is rendered under four boundary manipulations:
      Low: first target event immediately followed by the second target event.
      Temporal: a 3-second pause separates the target events.
      Visual: a 1-second visual marker separates the target events.
      Audio: a 1-second silent gap with a beep at the boundary separates the events.

    A later unrelated distractor motion is included in every condition.
    """
    first_start = START_HOLD
    first_end = first_start + EVENT_DURATION

    if condition == "low_boundary":
        boundary_start = first_end
        boundary_end = first_end
        second_start = first_end
        gap_frames = 0
        visual_marker = "none"
        audio_marker = "none"

    elif condition == "temporal_boundary":
        boundary_start = first_end
        boundary_end = first_end + TEMPORAL_GAP_FRAMES
        second_start = boundary_end
        gap_frames = TEMPORAL_GAP_FRAMES
        visual_marker = "none"
        audio_marker = "none"

    elif condition == "visual_boundary":
        boundary_start = first_end
        boundary_end = first_end + VISUAL_MARKER_FRAMES
        second_start = boundary_end
        gap_frames = 0
        visual_marker = "white_flash_black_dot"
        audio_marker = "none"

    elif condition == "audio_boundary":
        boundary_start = first_end
        boundary_end = first_end + VISUAL_MARKER_FRAMES
        second_start = boundary_end
        gap_frames = 0
        visual_marker = "none"
        audio_marker = "beep"

    else:
        raise ValueError(f"Unknown condition: {condition}")

    second_end = second_start + EVENT_DURATION
    unrelated_end = UNRELATED_EVENT_START + UNRELATED_EVENT_DURATION

    if max(second_end, unrelated_end) >= TOTAL_FRAMES:
        raise ValueError("Timing exceeds total video length.")

    return {
        "first_start": first_start,
        "first_end": first_end,
        "second_start": second_start,
        "second_end": second_end,
        "boundary_start": boundary_start,
        "boundary_end": boundary_end,
        "gap_frames": gap_frames,
        "visual_marker": visual_marker,
        "audio_marker": audio_marker,
        "unrelated_start": UNRELATED_EVENT_START,
        "unrelated_end": unrelated_end,
    }


def object_text(obj):
    return f"the {obj['color']} {obj['shape']}"


def event_text(obj):
    return f"The {obj['color']} {obj['shape']} moved {obj['path']['direction']}."


def make_base_specs():
    base_specs = []

    first_object_ids = [1, 2] * (SAMPLES_PER_CONDITION // 2)
    correct_relations = ["before", "after"] * (SAMPLES_PER_CONDITION // 2)
    correct_options = ["A", "B"] * (SAMPLES_PER_CONDITION // 2)
    random.shuffle(first_object_ids)
    random.shuffle(correct_relations)
    random.shuffle(correct_options)

    if len(first_object_ids) < SAMPLES_PER_CONDITION:
        first_object_ids.append(random.choice([1, 2]))
        correct_relations.append(random.choice(["before", "after"]))
        correct_options.append(random.choice(["A", "B"]))

    for base_id in range(1, SAMPLES_PER_CONDITION + 1):
        existing_points = []
        colors = random.sample(list(COLORS.keys()), 2)
        shapes = random.choices(SHAPES, k=2)

        object_1 = {
            "id": 1,
            "shape": shapes[0],
            "color": colors[0],
            "path": sample_motion(existing_points),
        }
        object_2 = {
            "id": 2,
            "shape": shapes[1],
            "color": colors[1],
            "path": sample_motion(existing_points),
        }

        first_object_id = first_object_ids[base_id - 1]
        second_object_id = 2 if first_object_id == 1 else 1
        first_object = object_1 if first_object_id == 1 else object_2
        second_object = object_2 if first_object_id == 1 else object_1

        correct_relation = correct_relations[base_id - 1]
        incorrect_relation = "after" if correct_relation == "before" else "before"

        if correct_relation == "before":
            relation_subject = "first_event_object"
            subject_text = object_text(first_object)
            object_text_for_relation = object_text(second_object)
        else:
            relation_subject = "second_event_object"
            subject_text = object_text(second_object)
            object_text_for_relation = object_text(first_object)

        correct_sentence = f"{subject_text.capitalize()} moves {correct_relation} {object_text_for_relation}."
        incorrect_sentence = f"{subject_text.capitalize()} moves {incorrect_relation} {object_text_for_relation}."

        if correct_options[base_id - 1] == "A":
            option_a = correct_sentence
            option_b = incorrect_sentence
            correct_option = "A"
        else:
            option_a = incorrect_sentence
            option_b = correct_sentence
            correct_option = "B"

        distractors = []
        distractor_count = random.randint(*DISTRACTOR_COUNT_RANGE)
        distractor_colors = [color for color in COLORS if color not in colors]

        for distractor_idx in range(distractor_count):
            color = random.choice(distractor_colors)
            shape = random.choice(SHAPES)

            if distractor_idx == 0:
                motion_kind = "unrelated_motion"
                path = sample_motion(existing_points)
            else:
                motion_kind = "static"
                point = sample_static_point(existing_points)
                path = {
                    "direction": "none",
                    "from": point,
                    "to": point,
                }

            distractors.append({
                "id": distractor_idx + 1,
                "shape": shape,
                "color": color,
                "motion_kind": motion_kind,
                "path": path,
            })

        base_specs.append({
            "base_id": base_id,
            "object_1": object_1,
            "object_2": object_2,
            "first_object_id": first_object_id,
            "second_object_id": second_object_id,
            "first_event": event_text(first_object),
            "second_event": event_text(second_object),
            "relation_subject": relation_subject,
            "correct_relation": correct_relation,
            "incorrect_relation": incorrect_relation,
            "option_A": option_a,
            "option_B": option_b,
            "correct_option": correct_option,
            "distractors": distractors,
        })

    return base_specs


def timed_distractors(base_spec, timing):
    timed = []

    for distractor in base_spec["distractors"]:
        if distractor["motion_kind"] == "unrelated_motion":
            motion = make_motion(
                distractor["path"],
                timing["unrelated_start"],
                timing["unrelated_end"],
            )
        else:
            motion = make_static_motion(distractor["path"]["from"])

        timed.append({
            "id": distractor["id"],
            "shape": distractor["shape"],
            "color": distractor["color"],
            "motion_kind": distractor["motion_kind"],
            "motion": motion,
        })

    return timed


def make_sample(sample_id, condition, base_spec):
    timing = get_timing(condition)

    object_1 = base_spec["object_1"]
    object_2 = base_spec["object_2"]

    if base_spec["first_object_id"] == 1:
        object_1_start = timing["first_start"]
        object_1_end = timing["first_end"]
        object_2_start = timing["second_start"]
        object_2_end = timing["second_end"]
    else:
        object_1_start = timing["second_start"]
        object_1_end = timing["second_end"]
        object_2_start = timing["first_start"]
        object_2_end = timing["first_end"]

    object_1_motion = make_motion(object_1["path"], object_1_start, object_1_end)
    object_2_motion = make_motion(object_2["path"], object_2_start, object_2_end)

    base_name = f"sample_{sample_id:03d}_{condition}"
    final_video_path = VIDEO_DIR / f"{base_name}.mp4"

    spec = {
        "shape_1": object_1["shape"],
        "shape_2": object_2["shape"],
        "color_1": object_1["color"],
        "color_2": object_2["color"],
        "object_1_motion": object_1_motion,
        "object_2_motion": object_2_motion,
        "distractors": timed_distractors(base_spec, timing),
        "boundary_start_frame": timing["boundary_start"],
        "boundary_end_frame": timing["boundary_end"],
    }

    if condition == "audio_boundary":
        silent_path = VIDEO_DIR / f"{base_name}_silent.mp4"
        make_silent_video(silent_path, condition, spec)

        beep_time = timing["boundary_start"] / FPS
        add_beep_to_video(
            video_path=silent_path,
            output_path=final_video_path,
            beep_time=beep_time,
            duration=AUDIO_MARKER_DURATION,
        )

        silent_path.unlink(missing_ok=True)

    else:
        make_silent_video(final_video_path, condition, spec)

    annotation = {
        "video_id": final_video_path.name,
        "video_path": str(final_video_path),
        "condition": condition,
        "boundary_type": condition.replace("_boundary", ""),
        "base_sample_id": base_spec["base_id"],
        "fps": FPS,
        "duration_sec": DURATION_SEC,
        "total_frames": TOTAL_FRAMES,

        "first_event_start_frame": timing["first_start"],
        "first_event_end_frame": timing["first_end"],
        "second_event_start_frame": timing["second_start"],
        "second_event_end_frame": timing["second_end"],
        "unrelated_event_start_frame": timing["unrelated_start"],
        "unrelated_event_end_frame": timing["unrelated_end"],

        "boundary_start_frame": timing["boundary_start"],
        "boundary_end_frame": timing["boundary_end"],
        "gap_frames": timing["gap_frames"],
        "visual_marker": timing["visual_marker"],
        "audio_marker": timing["audio_marker"],

        "shape_1": object_1["shape"],
        "color_1": object_1["color"],
        "direction_1": object_1["path"]["direction"],
        "object_1": object_text(object_1),
        "object_1_start_frame": object_1_start,
        "object_1_end_frame": object_1_end,
        "object_1_from": object_1["path"]["from"],
        "object_1_to": object_1["path"]["to"],

        "shape_2": object_2["shape"],
        "color_2": object_2["color"],
        "direction_2": object_2["path"]["direction"],
        "object_2": object_text(object_2),
        "object_2_start_frame": object_2_start,
        "object_2_end_frame": object_2_end,
        "object_2_from": object_2["path"]["from"],
        "object_2_to": object_2["path"]["to"],

        "first_object_id": base_spec["first_object_id"],
        "second_object_id": base_spec["second_object_id"],
        "event_1": base_spec["first_event"],
        "event_2": base_spec["second_event"],
        "prompt_relation_type": "before_after_balanced",
        "relation_subject": base_spec["relation_subject"],
        "correct_relation": base_spec["correct_relation"],
        "incorrect_relation": base_spec["incorrect_relation"],
        "option_A": base_spec["option_A"],
        "option_B": base_spec["option_B"],
        "correct_option": base_spec["correct_option"],

        "distractor_count": len(base_spec["distractors"]),
        "distractors": [
            {
                "id": d["id"],
                "shape": d["shape"],
                "color": d["color"],
                "motion_kind": d["motion_kind"],
                "direction": d["path"]["direction"],
                "from": d["path"]["from"],
                "to": d["path"]["to"],
            }
            for d in base_spec["distractors"]
        ],
    }

    return annotation


def cleanup_generated_videos():
    for path in VIDEO_DIR.glob("sample_*.mp4"):
        path.unlink()
    for path in VIDEO_DIR.glob("sample_*_silent.mp4"):
        path.unlink()


def main():
    random.seed(42)

    cleanup_generated_videos()

    annotations = []
    sample_id = 1
    base_specs = make_base_specs()

    for condition in CONDITIONS:
        for base_spec in base_specs:
            ann = make_sample(sample_id, condition, base_spec)
            annotations.append(ann)
            sample_id += 1

    with open(ANNOTATION_PATH, "w", encoding="utf-8") as f:
        for item in annotations:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Generated {len(annotations)} videos.")
    print(f"Videos saved to: {VIDEO_DIR}")
    print(f"Annotations saved to: {ANNOTATION_PATH}")

    print("\nCondition counts:")
    for condition in CONDITIONS:
        print(f"- {condition}: {SAMPLES_PER_CONDITION}")


if __name__ == "__main__":
    main()
