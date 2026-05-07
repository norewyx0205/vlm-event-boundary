import cv2
import json
import random
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
DURATION_SEC = 10
TOTAL_FRAMES = FPS * DURATION_SEC  # 150 frames

EVENT_DURATION = 30  # 2 seconds
START_HOLD = 15      # 1 second

TEMPORAL_GAP_FRAMES = 45  # 3 seconds
VISUAL_MARKER_FRAMES = 15 # 1 second
AUDIO_MARKER_DURATION = 0.35

SAMPLES_PER_CONDITION = 5

CONDITIONS = [
    "low_boundary",
    "temporal_boundary",
    "visual_boundary",
    "audio_boundary", # later
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


# =========================
# Audio utility
# =========================

import wave
import subprocess
import imageio_ffmpeg


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

def make_silent_video(video_path, condition, spec):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (W, H))

    e1 = spec["event_1_motion"]
    e2 = spec["event_2_motion"]

    boundary_start = spec["boundary_start_frame"]
    boundary_end = spec["boundary_end_frame"]

    for f in range(TOTAL_FRAMES):
        frame = np.ones((H, W, 3), dtype=np.uint8) * 245

        x1, y1 = object_position(e1, f)
        x2, y2 = object_position(e2, f)

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
    All videos are 10 seconds / 150 frames.

    Low:
      E1 -> E2, then long end hold.

    Temporal:
      E1 -> pause -> E2.

    Visual:
      E1 -> visual marker -> E2.

    Audio:
      E1 -> audio beep -> E2.

    The total length is fixed across conditions.
    """
    e1_start = START_HOLD
    e1_end = e1_start + EVENT_DURATION

    if condition == "low_boundary":
        boundary_start = e1_end
        boundary_end = e1_end
        e2_start = e1_end
        gap_frames = 0
        visual_marker = "none"
        audio_marker = "none"

    elif condition == "temporal_boundary":
        boundary_start = e1_end
        boundary_end = e1_end + TEMPORAL_GAP_FRAMES
        e2_start = boundary_end
        gap_frames = TEMPORAL_GAP_FRAMES
        visual_marker = "none"
        audio_marker = "none"

    elif condition == "visual_boundary":
        boundary_start = e1_end
        boundary_end = e1_end + VISUAL_MARKER_FRAMES
        e2_start = boundary_end
        gap_frames = 0
        visual_marker = "white_flash_black_dot"
        audio_marker = "none"

    elif condition == "audio_boundary":
        boundary_start = e1_end
        boundary_end = e1_end + VISUAL_MARKER_FRAMES
        e2_start = boundary_end
        gap_frames = 0
        visual_marker = "none"
        audio_marker = "beep"

    else:
        raise ValueError(f"Unknown condition: {condition}")

    e2_end = e2_start + EVENT_DURATION

    if e2_end >= TOTAL_FRAMES:
        raise ValueError("Timing exceeds total video length.")

    return {
        "e1_start": e1_start,
        "e1_end": e1_end,
        "e2_start": e2_start,
        "e2_end": e2_end,
        "boundary_start": boundary_start,
        "boundary_end": boundary_end,
        "gap_frames": gap_frames,
        "visual_marker": visual_marker,
        "audio_marker": audio_marker,
    }


def make_base_specs():
    base_specs = []

    for base_id in range(1, SAMPLES_PER_CONDITION + 1):
        colors = random.sample(list(COLORS.keys()), 2)
        shapes = random.sample(SHAPES, 2)

        color_1, color_2 = colors
        shape_1, shape_2 = shapes

        object_1_text = f"the {color_1} {shape_1}"
        object_2_text = f"the {color_2} {shape_2}"
        event_1_text = f"The {color_1} {shape_1} moved right."
        event_2_text = f"The {color_2} {shape_2} moved left."

        correct_sentence = f"The {color_1} {shape_1} moves before the {color_2} {shape_2}."
        reversed_sentence = f"The {color_1} {shape_1} moves after the {color_2} {shape_2}."

        if random.random() < 0.5:
            option_a = correct_sentence
            option_b = reversed_sentence
            correct_option = "A"
        else:
            option_a = reversed_sentence
            option_b = correct_sentence
            correct_option = "B"

        base_specs.append({
            "base_id": base_id,
            "shape_1": shape_1,
            "shape_2": shape_2,
            "color_1": color_1,
            "color_2": color_2,
            "object_1": object_1_text,
            "object_2": object_2_text,
            "event_1": event_1_text,
            "event_2": event_2_text,
            "option_A": option_a,
            "option_B": option_b,
            "correct_option": correct_option,
        })

    return base_specs


def make_sample(sample_id, condition, base_spec):
    timing = get_timing(condition)

    # Event 1: object 1 moves right.
    event_1_motion = {
        "start": timing["e1_start"],
        "end": timing["e1_end"],
        "from": (120, 180),
        "to": (260, 180),
    }

    # Event 2: object 2 moves left.
    event_2_motion = {
        "start": timing["e2_start"],
        "end": timing["e2_end"],
        "from": (390, 330),
        "to": (250, 330),
    }

    base_name = f"sample_{sample_id:03d}_{condition}"
    final_video_path = VIDEO_DIR / f"{base_name}.mp4"

    spec = {
        "shape_1": base_spec["shape_1"],
        "shape_2": base_spec["shape_2"],
        "color_1": base_spec["color_1"],
        "color_2": base_spec["color_2"],
        "event_1_motion": event_1_motion,
        "event_2_motion": event_2_motion,
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

        "event_1_start_frame": timing["e1_start"],
        "event_1_end_frame": timing["e1_end"],
        "event_2_start_frame": timing["e2_start"],
        "event_2_end_frame": timing["e2_end"],

        "boundary_start_frame": timing["boundary_start"],
        "boundary_end_frame": timing["boundary_end"],
        "gap_frames": timing["gap_frames"],
        "visual_marker": timing["visual_marker"],
        "audio_marker": timing["audio_marker"],

        "shape_1": base_spec["shape_1"],
        "color_1": base_spec["color_1"],
        "shape_2": base_spec["shape_2"],
        "color_2": base_spec["color_2"],
        "object_1": base_spec["object_1"],
        "object_2": base_spec["object_2"],

        "event_1": base_spec["event_1"],
        "event_2": base_spec["event_2"],
        "prompt_relation_type": "before_after",
        "option_A": base_spec["option_A"],
        "option_B": base_spec["option_B"],
        "correct_option": base_spec["correct_option"],
    }

    return annotation


def main():
    random.seed(42)

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
