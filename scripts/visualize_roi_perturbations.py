import argparse
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    from .common import PROJECT_ROOT, read_jsonl
except ImportError:
    from common import PROJECT_ROOT, read_jsonl


PERTURBATION_ORDER = (
    "original",
    "mask_target_1",
    "mask_target_2",
    "mask_targets",
    "mask_distractors",
    "remove_visual_marker",
)
PERTURBATION_LABELS = {
    "original": "Original",
    "mask_target_1": "Mask target 1",
    "mask_target_2": "Mask target 2",
    "mask_targets": "Mask both targets",
    "mask_distractors": "Mask distractors",
    "remove_visual_marker": "Remove marker",
}
CONDITION_PRIORITY = {
    "visual_boundary": 0,
    "temporal_boundary": 1,
    "audio_boundary": 2,
    "low_boundary": 3,
}


def resolve_path(value):
    path = Path(value)
    if path.exists():
        return path
    candidate = PROJECT_ROOT / path
    return candidate if candidate.exists() else path


def centered_text(image, text, center_x, y, scale=0.5, thickness=1):
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(
        image,
        text,
        (round(center_x - size[0] / 2), y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (42, 42, 42),
        thickness,
        cv2.LINE_AA,
    )


def read_frame(path, frame_idx):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def midpoint(start, end):
    if start is None or end is None:
        return None
    return round((int(start) + int(end)) / 2)


def representative_frames(row):
    events = row.get("event_timing") or {}
    boundary = row.get("boundary_timing") or {}
    frames = [
        (
            "Event 1",
            midpoint(events.get("first_event_start_frame"), events.get("first_event_end_frame")),
        ),
        (
            "Boundary",
            midpoint(boundary.get("boundary_start_frame"), boundary.get("boundary_end_frame")),
        ),
        (
            "Event 2",
            midpoint(events.get("second_event_start_frame"), events.get("second_event_end_frame")),
        ),
    ]
    return [(label, frame_idx) for label, frame_idx in frames if frame_idx is not None]


def select_group(rows, condition=None, source_video_id=None):
    candidates = [row for row in rows if row.get("prompt_variant") == "original"]
    if condition:
        candidates = [row for row in candidates if row.get("condition") == condition]
    if source_video_id:
        candidates = [
            row for row in candidates
            if (row.get("source_video_id") or row.get("video_id")) == source_video_id
        ]

    grouped = defaultdict(list)
    for row in candidates:
        grouped[row.get("source_video_id") or row.get("video_id")].append(row)
    if not grouped:
        raise ValueError("No complete ROI perturbation group matched the requested filters.")

    def group_rank(item):
        _, group = item
        perturbation_count = len({row.get("perturbation_type") for row in group})
        condition_rank = CONDITION_PRIORITY.get(group[0].get("condition"), 99)
        return (-perturbation_count, condition_rank, str(group[0].get("base_sample_id", "")))

    return sorted(grouped.items(), key=group_rank)[0]


def safe_title(value):
    return re.sub(r"[_-]+", " ", str(value)).strip().title()


def write_contact_sheet(rows, output_path, condition=None, source_video_id=None):
    selected_id, group = select_group(rows, condition, source_video_id)
    by_perturbation = {row.get("perturbation_type"): row for row in group}
    perturbations = [value for value in PERTURBATION_ORDER if value in by_perturbation]
    if not perturbations:
        raise ValueError("Selected rows do not contain perturbation_type values.")

    reference = by_perturbation.get("original", group[0])
    frame_specs = representative_frames(reference)
    cell_size = 230
    row_label_width = 112
    header_height = 82
    footer_height = 46
    width = row_label_width + cell_size * len(perturbations)
    height = header_height + cell_size * len(frame_specs) + footer_height
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    sample_label = reference.get("base_sample_id")
    if sample_label not in (None, ""):
        sample_label = f"sample {int(sample_label):03d}"
    else:
        sample_label = Path(str(selected_id)).stem
    title = (
        f"ROI perturbation QA | {safe_title(reference.get('condition', ''))} | "
        f"{sample_label}"
    )
    centered_text(image, title, width // 2, 31, scale=0.63, thickness=2)
    for column, perturbation in enumerate(perturbations):
        center_x = row_label_width + column * cell_size + cell_size // 2
        centered_text(
            image,
            PERTURBATION_LABELS.get(perturbation, safe_title(perturbation)),
            center_x,
            66,
            scale=0.44,
        )

    for row_idx, (frame_label, frame_idx) in enumerate(frame_specs):
        y1 = header_height + row_idx * cell_size
        centered_text(image, frame_label, row_label_width // 2, y1 + cell_size // 2 - 7, scale=0.5)
        centered_text(image, f"frame {frame_idx}", row_label_width // 2, y1 + cell_size // 2 + 18, scale=0.42)
        for column, perturbation in enumerate(perturbations):
            row = by_perturbation[perturbation]
            frame = read_frame(resolve_path(row["video_path"]), frame_idx)
            if frame is None:
                continue
            frame = cv2.resize(frame, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
            x1 = row_label_width + column * cell_size
            image[y1 : y1 + cell_size, x1 : x1 + cell_size] = frame
            cv2.rectangle(image, (x1, y1), (x1 + cell_size - 1, y1 + cell_size - 1), (215, 215, 215), 1)

    parameters = reference.get("perturbation_parameters") or {}
    footer = (
        f"mode={parameters.get('mask_mode', 'unknown')} | "
        f"scope={parameters.get('mask_scope', 'unknown')} | "
        f"padding={parameters.get('mask_padding', 'unknown')} px"
    )
    centered_text(image, footer, width // 2, height - 17, scale=0.46)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write preview image: {output_path}")
    return selected_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--source_video_id", default=None)
    args = parser.parse_args()

    selected_id = write_contact_sheet(
        read_jsonl(args.annotation_path),
        Path(args.output_path),
        condition=args.condition,
        source_video_id=args.source_video_id,
    )
    print(f"Wrote ROI perturbation preview for {selected_id} to {args.output_path}")


if __name__ == "__main__":
    main()
