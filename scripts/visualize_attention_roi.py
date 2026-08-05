import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


ROI_ORDER = (
    "target_1",
    "target_2",
    "distractors",
    "visual_marker",
    "boundary_flash",
    "background",
)
ROI_LABELS = {
    "target_1": "Target 1",
    "target_2": "Target 2",
    "distractors": "Distractors",
    "visual_marker": "Visual marker",
    "boundary_flash": "Boundary flash",
    "background": "Background",
}
PHASE_LABELS = {
    "pre_event": "Pre-event",
    "event_1": "Event 1",
    "boundary": "Boundary",
    "event_2": "Event 2",
    "post_event": "Post-event",
    "mixed": "Mixed phase",
}
PHASE_COLORS = {
    "pre_event": (244, 244, 244),
    "event_1": (232, 242, 252),
    "boundary": (239, 233, 249),
    "event_2": (232, 247, 238),
    "post_event": (244, 244, 244),
    "mixed": (235, 240, 240),
}


def safe_name(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "attention_probe"


def resolve_video_path(path, project_root=None):
    path = Path(path)
    if path.exists():
        return path
    if project_root is not None:
        candidate = Path(project_root) / path
        if candidate.exists():
            return candidate
    return path


def read_video_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def centered_text(image, text, center_x, y, scale=0.52, color=(35, 35, 35), thickness=1):
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(
        image,
        text,
        (round(center_x - size[0] / 2), y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def split_label(label, max_length=18):
    if len(label) <= max_length:
        return [label]
    words = label.split()
    if len(words) <= 1:
        return [label]
    midpoint = len(words) // 2
    return [" ".join(words[:midpoint]), " ".join(words[midpoint:])]


def attention_result_subtitle(result):
    sample_id = result.get("base_sample_id") or result.get("video_id") or "sample"
    if isinstance(sample_id, int) or str(sample_id).isdigit():
        sample_id = f"Sample {int(sample_id):03d}"
    condition = str(result.get("condition") or result.get("boundary_type") or "").replace("_", " ")
    variant = str(result.get("prompt_variant") or "").replace("_", " ")
    return " | ".join(part for part in (str(sample_id), condition.title(), variant.title()) if part)


def attention_semantic_label(result):
    semantics = str(
        result.get("attention_semantics")
        or (result.get("decision_query") or {}).get("attention_semantics")
        or ""
    )
    if "decision" in semantics or result.get("decision_query"):
        return "Decision-position"
    return "Generated-answer-token"


def choose_temporal_indices(result, max_frames):
    attention = np.asarray(result.get("temporal_attention") or [], dtype=np.float32)
    phases = result.get("temporal_phases") or []
    if attention.size == 0:
        return []

    selected = []
    for phase in PHASE_LABELS:
        candidates = [idx for idx, value in enumerate(phases) if value == phase]
        if candidates:
            selected.append(max(candidates, key=lambda idx: float(attention[idx])))
    for idx in np.argsort(attention)[::-1].tolist():
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= max_frames:
            break
    return sorted(selected[:max_frames])


def attention_overlay(frame, heatmap, scale_ceiling, alpha):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    normalized = np.clip(heatmap / max(scale_ceiling, 1e-12), 0.0, 1.0)
    normalized = cv2.resize(
        normalized,
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    colored = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
    return cv2.addWeighted(frame, 1.0 - alpha, colored, alpha, 0)


def write_attention_overlay(result, output_path, project_root=None, max_frames=6, alpha=0.45):
    maps = np.asarray(result.get("attention_map") or [], dtype=np.float32)
    source_frames = result.get("source_frame_indices") or []
    if maps.ndim != 3 or not source_frames:
        return False
    video_path = resolve_video_path(result.get("video_path", ""), project_root)
    indices = choose_temporal_indices(result, max_frames)
    if not indices:
        return False

    positive = maps[maps > 0]
    scale_ceiling = float(np.percentile(positive, 99)) if positive.size else 1.0
    cell_w, cell_h = 250, 250
    label_h, title_h = 62, 82
    width = cell_w * len(indices)
    image = np.full((title_h + cell_h + label_h, width, 3), 255, dtype=np.uint8)
    centered_text(image, f"{attention_semantic_label(result)} attention over sampled video frames", width // 2, 30, scale=0.65, thickness=2)
    centered_text(image, attention_result_subtitle(result), width // 2, 57, scale=0.44, color=(75, 75, 75))

    phases = result.get("temporal_phases") or []
    source_groups = result.get("source_frame_groups") or []
    temporal_attention = result.get("temporal_attention") or []
    for column, temporal_idx in enumerate(indices):
        frame_idx = int(source_frames[temporal_idx])
        frame = read_video_frame(video_path, frame_idx)
        if frame is None:
            continue
        frame = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        overlay = attention_overlay(frame, maps[temporal_idx], scale_ceiling, alpha)
        x1 = column * cell_w
        image[title_h : title_h + cell_h, x1 : x1 + cell_w] = overlay
        phase = phases[temporal_idx] if temporal_idx < len(phases) else ""
        mass = temporal_attention[temporal_idx] if temporal_idx < len(temporal_attention) else 0.0
        frame_group = (
            source_groups[temporal_idx]
            if temporal_idx < len(source_groups)
            else [frame_idx]
        )
        frame_label = "/".join(str(value) for value in frame_group)
        centered_text(
            image,
            f"Frames {frame_label}",
            x1 + cell_w // 2,
            title_h + cell_h + 23,
        )
        centered_text(
            image,
            f"{PHASE_LABELS.get(phase, phase)} | mass {mass:.3f}",
            x1 + cell_w // 2,
            title_h + cell_h + 48,
            scale=0.46,
            color=(70, 70, 70),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return True


def phase_runs(phases):
    if not phases:
        return []
    runs = []
    start = 0
    for idx in range(1, len(phases) + 1):
        if idx == len(phases) or phases[idx] != phases[start]:
            runs.append((start, idx, phases[start]))
            start = idx
    return runs


def write_temporal_attention(result, output_path):
    values = np.asarray(result.get("temporal_attention") or [], dtype=np.float32)
    phases = result.get("temporal_phases") or []
    source_frames = result.get("source_frame_indices") or []
    if values.size == 0:
        return False
    width, height = 1200, 570
    left, right, top, bottom = 95, 55, 102, 95
    plot_w, plot_h = width - left - right, height - top - bottom
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    centered_text(
        image,
        f"{attention_semantic_label(result)} temporal attention profile",
        width // 2,
        32,
        scale=0.72,
        thickness=2,
    )
    centered_text(image, attention_result_subtitle(result), width // 2, 59, scale=0.44, color=(75, 75, 75))

    for start, end, phase in phase_runs(phases):
        x1 = left + round(start / max(1, len(values)) * plot_w)
        x2 = left + round(end / max(1, len(values)) * plot_w)
        cv2.rectangle(image, (x1, top), (x2, top + plot_h), PHASE_COLORS.get(phase, (246, 246, 246)), -1)
        if x2 - x1 > 38:
            centered_text(
                image,
                PHASE_LABELS.get(phase, phase),
                (x1 + x2) // 2,
                top + 22,
                scale=0.38,
                color=(90, 90, 90),
            )

    maximum = max(0.01, float(values.max()) * 1.12)
    for fraction in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + plot_h - round(fraction * plot_h)
        cv2.line(image, (left, y), (left + plot_w, y), (220, 220, 220), 1)
        cv2.putText(
            image,
            f"{maximum * fraction:.3f}",
            (18, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (55, 55, 55),
            1,
            cv2.LINE_AA,
        )

    points = []
    for idx, value in enumerate(values):
        x = left + round(idx / max(1, len(values) - 1) * plot_w)
        y = top + plot_h - round(float(value) / maximum * plot_h)
        points.append((x, y))
    for first, second in zip(points, points[1:]):
        cv2.line(image, first, second, (180, 95, 24), 3, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, point, 4, (180, 95, 24), -1, cv2.LINE_AA)

    cv2.line(image, (left, top), (left, top + plot_h), (45, 45, 45), 2)
    cv2.line(image, (left, top + plot_h), (left + plot_w, top + plot_h), (45, 45, 45), 2)
    tick_indices = sorted(set(np.linspace(0, len(values) - 1, min(7, len(values)), dtype=int).tolist()))
    for idx in tick_indices:
        x = left + round(idx / max(1, len(values) - 1) * plot_w)
        label = str(source_frames[idx]) if idx < len(source_frames) else str(idx)
        centered_text(image, label, x, top + plot_h + 34, scale=0.46)
    centered_text(image, "Source frame", left + plot_w // 2, height - 22, scale=0.58, thickness=2)
    cv2.putText(image, "Visual", (14, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)
    cv2.putText(image, "attention", (14, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return True


def write_layer_roi_heatmap(result, output_path):
    profiles = result.get("layer_roi_profiles") or []
    if not profiles:
        return False
    present = {
        roi
        for profile in profiles
        for roi in (profile.get("spatial_roi") or {})
        if roi in ROI_LABELS
    }
    rois = [roi for roi in ROI_ORDER if roi in present]
    if not rois:
        return False
    matrix = np.asarray(
        [
            [
                float((profile.get("spatial_roi") or {}).get(roi, {}).get("enrichment") or 0.0)
                for roi in rois
            ]
            for profile in profiles
        ],
        dtype=np.float32,
    )
    clipped = np.clip(matrix, 0.0, 3.0) / 3.0
    heat = cv2.applyColorMap(np.uint8(clipped * 255), cv2.COLORMAP_VIRIDIS)

    cell_w = max(120, min(170, 760 // max(1, len(rois))))
    cell_h = max(14, min(25, 520 // max(1, len(profiles))))
    left, right, top, bottom = 115, 135, 108, 125
    width = left + cell_w * len(rois) + right
    height = top + cell_h * len(profiles) + bottom
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    centered_text(
        image,
        "Layer-wise ROI attention enrichment",
        width // 2,
        34,
        scale=0.66,
        thickness=2,
    )
    centered_text(
        image,
        attention_result_subtitle(result),
        width // 2,
        61,
        scale=0.44,
        color=(75, 75, 75),
    )

    heat = cv2.resize(heat, (cell_w * len(rois), cell_h * len(profiles)), interpolation=cv2.INTER_NEAREST)
    image[top : top + heat.shape[0], left : left + heat.shape[1]] = heat
    colorbar_x = left + heat.shape[1] + 28
    colorbar = np.linspace(255, 0, heat.shape[0], dtype=np.uint8).reshape(-1, 1)
    colorbar = cv2.applyColorMap(colorbar, cv2.COLORMAP_VIRIDIS)
    colorbar = cv2.resize(colorbar, (18, heat.shape[0]), interpolation=cv2.INTER_NEAREST)
    image[top : top + heat.shape[0], colorbar_x : colorbar_x + 18] = colorbar
    cv2.putText(image, "3x", (colorbar_x + 25, top + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (55, 55, 55), 1, cv2.LINE_AA)
    cv2.putText(image, "1x", (colorbar_x + 25, top + round(heat.shape[0] * 2 / 3) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (55, 55, 55), 1, cv2.LINE_AA)
    cv2.putText(image, "0x", (colorbar_x + 25, top + heat.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (55, 55, 55), 1, cv2.LINE_AA)
    for idx, profile in enumerate(profiles):
        if idx % max(1, len(profiles) // 8) == 0 or idx == len(profiles) - 1:
            y = top + idx * cell_h + cell_h // 2 + 5
            cv2.putText(
                image,
                str(profile.get("layer", idx)),
                (65, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (50, 50, 50),
                1,
                cv2.LINE_AA,
            )
    cv2.putText(image, "Layer", (18, top - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 50, 50), 1, cv2.LINE_AA)
    for idx, roi in enumerate(rois):
        center = left + idx * cell_w + cell_w // 2
        for line_idx, line in enumerate(split_label(ROI_LABELS[roi])):
            centered_text(image, line, center, top + heat.shape[0] + 32 + line_idx * 22, scale=0.45)
    centered_text(image, "Enrichment: observed attention / token-area share (clipped at 3x)", width // 2, height - 20, scale=0.48, color=(75, 75, 75))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return True


def write_padding_sensitivity(result, output_path):
    output_path = Path(output_path)
    profiles = result.get("roi_padding_sensitivity") or {}
    if len(profiles) < 2:
        return False
    paddings = sorted(int(value) for value in profiles)
    present = {
        roi
        for profile in profiles.values()
        for roi in profile
        if roi in ROI_LABELS and roi != "background"
    }
    rois = [roi for roi in ROI_ORDER if roi in present]
    if not rois:
        return False

    width, height = 1100, 650
    left, right, top, bottom = 90, 245, 105, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    centered_text(image, "ROI-padding sensitivity of attention enrichment", width // 2, 32, scale=0.68, thickness=2)
    centered_text(image, attention_result_subtitle(result), width // 2, 59, scale=0.44, color=(75, 75, 75))

    values = [
        float((profiles[str(padding)].get(roi) or {}).get("enrichment") or 0.0)
        for padding in paddings
        for roi in rois
    ]
    maximum = max(1.0, min(4.0, max(values, default=1.0) * 1.12))
    for fraction in np.linspace(0.0, 1.0, 5):
        y = top + plot_h - round(float(fraction) * plot_h)
        cv2.line(image, (left, y), (left + plot_w, y), (225, 225, 225), 1)
        cv2.putText(image, f"{maximum * fraction:.1f}x", (22, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (55, 55, 55), 1, cv2.LINE_AA)
    cv2.line(image, (left, top), (left, top + plot_h), (45, 45, 45), 2)
    cv2.line(image, (left, top + plot_h), (left + plot_w, top + plot_h), (45, 45, 45), 2)

    colors = [(44, 160, 44), (214, 120, 28), (50, 60, 210), (155, 80, 170), (50, 150, 170)]
    for roi_index, roi in enumerate(rois):
        points = []
        color = colors[roi_index % len(colors)]
        for padding_index, padding in enumerate(paddings):
            value = float((profiles[str(padding)].get(roi) or {}).get("enrichment") or 0.0)
            x = left + round(padding_index / max(1, len(paddings) - 1) * plot_w)
            y = top + plot_h - round(min(value, maximum) / maximum * plot_h)
            points.append((x, y))
            cv2.circle(image, (x, y), 5, color, -1, cv2.LINE_AA)
        for first, second in zip(points, points[1:]):
            cv2.line(image, first, second, color, 3, cv2.LINE_AA)
        legend_y = top + roi_index * 34
        cv2.line(image, (left + plot_w + 28, legend_y), (left + plot_w + 55, legend_y), color, 3)
        cv2.putText(image, ROI_LABELS[roi], (left + plot_w + 65, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (45, 45, 45), 1, cv2.LINE_AA)

    for padding_index, padding in enumerate(paddings):
        x = left + round(padding_index / max(1, len(paddings) - 1) * plot_w)
        centered_text(image, str(padding), x, top + plot_h + 34, scale=0.48)
    centered_text(image, "ROI padding (source pixels)", left + plot_w // 2, height - 25, scale=0.56, thickness=2)
    cv2.putText(image, "Enrichment", (16, top - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (45, 45, 45), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return True


def write_visualizations(results, output_dir, project_root=None, max_frames=6, alpha=0.45):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for result in results:
        stem = safe_name(result.get("eval_id") or result.get("video_id"))
        paths = {
            "attention_overlay": output_dir / f"{stem}_attention_overlay.png",
            "temporal_attention": output_dir / f"{stem}_temporal_attention.png",
            "layer_roi_enrichment": output_dir / f"{stem}_layer_roi_enrichment.png",
            "roi_padding_sensitivity": output_dir / f"{stem}_roi_padding_sensitivity.png",
        }
        if write_attention_overlay(result, paths["attention_overlay"], project_root, max_frames, alpha):
            written.append(str(paths["attention_overlay"]))
        if write_temporal_attention(result, paths["temporal_attention"]):
            written.append(str(paths["temporal_attention"]))
        if write_layer_roi_heatmap(result, paths["layer_roi_enrichment"]):
            written.append(str(paths["layer_roi_enrichment"]))
        if write_padding_sensitivity(result, paths["roi_padding_sensitivity"]):
            written.append(str(paths["roi_padding_sensitivity"]))
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--project_root", default=None)
    parser.add_argument("--max_frames", type=int, default=6)
    parser.add_argument("--heatmap_alpha", type=float, default=0.45)
    args = parser.parse_args()
    if not 0.0 <= args.heatmap_alpha <= 1.0:
        parser.error("--heatmap_alpha must be between 0 and 1.")
    results = json.loads(Path(args.input_path).read_text(encoding="utf-8"))
    written = write_visualizations(
        results,
        args.output_dir,
        project_root=args.project_root,
        max_frames=args.max_frames,
        alpha=args.heatmap_alpha,
    )
    print(f"Wrote {len(written)} attention visualization files to {args.output_dir}")


if __name__ == "__main__":
    main()
