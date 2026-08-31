import argparse
import csv
import json
import math
import random
import zipfile
from collections import defaultdict
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
LAYER_STAGES = {
    "early": range(0, 12),
    "middle": range(12, 24),
    "late": range(24, 36),
}
EPSILON = 1e-12
CONTRAST_METRICS = (
    "mean_delta_log_visual_mass_t1_over_t2",
    "mean_log_area_ratio_t1_over_t2",
    "mean_delta_log_enrichment_t1_over_t2",
    "mean_decomposition_residual",
)
FEATURE_ORDER = ("full", "color_only", "shape_only", "size_only")
CONDITION_ORDER = (
    "low_boundary",
    "temporal_boundary",
    "visual_boundary",
    "audio_boundary",
)
STAGE_ORDER = ("early", "middle", "late")


def read_attention_results(path):
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".zip":
        rows = []
        sources = []
        with zipfile.ZipFile(path) as archive:
            for member in sorted(archive.namelist()):
                if not member.endswith(".json") or member.endswith("_config.json"):
                    continue
                try:
                    payload = json.loads(archive.read(member).decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
                    continue
                if not isinstance(payload, list) or not payload:
                    continue
                if not all(
                    isinstance(row, dict) and row.get("layer_roi_profiles")
                    for row in payload
                ):
                    continue
                source = f"{path}::{member}"
                rows.extend((source, row) for row in payload)
                sources.append(source)
        return rows, sources
    files = [path] if path.is_file() else sorted(path.rglob("*.json"))
    rows = []
    sources = []
    for input_file in files:
        try:
            payload = json.loads(input_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, list) or not payload:
            continue
        if not all(isinstance(row, dict) and row.get("layer_roi_profiles") for row in payload):
            continue
        rows.extend((input_file, row) for row in payload)
        sources.append(input_file)
    return rows, sources


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def value(metrics, explicit_name, legacy_name, default=0.0):
    item = metrics.get(explicit_name)
    if item is None:
        item = metrics.get(legacy_name, default)
    return float(item) if item is not None else default


def object_label(target):
    label = str(
        target.get("reference_label")
        or target.get("label")
        or " ".join(
            part
            for part in (target.get("color"), target.get("shape"))
            if part
        )
        or f"target {target.get('id')}"
    ).strip()
    if label.lower().startswith("the "):
        label = label[4:]
    return label


def prompt_subject_id(result, targets):
    sentences = [
        result.get("option_A"),
        result.get("option_B"),
        result.get("correct_sentence"),
    ]
    for sentence in sentences:
        sentence = str(sentence or "").strip().lower()
        for target in targets:
            labels = {
                object_label(target).lower(),
                str(target.get("label") or "").strip().lower(),
            }
            for label in labels:
                if label and (sentence.startswith(label) or sentence.startswith(f"the {label}")):
                    return int(target.get("id"))
    return None


def target_roles(result):
    targets = sorted(result.get("target_objects") or [], key=lambda item: int(item["id"]))
    subject_id = prompt_subject_id(result, targets)
    starts = [int(target.get("start_frame") or 0) for target in targets]
    first_start = min(starts) if starts else None
    last_start = max(starts) if starts else None
    roles = {}
    for target in targets:
        target_id = int(target["id"])
        start = int(target.get("start_frame") or 0)
        if first_start == last_start:
            mover_role = "co-mover"
        else:
            mover_role = "first mover" if start == first_start else "second mover"
        subject_role = (
            "subject"
            if subject_id == target_id
            else "non-subject"
            if subject_id is not None
            else "subject unknown"
        )
        semantic_label = object_label(target)
        roles[f"target_{target_id}"] = {
            "target_id": target_id,
            "semantic_label": semantic_label,
            "mover_role": mover_role,
            "subject_role": subject_role,
            "direction": target.get("direction"),
            "radius": target.get("radius"),
            "size_label": target.get("reference_label") or target.get("size_label"),
            "display_label": (
                f"T{target_id} | {semantic_label} | {mover_role} | {subject_role}"
            ),
        }
    return roles


def layer_stage(layer):
    for name, indices in LAYER_STAGES.items():
        if layer in indices:
            return name
    return "outside_fixed_qwen3_stages"


def base_metadata(source_path, result):
    return {
        "source_attention_file": str(source_path),
        "eval_id": result.get("eval_id"),
        "video_id": result.get("video_id"),
        "base_sample_id": result.get("base_sample_id"),
        "dataset_version": result.get("dataset_version"),
        "feature_variant": result.get("feature_variant"),
        "size_scene_variant": result.get("size_scene_variant"),
        "condition": result.get("condition"),
        "prompt_variant": result.get("prompt_variant"),
        "correct_option": result.get("correct_option"),
        "prediction": result.get("prediction"),
        "is_correct": result.get("is_correct"),
        "attention_pairing_id": result.get("attention_pairing_id"),
        "attention_case_bundle_id": result.get("attention_case_bundle_id"),
        "attention_case_bundle_size": result.get("attention_case_bundle_size"),
        "attention_case_bundle_pair_count": result.get(
            "attention_case_bundle_pair_count"
        ),
        "attention_case_bundle_row_count": result.get(
            "attention_case_bundle_row_count"
        ),
        "attention_archived_pair_outcome": result.get("attention_archived_pair_outcome"),
        "attention_selection_first_mover": result.get("attention_selection_first_mover"),
        "attention_semantics": result.get("attention_semantics"),
        "head_reduction": result.get("head_reduction"),
        "roi_assignment_method": result.get("roi_assignment_method"),
        "roi_padding": result.get("roi_padding"),
    }


def flatten_roi_metrics(source_path, result):
    output = []
    roles = target_roles(result)
    metadata = base_metadata(source_path, result)
    for profile in result.get("layer_roi_profiles") or []:
        layer = int(profile.get("layer", 0))
        visual_fraction = float(profile.get("visual_attention_fraction") or 0.0)
        for roi in ROI_ORDER:
            metrics = (profile.get("spatial_roi") or {}).get(roi)
            if not metrics:
                continue
            role = roles.get(roi, {})
            visual_mass = value(
                metrics,
                "visual_normalized_attention_mass",
                "normalized_visual_attention",
            )
            explicit_all_token_share = metrics.get("all_token_attention_share")
            all_token_share = (
                float(explicit_all_token_share)
                if explicit_all_token_share is not None
                else visual_fraction * visual_mass
            )
            output.append({
                **metadata,
                "layer": layer,
                "layer_stage": layer_stage(layer),
                "roi": roi,
                "roi_display_label": role.get("display_label", roi.replace("_", " ").title()),
                "semantic_label": role.get("semantic_label"),
                "mover_role": role.get("mover_role"),
                "subject_role": role.get("subject_role"),
                "direction": role.get("direction"),
                "radius": role.get("radius"),
                "size_label": role.get("size_label"),
                "visual_attention_fraction": visual_fraction,
                "all_token_attention_share": all_token_share,
                "all_token_attention_share_source": (
                    "explicit"
                    if explicit_all_token_share is not None
                    else "reconstructed_visual_fraction_x_visual_mass"
                ),
                "visual_normalized_attention_mass": visual_mass,
                "effective_token_area_share": value(
                    metrics, "effective_token_area_share", "token_fraction"
                ),
                "effective_token_count": value(
                    metrics, "effective_token_count", "token_count"
                ),
                "mean_attention_per_effective_token": value(
                    metrics,
                    "mean_attention_per_effective_token",
                    "mean_attention_per_token",
                ),
                "area_normalized_enrichment": value(
                    metrics, "area_normalized_enrichment", "enrichment"
                ),
                "metric_schema": (
                    "explicit_mass_area_v2"
                    if "all_token_attention_share" in metrics
                    else "legacy_v1_aliases"
                ),
            })
    return output


def log_ratio(first, second):
    return math.log((float(first) + EPSILON) / (float(second) + EPSILON))


def target_contrasts(source_path, result):
    roles = target_roles(result)
    metadata = base_metadata(source_path, result)
    output = []
    for profile in result.get("layer_roi_profiles") or []:
        spatial = profile.get("spatial_roi") or {}
        target_1 = spatial.get("target_1")
        target_2 = spatial.get("target_2")
        if not target_1 or not target_2:
            continue
        layer = int(profile.get("layer", 0))
        mass_1 = value(
            target_1,
            "visual_normalized_attention_mass",
            "normalized_visual_attention",
        )
        mass_2 = value(
            target_2,
            "visual_normalized_attention_mass",
            "normalized_visual_attention",
        )
        area_1 = value(target_1, "effective_token_area_share", "token_fraction")
        area_2 = value(target_2, "effective_token_area_share", "token_fraction")
        enrichment_1 = value(
            target_1, "area_normalized_enrichment", "enrichment"
        )
        enrichment_2 = value(
            target_2, "area_normalized_enrichment", "enrichment"
        )
        delta_mass = log_ratio(mass_1, mass_2)
        log_area_ratio = log_ratio(area_1, area_2)
        delta_enrichment = log_ratio(enrichment_1, enrichment_2)
        output.append({
            **metadata,
            "layer": layer,
            "layer_stage": layer_stage(layer),
            "target_1_role": (roles.get("target_1") or {}).get("display_label"),
            "target_2_role": (roles.get("target_2") or {}).get("display_label"),
            "target_1_mover_role": (roles.get("target_1") or {}).get("mover_role"),
            "target_2_mover_role": (roles.get("target_2") or {}).get("mover_role"),
            "target_1_subject_role": (roles.get("target_1") or {}).get("subject_role"),
            "target_2_subject_role": (roles.get("target_2") or {}).get("subject_role"),
            "target_1_visual_mass": mass_1,
            "target_2_visual_mass": mass_2,
            "target_1_area_share": area_1,
            "target_2_area_share": area_2,
            "target_1_enrichment": enrichment_1,
            "target_2_enrichment": enrichment_2,
            "delta_log_visual_mass_t1_over_t2": delta_mass,
            "log_area_ratio_t1_over_t2": log_area_ratio,
            "delta_log_enrichment_t1_over_t2": delta_enrichment,
            "decomposition_residual": (
                delta_enrichment - (delta_mass - log_area_ratio)
            ),
            "target_1_has_more_total_visual_mass": mass_1 > mass_2,
            "target_1_has_higher_enrichment": enrichment_1 > enrichment_2,
        })
    return output


def stage_contrasts(layer_rows):
    grouped = defaultdict(list)
    for row in layer_rows:
        key = (
            row["source_attention_file"],
            row["eval_id"],
            row["layer_stage"],
        )
        grouped[key].append(row)
    output = []
    metrics = (
        "delta_log_visual_mass_t1_over_t2",
        "log_area_ratio_t1_over_t2",
        "delta_log_enrichment_t1_over_t2",
        "decomposition_residual",
    )
    for (_source, _eval_id, stage), rows in sorted(grouped.items()):
        if stage == "outside_fixed_qwen3_stages":
            continue
        summary = {
            key: value
            for key, value in rows[0].items()
            if key not in metrics and key not in {"layer", "layer_stage"}
        }
        summary["layer_stage"] = stage
        summary["layer_count"] = len(rows)
        for metric in metrics:
            summary[f"mean_{metric}"] = sum(row[metric] for row in rows) / len(rows)
        output.append(summary)
    return output


def paired_stage_contrasts(stage_rows):
    grouped = defaultdict(list)
    for row in stage_rows:
        key = (
            str(row.get("source_attention_file")),
            str(row.get("feature_variant")),
            str(row.get("size_scene_variant")),
            str(row.get("base_sample_id")),
            str(row.get("condition")),
            str(row.get("layer_stage")),
        )
        grouped[key].append(row)

    output = []
    excluded = set(CONTRAST_METRICS) | {
        "eval_id",
        "video_id",
        "prompt_variant",
        "correct_option",
        "prediction",
        "is_correct",
        "target_1_subject_role",
        "target_2_subject_role",
    }
    for _key, rows in sorted(grouped.items()):
        summary = {
            key: value
            for key, value in rows[0].items()
            if key not in excluded
        }
        variants = sorted({str(row.get("prompt_variant")) for row in rows})
        summary.update({
            "eval_ids": " | ".join(sorted(str(row.get("eval_id")) for row in rows)),
            "prompt_variants": " | ".join(variants),
            "prompt_rows": len(rows),
            "mirrored_pair_complete": variants == ["original", "swapped"],
        })
        for metric in CONTRAST_METRICS:
            values = [float(row[metric]) for row in rows]
            summary[metric] = sum(values) / len(values)
        output.append(summary)
    return output


def stable_seed(seed, values):
    text = "|".join(str(value) for value in values)
    return seed + sum((index + 1) * ord(character) for index, character in enumerate(text))


def bootstrap_interval(values, seed=42, iterations=2000):
    values = [float(value) for value in values]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    generator = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [values[generator.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def summarize_contrasts(rows, dimensions, seed=42):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key)) for key in dimensions)].append(row)
    output = []
    for dimension_values, group_rows in sorted(grouped.items()):
        summary = dict(zip(dimensions, dimension_values))
        base_ids = sorted({str(row.get("base_sample_id")) for row in group_rows})
        summary.update({
            "n_base_samples": len(base_ids),
            "n_rows": len(group_rows),
            "base_sample_ids": " | ".join(base_ids),
            "bootstrap_unit": "base_sample_id",
            "bootstrap_iterations": 2000,
        })
        for metric in CONTRAST_METRICS:
            by_base = defaultdict(list)
            for row in group_rows:
                by_base[str(row.get("base_sample_id"))].append(float(row[metric]))
            base_values = [sum(values) / len(values) for values in by_base.values()]
            mean_value = sum(base_values) / len(base_values)
            low, high = bootstrap_interval(
                base_values,
                seed=stable_seed(seed, (*dimension_values, metric)),
            )
            summary[f"{metric}_mean"] = mean_value
            summary[f"{metric}_ci_low"] = low
            summary[f"{metric}_ci_high"] = high
        output.append(summary)
    return output


def feature_calibration_tables(stage_rows, paired_rows):
    feature_rows = [row for row in paired_rows if row.get("feature_variant")]
    prompt_rows = [row for row in stage_rows if row.get("feature_variant")]
    return {
        "feature_stage_summary": summarize_contrasts(
            feature_rows,
            ("feature_variant", "layer_stage"),
        ),
        "feature_boundary_stage_summary": summarize_contrasts(
            feature_rows,
            ("feature_variant", "condition", "layer_stage"),
        ),
        "feature_mover_stage_summary": summarize_contrasts(
            feature_rows,
            ("feature_variant", "target_1_mover_role", "layer_stage"),
        ),
        "feature_prompt_stage_summary": summarize_contrasts(
            prompt_rows,
            ("feature_variant", "prompt_variant", "layer_stage"),
        ),
        "feature_pair_outcome_stage_summary": summarize_contrasts(
            feature_rows,
            ("feature_variant", "attention_archived_pair_outcome", "layer_stage"),
        ),
    }


def display_name(value):
    labels = {
        "full": "Full",
        "color_only": "Color only",
        "shape_only": "Shape only",
        "size_only": "Size only",
        "low_boundary": "Low",
        "temporal_boundary": "Temporal",
        "visual_boundary": "Visual",
        "audio_boundary": "Audio",
        "first mover": "T1 first",
        "second mover": "T2 first",
    }
    return labels.get(str(value), str(value).replace("_", " ").title())


def centered_text(image, text, center_x, y, scale=0.55, color=(40, 40, 40), thickness=1):
    size, _ = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(
        image,
        str(text),
        (round(center_x - size[0] / 2), y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def ordered_values(values, preferred):
    values = {str(value) for value in values if value not in {None, "None", ""}}
    output = [value for value in preferred if value in values]
    output.extend(sorted(values - set(output)))
    return output


def lookup_summary(rows, filters, metric):
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in filters.items()):
            return (
                float(row[f"{metric}_mean"]),
                float(row[f"{metric}_ci_low"]),
                float(row[f"{metric}_ci_high"]),
            )
    return None


def contrast_limit(rows, metrics):
    values = []
    for row in rows:
        for metric in metrics:
            for suffix in ("_mean", "_ci_low", "_ci_high"):
                value = row.get(f"{metric}{suffix}")
                if value is not None:
                    values.append(abs(float(value)))
    return max(0.25, max(values, default=0.0) * 1.15)


def write_feature_stage_plot(rows, output_path):
    features = ordered_values(
        (row.get("feature_variant") for row in rows), FEATURE_ORDER
    )
    stages = ordered_values((row.get("layer_stage") for row in rows), STAGE_ORDER)
    if len(features) < 2 or not stages:
        return False
    metrics = (
        "mean_delta_log_visual_mass_t1_over_t2",
        "mean_delta_log_enrichment_t1_over_t2",
    )
    metric_titles = (
        "Total visual-attention mass contrast",
        "Area-normalised enrichment contrast",
    )
    colors = {
        "early": (187, 116, 35),
        "middle": (53, 144, 72),
        "late": (57, 77, 201),
    }
    width, height = 1740, 790
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    centered_text(
        image,
        "Matched feature calibration: Target 1 versus Target 2",
        width // 2,
        38,
        scale=0.82,
        thickness=2,
    )
    centered_text(
        image,
        "Mirrored prompts are averaged within each video; intervals bootstrap base samples",
        width // 2,
        67,
        scale=0.48,
        color=(80, 80, 80),
    )
    limit = contrast_limit(rows, metrics)
    panel_width = 735
    panel_lefts = (105, 930)
    top, plot_height = 130, 465
    for panel_index, (metric, title) in enumerate(zip(metrics, metric_titles)):
        left = panel_lefts[panel_index]
        plot_width = panel_width
        centered_text(image, title, left + plot_width // 2, 106, scale=0.60, thickness=2)
        for tick in np.linspace(-limit, limit, 5):
            y = top + round((limit - tick) / (2 * limit) * plot_height)
            cv2.line(image, (left, y), (left + plot_width, y), (225, 225, 225), 1)
            cv2.putText(
                image,
                f"{tick:+.2f}",
                (left - 68, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (70, 70, 70),
                1,
                cv2.LINE_AA,
            )
        zero_y = top + round(plot_height / 2)
        cv2.line(image, (left, zero_y), (left + plot_width, zero_y), (75, 75, 75), 2)
        cv2.line(image, (left, top), (left, top + plot_height), (45, 45, 45), 2)
        x_positions = np.linspace(left + 75, left + plot_width - 75, len(features))
        for x, feature in zip(x_positions, features):
            centered_text(image, display_name(feature), int(x), top + plot_height + 34, scale=0.46)
        for stage_index, stage in enumerate(stages):
            points = []
            for x, feature in zip(x_positions, features):
                value = lookup_summary(
                    rows,
                    {"feature_variant": feature, "layer_stage": stage},
                    metric,
                )
                if value is None:
                    continue
                mean_value, low, high = value
                offset_x = int(x + (stage_index - (len(stages) - 1) / 2) * 12)
                mean_y = top + round((limit - mean_value) / (2 * limit) * plot_height)
                low_y = top + round((limit - low) / (2 * limit) * plot_height)
                high_y = top + round((limit - high) / (2 * limit) * plot_height)
                color = colors.get(stage, (80, 80, 80))
                cv2.line(image, (offset_x, high_y), (offset_x, low_y), color, 2)
                cv2.line(image, (offset_x - 4, high_y), (offset_x + 4, high_y), color, 2)
                cv2.line(image, (offset_x - 4, low_y), (offset_x + 4, low_y), color, 2)
                cv2.circle(image, (offset_x, mean_y), 6, color, -1, cv2.LINE_AA)
                points.append((offset_x, mean_y))
            for first, second in zip(points, points[1:]):
                cv2.line(image, first, second, colors.get(stage, (80, 80, 80)), 2, cv2.LINE_AA)
    legend_y = 650
    for index, stage in enumerate(stages):
        x = 600 + index * 190
        cv2.line(image, (x, legend_y), (x + 35, legend_y), colors.get(stage, (80, 80, 80)), 3)
        cv2.circle(image, (x + 17, legend_y), 5, colors.get(stage, (80, 80, 80)), -1)
        cv2.putText(
            image,
            display_name(stage),
            (x + 47, legend_y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
    centered_text(
        image,
        "Log ratio: positive = T1 > T2; negative = T2 > T1. T1/T2 denote matched trajectories, not a shared semantic feature.",
        width // 2,
        735,
        scale=0.46,
        color=(70, 70, 70),
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return True


def diverging_color(value, limit):
    fraction = min(1.0, abs(float(value)) / max(limit, EPSILON))
    neutral = np.array([245, 245, 242], dtype=np.float32)
    endpoint = (
        np.array([184, 98, 35], dtype=np.float32)
        if value < 0
        else np.array([70, 160, 58], dtype=np.float32)
    )
    return tuple(int(channel) for channel in neutral * (1 - fraction) + endpoint * fraction)


def write_feature_contrast_heatmaps(rows, grouping_key, preferred_columns, title, output_path):
    features = ordered_values(
        (row.get("feature_variant") for row in rows), FEATURE_ORDER
    )
    columns = ordered_values((row.get(grouping_key) for row in rows), preferred_columns)
    stages = ordered_values((row.get("layer_stage") for row in rows), STAGE_ORDER)
    if len(features) < 2 or not columns or not stages:
        return False
    metrics = (
        "mean_delta_log_visual_mass_t1_over_t2",
        "mean_delta_log_enrichment_t1_over_t2",
    )
    metric_labels = ("Visual-attention mass", "Area-normalised enrichment")
    limit = contrast_limit(rows, metrics)
    width, height = 1810, 1110
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    centered_text(image, title, width // 2, 38, scale=0.82, thickness=2)
    centered_text(
        image,
        "Cell values are mirrored-pair means; positive values favour T1 and negative values favour T2",
        width // 2,
        68,
        scale=0.48,
        color=(80, 80, 80),
    )
    panel_width, panel_height = 515, 360
    left_start, top_start = 105, 125
    horizontal_gap, vertical_gap = 60, 115
    for metric_index, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        for stage_index, stage in enumerate(stages):
            panel_left = left_start + stage_index * (panel_width + horizontal_gap)
            panel_top = top_start + metric_index * (panel_height + vertical_gap)
            centered_text(
                image,
                f"{metric_label} | {display_name(stage)} layers",
                panel_left + panel_width // 2,
                panel_top - 18,
                scale=0.50,
                thickness=2,
            )
            label_width = 112
            grid_left = panel_left + label_width
            grid_width = panel_width - label_width
            cell_width = grid_width / len(columns)
            cell_height = panel_height / len(features)
            for feature_index, feature in enumerate(features):
                y1 = round(panel_top + feature_index * cell_height)
                y2 = round(panel_top + (feature_index + 1) * cell_height)
                cv2.putText(
                    image,
                    display_name(feature),
                    (panel_left, round((y1 + y2) / 2) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (50, 50, 50),
                    1,
                    cv2.LINE_AA,
                )
                for column_index, column in enumerate(columns):
                    x1 = round(grid_left + column_index * cell_width)
                    x2 = round(grid_left + (column_index + 1) * cell_width)
                    item = lookup_summary(
                        rows,
                        {
                            "feature_variant": feature,
                            grouping_key: column,
                            "layer_stage": stage,
                        },
                        metric,
                    )
                    value = item[0] if item is not None else 0.0
                    cv2.rectangle(image, (x1, y1), (x2, y2), diverging_color(value, limit), -1)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    centered_text(
                        image,
                        f"{value:+.2f}" if item is not None else "NA",
                        (x1 + x2) // 2,
                        round((y1 + y2) / 2) + 6,
                        scale=0.43,
                        color=(25, 25, 25),
                        thickness=1,
                    )
            for column_index, column in enumerate(columns):
                x1 = round(grid_left + column_index * cell_width)
                x2 = round(grid_left + (column_index + 1) * cell_width)
                centered_text(
                    image,
                    display_name(column),
                    (x1 + x2) // 2,
                    panel_top + panel_height + 27,
                    scale=0.40,
                )
    legend_y = height - 72
    legend_left, legend_width = 590, 630
    for index in range(101):
        value = -limit + (2 * limit * index / 100)
        x1 = legend_left + round(index / 101 * legend_width)
        x2 = legend_left + round((index + 1) / 101 * legend_width)
        cv2.rectangle(image, (x1, legend_y), (x2, legend_y + 18), diverging_color(value, limit), -1)
    cv2.rectangle(image, (legend_left, legend_y), (legend_left + legend_width, legend_y + 18), (80, 80, 80), 1)
    centered_text(image, f"{-limit:.2f} (T2)", legend_left, legend_y + 44, scale=0.42)
    centered_text(image, "0", legend_left + legend_width // 2, legend_y + 44, scale=0.42)
    centered_text(image, f"+{limit:.2f} (T1)", legend_left + legend_width, legend_y + 44, scale=0.42)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return True


def archive_audit(source_rows):
    grouped = defaultdict(list)
    for source_path, result in source_rows:
        grouped[str(source_path)].append(result)
    output = []
    for source_path, rows in sorted(grouped.items()):
        values = lambda key: sorted({str(row.get(key)) for row in rows})
        semantics = values("attention_semantics")
        assignments = values("roi_assignment_method")
        paddings = values("roi_padding")
        frame_groups_present = all(bool(row.get("source_frame_groups")) for row in rows)
        metadata_complete = (
            semantics != ["None"]
            and assignments != ["None"]
            and paddings != ["None"]
            and frame_groups_present
        )
        output.append({
            "source_attention_file": source_path,
            "rows": len(rows),
            "attention_semantics": " | ".join(semantics),
            "roi_assignment_method": " | ".join(assignments),
            "roi_padding": " | ".join(paddings),
            "head_reduction": " | ".join(values("head_reduction")),
            "has_source_frame_groups": frame_groups_present,
            "has_fractional_metric_aliases": all(
                "all_token_attention_share"
                in (((row.get("layer_roi_profiles") or [{}])[0].get("spatial_roi") or {}).get("background") or {})
                for row in rows
            ),
            "layer_counts": " | ".join(
                sorted({str(len(row.get("layer_roi_profiles") or [])) for row in rows})
            ),
            "measurement_metadata_complete": metadata_complete,
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, nargs="+")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    source_rows = []
    sources = []
    for input_path in args.input_path:
        rows, input_sources = read_attention_results(input_path)
        source_rows.extend(rows)
        sources.extend(input_sources)
    if not source_rows:
        raise ValueError("No attention JSON arrays with layer_roi_profiles were found.")

    roi_rows = []
    contrast_rows = []
    for source_path, result in source_rows:
        roi_rows.extend(flatten_roi_metrics(source_path, result))
        contrast_rows.extend(target_contrasts(source_path, result))
    stage_rows = stage_contrasts(contrast_rows)
    paired_stage_rows = paired_stage_contrasts(stage_rows)
    calibration_tables = feature_calibration_tables(stage_rows, paired_stage_rows)
    audit_rows = archive_audit(source_rows)
    audit_signatures = {
        (
            row["attention_semantics"],
            row["roi_assignment_method"],
            row["roi_padding"],
            row["head_reduction"],
            row["has_source_frame_groups"],
        )
        for row in audit_rows
    }
    cross_archive_comparison_ready = (
        all(row["measurement_metadata_complete"] for row in audit_rows)
        and len(audit_signatures) == 1
    )

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "layer_roi_metrics.csv", roi_rows)
    write_csv(output_dir / "layer_target_contrasts.csv", contrast_rows)
    write_csv(output_dir / "stage_target_contrasts.csv", stage_rows)
    write_csv(output_dir / "paired_stage_target_contrasts.csv", paired_stage_rows)
    for table_name, table_rows in calibration_tables.items():
        write_csv(output_dir / f"{table_name}.csv", table_rows)
    write_csv(output_dir / "attention_archive_audit.csv", audit_rows)
    plot_paths = []
    feature_stage_rows = calibration_tables["feature_stage_summary"]
    feature_boundary_rows = calibration_tables["feature_boundary_stage_summary"]
    feature_mover_rows = calibration_tables["feature_mover_stage_summary"]
    feature_variants = ordered_values(
        (row.get("feature_variant") for row in paired_stage_rows), FEATURE_ORDER
    )
    if write_feature_stage_plot(
        feature_stage_rows,
        output_dir / "feature_stage_mass_vs_enrichment.png",
    ):
        plot_paths.append("feature_stage_mass_vs_enrichment.png")
    if write_feature_contrast_heatmaps(
        feature_boundary_rows,
        "condition",
        CONDITION_ORDER,
        "Matched feature calibration by boundary condition",
        output_dir / "feature_boundary_mass_vs_enrichment.png",
    ):
        plot_paths.append("feature_boundary_mass_vs_enrichment.png")
    if write_feature_contrast_heatmaps(
        feature_mover_rows,
        "target_1_mover_role",
        ("first mover", "second mover"),
        "Matched feature calibration by first-moving target",
        output_dir / "feature_first_mover_mass_vs_enrichment.png",
    ):
        plot_paths.append("feature_first_mover_mass_vs_enrichment.png")
    summary = {
        "source_files": sorted({str(path) for path in sources}),
        "attention_rows": len(source_rows),
        "layer_roi_rows": len(roi_rows),
        "layer_target_contrast_rows": len(contrast_rows),
        "stage_target_contrast_rows": len(stage_rows),
        "paired_stage_target_contrast_rows": len(paired_stage_rows),
        "complete_mirrored_stage_pairs": sum(
            bool(row.get("mirrored_pair_complete")) for row in paired_stage_rows
        ),
        "feature_variants": feature_variants,
        "feature_calibration_ready": len(feature_variants) >= 2,
        "feature_calibration_table_rows": {
            name: len(rows) for name, rows in calibration_tables.items()
        },
        "feature_calibration_plots": plot_paths,
        "aggregation_unit": (
            "Original/swapped prompts are averaged within each video pair; "
            "group summaries and bootstrap intervals use base_sample_id clusters."
        ),
        "layer_stage_definition": {
            name: [min(indices), max(indices)]
            for name, indices in LAYER_STAGES.items()
        },
        "contrast_definition": {
            "delta_mass": "log(T1 visual-normalised mass / T2 visual-normalised mass)",
            "delta_enrichment": "log(T1 area-normalised enrichment / T2 area-normalised enrichment)",
            "identity": "delta_enrichment = delta_mass - log(T1 area share / T2 area share)",
        },
        "cross_archive_comparison_ready": cross_archive_comparison_ready,
        "cross_archive_warning": (
            None
            if cross_archive_comparison_ready
            else "Attention semantics or ROI measurement metadata differ or are missing; do not interpret cross-archive heatmap differences as model effects."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(
        f"Analyzed {len(source_rows)} attention rows from {len(set(sources))} files; "
        f"wrote attention tables and {len(plot_paths)} calibration plots to {output_dir}"
    )


if __name__ == "__main__":
    main()
