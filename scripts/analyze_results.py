import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict, Counter

try:
    from .common import PROJECT_ROOT, read_jsonl
except ImportError:
    from common import PROJECT_ROOT, read_jsonl


FEATURE_VARIANTS = ("full", "shape_only", "color_only", "size_only")
FEATURE_LABELS = {
    "full": "Full",
    "shape_only": "Shape only",
    "color_only": "Color only",
    "size_only": "Size only",
}
SIZE_SCENE_VARIANTS = (
    "large_few",
    "large_many",
    "small_few",
    "small_many",
    "clear_large_few",
    "clear_large_many",
    "clear_small_few",
    "clear_small_many",
)
SIZE_SCENE_LABELS = {
    "large_few": "Large / few",
    "large_many": "Large / many",
    "small_few": "Small / few",
    "small_many": "Small / many",
    "clear_large_few": "Clear large / few",
    "clear_large_many": "Clear large / many",
    "clear_small_few": "Clear small / few",
    "clear_small_many": "Clear small / many",
}


def ordered_size_scene_variants(rows, key="size_scene_variant"):
    present = {row.get(key) for row in rows}
    ordered = [variant for variant in SIZE_SCENE_VARIANTS if variant in present]
    extras = sorted(variant for variant in present if variant and variant not in ordered)
    return ordered + extras


def axis_label_lines(label):
    label = str(label)
    if " / " in label:
        return label.split(" / ", 1)
    if len(label) <= 14 or " " not in label:
        return [label]
    words = label.split()
    midpoint = len(words) // 2
    return [" ".join(words[:midpoint]), " ".join(words[midpoint:])]


def put_centered_lines(image, lines, center_x, y, font, scale, color, thickness, line_gap=22):
    import cv2

    for line_idx, line in enumerate(lines):
        size, _ = cv2.getTextSize(line, font, scale, thickness)
        cv2.putText(
            image,
            line,
            (round(center_x - size[0] / 2), y + line_idx * line_gap),
            font,
            scale,
            color,
            thickness,
        )


def load_result_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(path.rglob("raw_results.jsonl"))


def dataset_name_from_path(path):
    parts = Path(path).parts
    if len(parts) >= 4 and parts[-1] == "raw_results.jsonl":
        return parts[-3]
    if len(parts) >= 3:
        return parts[-2]
    return ""


def latest_result_files(paths):
    latest = {}
    for path in paths:
        dataset_name = dataset_name_from_path(path)
        key = (str(Path(path).parents[2]), dataset_name)
        if key not in latest or path.parent.name > latest[key].parent.name:
            latest[key] = path
    return sorted(latest.values())


def load_rows(paths, dataset_name_prefix=None):
    rows = []
    for path in paths:
        dataset_name = dataset_name_from_path(path)
        for row in read_jsonl(path):
            row["_source_file"] = str(path)
            row.setdefault("dataset_name", dataset_name)
            if dataset_name_prefix and not row.get("dataset_name", "").startswith(dataset_name_prefix):
                continue
            rows.append(row)
    return rows


def grouped_accuracy(rows, keys):
    stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in rows:
        key = tuple(row.get(k, "") for k in keys)
        stats[key]["total"] += 1
        stats[key]["correct"] += int(row.get("is_correct", False))
    return [
        {
            **{key_name: key_value for key_name, key_value in zip(keys, key)},
            "total": item["total"],
            "correct": item["correct"],
            "accuracy": item["correct"] / item["total"] if item["total"] else None,
        }
        for key, item in sorted(stats.items(), key=lambda kv: kv[0])
    ]


def prediction_distribution(rows):
    counts = Counter(row.get("prediction", "UNKNOWN") for row in rows)
    total = sum(counts.values())
    return [
        {"prediction": pred, "count": count, "proportion": count / total if total else None}
        for pred, count in sorted(counts.items())
    ]


def infer_base_sample_id(row):
    value = row.get("base_sample_id")
    if value not in (None, ""):
        return str(value)
    match = re.search(r"sample_(\d+)_", row.get("video_id", ""))
    if match:
        return str(int(match.group(1)))
    return ""


def mean(values):
    return sum(values) / len(values) if values else None


def swap_consistency(rows):
    by_video = defaultdict(dict)
    for row in rows:
        variant = row.get("prompt_variant")
        if variant in ("original", "swapped"):
            key = (
                row.get("_source_file", ""),
                row.get("dataset_version", ""),
                row.get("difficulty_level", ""),
                row.get("condition", ""),
                row.get("video_id", ""),
            )
            by_video[key][variant] = row

    counts = Counter()
    detail_rows = []
    for key, pair in by_video.items():
        if "original" not in pair or "swapped" not in pair:
            continue
        original_correct = bool(pair["original"].get("is_correct"))
        swapped_correct = bool(pair["swapped"].get("is_correct"))

        if original_correct and swapped_correct:
            category = "both_correct"
        elif not original_correct and not swapped_correct:
            category = "both_wrong"
        elif original_correct and not swapped_correct:
            category = "original_correct_swapped_wrong"
        else:
            category = "original_wrong_swapped_correct"

        counts[category] += 1
        source_file, dataset_version, difficulty_level, condition, video_id = key
        detail_rows.append({
            "source_file": source_file,
            "dataset_version": dataset_version,
            "difficulty_level": difficulty_level,
            "difficulty_name": pair["original"].get("difficulty_name", ""),
            "feature_variant": pair["original"].get("feature_variant", ""),
            "size_scene_variant": pair["original"].get("size_scene_variant", ""),
            "target_size_condition": pair["original"].get("target_size_condition", ""),
            "distractor_count_condition": pair["original"].get("distractor_count_condition", ""),
            "condition": condition,
            "video_id": video_id,
            "category": category,
            "original_prediction": pair["original"].get("prediction"),
            "swapped_prediction": pair["swapped"].get("prediction"),
        })

    summary = [
        {"category": category, "count": count}
        for category, count in sorted(counts.items())
    ]
    return summary, detail_rows


def grouped_counts(rows, keys, count_key):
    stats = defaultdict(Counter)
    for row in rows:
        key = tuple(row.get(k, "") for k in keys)
        stats[key][row.get(count_key, "")] += 1
    out = []
    for key, counts in sorted(stats.items(), key=lambda kv: kv[0]):
        total = sum(counts.values())
        base = {key_name: key_value for key_name, key_value in zip(keys, key)}
        for category, count in sorted(counts.items()):
            out.append({
                **base,
                count_key: category,
                "count": count,
                "proportion": count / total if total else None,
            })
    return out


def strict_pair_metrics(prompt_rows, swap_details, keys):
    prompt_stats = {
        tuple(row.get(key, "") for key in keys): row
        for row in grouped_accuracy(prompt_rows, keys)
    }
    pair_stats = defaultdict(Counter)
    for row in swap_details:
        key = tuple(row.get(key_name, "") for key_name in keys)
        category = row["category"]
        if category in {"original_correct_swapped_wrong", "original_wrong_swapped_correct"}:
            category = "exactly_one_correct"
        pair_stats[key][category] += 1

    output = []
    for key, counts in sorted(pair_stats.items()):
        total_pairs = sum(counts.values())
        both_correct = counts.get("both_correct", 0)
        exactly_one = counts.get("exactly_one_correct", 0)
        both_wrong = counts.get("both_wrong", 0)
        prompt_accuracy = prompt_stats.get(key, {}).get("accuracy")
        strict_accuracy = both_correct / total_pairs if total_pairs else None
        accuracy_strict_gap = (
            prompt_accuracy - strict_accuracy
            if prompt_accuracy is not None and strict_accuracy is not None
            else None
        )
        output.append({
            **{key_name: key_value for key_name, key_value in zip(keys, key)},
            "video_pairs": total_pairs,
            "prompt_accuracy": prompt_accuracy,
            "strict_both_correct": strict_accuracy,
            "accuracy_strict_gap_d": accuracy_strict_gap,
            "position_sensitive_rate": exactly_one / total_pairs if total_pairs else None,
            "both_wrong_rate": both_wrong / total_pairs if total_pairs else None,
            "both_correct_count": both_correct,
            "exactly_one_correct_count": exactly_one,
            "both_wrong_count": both_wrong,
        })
    return output


def paired_boundary_comparison(rows, baseline_condition="low_boundary"):
    by_unit = defaultdict(lambda: defaultdict(list))
    for row in rows:
        condition = row.get("condition", "")
        if not condition:
            continue
        key = (
            row.get("_source_file", ""),
            row.get("dataset_name", ""),
            row.get("dataset_version", ""),
            str(row.get("difficulty_level", "")),
            row.get("difficulty_name", ""),
            row.get("feature_variant", ""),
            row.get("size_scene_variant", ""),
            infer_base_sample_id(row),
        )
        by_unit[key][condition].append(int(bool(row.get("is_correct", False))))

    detail_rows = []
    for key, condition_values in by_unit.items():
        if baseline_condition not in condition_values:
            continue
        low_acc = mean(condition_values[baseline_condition])
        (
            source_file,
            dataset_name,
            dataset_version,
            difficulty_level,
            difficulty_name,
            feature_variant,
            size_scene_variant,
            base_sample_id,
        ) = key

        for condition, values in sorted(condition_values.items()):
            if condition == baseline_condition:
                continue
            condition_acc = mean(values)
            diff = condition_acc - low_acc
            if diff > 0:
                direction = "improved"
            elif diff < 0:
                direction = "worse"
            else:
                direction = "same"

            detail_rows.append({
                "source_file": source_file,
                "dataset_name": dataset_name,
                "dataset_version": dataset_version,
                "difficulty_level": difficulty_level,
                "difficulty_name": difficulty_name,
                "feature_variant": feature_variant,
                "size_scene_variant": size_scene_variant,
                "base_sample_id": base_sample_id,
                "comparison": f"{condition}_minus_{baseline_condition}",
                "condition": condition,
                "baseline_condition": baseline_condition,
                "condition_accuracy": condition_acc,
                "baseline_accuracy": low_acc,
                "difference": diff,
                "direction": direction,
            })

    grouped = defaultdict(list)
    for row in detail_rows:
        grouped[
            (
                row["difficulty_level"],
                row["difficulty_name"],
                row["feature_variant"],
                row["size_scene_variant"],
                row["comparison"],
            )
        ].append(row)

    summary_rows = []
    for key, items in sorted(grouped.items(), key=lambda kv: kv[0]):
        difficulty_level, difficulty_name, feature_variant, size_scene_variant, comparison = key
        differences = [item["difference"] for item in items]
        directions = Counter(item["direction"] for item in items)
        summary_rows.append({
            "difficulty_level": difficulty_level,
            "difficulty_name": difficulty_name,
            "feature_variant": feature_variant,
            "size_scene_variant": size_scene_variant,
            "comparison": comparison,
            "pairs": len(items),
            "mean_difference": mean(differences),
            "improved": directions.get("improved", 0),
            "same": directions.get("same", 0),
            "worse": directions.get("worse", 0),
        })

    return summary_rows, detail_rows


def result_model_group(row):
    source = Path(row.get("_source_file", ""))
    return source.parents[2].name if len(source.parents) >= 3 else ""


def paired_feature_comparison(rows):
    by_unit = defaultdict(lambda: defaultdict(list))
    for row in rows:
        feature_variant = row.get("feature_variant", "")
        if feature_variant not in FEATURE_VARIANTS or row.get("size_scene_variant"):
            continue
        key = (
            result_model_group(row),
            row.get("dataset_version", ""),
            infer_base_sample_id(row),
            row.get("condition", ""),
        )
        by_unit[key][feature_variant].append(int(bool(row.get("is_correct", False))))

    comparisons = [
        ("shape_only", "full"),
        ("color_only", "shape_only"),
        ("color_only", "full"),
        ("size_only", "full"),
        ("size_only", "shape_only"),
        ("size_only", "color_only"),
    ]
    detail_rows = []
    for key, feature_values in by_unit.items():
        model_group, dataset_version, base_sample_id, condition = key
        for feature_variant, baseline_variant in comparisons:
            if feature_variant not in feature_values or baseline_variant not in feature_values:
                continue
            feature_accuracy = mean(feature_values[feature_variant])
            baseline_accuracy = mean(feature_values[baseline_variant])
            difference = feature_accuracy - baseline_accuracy
            detail_rows.append({
                "model_group": model_group,
                "dataset_version": dataset_version,
                "base_sample_id": base_sample_id,
                "condition": condition,
                "comparison": f"{feature_variant}_minus_{baseline_variant}",
                "feature_variant": feature_variant,
                "baseline_variant": baseline_variant,
                "feature_accuracy": feature_accuracy,
                "baseline_accuracy": baseline_accuracy,
                "difference": difference,
                "direction": "improved" if difference > 0 else "worse" if difference < 0 else "same",
            })

    summary_rows = []
    for condition_scope in ("by_condition", "overall"):
        grouped = defaultdict(list)
        for row in detail_rows:
            key = (
                row["comparison"],
                row["condition"] if condition_scope == "by_condition" else "all",
            )
            grouped[key].append(row)

        for (comparison, condition), items in sorted(grouped.items()):
            directions = Counter(item["direction"] for item in items)
            summary_rows.append({
                "scope": condition_scope,
                "condition": condition,
                "comparison": comparison,
                "pairs": len(items),
                "mean_difference": mean([item["difference"] for item in items]),
                "improved": directions.get("improved", 0),
                "same": directions.get("same", 0),
                "worse": directions.get("worse", 0),
            })

    return summary_rows, detail_rows


def size_factorial_effects(size_scene_rows, strict_by_scene):
    prompt_cells = {
        (row["target_size_condition"], row["distractor_count_condition"]): row["accuracy"]
        for row in grouped_accuracy(
            size_scene_rows,
            ["target_size_condition", "distractor_count_condition"],
        )
    }
    strict_cells = {
        (row["target_size_condition"], row["distractor_count_condition"]): row["strict_both_correct"]
        for row in strict_by_scene
    }

    output = []
    for measure, cells in [
        ("prompt_accuracy", prompt_cells),
        ("strict_both_correct", strict_cells),
    ]:
        required = {
            ("large", "few"),
            ("large", "many"),
            ("small", "few"),
            ("small", "many"),
        }
        if not required.issubset(cells):
            continue
        large = mean([cells[("large", "few")], cells[("large", "many")]])
        small = mean([cells[("small", "few")], cells[("small", "many")]])
        few = mean([cells[("large", "few")], cells[("small", "few")]])
        many = mean([cells[("large", "many")], cells[("small", "many")]])
        interaction = (
            cells[("small", "many")]
            - cells[("small", "few")]
            - cells[("large", "many")]
            + cells[("large", "few")]
        )
        output.extend([
            {
                "measure": measure,
                "effect": "large_minus_small",
                "estimate": large - small,
                "interpretation": "positive means larger targets are easier",
            },
            {
                "measure": measure,
                "effect": "few_minus_many",
                "estimate": few - many,
                "interpretation": "positive means fewer distractors are easier",
            },
            {
                "measure": measure,
                "effect": "small_x_many_interaction",
                "estimate": interaction,
                "interpretation": "negative means the small+many combination has an extra cost",
            },
        ])
    return output


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.unlink(missing_ok=True)
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(
    rows,
    output_path,
    title="Accuracy by difficulty level and boundary condition",
    y_label="Accuracy",
):
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("opencv-python and numpy are required for plots; skipping plot.")
        return

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)

    level_values = sorted({
        int(row["difficulty_level"])
        for row in rows
        if str(row.get("difficulty_level", "")).isdigit()
    })
    if len(level_values) == 1:
        condition_order = [
            condition
            for condition in (
                "low_boundary",
                "temporal_boundary",
                "visual_boundary",
                "audio_boundary",
            )
            if condition in grouped
        ]
        condition_labels = {
            "low_boundary": "Low",
            "temporal_boundary": "Temporal",
            "visual_boundary": "Visual",
            "audio_boundary": "Audio",
        }
        metric_name = "Strict both-correct" if y_label == "Strict pair" else "Prompt accuracy"
        values = {
            (condition_labels.get(condition, condition), metric_name): float(items[0]["accuracy"])
            for condition, items in grouped.items()
            if items
        }
        make_grouped_bar_plot(
            [condition_labels.get(condition, condition) for condition in condition_order],
            [metric_name],
            values,
            f"{title} (Level {level_values[0]})",
            output_path,
            y_label=y_label,
        )
        return

    width, height = 1120, 620
    margin_left, margin_right = 95, 290
    margin_top, margin_bottom = 85, 85
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    axis_color = (40, 40, 40)
    cv2.line(image, (margin_left, margin_top), (margin_left, margin_top + plot_h), axis_color, 2)
    cv2.line(image, (margin_left, margin_top + plot_h), (margin_left + plot_w, margin_top + plot_h), axis_color, 2)

    if not level_values:
        level_values = [1]
    min_level = min(level_values)
    max_level = max(level_values)
    level_span = max(1, max_level - min_level)

    for level in level_values:
        x = margin_left + round((level - min_level) / level_span * plot_w)
        cv2.line(image, (x, margin_top + plot_h), (x, margin_top + plot_h + 6), axis_color, 1)
        cv2.putText(image, str(level), (x - 6, margin_top + plot_h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, axis_color, 2)

    for acc in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = margin_top + plot_h - round(acc * plot_h)
        cv2.line(image, (margin_left - 6, y), (margin_left, y), axis_color, 1)
        cv2.putText(image, f"{acc:.2f}", (20, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 1)

    colors = {
        "low_boundary": (31, 119, 180),
        "temporal_boundary": (44, 160, 44),
        "visual_boundary": (214, 39, 40),
        "audio_boundary": (148, 103, 189),
    }

    legend_x = margin_left + plot_w + 35
    legend_y = margin_top + 12
    cv2.putText(image, "Boundary", (legend_x, margin_top - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_color, 2)
    for idx, (condition, items) in enumerate(sorted(grouped.items())):
        items = sorted(items, key=lambda x: int(x["difficulty_level"]))
        points = []
        color = colors.get(condition, (80, 80, 80))

        for item in items:
            level = int(item["difficulty_level"])
            acc = float(item["accuracy"])
            x = margin_left + round((level - min_level) / level_span * plot_w)
            y = margin_top + plot_h - round(acc * plot_h)
            points.append((x, y))

        for p1, p2 in zip(points, points[1:]):
            cv2.line(image, p1, p2, color, 3)
        for point in points:
            cv2.circle(image, point, 6, color, -1)

        lx = legend_x
        ly = legend_y + idx * 30
        cv2.line(image, (lx, ly), (lx + 30, ly), color, 3)
        cv2.putText(image, condition, (lx + 38, ly + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 1)

    cv2.putText(image, title, (margin_left + 30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, axis_color, 2)
    cv2.putText(image, "Difficulty level", (margin_left + plot_w // 2 - 80, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, axis_color, 2)
    cv2.putText(image, y_label, (12, margin_top - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_color, 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def make_feature_plot(
    rows,
    output_path,
    title="Level 5 feature ablation by boundary condition",
    variants=None,
    labels=None,
    x_axis_label="Feature encoding",
    y_label="Prompt accuracy",
):
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("opencv-python and numpy are required for plots; skipping feature plot.")
        return

    variants = list(variants or FEATURE_VARIANTS)
    labels = labels or FEATURE_LABELS
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["condition"]][row["feature_variant"]] = float(row["accuracy"])

    width, height = max(1120, 145 * len(variants) + 390), 660
    margin_left, margin_right = 95, 290
    margin_top, margin_bottom = 85, 135
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    axis_color = (40, 40, 40)

    cv2.line(image, (margin_left, margin_top), (margin_left, margin_top + plot_h), axis_color, 2)
    cv2.line(image, (margin_left, margin_top + plot_h), (margin_left + plot_w, margin_top + plot_h), axis_color, 2)

    x_positions = {}
    for idx, variant in enumerate(variants):
        x = margin_left + round(idx / max(1, len(variants) - 1) * plot_w)
        x_positions[variant] = x
        cv2.line(image, (x, margin_top + plot_h), (x, margin_top + plot_h + 6), axis_color, 1)
        put_centered_lines(
            image,
            axis_label_lines(labels[variant]),
            x,
            margin_top + plot_h + 32,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            axis_color,
            1,
        )

    for acc in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = margin_top + plot_h - round(acc * plot_h)
        cv2.line(image, (margin_left - 6, y), (margin_left, y), axis_color, 1)
        cv2.putText(image, f"{acc:.2f}", (20, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 1)

    colors = {
        "low_boundary": (31, 119, 180),
        "temporal_boundary": (44, 160, 44),
        "visual_boundary": (214, 39, 40),
        "audio_boundary": (148, 103, 189),
    }
    legend_x = margin_left + plot_w + 35
    cv2.putText(image, "Boundary", (legend_x, margin_top - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_color, 2)

    for idx, condition in enumerate(sorted(grouped)):
        color = colors.get(condition, (80, 80, 80))
        points = []
        for variant in variants:
            if variant not in grouped[condition]:
                continue
            x = x_positions[variant]
            y = margin_top + plot_h - round(grouped[condition][variant] * plot_h)
            points.append((x, y))
        for p1, p2 in zip(points, points[1:]):
            cv2.line(image, p1, p2, color, 3)
        for point in points:
            cv2.circle(image, point, 6, color, -1)

        legend_y = margin_top + 12 + idx * 30
        cv2.line(image, (legend_x, legend_y), (legend_x + 30, legend_y), color, 3)
        cv2.putText(
            image,
            condition,
            (legend_x + 38, legend_y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            axis_color,
            1,
        )

    cv2.putText(
        image,
        title,
        (margin_left + 30, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        axis_color,
        2,
    )
    cv2.putText(
        image,
        x_axis_label,
        (margin_left + plot_w // 2 - max(60, len(x_axis_label) * 5), height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        axis_color,
        2,
    )
    cv2.putText(image, y_label, (12, margin_top - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_color, 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def make_grouped_bar_plot(categories, series, values, title, output_path, y_label="Rate"):
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("opencv-python and numpy are required for plots; skipping grouped bar plot.")
        return

    width, height = max(1120, 165 * len(categories) + 390), 690
    margin_left, margin_right = 95, 250
    margin_top, margin_bottom = 90, 155
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    axis_color = (40, 40, 40)
    palette = [(31, 119, 180), (44, 160, 44), (214, 39, 40), (148, 103, 189)]

    cv2.line(image, (margin_left, margin_top), (margin_left, margin_top + plot_h), axis_color, 2)
    cv2.line(image, (margin_left, margin_top + plot_h), (margin_left + plot_w, margin_top + plot_h), axis_color, 2)
    for rate in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = margin_top + plot_h - round(rate * plot_h)
        cv2.line(image, (margin_left - 6, y), (margin_left, y), axis_color, 1)
        cv2.putText(image, f"{rate:.2f}", (20, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 1)

    group_w = plot_w / max(1, len(categories))
    usable_w = group_w * 0.72
    bar_w = usable_w / max(1, len(series))
    for category_idx, category in enumerate(categories):
        group_center = margin_left + (category_idx + 0.5) * group_w
        group_start = group_center - usable_w / 2
        for series_idx, series_name in enumerate(series):
            value = values.get((category, series_name))
            if value is None:
                continue
            x1 = round(group_start + series_idx * bar_w + 4)
            x2 = round(group_start + (series_idx + 1) * bar_w - 4)
            y = margin_top + plot_h - round(value * plot_h)
            cv2.rectangle(image, (x1, y), (x2, margin_top + plot_h), palette[series_idx], -1)
            cv2.putText(
                image,
                f"{value:.2f}",
                (x1, max(margin_top + 18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                axis_color,
                1,
            )
        put_centered_lines(
            image,
            axis_label_lines(category),
            group_center,
            margin_top + plot_h + 34,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            axis_color,
            1,
        )

    legend_x = margin_left + plot_w + 30
    cv2.putText(image, "Measure", (legend_x, margin_top - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_color, 2)
    for idx, series_name in enumerate(series):
        y = margin_top + 14 + idx * 34
        cv2.rectangle(image, (legend_x, y - 12), (legend_x + 25, y + 8), palette[idx], -1)
        cv2.putText(image, series_name, (legend_x + 36, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52, axis_color, 1)

    cv2.putText(image, title, (margin_left, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, axis_color, 2)
    cv2.putText(image, y_label, (12, margin_top - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, axis_color, 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def make_pair_outcome_plot(rows, output_path, variants=None, labels=None, title=None):
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("opencv-python and numpy are required for plots; skipping pair outcome plot.")
        return

    variants = list(variants or FEATURE_VARIANTS)
    labels = labels or FEATURE_LABELS
    categories = ["both_correct", "exactly_one_correct", "both_wrong"]
    colors = {
        "both_correct": (44, 160, 44),
        "exactly_one_correct": (31, 119, 180),
        "both_wrong": (214, 39, 40),
    }
    values = {
        (row["feature_variant"], category): float(row[value_name])
        for row in rows
        for category, value_name in [
            ("both_correct", "strict_both_correct"),
            ("exactly_one_correct", "position_sensitive_rate"),
            ("both_wrong", "both_wrong_rate"),
        ]
    }

    width, height = max(1050, 150 * len(variants) + 390), 690
    margin_left, margin_right = 100, 270
    margin_top, margin_bottom = 90, 145
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    axis_color = (40, 40, 40)
    cv2.line(image, (margin_left, margin_top), (margin_left, margin_top + plot_h), axis_color, 2)
    cv2.line(image, (margin_left, margin_top + plot_h), (margin_left + plot_w, margin_top + plot_h), axis_color, 2)

    for rate in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = margin_top + plot_h - round(rate * plot_h)
        cv2.line(image, (margin_left - 6, y), (margin_left, y), axis_color, 1)
        cv2.putText(image, f"{rate:.2f}", (20, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 1)

    group_w = plot_w / len(variants)
    bar_w = min(120, group_w * 0.55)
    for idx, variant in enumerate(variants):
        center = margin_left + (idx + 0.5) * group_w
        x1, x2 = round(center - bar_w / 2), round(center + bar_w / 2)
        bottom_y = margin_top + plot_h
        for category in categories:
            value = values.get((variant, category), 0.0)
            segment_h = round(value * plot_h)
            top_y = bottom_y - segment_h
            cv2.rectangle(image, (x1, top_y), (x2, bottom_y), colors[category], -1)
            if value >= 0.04:
                cv2.putText(
                    image,
                    f"{value:.2f}",
                    (x1 + 8, top_y + max(18, segment_h // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                )
            bottom_y = top_y
        put_centered_lines(
            image,
            axis_label_lines(labels[variant]),
            center,
            margin_top + plot_h + 34,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            axis_color,
            1,
        )

    legend_x = margin_left + plot_w + 30
    cv2.putText(image, "Pair outcome", (legend_x, margin_top - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_color, 2)
    legend_labels = {
        "both_correct": "Both correct",
        "exactly_one_correct": "Exactly one correct",
        "both_wrong": "Both wrong",
    }
    for idx, category in enumerate(categories):
        y = margin_top + 14 + idx * 34
        cv2.rectangle(image, (legend_x, y - 12), (legend_x + 25, y + 8), colors[category], -1)
        cv2.putText(image, legend_labels[category], (legend_x + 36, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, axis_color, 1)

    cv2.putText(
        image,
        title or "Mirrored-pair outcomes by feature condition",
        (margin_left, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        axis_color,
        2,
    )
    cv2.putText(image, "Proportion", (12, margin_top - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, axis_color, 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def make_size_interaction_plot(rows, value_key, title, output_path):
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("opencv-python and numpy are required for plots; skipping size interaction plot.")
        return

    values = {
        (row["target_size_condition"], row["distractor_count_condition"]): row[value_key]
        for row in rows
    }
    if not all(
        (size, count) in values
        for size in ("large", "small")
        for count in ("few", "many")
    ):
        return

    width, height = 980, 620
    margin_left, margin_right = 100, 250
    margin_top, margin_bottom = 90, 95
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    axis_color = (40, 40, 40)

    for rate in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = margin_top + plot_h - round(rate * plot_h)
        cv2.line(image, (margin_left, y), (margin_left + plot_w, y), (225, 225, 225), 1)
        cv2.putText(image, f"{rate:.2f}", (20, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 1)
    cv2.line(image, (margin_left, margin_top), (margin_left, margin_top + plot_h), axis_color, 2)
    cv2.line(image, (margin_left, margin_top + plot_h), (margin_left + plot_w, margin_top + plot_h), axis_color, 2)

    x_positions = {
        "few": margin_left + round(plot_w * 0.18),
        "many": margin_left + round(plot_w * 0.82),
    }
    for count, label in [("few", "Few distractors"), ("many", "Many distractors")]:
        x = x_positions[count]
        cv2.line(image, (x, margin_top + plot_h), (x, margin_top + plot_h + 6), axis_color, 1)
        cv2.putText(
            image,
            label,
            (x - 72, margin_top + plot_h + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            axis_color,
            1,
        )

    series = [
        ("large", "Large targets", (31, 119, 180)),
        ("small", "Small targets", (44, 160, 44)),
    ]
    for series_index, (size, label, color) in enumerate(series):
        x_offset = -4 if series_index == 0 else 4
        points = []
        for count in ("few", "many"):
            value = float(values[(size, count)])
            x = x_positions[count] + x_offset
            y = margin_top + plot_h - round(value * plot_h)
            points.append((x, y))
            cv2.circle(image, (x, y), 7, color, -1)
            cv2.putText(
                image,
                f"{value:.2f}",
                (x - 22, max(margin_top + 18, y - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                axis_color,
                1,
            )
        cv2.line(image, points[0], points[1], color, 3)

    legend_x = margin_left + plot_w + 35
    cv2.putText(image, "Target size", (legend_x, margin_top - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_color, 2)
    for idx, (_, label, color) in enumerate(series):
        y = margin_top + 14 + idx * 36
        cv2.line(image, (legend_x, y), (legend_x + 30, y), color, 3)
        cv2.circle(image, (legend_x + 15, y), 5, color, -1)
        cv2.putText(image, label, (legend_x + 42, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52, axis_color, 1)

    cv2.putText(image, title, (margin_left, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, axis_color, 2)
    cv2.putText(image, "Accuracy", (14, margin_top - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, axis_color, 2)
    cv2.putText(
        image,
        "Distractor count",
        (margin_left + plot_w // 2 - 75, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        axis_color,
        2,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="raw_results.jsonl file or directory containing run folders.")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "analysis"))
    parser.add_argument("--dataset_name_prefix", default=None, help="Only analyze rows/runs whose dataset_name starts with this prefix.")
    parser.add_argument("--latest_per_dataset", action="store_true", help="Use only the latest timestamped run for each dataset.")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    result_files = load_result_files(args.input)
    if not result_files:
        raise FileNotFoundError(f"No raw_results.jsonl files found under {args.input}")
    if args.dataset_name_prefix:
        result_files = [
            path
            for path in result_files
            if dataset_name_from_path(path).startswith(args.dataset_name_prefix)
        ]
        if not result_files:
            raise ValueError(
                f"No raw result files matched dataset_name_prefix={args.dataset_name_prefix!r}."
            )
    if args.latest_per_dataset:
        result_files = latest_result_files(result_files)

    rows = load_rows(result_files, dataset_name_prefix=args.dataset_name_prefix)
    if not rows:
        raise ValueError("No rows matched the requested input/filter.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_difficulty = grouped_accuracy(rows, ["difficulty_level"])
    by_difficulty_condition = grouped_accuracy(rows, ["difficulty_level", "condition"])
    by_correct_option = grouped_accuracy(rows, ["correct_option"])
    by_prompt_variant = grouped_accuracy(rows, ["prompt_variant"])
    pred_dist = prediction_distribution(rows)
    swap_summary, swap_details = swap_consistency(rows)
    swap_by_level_condition = grouped_counts(
        swap_details,
        ["difficulty_level", "difficulty_name", "condition"],
        "category",
    )
    strict_overall = strict_pair_metrics(rows, swap_details, [])
    strict_by_difficulty = strict_pair_metrics(rows, swap_details, ["difficulty_level"])
    strict_by_condition = strict_pair_metrics(rows, swap_details, ["condition"])
    strict_by_difficulty_condition = strict_pair_metrics(
        rows,
        swap_details,
        ["difficulty_level", "condition"],
    )
    paired_summary, paired_details = paired_boundary_comparison(rows)
    feature_rows = [
        row
        for row in rows
        if row.get("feature_variant") in FEATURE_VARIANTS
        and not row.get("size_scene_variant")
    ]
    size_scene_rows = [row for row in rows if row.get("size_scene_variant") in SIZE_SCENE_VARIANTS]
    by_feature = grouped_accuracy(feature_rows, ["feature_variant"]) if feature_rows else []
    by_feature_condition = grouped_accuracy(feature_rows, ["feature_variant", "condition"]) if feature_rows else []
    by_feature_correct_option = grouped_accuracy(feature_rows, ["feature_variant", "correct_option"]) if feature_rows else []
    by_feature_prompt_variant = grouped_accuracy(feature_rows, ["feature_variant", "prompt_variant"]) if feature_rows else []
    feature_swap_details = [
        row
        for row in swap_details
        if row.get("feature_variant") in FEATURE_VARIANTS
        and not row.get("size_scene_variant")
    ]
    size_scene_swap_details = [
        row for row in swap_details if row.get("size_scene_variant") in SIZE_SCENE_VARIANTS
    ]
    swap_by_feature_condition = grouped_counts(
        feature_swap_details,
        ["feature_variant", "condition"],
        "category",
    )
    strict_by_feature = strict_pair_metrics(feature_rows, feature_swap_details, ["feature_variant"])
    strict_by_feature_condition = strict_pair_metrics(
        feature_rows,
        feature_swap_details,
        ["feature_variant", "condition"],
    )
    paired_feature_summary, paired_feature_details = paired_feature_comparison(feature_rows)
    by_size_scene = (
        grouped_accuracy(
            size_scene_rows,
            ["size_scene_variant", "target_size_condition", "distractor_count_condition"],
        )
        if size_scene_rows else []
    )
    by_size_scene_condition = (
        grouped_accuracy(
            size_scene_rows,
            [
                "size_scene_variant",
                "target_size_condition",
                "distractor_count_condition",
                "condition",
            ],
        )
        if size_scene_rows else []
    )
    by_size_scene_correct_option = (
        grouped_accuracy(size_scene_rows, ["size_scene_variant", "correct_option"])
        if size_scene_rows else []
    )
    strict_by_size_scene = strict_pair_metrics(
        size_scene_rows,
        size_scene_swap_details,
        ["size_scene_variant", "target_size_condition", "distractor_count_condition"],
    )
    strict_by_size_scene_condition = strict_pair_metrics(
        size_scene_rows,
        size_scene_swap_details,
        ["size_scene_variant", "condition"],
    )
    size_factorial_summary = size_factorial_effects(size_scene_rows, strict_by_size_scene)
    diagnostic_rows = [row for row in rows if row.get("diagnostic_type")]
    diagnostic_swap_details = [row for row in swap_details if row.get("diagnostic_type")]
    by_diagnostic_type = (
        grouped_accuracy(diagnostic_rows, ["diagnostic_type"]) if diagnostic_rows else []
    )
    by_diagnostic_type_condition = (
        grouped_accuracy(diagnostic_rows, ["diagnostic_type", "condition"])
        if diagnostic_rows else []
    )
    by_diagnostic_type_correct_option = (
        grouped_accuracy(diagnostic_rows, ["diagnostic_type", "correct_option"])
        if diagnostic_rows else []
    )
    strict_by_diagnostic_type = strict_pair_metrics(
        diagnostic_rows,
        diagnostic_swap_details,
        ["diagnostic_type"],
    )
    strict_by_diagnostic_type_condition = strict_pair_metrics(
        diagnostic_rows,
        diagnostic_swap_details,
        ["diagnostic_type", "condition"],
    )

    write_csv(output_dir / "accuracy_by_difficulty.csv", by_difficulty)
    write_csv(output_dir / "accuracy_by_difficulty_condition.csv", by_difficulty_condition)
    write_csv(output_dir / "accuracy_by_correct_option.csv", by_correct_option)
    write_csv(output_dir / "accuracy_by_prompt_variant.csv", by_prompt_variant)
    write_csv(output_dir / "prediction_distribution.csv", pred_dist)
    write_csv(output_dir / "swap_consistency_summary.csv", swap_summary)
    write_csv(output_dir / "swap_consistency_details.csv", swap_details)
    write_csv(output_dir / "swap_consistency_by_level_condition.csv", swap_by_level_condition)
    write_csv(output_dir / "strict_pair_overall.csv", strict_overall)
    write_csv(output_dir / "strict_pair_by_difficulty.csv", strict_by_difficulty)
    write_csv(output_dir / "strict_pair_by_condition.csv", strict_by_condition)
    write_csv(
        output_dir / "strict_pair_by_difficulty_condition.csv",
        strict_by_difficulty_condition,
    )
    write_csv(output_dir / "paired_boundary_summary.csv", paired_summary)
    write_csv(output_dir / "paired_boundary_details.csv", paired_details)
    write_csv(output_dir / "accuracy_by_feature_variant.csv", by_feature)
    write_csv(output_dir / "accuracy_by_feature_variant_condition.csv", by_feature_condition)
    write_csv(output_dir / "accuracy_by_feature_variant_correct_option.csv", by_feature_correct_option)
    write_csv(output_dir / "accuracy_by_feature_variant_prompt_variant.csv", by_feature_prompt_variant)
    write_csv(output_dir / "swap_consistency_by_feature_condition.csv", swap_by_feature_condition)
    write_csv(output_dir / "strict_pair_by_feature_variant.csv", strict_by_feature)
    write_csv(output_dir / "strict_pair_by_feature_variant_condition.csv", strict_by_feature_condition)
    write_csv(output_dir / "paired_feature_summary.csv", paired_feature_summary)
    write_csv(output_dir / "paired_feature_details.csv", paired_feature_details)
    write_csv(output_dir / "accuracy_by_size_scene.csv", by_size_scene)
    write_csv(output_dir / "accuracy_by_size_scene_condition.csv", by_size_scene_condition)
    write_csv(output_dir / "accuracy_by_size_scene_correct_option.csv", by_size_scene_correct_option)
    write_csv(output_dir / "strict_pair_by_size_scene.csv", strict_by_size_scene)
    write_csv(output_dir / "strict_pair_by_size_scene_condition.csv", strict_by_size_scene_condition)
    write_csv(output_dir / "size_factorial_effects.csv", size_factorial_summary)
    write_csv(output_dir / "accuracy_by_diagnostic_type.csv", by_diagnostic_type)
    write_csv(output_dir / "accuracy_by_diagnostic_type_condition.csv", by_diagnostic_type_condition)
    write_csv(output_dir / "accuracy_by_diagnostic_type_correct_option.csv", by_diagnostic_type_correct_option)
    write_csv(output_dir / "strict_pair_by_diagnostic_type.csv", strict_by_diagnostic_type)
    write_csv(output_dir / "strict_pair_by_diagnostic_type_condition.csv", strict_by_diagnostic_type_condition)

    summary = {
        "input": args.input,
        "dataset_name_prefix": args.dataset_name_prefix,
        "result_files": [str(path) for path in result_files],
        "row_count": len(rows),
        "accuracy_by_difficulty": by_difficulty,
        "accuracy_by_difficulty_condition": by_difficulty_condition,
        "accuracy_by_correct_option": by_correct_option,
        "accuracy_by_prompt_variant": by_prompt_variant,
        "prediction_distribution": pred_dist,
        "swap_consistency": swap_summary,
        "strict_pair_overall": strict_overall,
        "strict_pair_by_difficulty": strict_by_difficulty,
        "strict_pair_by_condition": strict_by_condition,
        "strict_pair_by_difficulty_condition": strict_by_difficulty_condition,
        "paired_boundary_summary": paired_summary,
        "accuracy_by_feature_variant": by_feature,
        "accuracy_by_feature_variant_condition": by_feature_condition,
        "accuracy_by_feature_variant_correct_option": by_feature_correct_option,
        "accuracy_by_feature_variant_prompt_variant": by_feature_prompt_variant,
        "strict_pair_by_feature_variant": strict_by_feature,
        "strict_pair_by_feature_variant_condition": strict_by_feature_condition,
        "paired_feature_summary": paired_feature_summary,
        "accuracy_by_size_scene": by_size_scene,
        "accuracy_by_size_scene_condition": by_size_scene_condition,
        "accuracy_by_size_scene_correct_option": by_size_scene_correct_option,
        "strict_pair_by_size_scene": strict_by_size_scene,
        "strict_pair_by_size_scene_condition": strict_by_size_scene_condition,
        "size_factorial_effects": size_factorial_summary,
        "accuracy_by_diagnostic_type": by_diagnostic_type,
        "accuracy_by_diagnostic_type_condition": by_diagnostic_type_condition,
        "accuracy_by_diagnostic_type_correct_option": by_diagnostic_type_correct_option,
        "strict_pair_by_diagnostic_type": strict_by_diagnostic_type,
        "strict_pair_by_diagnostic_type_condition": strict_by_diagnostic_type_condition,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.plots:
        present_size_scene_variants = ordered_size_scene_variants(size_scene_rows)
        present_size_scene_labels = {
            variant: SIZE_SCENE_LABELS.get(variant, variant)
            for variant in present_size_scene_variants
        }
        make_plot(by_difficulty_condition, output_dir / "accuracy_by_difficulty_condition.png")
        if strict_by_difficulty_condition:
            make_plot(
                [
                    {
                        "difficulty_level": row["difficulty_level"],
                        "condition": row["condition"],
                        "accuracy": row["strict_both_correct"],
                    }
                    for row in strict_by_difficulty_condition
                ],
                output_dir / "strict_pair_by_difficulty_condition.png",
                title="Strict both-correct by difficulty level and boundary",
                y_label="Strict pair",
            )
        if strict_by_difficulty:
            difficulty_values = {}
            difficulty_categories = []
            for row in strict_by_difficulty:
                label = f"Level {row['difficulty_level']}"
                difficulty_categories.append(label)
                difficulty_values[(label, "Prompt accuracy")] = row["prompt_accuracy"]
                difficulty_values[(label, "Strict both-correct")] = row["strict_both_correct"]
            make_grouped_bar_plot(
                difficulty_categories,
                ["Prompt accuracy", "Strict both-correct"],
                difficulty_values,
                "Prompt accuracy versus strict pair accuracy by difficulty",
                output_dir / "accuracy_vs_strict_pair_by_difficulty.png",
            )
        if strict_by_condition:
            condition_labels = {
                "low_boundary": "Low",
                "temporal_boundary": "Temporal",
                "visual_boundary": "Visual",
                "audio_boundary": "Audio",
            }
            condition_order = [
                condition
                for condition in (
                    "low_boundary",
                    "temporal_boundary",
                    "visual_boundary",
                    "audio_boundary",
                )
                if any(row["condition"] == condition for row in strict_by_condition)
            ]
            condition_values = {}
            for row in strict_by_condition:
                label = condition_labels.get(row["condition"], row["condition"])
                condition_values[(label, "Prompt accuracy")] = row["prompt_accuracy"]
                condition_values[(label, "Strict both-correct")] = row["strict_both_correct"]
            make_grouped_bar_plot(
                [condition_labels.get(condition, condition) for condition in condition_order],
                ["Prompt accuracy", "Strict both-correct"],
                condition_values,
                "Prompt accuracy versus strict pair accuracy by boundary",
                output_dir / "accuracy_vs_strict_pair_by_boundary.png",
            )
        if by_feature_condition:
            make_feature_plot(
                by_feature_condition,
                output_dir / "accuracy_by_feature_variant_condition.png",
                title="Prompt accuracy by feature variant and boundary",
                variants=FEATURE_VARIANTS,
                labels=FEATURE_LABELS,
                y_label="Prompt accuracy",
            )
        if strict_by_feature:
            accuracy_strict_values = {}
            for row in strict_by_feature:
                label = FEATURE_LABELS[row["feature_variant"]]
                accuracy_strict_values[(label, "Prompt accuracy")] = row["prompt_accuracy"]
                accuracy_strict_values[(label, "Strict both-correct")] = row["strict_both_correct"]
            make_grouped_bar_plot(
                [FEATURE_LABELS[variant] for variant in FEATURE_VARIANTS],
                ["Prompt accuracy", "Strict both-correct"],
                accuracy_strict_values,
                "Prompt accuracy versus strict mirrored-pair accuracy",
                output_dir / "accuracy_vs_strict_pair_by_feature.png",
            )

            pair_outcome_rows = strict_by_feature
            make_pair_outcome_plot(
                pair_outcome_rows,
                output_dir / "pair_outcomes_by_feature.png",
            )

            correct_option_values = {}
            for row in by_feature_correct_option:
                label = FEATURE_LABELS[row["feature_variant"]]
                correct_option_values[(label, f"Correct option {row['correct_option']}")] = row["accuracy"]
            make_grouped_bar_plot(
                [FEATURE_LABELS[variant] for variant in FEATURE_VARIANTS],
                ["Correct option A", "Correct option B"],
                correct_option_values,
                "Response-position sensitivity by feature condition",
                output_dir / "correct_option_bias_by_feature.png",
            )

            visual_values = {}
            for row in by_feature_condition:
                if row["condition"] not in {"low_boundary", "visual_boundary"}:
                    continue
                label = FEATURE_LABELS[row["feature_variant"]]
                series_name = "Low boundary" if row["condition"] == "low_boundary" else "Visual boundary"
                visual_values[(label, series_name)] = row["accuracy"]
            make_grouped_bar_plot(
                [FEATURE_LABELS[variant] for variant in FEATURE_VARIANTS],
                ["Low boundary", "Visual boundary"],
                visual_values,
                "Visual-boundary effect by feature condition",
                output_dir / "visual_boundary_effect_by_feature.png",
            )

        if strict_by_feature_condition:
            strict_condition_plot_rows = [
                {
                    "feature_variant": row["feature_variant"],
                    "condition": row["condition"],
                    "accuracy": row["strict_both_correct"],
                }
                for row in strict_by_feature_condition
            ]
            make_feature_plot(
                strict_condition_plot_rows,
                output_dir / "strict_pair_by_feature_variant_condition.png",
                title="Strict both-correct by feature variant and boundary",
                variants=FEATURE_VARIANTS,
                labels=FEATURE_LABELS,
                y_label="Strict pair",
            )

        if by_size_scene_condition:
            size_boundary_rows = [
                {
                    "feature_variant": row["size_scene_variant"],
                    "condition": row["condition"],
                    "accuracy": row["accuracy"],
                }
                for row in by_size_scene_condition
            ]
            make_feature_plot(
                size_boundary_rows,
                output_dir / "accuracy_by_size_scene_condition.png",
                title="Size-only 2x2 pilot by boundary condition",
                variants=present_size_scene_variants,
                labels=present_size_scene_labels,
                x_axis_label="Target size / distractor count",
                y_label="Prompt accuracy",
            )

        if strict_by_size_scene_condition:
            strict_size_boundary_rows = [
                {
                    "feature_variant": row["size_scene_variant"],
                    "condition": row["condition"],
                    "accuracy": row["strict_both_correct"],
                }
                for row in strict_by_size_scene_condition
            ]
            make_feature_plot(
                strict_size_boundary_rows,
                output_dir / "strict_pair_by_size_scene_condition.png",
                title="Size-only 2x2 strict both-correct by boundary",
                variants=present_size_scene_variants,
                labels=present_size_scene_labels,
                x_axis_label="Target size / distractor count",
                y_label="Strict pair",
            )

        if strict_by_size_scene:
            scene_accuracy_values = {}
            for row in strict_by_size_scene:
                label = SIZE_SCENE_LABELS[row["size_scene_variant"]]
                scene_accuracy_values[(label, "Prompt accuracy")] = row["prompt_accuracy"]
                scene_accuracy_values[(label, "Strict both-correct")] = row["strict_both_correct"]
            make_grouped_bar_plot(
                [present_size_scene_labels[variant] for variant in present_size_scene_variants],
                ["Prompt accuracy", "Strict both-correct"],
                scene_accuracy_values,
                "Size-only scene accuracy versus strict pair accuracy",
                output_dir / "accuracy_vs_strict_pair_by_size_scene.png",
            )

            scene_pair_rows = [
                {**row, "feature_variant": row["size_scene_variant"]}
                for row in strict_by_size_scene
            ]
            make_pair_outcome_plot(
                scene_pair_rows,
                output_dir / "pair_outcomes_by_size_scene.png",
                variants=present_size_scene_variants,
                labels=present_size_scene_labels,
                title="Mirrored-pair outcomes in the size-only 2x2 pilot",
            )
            make_size_interaction_plot(
                by_size_scene,
                "accuracy",
                "Target-size and distractor-count interaction: prompt accuracy",
                output_dir / "size_crowding_interaction_prompt_accuracy.png",
            )
            make_size_interaction_plot(
                strict_by_size_scene,
                "strict_both_correct",
                "Target-size and distractor-count interaction: strict pair accuracy",
                output_dir / "size_crowding_interaction_strict_pair.png",
            )
        if by_diagnostic_type_condition:
            diagnostic_variants = sorted({row["diagnostic_type"] for row in by_diagnostic_type_condition})
            diagnostic_labels = {
                name: name.replace("_", " ").title()
                for name in diagnostic_variants
            }
            diagnostic_plot_rows = [
                {
                    "feature_variant": row["diagnostic_type"],
                    "condition": row["condition"],
                    "accuracy": row["accuracy"],
                }
                for row in by_diagnostic_type_condition
            ]
            make_feature_plot(
                diagnostic_plot_rows,
                output_dir / "accuracy_by_diagnostic_type_condition.png",
                title="Diagnostic prompt accuracy by boundary",
                variants=diagnostic_variants,
                labels=diagnostic_labels,
                x_axis_label="Diagnostic type",
                y_label="Prompt accuracy",
            )
        if strict_by_diagnostic_type_condition:
            diagnostic_variants = sorted({row["diagnostic_type"] for row in strict_by_diagnostic_type_condition})
            diagnostic_labels = {
                name: name.replace("_", " ").title()
                for name in diagnostic_variants
            }
            diagnostic_plot_rows = [
                {
                    "feature_variant": row["diagnostic_type"],
                    "condition": row["condition"],
                    "accuracy": row["strict_both_correct"],
                }
                for row in strict_by_diagnostic_type_condition
            ]
            make_feature_plot(
                diagnostic_plot_rows,
                output_dir / "strict_pair_by_diagnostic_type_condition.png",
                title="Diagnostic strict both-correct by boundary",
                variants=diagnostic_variants,
                labels=diagnostic_labels,
                x_axis_label="Diagnostic type",
                y_label="Strict pair",
            )

    print(f"Analyzed {len(rows)} rows from {len(result_files)} raw result file(s).")
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
