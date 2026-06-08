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
                row["comparison"],
            )
        ].append(row)

    summary_rows = []
    for key, items in sorted(grouped.items(), key=lambda kv: kv[0]):
        difficulty_level, difficulty_name, feature_variant, comparison = key
        differences = [item["difference"] for item in items]
        directions = Counter(item["direction"] for item in items)
        summary_rows.append({
            "difficulty_level": difficulty_level,
            "difficulty_name": difficulty_name,
            "feature_variant": feature_variant,
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
        if feature_variant not in {"full", "shape_only", "color_only"}:
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


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows, output_path):
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("opencv-python and numpy are required for plots; skipping plot.")
        return

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)

    width, height = 1120, 620
    margin_left, margin_right = 95, 290
    margin_top, margin_bottom = 85, 85
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    axis_color = (40, 40, 40)
    cv2.line(image, (margin_left, margin_top), (margin_left, margin_top + plot_h), axis_color, 2)
    cv2.line(image, (margin_left, margin_top + plot_h), (margin_left + plot_w, margin_top + plot_h), axis_color, 2)

    level_values = sorted({int(row["difficulty_level"]) for row in rows if str(row.get("difficulty_level", "")).isdigit()})
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

    cv2.putText(image, "Accuracy by difficulty level and boundary condition", (margin_left, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, axis_color, 2)
    cv2.putText(image, "Difficulty level", (margin_left + plot_w // 2 - 80, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, axis_color, 2)
    cv2.putText(image, "Accuracy", (12, margin_top - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_color, 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def make_feature_plot(rows, output_path):
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("opencv-python and numpy are required for plots; skipping feature plot.")
        return

    variants = ["full", "shape_only", "color_only"]
    labels = {
        "full": "Full",
        "shape_only": "Shape only",
        "color_only": "Color only",
    }
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["condition"]][row["feature_variant"]] = float(row["accuracy"])

    width, height = 1120, 620
    margin_left, margin_right = 95, 290
    margin_top, margin_bottom = 85, 95
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    axis_color = (40, 40, 40)

    cv2.line(image, (margin_left, margin_top), (margin_left, margin_top + plot_h), axis_color, 2)
    cv2.line(image, (margin_left, margin_top + plot_h), (margin_left + plot_w, margin_top + plot_h), axis_color, 2)

    x_positions = {}
    for idx, variant in enumerate(variants):
        x = margin_left + round(idx / (len(variants) - 1) * plot_w)
        x_positions[variant] = x
        cv2.line(image, (x, margin_top + plot_h), (x, margin_top + plot_h + 6), axis_color, 1)
        cv2.putText(
            image,
            labels[variant],
            (x - 48, margin_top + plot_h + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
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
        "Level 5 feature ablation by boundary condition",
        (margin_left, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        axis_color,
        2,
    )
    cv2.putText(image, "Feature encoding", (margin_left + plot_w // 2 - 75, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, axis_color, 2)
    cv2.putText(image, "Accuracy", (12, margin_top - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_color, 2)
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
    paired_summary, paired_details = paired_boundary_comparison(rows)
    feature_rows = [row for row in rows if row.get("feature_variant") in {"full", "shape_only", "color_only"}]
    by_feature = grouped_accuracy(feature_rows, ["feature_variant"]) if feature_rows else []
    by_feature_condition = grouped_accuracy(feature_rows, ["feature_variant", "condition"]) if feature_rows else []
    by_feature_correct_option = grouped_accuracy(feature_rows, ["feature_variant", "correct_option"]) if feature_rows else []
    by_feature_prompt_variant = grouped_accuracy(feature_rows, ["feature_variant", "prompt_variant"]) if feature_rows else []
    swap_by_feature_condition = grouped_counts(
        [row for row in swap_details if row.get("feature_variant")],
        ["feature_variant", "condition"],
        "category",
    )
    paired_feature_summary, paired_feature_details = paired_feature_comparison(feature_rows)

    write_csv(output_dir / "accuracy_by_difficulty.csv", by_difficulty)
    write_csv(output_dir / "accuracy_by_difficulty_condition.csv", by_difficulty_condition)
    write_csv(output_dir / "accuracy_by_correct_option.csv", by_correct_option)
    write_csv(output_dir / "accuracy_by_prompt_variant.csv", by_prompt_variant)
    write_csv(output_dir / "prediction_distribution.csv", pred_dist)
    write_csv(output_dir / "swap_consistency_summary.csv", swap_summary)
    write_csv(output_dir / "swap_consistency_details.csv", swap_details)
    write_csv(output_dir / "swap_consistency_by_level_condition.csv", swap_by_level_condition)
    write_csv(output_dir / "paired_boundary_summary.csv", paired_summary)
    write_csv(output_dir / "paired_boundary_details.csv", paired_details)
    write_csv(output_dir / "accuracy_by_feature_variant.csv", by_feature)
    write_csv(output_dir / "accuracy_by_feature_variant_condition.csv", by_feature_condition)
    write_csv(output_dir / "accuracy_by_feature_variant_correct_option.csv", by_feature_correct_option)
    write_csv(output_dir / "accuracy_by_feature_variant_prompt_variant.csv", by_feature_prompt_variant)
    write_csv(output_dir / "swap_consistency_by_feature_condition.csv", swap_by_feature_condition)
    write_csv(output_dir / "paired_feature_summary.csv", paired_feature_summary)
    write_csv(output_dir / "paired_feature_details.csv", paired_feature_details)

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
        "paired_boundary_summary": paired_summary,
        "accuracy_by_feature_variant": by_feature,
        "accuracy_by_feature_variant_condition": by_feature_condition,
        "accuracy_by_feature_variant_correct_option": by_feature_correct_option,
        "accuracy_by_feature_variant_prompt_variant": by_feature_prompt_variant,
        "paired_feature_summary": paired_feature_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.plots:
        make_plot(by_difficulty_condition, output_dir / "accuracy_by_difficulty_condition.png")
        if by_feature_condition:
            make_feature_plot(by_feature_condition, output_dir / "accuracy_by_feature_variant_condition.png")

    print(f"Analyzed {len(rows)} rows from {len(result_files)} raw result file(s).")
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
