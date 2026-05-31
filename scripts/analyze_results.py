import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict, Counter


def load_result_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(path.rglob("raw_results.jsonl"))


def dataset_name_from_path(path):
    parts = Path(path).parts
    if len(parts) >= 3:
        return parts[-2]
    return ""


def load_rows(paths, dataset_name_prefix=None):
    rows = []
    for path in paths:
        dataset_name = dataset_name_from_path(path)
        if dataset_name_prefix and not dataset_name.startswith(dataset_name_prefix):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
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

    width, height = 900, 560
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 50, 80
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    axis_color = (40, 40, 40)
    cv2.line(image, (margin_left, margin_top), (margin_left, margin_top + plot_h), axis_color, 2)
    cv2.line(image, (margin_left, margin_top + plot_h), (margin_left + plot_w, margin_top + plot_h), axis_color, 2)

    for level in [1, 2, 3, 4]:
        x = margin_left + round((level - 1) / 3 * plot_w)
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

    legend_y = 28
    for idx, (condition, items) in enumerate(sorted(grouped.items())):
        items = sorted(items, key=lambda x: int(x["difficulty_level"]))
        points = []
        color = colors.get(condition, (80, 80, 80))

        for item in items:
            level = int(item["difficulty_level"])
            acc = float(item["accuracy"])
            x = margin_left + round((level - 1) / 3 * plot_w)
            y = margin_top + plot_h - round(acc * plot_h)
            points.append((x, y))

        for p1, p2 in zip(points, points[1:]):
            cv2.line(image, p1, p2, color, 3)
        for point in points:
            cv2.circle(image, point, 6, color, -1)

        lx = 560
        ly = legend_y + idx * 24
        cv2.line(image, (lx, ly), (lx + 30, ly), color, 3)
        cv2.putText(image, condition, (lx + 38, ly + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 1)

    cv2.putText(image, "Accuracy by difficulty level and boundary condition", (margin_left, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, axis_color, 2)
    cv2.putText(image, "Difficulty level", (margin_left + plot_w // 2 - 80, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, axis_color, 2)
    cv2.putText(image, "Accuracy", (8, margin_top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, axis_color, 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="raw_results.jsonl file or directory containing run folders.")
    parser.add_argument("--output_dir", default="analysis")
    parser.add_argument("--dataset_name_prefix", default=None, help="Only analyze rows/runs whose dataset_name starts with this prefix.")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    result_files = load_result_files(args.input)
    if not result_files:
        raise FileNotFoundError(f"No raw_results.jsonl files found under {args.input}")

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

    write_csv(output_dir / "accuracy_by_difficulty.csv", by_difficulty)
    write_csv(output_dir / "accuracy_by_difficulty_condition.csv", by_difficulty_condition)
    write_csv(output_dir / "accuracy_by_correct_option.csv", by_correct_option)
    write_csv(output_dir / "accuracy_by_prompt_variant.csv", by_prompt_variant)
    write_csv(output_dir / "prediction_distribution.csv", pred_dist)
    write_csv(output_dir / "swap_consistency_summary.csv", swap_summary)
    write_csv(output_dir / "swap_consistency_details.csv", swap_details)

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
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.plots:
        make_plot(by_difficulty_condition, output_dir / "accuracy_by_difficulty_condition.png")

    print(f"Analyzed {len(rows)} rows from {len(result_files)} raw result file(s).")
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
