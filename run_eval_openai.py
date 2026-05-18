import argparse
import base64
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import cv2


def load_data(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def slugify(text):
    keep = []

    for ch in text:
        if ch.isalnum():
            keep.append(ch)
        elif ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")

    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unnamed"


def summarize_grouped_stats(grouped_stats):
    summary = {}

    for key, stats in sorted(grouped_stats.items()):
        total = stats["total"]
        correct = stats["correct"]
        summary[key] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else None,
        }

    return summary


def format_summary_section(title, summary):
    lines = [title]

    for key, stats in summary.items():
        acc = stats["accuracy"]
        acc_text = "nan" if acc is None else f"{acc:.2f}"
        lines.append(f"{key}: {acc_text} ({stats['correct']}/{stats['total']})")

    return "\n".join(lines)


def parse_answer(raw_text):
    text = raw_text.strip().upper()

    if text.startswith("A"):
        return "A"
    if text.startswith("B"):
        return "B"
    if "ANSWER: A" in text or "OPTION A" in text:
        return "A"
    if "ANSWER: B" in text or "OPTION B" in text:
        return "B"

    return "UNKNOWN"


def extract_frame_images(video_path, frame_count, max_width, jpeg_quality):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    if frame_count == 1:
        frame_indices = [total_frames // 2]
    else:
        frame_indices = [
            round(i * (total_frames - 1) / (frame_count - 1))
            for i in range(frame_count)
        ]

    images = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()

        if not ok:
            continue

        height, width = frame.shape[:2]

        if width > max_width:
            scale = max_width / width
            new_size = (max_width, round(height * scale))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

        if not ok:
            continue

        images.append({
            "frame_index": idx,
            "image_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
        })

    cap.release()

    if not images:
        raise RuntimeError(f"Could not extract frames from video: {video_path}")

    return images


def make_prompt(option_a, option_b):
    return f"""
You are given a sequence of frames sampled in chronological order from a video.
Watch the frame sequence carefully.

Which statement correctly describes the order of events?

A: {option_a}
B: {option_b}

Answer with only A or B.
""".strip()


def ask_model(client, model_name, video_path, option_a, option_b, frame_count, max_width, jpeg_quality, image_detail):
    images = extract_frame_images(video_path, frame_count, max_width, jpeg_quality)

    content = [{"type": "input_text", "text": make_prompt(option_a, option_b)}]

    for image in images:
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{image['image_b64']}",
            "detail": image_detail,
        })

    response = client.responses.create(
        model=model_name,
        input=[{
            "role": "user",
            "content": content,
        }],
        max_output_tokens=10,
    )

    raw_text = response.output_text
    pred = parse_answer(raw_text)
    return pred, raw_text, [image["frame_index"] for image in images]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        default="gpt-4.1",
        help="OpenAI vision-capable model id, e.g. gpt-4.1 or gpt-4o.",
    )
    parser.add_argument(
        "--annotation-path",
        default="/content/vlm-event-boundary/synthetic_boundary_videos/annotations.jsonl",
        help="Path to annotations.jsonl.",
    )
    parser.add_argument(
        "--result-dir",
        default="/content/vlm-event-boundary/results",
        help="Root directory for timestamped result folders.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional raw result filename. Defaults to raw_results.jsonl inside the run folder.",
    )
    parser.add_argument(
        "--experiment-version",
        default=None,
        help="Dataset or experiment version label, e.g. baseline_gpt, hard_v1_gpt.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for quick tests.",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=12,
        help="Number of frames sampled uniformly from each video.",
    )
    parser.add_argument(
        "--max-frame-width",
        type=int,
        default=384,
        help="Resize sampled frames to this maximum width before sending.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality for sampled frames.",
    )
    parser.add_argument(
        "--image-detail",
        choices=["low", "high", "auto"],
        default="low",
        help="OpenAI image detail setting.",
    )
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Missing openai package. Install it with `pip install openai`.") from exc

    annotation_path = Path(args.annotation_path)
    inferred_version = annotation_path.parent.name
    experiment_version = args.experiment_version or inferred_version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = slugify(args.model_name)
    version_slug = slugify(experiment_version)

    run_dir = Path(args.result_dir) / version_slug / model_slug / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    output_name = args.output_name or "raw_results.jsonl"
    raw_result_path = run_dir / output_name
    summary_json_path = run_dir / "summary.json"
    summary_txt_path = run_dir / "summary.txt"
    config_path = run_dir / "config.json"

    client = OpenAI()
    data = load_data(annotation_path)

    if args.max_samples is not None:
        data = data[:args.max_samples]

    results = []
    stats = defaultdict(lambda: {"total": 0, "correct": 0})
    option_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    variant_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    prediction_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for item in data:
        print(f"Processing {item.get('eval_id', item['video_id'])}")

        pred, raw_response, frame_indices = ask_model(
            client=client,
            model_name=args.model_name,
            video_path=item["video_path"],
            option_a=item["option_A"],
            option_b=item["option_B"],
            frame_count=args.frame_count,
            max_width=args.max_frame_width,
            jpeg_quality=args.jpeg_quality,
            image_detail=args.image_detail,
        )

        correct = pred == item["correct_option"]

        result = {
            "video_id": item["video_id"],
            "eval_id": item.get("eval_id", item["video_id"]),
            "video_path": item["video_path"],
            "dataset": item.get("dataset", ""),
            "condition": item["condition"],
            "boundary_type": item.get("boundary_type", ""),
            "prompt_variant": item.get("prompt_variant", ""),
            "correct_option": item["correct_option"],
            "prediction": pred,
            "is_correct": correct,
            "option_A": item["option_A"],
            "option_B": item["option_B"],
            "raw_response": raw_response,
            "sampled_frame_indices": frame_indices,
        }

        results.append(result)

        stats[item["condition"]]["total"] += 1
        stats[item["condition"]]["correct"] += int(correct)
        option_stats[item["correct_option"]]["total"] += 1
        option_stats[item["correct_option"]]["correct"] += int(correct)
        variant = item.get("prompt_variant", "none")
        variant_stats[variant]["total"] += 1
        variant_stats[variant]["correct"] += int(correct)
        prediction_stats[pred]["total"] += 1
        prediction_stats[pred]["correct"] += int(correct)

        print(
            "option_A:" + item["option_A"],
            "option_B:" + item["option_B"],
            item["video_id"],
            "pred=", pred,
            "correct=", item["correct_option"],
            "is_correct=", correct,
            "raw=", repr(raw_response),
        )

    with open(raw_result_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_condition = summarize_grouped_stats(stats)
    by_correct_option = summarize_grouped_stats(option_stats)
    by_prompt_variant = summarize_grouped_stats(variant_stats)
    by_prediction = summarize_grouped_stats(prediction_stats)

    overall_correct = sum(int(r["is_correct"]) for r in results)
    overall_total = len(results)
    summary = {
        "timestamp": timestamp,
        "model_name": args.model_name,
        "annotation_path": str(annotation_path),
        "experiment_version": experiment_version,
        "run_dir": str(run_dir),
        "max_samples": args.max_samples,
        "frame_count": args.frame_count,
        "max_frame_width": args.max_frame_width,
        "jpeg_quality": args.jpeg_quality,
        "image_detail": args.image_detail,
        "overall": {
            "total": overall_total,
            "correct": overall_correct,
            "accuracy": overall_correct / overall_total if overall_total else None,
        },
        "by_condition": by_condition,
        "by_correct_option": by_correct_option,
        "by_prompt_variant": by_prompt_variant,
        "by_prediction": by_prediction,
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    config = {
        "model_name": args.model_name,
        "annotation_path": str(annotation_path),
        "result_dir": args.result_dir,
        "experiment_version": experiment_version,
        "output_name": output_name,
        "max_samples": args.max_samples,
        "frame_count": args.frame_count,
        "max_frame_width": args.max_frame_width,
        "jpeg_quality": args.jpeg_quality,
        "image_detail": args.image_detail,
        "timestamp": timestamp,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    summary_text = "\n\n".join([
        f"Run: {timestamp}",
        f"Model: {args.model_name}",
        f"Experiment version: {experiment_version}",
        f"Annotation path: {annotation_path}",
        f"Frame count: {args.frame_count}",
        f"Overall: {summary['overall']['accuracy']:.2f} ({overall_correct}/{overall_total})",
        format_summary_section("=== FINAL RESULTS ===", by_condition),
        format_summary_section("=== BY CORRECT OPTION ===", by_correct_option),
        format_summary_section("=== BY PROMPT VARIANT ===", by_prompt_variant),
        format_summary_section("=== BY PREDICTION ===", by_prediction),
    ])

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    print(f"\nSaved raw results to {raw_result_path}")
    print(f"Saved summary JSON to {summary_json_path}")
    print(f"Saved summary text to {summary_txt_path}")
    print(f"Saved run config to {config_path}")

    print("\n" + format_summary_section("=== FINAL RESULTS ===", by_condition))
    print("\n" + format_summary_section("=== BY CORRECT OPTION ===", by_correct_option))
    print("\n" + format_summary_section("=== BY PROMPT VARIANT ===", by_prompt_variant))
    print("\n" + format_summary_section("=== BY PREDICTION ===", by_prediction))


if __name__ == "__main__":
    main()
