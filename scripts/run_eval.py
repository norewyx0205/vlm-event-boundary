import argparse
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None


KEY_FIELDS = [
    "eval_id",
    "video_id",
    "video_path",
    "dataset_version",
    "difficulty_level",
    "difficulty_name",
    "condition",
    "boundary_type",
    "prompt_variant",
    "correct_option",
    "option_A",
    "option_B",
]


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


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def load_model(model_name):
    if AutoModelForImageTextToText is not None:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                dtype=torch.float16,
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained(model_name)
            return model, processor
        except Exception as exc:
            print(f"AutoModelForImageTextToText failed; falling back to Qwen2VL class: {exc}")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def ask_model(model, processor, video_path, option_a, option_b):
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise ImportError("Missing qwen_vl_utils. Install with `pip install qwen-vl-utils`.") from exc

    prompt = f"""
Watch the video carefully.

Which statement correctly describes the order of events?

A: {option_a}
B: {option_b}

Answer with only A or B.
""".strip()

    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return parse_answer(raw_text), raw_text


def update_stats(bucket, key, correct):
    bucket[key]["total"] += 1
    bucket[key]["correct"] += int(correct)


def summarize(stats):
    out = {}
    for key, item in sorted(stats.items(), key=lambda kv: str(kv[0])):
        total = item["total"]
        correct = item["correct"]
        out[str(key)] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else None,
        }
    return out


def print_section(title, summary):
    print(f"\n{title}")
    for key, item in summary.items():
        print(f"{key}: {item['accuracy']:.2f} ({item['correct']}/{item['total']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    safe_model_name = slugify(args.model_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / safe_model_name / args.dataset_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_result_path = run_dir / "raw_results.jsonl"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"

    model, processor = load_model(args.model_name)
    data = load_data(args.annotation_path)
    if args.max_samples is not None:
        data = data[:args.max_samples]

    results = []
    by_difficulty = defaultdict(lambda: {"total": 0, "correct": 0})
    by_difficulty_condition = defaultdict(lambda: {"total": 0, "correct": 0})
    by_condition = defaultdict(lambda: {"total": 0, "correct": 0})
    by_correct_option = defaultdict(lambda: {"total": 0, "correct": 0})
    by_prompt_variant = defaultdict(lambda: {"total": 0, "correct": 0})
    by_prediction = defaultdict(lambda: {"total": 0, "correct": 0})

    for item in data:
        print(f"Processing {item.get('eval_id', item['video_id'])}")
        pred, raw_response = ask_model(
            model,
            processor,
            item["video_path"],
            item["option_A"],
            item["option_B"],
        )
        correct = pred == item["correct_option"]

        result = {field: item.get(field, "") for field in KEY_FIELDS}
        result.update({
            "dataset_name": args.dataset_name,
            "prediction": pred,
            "is_correct": correct,
            "raw_response": raw_response,
        })
        results.append(result)

        diff = item.get("difficulty_level", "unknown")
        cond = item.get("condition", "unknown")
        update_stats(by_difficulty, diff, correct)
        update_stats(by_difficulty_condition, f"{diff}::{cond}", correct)
        update_stats(by_condition, cond, correct)
        update_stats(by_correct_option, item.get("correct_option", "unknown"), correct)
        update_stats(by_prompt_variant, item.get("prompt_variant", "unknown"), correct)
        update_stats(by_prediction, pred, correct)

        print(
            item["video_id"],
            "pred=", pred,
            "correct=", item["correct_option"],
            "is_correct=", correct,
            "raw=", repr(raw_response),
        )

    with open(raw_result_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    overall_correct = sum(int(r["is_correct"]) for r in results)
    overall_total = len(results)
    summary = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "annotation_path": args.annotation_path,
        "run_dir": str(run_dir),
        "timestamp": timestamp,
        "overall": {
            "total": overall_total,
            "correct": overall_correct,
            "accuracy": overall_correct / overall_total if overall_total else None,
        },
        "by_difficulty_level": summarize(by_difficulty),
        "by_difficulty_level_condition": summarize(by_difficulty_condition),
        "by_condition": summarize(by_condition),
        "by_correct_option": summarize(by_correct_option),
        "by_prompt_variant": summarize(by_prompt_variant),
        "by_prediction": summarize(by_prediction),
    }
    config = vars(args) | {"timestamp": timestamp, "run_dir": str(run_dir)}

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved raw results to {raw_result_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved config to {config_path}")
    print_section("=== BY DIFFICULTY LEVEL ===", summary["by_difficulty_level"])
    print_section("=== BY CONDITION ===", summary["by_condition"])
    print_section("=== BY CORRECT OPTION ===", summary["by_correct_option"])
    print_section("=== BY PROMPT VARIANT ===", summary["by_prompt_variant"])
    print_section("=== BY PREDICTION ===", summary["by_prediction"])


if __name__ == "__main__":
    main()
