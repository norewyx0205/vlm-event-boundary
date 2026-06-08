import argparse
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

try:
    from .common import PROJECT_ROOT, read_jsonl, slugify, summarize_counts, write_jsonl
except ImportError:
    from common import PROJECT_ROOT, read_jsonl, slugify, summarize_counts, write_jsonl


KEY_FIELDS = [
    "eval_id",
    "video_id",
    "video_path",
    "dataset_version",
    "difficulty_level",
    "difficulty_name",
    "feature_variant",
    "feature_encoding",
    "condition",
    "boundary_type",
    "base_sample_id",
    "pairing_id",
    "prompt_variant",
    "correct_option",
    "option_A",
    "option_B",
]


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


def model_kwargs(load_in_4bit):
    kwargs = {
        "dtype": torch.float16,
        "device_map": "auto",
    }
    if load_in_4bit:
        if BitsAndBytesConfig is None:
            raise ImportError("4-bit loading requires transformers BitsAndBytesConfig and bitsandbytes.")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs.pop("dtype", None)
    return kwargs


def load_model(model_name, load_in_4bit=False):
    kwargs = model_kwargs(load_in_4bit)
    if AutoModelForImageTextToText is not None:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                **kwargs,
            )
            processor = AutoProcessor.from_pretrained(model_name)
            return model, processor
        except Exception as exc:
            print(f"AutoModelForImageTextToText failed; falling back to Qwen2VL class: {exc}")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        **kwargs,
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def process_video_inputs(messages):
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise ImportError("Missing qwen_vl_utils. Install with `pip install qwen-vl-utils`.") from exc

    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
    except TypeError:
        image_inputs, video_inputs = process_vision_info(messages)
        return image_inputs, video_inputs, {}

    if video_inputs is not None and video_inputs and isinstance(video_inputs[0], tuple):
        video_inputs, video_metadata = zip(*video_inputs)
        video_inputs = list(video_inputs)
        video_kwargs["video_metadata"] = list(video_metadata)

    return image_inputs, video_inputs, video_kwargs


def ask_model(model, processor, video_path, option_a, option_b, video_fps=None, video_max_pixels=None):
    prompt = f"""
Watch the video carefully.

Which statement correctly describes the order of events?

A: {option_a}
B: {option_b}

Answer with only A or B.
""".strip()

    video_content = {
        "type": "video",
        "video": video_path,
    }
    if video_fps is not None:
        video_content["fps"] = video_fps
    if video_max_pixels is not None:
        video_content["max_pixels"] = video_max_pixels

    messages = [{
        "role": "user",
        "content": [
            video_content,
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_video_inputs(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        **video_kwargs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
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
    del inputs, generated_ids, generated_ids_trimmed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return parse_answer(raw_text), raw_text


def update_stats(bucket, key, correct):
    bucket[key]["total"] += 1
    bucket[key]["correct"] += int(correct)


def print_section(title, summary):
    print(f"\n{title}")
    for key, item in summary.items():
        print(f"{key}: {item['accuracy']:.2f} ({item['correct']}/{item['total']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", "--annotation-path", required=True)
    parser.add_argument("--model_name", "--model-name", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--dataset_name", "--dataset-name", default=None)
    parser.add_argument("--output_dir", "--output-dir", "--result-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--max_samples", "--max-samples", type=int, default=None)
    parser.add_argument(
        "--video_fps",
        "--video-fps",
        type=float,
        default=None,
        help="Optional frames sampled per second from each video. Lower values reduce GPU memory use.",
    )
    parser.add_argument(
        "--video_max_pixels",
        "--video-max-pixels",
        type=int,
        default=None,
        help="Optional maximum pixels per sampled video frame. Lower values reduce visual tokens and GPU memory use.",
    )
    parser.add_argument(
        "--load_in_4bit",
        "--load-in-4bit",
        action="store_true",
        help="Load the model with bitsandbytes 4-bit quantization. Recommended for Qwen3-VL-8B on Colab T4.",
    )
    parser.add_argument(
        "--experiment_version",
        "--experiment-version",
        default=None,
        help="Backward-compatible alias used as dataset_name when --dataset_name is omitted.",
    )
    parser.add_argument(
        "--output_name",
        "--output-name",
        default="raw_results.jsonl",
        help="Backward-compatible raw result filename option.",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name or args.experiment_version or Path(args.annotation_path).parent.name
    dataset_slug = slugify(dataset_name)
    safe_model_name = slugify(args.model_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / safe_model_name / dataset_slug / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_result_path = run_dir / args.output_name
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"

    model, processor = load_model(args.model_name, load_in_4bit=args.load_in_4bit)
    data = read_jsonl(args.annotation_path)
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
            args.video_fps,
            args.video_max_pixels,
        )
        correct = pred == item["correct_option"]

        result = {field: item.get(field, "") for field in KEY_FIELDS}
        result.update({
            "dataset_name": dataset_name,
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

    write_jsonl(raw_result_path, results)

    overall_correct = sum(int(r["is_correct"]) for r in results)
    overall_total = len(results)
    summary = {
        "model_name": args.model_name,
        "dataset_name": dataset_name,
        "annotation_path": args.annotation_path,
        "run_dir": str(run_dir),
        "timestamp": timestamp,
        "overall": {
            "total": overall_total,
            "correct": overall_correct,
            "accuracy": overall_correct / overall_total if overall_total else None,
        },
        "by_difficulty_level": summarize_counts(by_difficulty),
        "by_difficulty_level_condition": summarize_counts(by_difficulty_condition),
        "by_condition": summarize_counts(by_condition),
        "by_correct_option": summarize_counts(by_correct_option),
        "by_prompt_variant": summarize_counts(by_prompt_variant),
        "by_prediction": summarize_counts(by_prediction),
    }
    config = vars(args) | {
        "dataset_name": dataset_name,
        "dataset_slug": dataset_slug,
        "timestamp": timestamp,
        "run_dir": str(run_dir),
    }

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
