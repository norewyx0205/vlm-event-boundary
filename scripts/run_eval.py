import argparse
import hashlib
import json
import os
import platform
import random
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import transformers
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
    "size_scene_variant",
    "size_contrast_condition",
    "target_size_condition",
    "distractor_count_condition",
    "diagnostic_type",
    "diagnostic_family",
    "diagnostic_source_eval_id",
    "diagnostic_prompt_version",
    "diagnostic_axis",
    "diagnostic_relation",
    "diagnostic_subject_reference_label",
    "diagnostic_reference_reference_label",
    "perturbation_type",
    "perturbation_target",
    "condition",
    "boundary_type",
    "base_sample_id",
    "pairing_id",
    "target_1_radius",
    "target_2_radius",
    "target_1_size_label",
    "target_2_size_label",
    "target_1_reference_label",
    "target_2_reference_label",
    "distractor_radius",
    "distractor_size_label",
    "shared_target_attribute",
    "distractor_count",
    "static_distractor_count",
    "moving_distractor_count",
    "prompt_variant",
    "correct_option",
    "option_A",
    "option_B",
    "question",
]


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def tensor_shape_metadata(tensor):
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape) if hasattr(tensor, "shape") else None,
        "dtype": str(getattr(tensor, "dtype", "")),
        "device": str(getattr(tensor, "device", "")),
    }


def video_input_metadata(video_inputs):
    metadata = []
    if not video_inputs:
        return metadata
    for item in video_inputs:
        shape = list(item.shape) if hasattr(item, "shape") else None
        entry = {
            "shape": shape,
            "dtype": str(getattr(item, "dtype", "")),
            "frame_count_from_first_dim": shape[0] if shape and len(shape) >= 4 else None,
        }
        metadata.append(entry)
    return metadata


def count_video_tokens(inputs):
    token_types = getattr(inputs, "mm_token_type_ids", None)
    if token_types is None:
        return None
    return int((token_types.detach().cpu() == 2).sum().item())


def processor_input_metadata(inputs, video_inputs, video_kwargs):
    video_grid = None
    video_grid_cells = None
    if getattr(inputs, "video_grid_thw", None) is not None:
        video_grid = inputs.video_grid_thw.detach().cpu().tolist()
        video_grid_cells = [
            int(grid[0] * grid[1] * grid[2])
            for grid in video_grid
            if len(grid) == 3
        ]

    safe_video_kwargs = {
        key: json_safe(value)
        for key, value in video_kwargs.items()
        if key != "video_metadata"
    }
    video_metadata = video_kwargs.get("video_metadata")

    return {
        "video_kwargs": safe_video_kwargs,
        "video_metadata": str(video_metadata) if video_metadata is not None else None,
        "video_inputs": video_input_metadata(video_inputs),
        "pixel_values_videos": tensor_shape_metadata(getattr(inputs, "pixel_values_videos", None)),
        "video_grid_thw": video_grid,
        "visual_tokens_from_grid_thw": video_grid_cells,
        "visual_token_count_from_grid_thw": sum(video_grid_cells) if video_grid_cells else None,
        "video_token_count_from_mm_token_type_ids": count_video_tokens(inputs),
        "input_ids": tensor_shape_metadata(getattr(inputs, "input_ids", None)),
        "attention_mask": tensor_shape_metadata(getattr(inputs, "attention_mask", None)),
        "mm_token_type_ids": tensor_shape_metadata(getattr(inputs, "mm_token_type_ids", None)),
    }


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


def configure_reproducibility(seed, deterministic=False, deterministic_warn_only=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=deterministic_warn_only)
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")


def package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_metadata(model):
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [
            torch.cuda.get_device_name(device_index)
            for device_index in range(torch.cuda.device_count())
        ]
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "accelerate_version": package_version("accelerate"),
        "qwen_vl_utils_version": package_version("qwen-vl-utils"),
        "decord_version": package_version("decord"),
        "av_version": package_version("av"),
        "opencv_version": package_version("opencv-python"),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_names": gpu_names,
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
    }


def model_kwargs(load_in_4bit, model_revision=None, attn_implementation=None):
    kwargs = {
        "dtype": torch.float16,
        "device_map": "auto",
    }
    if model_revision:
        kwargs["revision"] = model_revision
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
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


def load_model(
    model_name,
    load_in_4bit=False,
    model_revision=None,
    attn_implementation=None,
):
    kwargs = model_kwargs(load_in_4bit, model_revision, attn_implementation)
    processor_kwargs = {"revision": model_revision} if model_revision else {}
    if AutoModelForImageTextToText is not None:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                **kwargs,
            )
            processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
            model.eval()
            return model, processor
        except Exception as exc:
            print(f"AutoModelForImageTextToText failed; falling back to Qwen2VL class: {exc}")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        **kwargs,
    )
    processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
    model.eval()
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


def build_messages(
    video_path,
    option_a,
    option_b,
    video_fps=None,
    video_num_frames=None,
    video_max_pixels=None,
    question=None,
):
    prompt_question = question or "Which statement correctly describes the order of events?"
    prompt = f"""
Watch the video carefully.

{prompt_question}

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
    if video_num_frames is not None:
        video_content["num_frames"] = video_num_frames
    if video_max_pixels is not None:
        video_content["max_pixels"] = video_max_pixels

    return [{
        "role": "user",
        "content": [
            video_content,
            {"type": "text", "text": prompt},
        ],
    }]


def ask_model(
    model,
    processor,
    video_path,
    option_a,
    option_b,
    video_fps=None,
    video_num_frames=None,
    video_max_pixels=None,
    max_new_tokens=10,
    empty_cache_each_sample=False,
    prepared_vision=None,
    question=None,
):
    messages = build_messages(
        video_path,
        option_a,
        option_b,
        video_fps,
        video_num_frames,
        video_max_pixels,
        question,
    )

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prepared_vision is None:
        image_inputs, video_inputs, video_kwargs = process_video_inputs(messages)
    else:
        image_inputs, video_inputs, video_kwargs = prepared_vision
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        **video_kwargs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    input_metadata = processor_input_metadata(inputs, video_inputs, video_kwargs)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
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
    if empty_cache_each_sample and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return parse_answer(raw_text), raw_text, input_metadata


def update_stats(bucket, key, correct):
    bucket[key]["total"] += 1
    bucket[key]["correct"] += int(correct)


def print_section(title, summary):
    print(f"\n{title}")
    for key, item in summary.items():
        print(f"{key}: {item['accuracy']:.2f} ({item['correct']}/{item['total']})")


def evaluate_annotation(model, processor, args, annotation_path, dataset_name, timestamp):
    dataset_slug = slugify(dataset_name)
    safe_model_name = slugify(args.model_name)
    run_dir = Path(args.output_dir) / safe_model_name / dataset_slug / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_result_path = run_dir / args.output_name
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"

    data = read_jsonl(annotation_path)
    if args.max_samples is not None:
        data = data[:args.max_samples]

    results = []
    by_difficulty = defaultdict(lambda: {"total": 0, "correct": 0})
    by_difficulty_condition = defaultdict(lambda: {"total": 0, "correct": 0})
    by_condition = defaultdict(lambda: {"total": 0, "correct": 0})
    by_correct_option = defaultdict(lambda: {"total": 0, "correct": 0})
    by_prompt_variant = defaultdict(lambda: {"total": 0, "correct": 0})
    by_prediction = defaultdict(lambda: {"total": 0, "correct": 0})
    cached_video_path = None
    cached_vision = None

    for item in data:
        print(f"Processing {item.get('eval_id', item['video_id'])}")
        if args.disable_video_cache or item["video_path"] != cached_video_path:
            messages = build_messages(
                item["video_path"],
                item["option_A"],
                item["option_B"],
                args.video_fps,
                args.video_num_frames,
                args.video_max_pixels,
                item.get("question"),
            )
            cached_vision = process_video_inputs(messages)
            cached_video_path = item["video_path"]
        pred, raw_response, input_metadata = ask_model(
            model,
            processor,
            item["video_path"],
            item["option_A"],
            item["option_B"],
            args.video_fps,
            args.video_num_frames,
            args.video_max_pixels,
            args.max_new_tokens,
            args.empty_cache_each_sample,
            cached_vision,
            item.get("question"),
        )
        correct = pred == item["correct_option"]

        result = {field: item.get(field, "") for field in KEY_FIELDS}
        result.update({
            "dataset_name": dataset_name,
            "prediction": pred,
            "is_correct": correct,
            "raw_response": raw_response,
            "input_metadata": input_metadata,
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
        "annotation_path": str(annotation_path),
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
        "annotation_path": str(annotation_path),
        "annotation_sha256": file_sha256(annotation_path),
        "dataset_name": dataset_name,
        "dataset_slug": dataset_slug,
        "timestamp": timestamp,
        "run_dir": str(run_dir),
        "model_load": {
            "dtype": "float16" if not args.load_in_4bit else None,
            "load_in_4bit": args.load_in_4bit,
            "attn_implementation": args.attn_implementation,
            "model_revision": args.model_revision,
        },
        "video_sampling_request": {
            "fps": args.video_fps,
            "num_frames": args.video_num_frames,
            "max_pixels": args.video_max_pixels,
            "min_pixels": None,
        },
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
        },
        "output_parsing": {
            "accepted_patterns": [
                "leading A",
                "leading B",
                "contains ANSWER: A",
                "contains ANSWER: B",
                "contains OPTION A",
                "contains OPTION B",
            ],
            "fallback": "UNKNOWN",
            "unknown_scored_as_correct": False,
        },
        "environment": environment_metadata(model),
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
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", "--annotation-path", default=None)
    parser.add_argument(
        "--annotation_root",
        "--annotation-root",
        default=None,
        help="Evaluate every immediate child annotations.jsonl under this directory with one model load.",
    )
    parser.add_argument("--model_name", "--model-name", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument(
        "--model_revision",
        "--model-revision",
        default=None,
        help="Optional Hugging Face branch, tag, or commit hash. Use a commit hash for long-term reproducibility.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for Python, NumPy, PyTorch, and all CUDA devices.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Require deterministic PyTorch/CUDA algorithms and disable TF32.",
    )
    parser.add_argument(
        "--deterministic_warn_only",
        "--deterministic-warn-only",
        action="store_true",
        help="Warn instead of failing if a deterministic implementation is unavailable.",
    )
    parser.add_argument(
        "--attn_implementation",
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default=None,
        help="Optional fixed attention backend. Use eager for the strongest cross-run reproducibility.",
    )
    parser.add_argument("--dataset_name", "--dataset-name", default=None)
    parser.add_argument(
        "--dataset_name_prefix",
        "--dataset-name-prefix",
        default=None,
        help="Prefix for dataset names when --annotation_root evaluates multiple datasets.",
    )
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
        "--video_num_frames",
        "--video-num-frames",
        type=int,
        default=None,
        help="Optional fixed number of frames sampled from each video. Mutually exclusive with --video_fps.",
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
    parser.add_argument("--max_new_tokens", "--max-new-tokens", type=int, default=10)
    parser.add_argument(
        "--empty_cache_each_sample",
        "--empty-cache-each-sample",
        action="store_true",
        help="Force torch.cuda.empty_cache() after every prompt. Usually slower; use only for memory pressure.",
    )
    parser.add_argument(
        "--disable_video_cache",
        "--disable-video-cache",
        action="store_true",
        help="Decode every prompt video separately instead of reusing the adjacent original/swapped video input.",
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

    if bool(args.annotation_path) == bool(args.annotation_root):
        parser.error("Provide exactly one of --annotation_path or --annotation_root.")
    if args.video_fps is not None and args.video_num_frames is not None:
        parser.error("Use only one temporal sampling control: --video_fps or --video_num_frames.")

    if args.annotation_root:
        annotation_paths = sorted(Path(args.annotation_root).glob("*/annotations.jsonl"))
        if not annotation_paths:
            raise FileNotFoundError(f"No */annotations.jsonl files found under {args.annotation_root}")
    else:
        annotation_paths = [Path(args.annotation_path)]

    configure_reproducibility(
        args.seed,
        deterministic=args.deterministic,
        deterministic_warn_only=args.deterministic_warn_only,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model, processor = load_model(
        args.model_name,
        load_in_4bit=args.load_in_4bit,
        model_revision=args.model_revision,
        attn_implementation=args.attn_implementation,
    )

    for annotation_path in annotation_paths:
        if len(annotation_paths) == 1:
            dataset_name = args.dataset_name or args.experiment_version or annotation_path.parent.name
        else:
            prefix = args.dataset_name_prefix or ""
            dataset_name = f"{prefix}{annotation_path.parent.name}"
        print(f"\nRunning {dataset_name} from {annotation_path}")
        evaluate_annotation(
            model,
            processor,
            args,
            annotation_path,
            dataset_name,
            timestamp,
        )


if __name__ == "__main__":
    main()
