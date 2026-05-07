import json
from pathlib import Path
from collections import defaultdict

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
ANNOTATION_PATH = "/content/vlm-event-boundary/synthetic_boundary_videos/annotations.jsonl"

RESULT_DIR = Path("/content/vlm-event-boundary/results")
RESULT_DIR.mkdir(exist_ok=True)
RAW_RESULT_PATH = RESULT_DIR / "qwen2vl_2b_raw_results.jsonl"


model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)


def load_data(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


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


def ask_model(video_path, option_a, option_b):
    prompt = f"""
Watch the video carefully.

Which statement correctly describes the order of events?

A: {option_a}
B: {option_b}

Answer with only A or B.
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    raw_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    pred = parse_answer(raw_text)
    return pred, raw_text


def main():
    data = load_data(ANNOTATION_PATH)

    # For quick testing, uncomment this:
    # data = data[:6]

    results = []
    stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for item in data:
        print(f"Processing {item['video_id']}")

        pred, raw_response = ask_model(
            item["video_path"],
            item["option_A"],
            item["option_B"]
        )

        correct = pred == item["correct_option"]

        result = {
            "video_id": item["video_id"],
            "video_path": item["video_path"],
            "condition": item["condition"],
            "boundary_type": item.get("boundary_type", ""),
            "correct_option": item["correct_option"],
            "prediction": pred,
            "is_correct": correct,
            "option_A": item["option_A"],
            "option_B": item["option_B"],
            "raw_response": raw_response,
        }

        results.append(result)

        stats[item["condition"]]["total"] += 1
        stats[item["condition"]]["correct"] += int(correct)

        print(
            item["video_id"],
            "pred=", pred,
            "correct=", item["correct_option"],
            "is_correct=", correct,
            "raw=", repr(raw_response)
        )

    with open(RAW_RESULT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved raw results to {RAW_RESULT_PATH}")

    print("\n=== FINAL RESULTS ===")
    for condition, s in stats.items():
        acc = s["correct"] / s["total"]
        print(f"{condition}: {acc:.2f} ({s['correct']}/{s['total']})")


if __name__ == "__main__":
    main()
