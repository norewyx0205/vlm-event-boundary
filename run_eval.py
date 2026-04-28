import json
from collections import defaultdict

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# ====== load model ======
# model_name = "Qwen/Qwen2-VL-7B-Instruct" #too heavy to download, try 2B instead, as below
model_name = "Qwen/Qwen2-VL-2B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_name)

# ====== load data ======
ANNOTATION_PATH = "synthetic_boundary_videos/annotations.jsonl"

data = []
with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

# ====== evaluation ======
results = []

from qwen_vl_utils import process_vision_info

def ask_model(video_path, option_a, option_b):
    prompt = f"""
Watch the video carefully.

Which description correctly matches the order of events?

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

    generated_ids = model.generate(**inputs, max_new_tokens=10)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    raw_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    raw_upper = raw_text.strip().upper()

    if raw_upper.startswith("A") or "ANSWER: A" in raw_upper:
        pred = "A"
    elif raw_upper.startswith("B") or "ANSWER: B" in raw_upper:
        pred = "B"
    else:
        pred = "UNKNOWN"

    return pred, raw_text

# ====== run ======
for item in data:
    pred, raw_response = ask_model(
        item["video_path"],
        item["option_A"],
        item["option_B"]
    )

    correct = (pred == item["correct_option"])

    results.append({
        "condition": item["condition"],
        "correct": correct,
        "raw_response": raw_response
    })

    print(item["video_id"], pred, item["correct_option"], correct)

# ====== aggregate ======
stats = defaultdict(lambda: {"total": 0, "correct": 0})

for r in results:
    stats[r["condition"]]["total"] += 1
    if r["correct"]:
        stats[r["condition"]]["correct"] += 1

print("\n=== FINAL RESULTS ===")

for cond, s in stats.items():
    acc = s["correct"] / s["total"]
    print(f"{cond}: {acc:.2f} ({s['correct']}/{s['total']})")