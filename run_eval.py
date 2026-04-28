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

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=10)
    text = processor.decode(outputs[0], skip_special_tokens=True)

    # 简单解析
    text = text.strip().upper()
    if "A" in text and "B" not in text:
        return "A"
    if "B" in text and "A" not in text:
        return "B"

    return "UNKNOWN"


# ====== run ======
for item in data:
    pred = ask_model(
        item["video_path"],
        item["option_A"],
        item["option_B"]
    )

    correct = (pred == item["correct_option"])

    results.append({
        "condition": item["condition"],
        "correct": correct
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