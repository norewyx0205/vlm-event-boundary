import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def slugify(text):
    keep = []
    for ch in str(text):
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


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_counts(stats):
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
