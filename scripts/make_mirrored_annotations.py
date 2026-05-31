import argparse
import json
from pathlib import Path


def make_eval_rows(row):
    if "correct_sentence" not in row or "incorrect_sentence" not in row:
        raise ValueError("Input rows must contain correct_sentence and incorrect_sentence.")

    base_eval_id = row.get("eval_id_base") or row.get("eval_id") or Path(row["video_id"]).stem

    original = dict(row)
    original.update({
        "eval_id": f"{base_eval_id}_original",
        "prompt_variant": "original",
        "option_A": row["correct_sentence"],
        "option_B": row["incorrect_sentence"],
        "correct_option": "A",
    })

    swapped = dict(row)
    swapped.update({
        "eval_id": f"{base_eval_id}_swapped",
        "prompt_variant": "swapped",
        "option_A": row["incorrect_sentence"],
        "option_B": row["correct_sentence"],
        "correct_option": "B",
    })

    original.pop("eval_id_base", None)
    swapped.pop("eval_id_base", None)
    return [original, swapped]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Video-level JSONL annotation file.")
    parser.add_argument("--output_path", required=True, help="Evaluation-level mirrored JSONL output file.")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with open(input_path, "r", encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            for eval_row in make_eval_rows(row):
                dst.write(json.dumps(eval_row, ensure_ascii=False) + "\n")
                rows_written += 1

    print(f"Wrote {rows_written} mirrored evaluation rows to {output_path}")


if __name__ == "__main__":
    main()
