import argparse
from pathlib import Path

try:
    from .common import read_jsonl, write_jsonl
except ImportError:
    from common import read_jsonl, write_jsonl


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

    output_rows = []
    for row in read_jsonl(input_path):
        output_rows.extend(make_eval_rows(row))

    write_jsonl(output_path, output_rows)
    print(f"Wrote {len(output_rows)} mirrored evaluation rows to {output_path}")


if __name__ == "__main__":
    main()
