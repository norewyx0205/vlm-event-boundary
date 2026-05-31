import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_rows(root):
    rows = []
    for ann_path in sorted(Path(root).glob("level_*/annotations.jsonl")):
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def target_identity(row):
    return tuple(
        (obj["id"], obj["color"], obj["shape"])
        for obj in sorted(row["target_objects"], key=lambda item: item["id"])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ladder_v1")
    args = parser.parse_args()

    rows = load_rows(args.root)
    if not rows:
        raise FileNotFoundError(f"No ladder annotations found under {args.root}")

    by_level = defaultdict(list)
    by_video = defaultdict(list)
    identity_by_base = defaultdict(set)
    identity_by_base_condition = defaultdict(set)

    for row in rows:
        by_level[row["difficulty_name"]].append(row)
        by_video[(row["difficulty_name"], row["video_id"])].append(row)

        if row["prompt_variant"] == "original":
            identity = target_identity(row)
            identity_by_base[row["base_sample_id"]].add(identity)
            identity_by_base_condition[(row["base_sample_id"], row["condition"])].add(identity)

    bad_video_pairs = []
    for key, pair in by_video.items():
        variants = {row["prompt_variant"]: row for row in pair}
        if len(pair) != 2 or set(variants) != {"original", "swapped"}:
            bad_video_pairs.append(key)
            continue
        original = variants["original"]
        swapped = variants["swapped"]
        if not (
            original["correct_option"] == "A"
            and swapped["correct_option"] == "B"
            and original["option_A"] == swapped["option_B"]
            and original["option_B"] == swapped["option_A"]
        ):
            bad_video_pairs.append(key)

    bad_base_identity = {
        base_id: identities
        for base_id, identities in identity_by_base.items()
        if len(identities) != 1
    }
    bad_base_condition_identity = {
        key: identities
        for key, identities in identity_by_base_condition.items()
        if len(identities) != 1
    }

    print(f"rows: {len(rows)}")
    print(f"videos: {len(by_video)}")
    print(f"mirrored_prompt_pairs_ok: {not bad_video_pairs}")
    print(f"target_identity_consistent_by_base: {not bad_base_identity}")
    print(f"target_identity_consistent_by_base_condition: {not bad_base_condition_identity}")

    for level_name, level_rows in sorted(by_level.items()):
        level_videos = {
            row["video_id"]
            for row in level_rows
            if row["prompt_variant"] == "original"
        }
        print(f"\n{level_name}")
        print(f"  eval_rows: {len(level_rows)}")
        print(f"  videos: {len(level_videos)}")
        print(f"  correct_options: {dict(Counter(row['correct_option'] for row in level_rows))}")
        print(f"  correct_relations: {dict(Counter(row['correct_relation'] for row in level_rows))}")
        print(f"  first_object_ids: {dict(Counter(row['first_object_id'] for row in level_rows))}")
        print(f"  static_distractors: {dict(Counter(row['static_distractor_count'] for row in level_rows))}")
        print(f"  moving_distractors: {dict(Counter(row['moving_distractor_count'] for row in level_rows))}")

    if bad_video_pairs or bad_base_identity or bad_base_condition_identity:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
