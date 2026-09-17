import argparse
from collections import defaultdict
from pathlib import Path

try:
    from .activation_patching_core import atomic_write_json, atomic_write_jsonl
    from .common import read_jsonl
except ImportError:
    from activation_patching_core import atomic_write_json, atomic_write_jsonl
    from common import read_jsonl


def descending_percentile(values):
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    denominator = max(1, len(values) - 1)
    scores = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + end - 1) / 2
        score = 1.0 - average_rank / denominator
        for index in order[cursor:end]:
            scores[index] = score
        cursor = end
    return scores


def rank_pair(rows):
    eligible = [
        dict(row)
        for row in rows
        if row.get("status") == "ok"
        and row.get("cosine_distance") is not None
        and row.get("relative_l2") is not None
    ]
    if not eligible:
        return []
    cosine_scores = descending_percentile(
        [float(row["cosine_distance"]) for row in eligible]
    )
    l2_scores = descending_percentile(
        [float(row["relative_l2"]) for row in eligible]
    )
    for row, cosine_score, l2_score in zip(eligible, cosine_scores, l2_scores):
        row["cosine_percentile_score"] = cosine_score
        row["relative_l2_percentile_score"] = l2_score
        row["candidate_score"] = (cosine_score + l2_score) / 2
    eligible.sort(
        key=lambda row: (
            -row["candidate_score"],
            -float(row["cosine_distance"]),
            -float(row["relative_l2"]),
            int(row["layer"]),
            row["token_group"],
        )
    )
    return eligible


def select_candidates(rows, top_k_per_pair=6, max_per_token_group=1):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["phase3_pair_id"]].append(row)
    selected = []
    audits = []
    for pair_key in sorted(grouped):
        ranked = rank_pair(grouped[pair_key])
        group_counts = defaultdict(int)
        pair_selected = []
        for row in ranked:
            group = row["token_group"]
            if group_counts[group] >= max_per_token_group:
                continue
            candidate = dict(row)
            candidate["candidate_rank"] = len(pair_selected) + 1
            candidate["selection_rule"] = (
                "mean_of_within_pair_cosine_and_relative_l2_descending_percentiles"
            )
            candidate["top_k_per_pair"] = top_k_per_pair
            candidate["max_per_token_group"] = max_per_token_group
            pair_selected.append(candidate)
            group_counts[group] += 1
            if len(pair_selected) == top_k_per_pair:
                break
        if len(pair_selected) < top_k_per_pair:
            raise ValueError(
                f"Pair {pair_key} yielded only {len(pair_selected)} candidates; "
                f"requested {top_k_per_pair}."
            )
        selected.extend(pair_selected)
        audits.append({
            "phase3_pair_id": pair_key,
            "eligible_locations": len(ranked),
            "selected_locations": len(pair_selected),
            "selected_token_groups": [row["token_group"] for row in pair_selected],
            "selected_layers": [row["layer"] for row in pair_selected],
        })
    return selected, audits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--divergence_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--top_k_per_pair", type=int, default=6)
    parser.add_argument("--max_per_token_group", type=int, default=1)
    args = parser.parse_args()
    if args.top_k_per_pair <= 0 or args.max_per_token_group <= 0:
        parser.error("Candidate limits must be positive.")
    rows = read_jsonl(args.divergence_path)
    selected, audits = select_candidates(
        rows,
        top_k_per_pair=args.top_k_per_pair,
        max_per_token_group=args.max_per_token_group,
    )
    output_path = Path(args.output_path)
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else output_path.with_name(f"{output_path.stem}_summary.json")
    )
    atomic_write_jsonl(output_path, selected)
    atomic_write_json(summary_path, {
        "selection_schema": "phase3_divergence_guided_topk_v1",
        "divergence_path": str(args.divergence_path),
        "top_k_per_pair": args.top_k_per_pair,
        "max_per_token_group": args.max_per_token_group,
        "candidate_rows": len(selected),
        "matched_pairs": len(audits),
        "patch_runs_expected": len(selected) * 2,
        "ranking_rule": (
            "For each matched pair, rank all valid layer/token-group locations by "
            "the mean of descending within-pair percentile scores for cosine distance "
            "and relative L2; greedily retain top-k with a per-token-group cap."
        ),
        "pair_audit": audits,
    })
    print(
        f"Selected {len(selected)} patch candidates across {len(audits)} pairs; "
        f"expected bidirectional patches={len(selected) * 2}."
    )


if __name__ == "__main__":
    main()
