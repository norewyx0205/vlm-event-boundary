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


def rank_pair(rows, required_patch_method="positionwise_replace"):
    eligible = [
        dict(row)
        for row in rows
        if row.get("status") == "ok"
        and row.get("cosine_distance") is not None
        and row.get("relative_l2") is not None
        and row.get("patch_method_eligibility") == required_patch_method
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
    denominator = max(1, len(eligible) - 1)
    for index, row in enumerate(eligible):
        row["divergence_rank_fraction"] = index / denominator
        if index / denominator < 1 / 3:
            row["divergence_stratum"] = "high"
        elif index / denominator < 2 / 3:
            row["divergence_stratum"] = "medium"
        else:
            row["divergence_stratum"] = "low"
    return eligible


def stratified_group_capped_selection(ranked, quotas, max_per_token_group):
    options = {}
    for stratum, _, _ in quotas:
        rows = [row for row in ranked if row["divergence_stratum"] == stratum]
        ordered = list(reversed(rows)) if stratum == "low" else rows
        retained = []
        retained_by_group = defaultdict(int)
        for row in ordered:
            group = row["token_group"]
            if retained_by_group[group] >= max_per_token_group:
                continue
            retained.append(row)
            retained_by_group[group] += 1
        options[stratum] = retained
    slots = [
        (stratum, selection_role, slot_index)
        for stratum, quota, selection_role in quotas
        for slot_index in range(quota)
    ]
    search_order = sorted(
        range(len(slots)),
        key=lambda index: (len(options[slots[index][0]]), index),
    )
    assignment = {}
    group_counts = defaultdict(int)
    selected_locations = set()

    def search(cursor):
        if cursor == len(search_order):
            return True
        slot_index = search_order[cursor]
        stratum, _, _ = slots[slot_index]
        for row in options[stratum]:
            location = (int(row["layer"]), row["token_group"])
            group = row["token_group"]
            if location in selected_locations or group_counts[group] >= max_per_token_group:
                continue
            assignment[slot_index] = row
            selected_locations.add(location)
            group_counts[group] += 1
            if search(cursor + 1):
                return True
            group_counts[group] -= 1
            selected_locations.remove(location)
            del assignment[slot_index]
        return False

    if not search(0):
        availability = {
            stratum: len({row["token_group"] for row in options[stratum]})
            for stratum, _, _ in quotas
        }
        raise ValueError(
            "Could not satisfy stratified candidate quotas under the token-group "
            f"cap; available groups by stratum={availability}."
        )
    return [
        (assignment[index], slots[index][1])
        for index in range(len(slots))
    ]


def select_candidates(
    rows,
    top_k_per_pair=6,
    max_per_token_group=1,
    medium_k_per_pair=1,
    low_k_per_pair=1,
    required_patch_method="positionwise_replace",
):
    high_k_per_pair = top_k_per_pair - medium_k_per_pair - low_k_per_pair
    if high_k_per_pair <= 0:
        raise ValueError(
            "top_k_per_pair must exceed the combined medium/low comparison quotas."
        )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["phase3_pair_id"]].append(row)
    selected = []
    audits = []
    for pair_key in sorted(grouped):
        ranked = rank_pair(grouped[pair_key], required_patch_method)
        pair_selected = []
        quotas = (
            ("high", high_k_per_pair, "primary_high_divergence"),
            ("medium", medium_k_per_pair, "range_comparison_medium"),
            ("low", low_k_per_pair, "range_comparison_low"),
        )
        try:
            chosen = stratified_group_capped_selection(
                ranked, quotas, max_per_token_group
            )
        except ValueError as exc:
            raise ValueError(f"Pair {pair_key}: {exc}") from exc
        for row, selection_role in chosen:
            candidate = dict(row)
            candidate["candidate_rank"] = len(pair_selected) + 1
            candidate["selection_role"] = selection_role
            candidate["selection_rule"] = (
                "fixed_budget_stratified_divergence_sampling_with_group_cap"
            )
            candidate["top_k_per_pair"] = top_k_per_pair
            candidate["high_k_per_pair"] = high_k_per_pair
            candidate["medium_k_per_pair"] = medium_k_per_pair
            candidate["low_k_per_pair"] = low_k_per_pair
            candidate["max_per_token_group"] = max_per_token_group
            candidate["required_patch_method"] = required_patch_method
            pair_selected.append(candidate)
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
            "selected_divergence_strata": [
                row["divergence_stratum"] for row in pair_selected
            ],
            "selected_roles": [row["selection_role"] for row in pair_selected],
            "excluded_non_positionwise_locations": sum(
                row.get("status") == "ok"
                and row.get("patch_method_eligibility") != required_patch_method
                for row in grouped[pair_key]
            ),
        })
    return selected, audits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--divergence_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--top_k_per_pair", type=int, default=6)
    parser.add_argument("--max_per_token_group", type=int, default=1)
    parser.add_argument("--medium_k_per_pair", type=int, default=1)
    parser.add_argument("--low_k_per_pair", type=int, default=1)
    parser.add_argument(
        "--required_patch_method",
        default="positionwise_replace",
        choices=("positionwise_replace", "pooled_mean_delta"),
    )
    args = parser.parse_args()
    if (
        args.top_k_per_pair <= 0
        or args.max_per_token_group <= 0
        or args.medium_k_per_pair < 0
        or args.low_k_per_pair < 0
    ):
        parser.error("Candidate limits must be positive.")
    rows = read_jsonl(args.divergence_path)
    selected, audits = select_candidates(
        rows,
        top_k_per_pair=args.top_k_per_pair,
        max_per_token_group=args.max_per_token_group,
        medium_k_per_pair=args.medium_k_per_pair,
        low_k_per_pair=args.low_k_per_pair,
        required_patch_method=args.required_patch_method,
    )
    output_path = Path(args.output_path)
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else output_path.with_name(f"{output_path.stem}_summary.json")
    )
    atomic_write_jsonl(output_path, selected)
    atomic_write_json(summary_path, {
        "selection_schema": "phase3_stratified_position_aligned_v2",
        "divergence_path": str(args.divergence_path),
        "top_k_per_pair": args.top_k_per_pair,
        "max_per_token_group": args.max_per_token_group,
        "high_k_per_pair": (
            args.top_k_per_pair - args.medium_k_per_pair - args.low_k_per_pair
        ),
        "medium_k_per_pair": args.medium_k_per_pair,
        "low_k_per_pair": args.low_k_per_pair,
        "required_patch_method": args.required_patch_method,
        "candidate_rows": len(selected),
        "matched_pairs": len(audits),
        "patch_runs_expected": len(selected) * 2,
        "ranking_rule": (
            "For each matched pair, rank position-aligned locations by the mean of "
            "within-pair cosine and relative-L2 percentile scores. Preserve a fixed "
            "six-location budget: four high-divergence primary candidates plus one "
            "medium and one low comparison candidate, with a token-group cap."
        ),
        "pair_audit": audits,
    })
    print(
        f"Selected {len(selected)} patch candidates across {len(audits)} pairs; "
        f"expected bidirectional patches={len(selected) * 2}."
    )


if __name__ == "__main__":
    main()
