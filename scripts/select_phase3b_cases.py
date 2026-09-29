"""Freeze mapping-eligible Phase 3B rescue cases before activation analysis."""

import argparse
from collections import Counter
from pathlib import Path

try:
    from .activation_patching_core import atomic_write_json, atomic_write_jsonl
    from .common import read_jsonl
    from .screen_phase3b_rescues import screen
except ImportError:
    from activation_patching_core import atomic_write_json, atomic_write_jsonl
    from common import read_jsonl
    from screen_phase3b_rescues import screen


CONDITIONS = ("low_boundary", "temporal_boundary")


def choose_independent_cases(candidates, mappings, count):
    grouped = {}
    for candidate in candidates:
        key = (candidate["base_sample_id"], candidate["prompt_variant"])
        if mappings.get(key, {}).get("eligible"):
            base = candidate["base_sample_id"]
            if base not in grouped or candidate["prompt_variant"] == "original":
                grouped[base] = candidate
    by_mover = {
        mover: sorted(
            (row for row in grouped.values() if int(row["first_object_id"]) == mover),
            key=lambda row: row["base_sample_id"],
        )
        for mover in (1, 2)
    }
    selected = []
    while len(selected) < count and any(by_mover.values()):
        mover = 1 if len([row for row in selected if row["first_object_id"] == 1]) <= len([
            row for row in selected if row["first_object_id"] == 2
        ]) else 2
        if not by_mover[mover]:
            mover = 3 - mover
        selected.append(by_mover[mover].pop(0))
    return selected


def manifest_rows(candidates, annotations, results, stratum):
    rows = []
    for candidate in candidates:
        pair_id = f"phase3b_base_{candidate['base_sample_id']:03d}_{candidate['prompt_variant']}"
        for condition in CONDITIONS:
            eval_id = candidate["low_eval_id" if condition == "low_boundary" else "temporal_eval_id"]
            annotation, archived = annotations[eval_id], results[eval_id]
            row = dict(annotation)
            row.update({
                "phase3b_pair_id": pair_id,
                "phase3b_analysis_stratum": stratum,
                "phase3b_prompt_pair_behavior": candidate["outcome"],
                "archived_prediction": archived["prediction"],
                "archived_is_correct": archived["is_correct"],
                "archived_input_metadata": archived.get("input_metadata"),
                "archived_raw_response": archived.get("raw_response"),
            })
            rows.append(row)
    return rows


def write_frozen_rows(path, rows):
    path = Path(path)
    if path.exists() and read_jsonl(path) != rows:
        raise RuntimeError(f"Frozen Phase 3B selection differs: {path}. Use a new output directory.")
    if not path.exists():
        atomic_write_jsonl(path, rows)


def frozen_representatives(primary):
    representatives = {}
    for mover in (1, 2):
        cases = sorted(
            (row for row in primary if int(row["first_object_id"]) == mover),
            key=lambda row: (int(row["base_sample_id"]), row["prompt_variant"]),
        )
        if cases:
            chosen = cases[(len(cases) - 1) // 2]
            representatives[f"target_{mover}_first"] = (
                f"phase3b_base_{chosen['base_sample_id']:03d}_{chosen['prompt_variant']}"
            )
    return representatives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation_paths", nargs="+", required=True)
    parser.add_argument("--result_paths", nargs="+", required=True)
    parser.add_argument("--mapping_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--primary_count", type=int, default=50)
    parser.add_argument("--reserve_count", type=int, default=10)
    parser.add_argument("--control_count", type=int, default=10)
    parser.add_argument("--mirrored_count", type=int, default=5)
    parser.add_argument("--allow_incomplete", action="store_true")
    parser.add_argument(
        "--selection_purpose", choices=("formal_primary", "technical_preflight"),
        default="formal_primary",
    )
    args = parser.parse_args()
    screened, rescue, annotations, results = screen(args.annotation_paths, args.result_paths)
    mappings = {
        (int(row["base_sample_id"]), row["prompt_variant"]): row
        for row in read_jsonl(args.mapping_path)
    }
    selected = choose_independent_cases(rescue, mappings, args.primary_count + args.reserve_count)
    if len(selected) < args.primary_count and not args.allow_incomplete:
        raise ValueError(
            f"Only {len(selected)} independent mapping-eligible rescues; need {args.primary_count}. "
            "Continue screening or use --allow_incomplete for a technical preflight."
        )
    primary = selected[:args.primary_count]
    reserves = selected[args.primary_count:args.primary_count + args.reserve_count]
    rescue_bases = {row["base_sample_id"] for row in rescue}
    controls = [
        row for row in screened
        if row["outcome"] == "stable_both_correct"
        and row["prompt_variant"] == "original"
        and row["base_sample_id"] not in rescue_bases
        and mappings.get((row["base_sample_id"], "original"), {}).get("eligible")
    ][:args.control_count]
    screened_lookup = {(row["base_sample_id"], row["prompt_variant"]): row for row in screened}
    mirrored_controls = []
    independent_mirrored_rescues = []
    for primary_case in primary:
        if len(mirrored_controls) + len(independent_mirrored_rescues) >= args.mirrored_count:
            break
        opposite = "swapped" if primary_case["prompt_variant"] == "original" else "original"
        key = (primary_case["base_sample_id"], opposite)
        if key in screened_lookup and mappings.get(key, {}).get("eligible"):
            counterpart = screened_lookup[key]
            destination = (
                independent_mirrored_rescues if counterpart["outcome"] == "temporal_rescue"
                else mirrored_controls
            )
            destination.append(counterpart)
    output_dir = Path(args.output_dir)
    for name, cases, stratum in (
        ("case_manifest.jsonl", primary, "primary_rescue"),
        ("reserve_case_manifest.jsonl", reserves, "reserve_rescue"),
        ("control_case_manifest.jsonl", controls, "stable_both_correct_control"),
        ("mirrored_case_manifest.jsonl", mirrored_controls, "mirrored_prompt_control"),
        ("independent_mirrored_rescue_manifest.jsonl", independent_mirrored_rescues, "independent_mirrored_rescue"),
    ):
        write_frozen_rows(output_dir / name, manifest_rows(cases, annotations, results, stratum))
    preflight = [
        next((row for row in primary if int(row["first_object_id"]) == mover), None)
        for mover in (1, 2)
    ]
    preflight = [row for row in preflight if row is not None]
    write_frozen_rows(
        output_dir / "preflight_case_manifest.jsonl",
        manifest_rows(preflight, annotations, results, "primary_rescue"),
    )

    eligible_b = sorted({row["base_sample_id"] for row in rescue if row["prompt_variant"] == "swapped" and mappings.get((row["base_sample_id"], "swapped"), {}).get("eligible")})
    eligible_a = sorted({row["base_sample_id"] for row in rescue if row["prompt_variant"] == "original" and mappings.get((row["base_sample_id"], "original"), {}).get("eligible")})
    extension = []
    if len(eligible_b) >= 25:
        b_bases = set(eligible_b[:25])
        a_bases = [base for base in eligible_a if base not in b_bases][:25]
        if len(a_bases) == 25:
            lookup = {(row["base_sample_id"], row["prompt_variant"]): row for row in rescue}
            extension = [lookup[(base, "original")] for base in a_bases]
            extension += [lookup[(base, "swapped")] for base in sorted(b_bases)]
    write_frozen_rows(
        output_dir / "balanced_extension_manifest.jsonl",
        manifest_rows(extension, annotations, results, "balanced_extension"),
    )
    write_frozen_rows(
        output_dir / "analysis_case_manifest.jsonl",
        manifest_rows(primary, annotations, results, "primary_rescue")
        + manifest_rows(mirrored_controls, annotations, results, "mirrored_prompt_control")
        + manifest_rows(independent_mirrored_rescues, annotations, results, "independent_mirrored_rescue")
        + manifest_rows(controls, annotations, results, "stable_both_correct_control"),
    )
    selected_keys = {
        (row["base_sample_id"], row["prompt_variant"])
        for row in primary + reserves + controls + mirrored_controls + independent_mirrored_rescues + extension
    }
    write_frozen_rows(
        output_dir / "selected_video_mappings.jsonl",
        [mappings[key] for key in sorted(selected_keys)],
    )
    summary = {
        "schema": "phase3b_case_selection_v1",
        "selection_purpose": args.selection_purpose,
        "requested_primary_count": args.primary_count,
        "representative_selection_method": "median_base_id_within_first_mover_stratum_before_patching",
        "representative_pair_ids": frozen_representatives(primary),
        "primary_bases": [row["base_sample_id"] for row in primary],
        "reserve_bases": [row["base_sample_id"] for row in reserves],
        "control_bases": [row["base_sample_id"] for row in controls],
        "mirrored_control_bases": [row["base_sample_id"] for row in mirrored_controls],
        "independent_mirrored_rescue_bases": [row["base_sample_id"] for row in independent_mirrored_rescues],
        "mirrored_prompt_pair_behaviors": dict(Counter(
            row["outcome"] for row in mirrored_controls + independent_mirrored_rescues
        )),
        "primary_prompt_variants": dict(Counter(row["prompt_variant"] for row in primary)),
        "primary_first_mover": dict(Counter(row["first_object_id"] for row in primary)),
        "preflight_first_movers": [row["first_object_id"] for row in preflight],
        "eligible_independent_rescue_bases": len({row["base_sample_id"] for row in rescue if mappings.get((row["base_sample_id"], row["prompt_variant"]), {}).get("eligible")}),
        "eligible_b_rescue_bases": len(eligible_b),
        "balanced_extension_cases": len(extension),
        "mapping_path": args.mapping_path,
        "annotation_paths": args.annotation_paths,
        "result_paths": args.result_paths,
    }
    atomic_write_json(output_dir / "case_selection_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
