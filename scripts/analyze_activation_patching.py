import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .activation_patching_core import atomic_write_json
    from .common import read_jsonl
except ImportError:
    from activation_patching_core import atomic_write_json
    from common import read_jsonl


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def adjacent_errors(path):
    path = Path(path)
    errors_path = path.with_name(f"{path.stem}_errors.json")
    if not errors_path.is_file():
        return [], str(errors_path)
    payload = json.loads(errors_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else [], str(errors_path)


def mean(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def median(values):
    values = [float(value) for value in values if value is not None]
    return float(np.median(values)) if values else None


def summarize(rows, keys, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    output = []
    for group_key, items in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        record = dict(zip(keys, group_key))
        record["n"] = len(items)
        for metric in metrics:
            values = [item.get(metric) for item in items]
            record[f"mean_{metric}"] = mean(values)
            record[f"median_{metric}"] = median(values)
        output.append(record)
    return output


def tied_ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2 + 1
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def spearman(left, right):
    pairs = [
        (float(x), float(y))
        for x, y in zip(left, right)
        if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 3:
        return None
    x_rank = np.asarray(tied_ranks([item[0] for item in pairs]), dtype=float)
    y_rank = np.asarray(tied_ranks([item[1] for item in pairs]), dtype=float)
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def correlations(patch_rows):
    groups = {"all": patch_rows}
    for direction in sorted({row.get("patch_direction") for row in patch_rows}):
        groups[direction] = [row for row in patch_rows if row.get("patch_direction") == direction]
    for stratum in sorted({row.get("analysis_stratum") for row in patch_rows}):
        groups[f"analysis_stratum:{stratum}"] = [
            row for row in patch_rows if row.get("analysis_stratum") == stratum
        ]
    for pair_key in sorted({row.get("phase3_pair_id") for row in patch_rows}):
        groups[f"pair:{pair_key}"] = [
            row for row in patch_rows if row.get("phase3_pair_id") == pair_key
        ]
    output = []
    for label, rows in groups.items():
        output.append({
            "subset": label,
            "n": len(rows),
            "spearman_cosine_vs_source_aligned_effect": spearman(
                [row.get("cosine_distance") for row in rows],
                [row.get("source_aligned_patch_effect") for row in rows],
            ),
            "spearman_relative_l2_vs_source_aligned_effect": spearman(
                [row.get("relative_l2") for row in rows],
                [row.get("source_aligned_patch_effect") for row in rows],
            ),
        })
    return output


def patch_method_family(row):
    if row.get("patch_method") == "positionwise_replace":
        return "standard_position_aligned_activation_patching"
    return "exploratory_pooled_group_mean_delta"


def normalize_analysis_metadata(rows):
    output = []
    for source in rows:
        row = dict(source)
        row["analysis_stratum"] = (
            row.get("analysis_stratum") or row.get("case_category") or "unclassified"
        )
        row["prompt_pair_behavior"] = row.get("prompt_pair_behavior") or "unavailable"
        row["divergence_stratum"] = row.get("divergence_stratum") or "unavailable"
        row["selection_role"] = row.get("selection_role") or "unavailable"
        row["intervention_family"] = row.get("intervention_family") or patch_method_family(row)
        row["standard_activation_patching"] = (
            row.get("patch_method") == "positionwise_replace"
        )
        output.append(row)
    return output


def setup_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    })
    return plt


def plot_divergence_by_layer(rows, output_path):
    plt = setup_plotting()
    valid = [row for row in rows if row.get("status") == "ok"]
    layers = sorted({int(row["layer"]) for row in valid})
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("cosine_distance", "relative_l2"),
        ("Cosine distance", "Relative L2 change"),
    ):
        values = [mean(row.get(metric) for row in valid if int(row["layer"]) == layer) for layer in layers]
        axis.plot(layers, values, color="#176B87", marker="o", markersize=3, linewidth=1.8)
        axis.set_title(title)
        axis.set_xlabel("Decoder layer")
        axis.set_ylabel("Mean pairwise divergence")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Low-boundary versus temporal-boundary representational divergence")
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_divergence_by_group(rows, output_path):
    plt = setup_plotting()
    valid = [row for row in rows if row.get("status") == "ok"]
    groups = sorted({row["token_group"] for row in valid})
    cosine = [mean(row["cosine_distance"] for row in valid if row["token_group"] == group) for group in groups]
    l2 = [mean(row["relative_l2"] for row in valid if row["token_group"] == group) for group in groups]
    y = np.arange(len(groups))
    figure, axes = plt.subplots(1, 2, figsize=(12, max(5, len(groups) * 0.38)), constrained_layout=True)
    axes[0].barh(y, cosine, color="#176B87")
    axes[1].barh(y, l2, color="#C47A1C")
    for axis, title in zip(axes, ("Cosine distance", "Relative L2 change")):
        axis.set_yticks(y, groups)
        axis.invert_yaxis()
        axis.set_xlabel("Mean divergence")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Representational divergence by token group")
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_patch_heatmap(rows, output_path):
    plt = setup_plotting()
    groups = sorted({row["token_group"] for row in rows})
    layers = sorted({int(row["layer"]) for row in rows})
    matrix = np.full((len(groups), len(layers)), np.nan)
    for y, group in enumerate(groups):
        for x, layer in enumerate(layers):
            matrix[y, x] = mean(
                row.get("source_aligned_patch_effect")
                for row in rows
                if row["token_group"] == group and int(row["layer"]) == layer
            )
    limit = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    limit = max(float(limit), 1e-6)
    figure, axis = plt.subplots(figsize=(12, max(5, len(groups) * 0.4)), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set_yticks(range(len(groups)), groups)
    axis.set_xticks(range(len(layers)), layers)
    axis.set_xlabel("Decoder layer")
    axis.set_title(
        "Source-aligned causal patch effect at selected locations\n"
        "Positive means movement toward the source condition"
    )
    figure.colorbar(image, ax=axis, label="Aligned change in correct-minus-incorrect margin")
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_divergence_vs_effect(rows, output_path):
    plt = setup_plotting()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = {"temporal_to_low": "#2A9D55", "low_to_temporal": "#8E3B8F"}
    markers = {"high": "o", "medium": "s", "low": "^", "unavailable": "x"}
    for axis, metric, label in zip(
        axes,
        ("cosine_distance", "relative_l2"),
        ("Cosine distance", "Relative L2 change"),
    ):
        for direction in ("temporal_to_low", "low_to_temporal"):
            for stratum in ("high", "medium", "low", "unavailable"):
                subset = [
                    row for row in rows
                    if row.get("patch_direction") == direction
                    and row.get("divergence_stratum") == stratum
                ]
                if not subset:
                    continue
                axis.scatter(
                    [row[metric] for row in subset],
                    [row["source_aligned_patch_effect"] for row in subset],
                    s=30,
                    alpha=0.72,
                    marker=markers[stratum],
                    color=colors[direction],
                    label=f"{direction.replace('_', ' ')} | {stratum}",
                )
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.set_xlabel(label)
        axis.set_ylabel("Source-aligned patch effect")
        axis.grid(alpha=0.2)
    axes[1].legend(frameon=False)
    figure.suptitle(
        "Divergence versus position-aligned patch effect\n"
        "High candidates are primary; medium/low candidates diagnose range restriction"
    )
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_patch_effect_by_analysis_stratum(rows, output_path):
    plt = setup_plotting()
    preferred_order = (
        "primary_original_rescue",
        "swapped_independent_rescue",
        "mirrored_prompt_control",
        "stable_both_correct_control",
        "original_prompt_other",
        "unclassified",
    )
    available = {row.get("analysis_stratum") for row in rows}
    strata = [item for item in preferred_order if item in available]
    strata.extend(sorted(available - set(strata)))
    directions = ("temporal_to_low", "low_to_temporal")
    colors = ("#2A9D55", "#8E3B8F")
    x = np.arange(len(strata))
    width = 0.36
    figure, axis = plt.subplots(
        figsize=(max(8, len(strata) * 1.7), 5.2), constrained_layout=True
    )
    for offset, direction, color in zip((-width / 2, width / 2), directions, colors):
        values = [
            mean(
                row.get("source_aligned_patch_effect")
                for row in rows
                if row.get("analysis_stratum") == stratum
                and row.get("patch_direction") == direction
            )
            for stratum in strata
        ]
        axis.bar(
            x + offset,
            [np.nan if value is None else value for value in values],
            width,
            color=color,
            label=direction.replace("_", " "),
        )
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set_xticks(x, [item.replace("_", "\n") for item in strata])
    axis.set_ylabel("Mean source-aligned margin change")
    axis.set_title("Position-aligned patch effects by behavioural analysis stratum")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--divergence_path", required=True)
    parser.add_argument("--candidate_path", required=True)
    parser.add_argument("--patching_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require_complete", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    divergence = read_jsonl(args.divergence_path)
    candidates = read_jsonl(args.candidate_path)
    patches = normalize_analysis_metadata(
        row for row in read_jsonl(args.patching_path) if row.get("status") == "ok"
    )
    positionwise_patches = [
        row for row in patches if row["standard_activation_patching"]
    ]
    pooled_mean_delta_patches = [
        row for row in patches if not row["standard_activation_patching"]
    ]
    primary_rescue_patches = [
        row for row in positionwise_patches
        if row.get("analysis_stratum") == "primary_original_rescue"
        and row.get("divergence_stratum") == "high"
    ]
    mirrored_control_patches = [
        row for row in positionwise_patches
        if row.get("analysis_stratum") == "mirrored_prompt_control"
    ]
    stable_control_patches = [
        row for row in positionwise_patches
        if row.get("analysis_stratum") == "stable_both_correct_control"
    ]
    swapped_rescue_patches = [
        row for row in positionwise_patches
        if row.get("analysis_stratum") == "swapped_independent_rescue"
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_patch_keys = {
        (
            row["phase3_pair_id"],
            direction,
            int(row["layer"]),
            row["token_group"],
        )
        for row in candidates
        for direction in ("temporal_to_low", "low_to_temporal")
    }
    actual_patch_keys = {
        (
            row["phase3_pair_id"],
            row["patch_direction"],
            int(row["layer"]),
            row["token_group"],
        )
        for row in patches
    }
    missing_patch_keys = sorted(expected_patch_keys - actual_patch_keys)
    missing_patch_rows = [
        {
            "phase3_pair_id": key[0],
            "patch_direction": key[1],
            "layer": key[2],
            "token_group": key[3],
        }
        for key in missing_patch_keys
    ]
    write_csv(
        output_dir / "missing_patch_locations.csv",
        missing_patch_rows,
        fieldnames=("phase3_pair_id", "patch_direction", "layer", "token_group"),
    )
    divergence_errors, divergence_error_path = adjacent_errors(args.divergence_path)
    patch_errors, patch_error_path = adjacent_errors(args.patching_path)
    patch_fieldnames = []
    for row in patches:
        for key in row:
            if key not in patch_fieldnames:
                patch_fieldnames.append(key)

    write_csv(output_dir / "pairwise_divergence.csv", divergence)
    write_csv(output_dir / "patching_results.csv", patches)
    write_csv(
        output_dir / "patching_results_positionwise_primary_method.csv",
        positionwise_patches,
        fieldnames=patch_fieldnames,
    )
    write_csv(
        output_dir / "patching_results_pooled_mean_delta_exploratory.csv",
        pooled_mean_delta_patches,
        fieldnames=patch_fieldnames,
    )
    write_csv(
        output_dir / "patching_results_primary_original_rescue_high.csv",
        primary_rescue_patches,
        fieldnames=patch_fieldnames,
    )
    write_csv(
        output_dir / "patching_results_mirrored_prompt_controls.csv",
        mirrored_control_patches,
        fieldnames=patch_fieldnames,
    )
    write_csv(
        output_dir / "patching_results_stable_both_correct_controls.csv",
        stable_control_patches,
        fieldnames=patch_fieldnames,
    )
    write_csv(
        output_dir / "patching_results_swapped_independent_rescue.csv",
        swapped_rescue_patches,
        fieldnames=patch_fieldnames,
    )
    write_csv(output_dir / "selected_patch_candidates.csv", candidates)
    divergence_layer = summarize(
        [row for row in divergence if row.get("status") == "ok"],
        ["layer"],
        ["cosine_distance", "relative_l2"],
    )
    divergence_group = summarize(
        [row for row in divergence if row.get("status") == "ok"],
        ["token_group"],
        ["cosine_distance", "relative_l2"],
    )
    patch_summary = summarize(
        positionwise_patches,
        ["patch_direction", "layer", "token_group"],
        ["margin_change", "source_aligned_patch_effect"],
    )
    divergence_case_summary = summarize(
        [row for row in divergence if row.get("status") == "ok"],
        [
            "analysis_stratum",
            "prompt_pair_behavior",
            "prompt_variant",
            "token_group",
        ],
        ["cosine_distance", "relative_l2"],
    )
    patch_case_summary = summarize(
        positionwise_patches,
        [
            "analysis_stratum",
            "prompt_pair_behavior",
            "divergence_stratum",
            "patch_direction",
        ],
        ["margin_change", "source_aligned_patch_effect"],
    )
    method_summary = summarize(
        patches,
        ["intervention_family", "analysis_stratum", "patch_direction"],
        ["margin_change", "source_aligned_patch_effect"],
    )
    correlation_rows = correlations(positionwise_patches)
    write_csv(output_dir / "divergence_by_layer.csv", divergence_layer)
    write_csv(output_dir / "divergence_by_token_group.csv", divergence_group)
    write_csv(output_dir / "patch_effect_by_layer_token_group.csv", patch_summary)
    write_csv(output_dir / "divergence_by_case_prompt_group.csv", divergence_case_summary)
    write_csv(output_dir / "patch_effect_by_case_prompt.csv", patch_case_summary)
    write_csv(output_dir / "patch_effect_by_intervention_family.csv", method_summary)
    write_csv(output_dir / "divergence_patch_correlations.csv", correlation_rows)

    if args.plots:
        plot_divergence_by_layer(divergence, output_dir / "divergence_by_layer.png")
        plot_divergence_by_group(divergence, output_dir / "divergence_by_token_group.png")
        if positionwise_patches:
            plot_patch_heatmap(
                positionwise_patches,
                output_dir / "patch_effect_by_layer_token_group.png",
            )
            plot_divergence_vs_effect(
                positionwise_patches,
                output_dir / "divergence_vs_patch_effect.png",
            )
            plot_patch_effect_by_analysis_stratum(
                positionwise_patches,
                output_dir / "patch_effect_by_analysis_stratum.png",
            )
        if primary_rescue_patches:
            plot_patch_heatmap(
                primary_rescue_patches,
                output_dir / "patch_effect_primary_original_rescue_high.png",
            )

    summary = {
        "analysis_schema": "phase3_activation_patching_analysis_v2_stratified",
        "divergence_rows": len(divergence),
        "valid_divergence_rows": sum(row.get("status") == "ok" for row in divergence),
        "missing_group_rows": sum(
            row.get("status") == "missing_token_group" for row in divergence
        ),
        "excluded_unmatched_phase_rows": sum(
            row.get("status") == "excluded_unmatched_phase" for row in divergence
        ),
        "candidate_rows": len(candidates),
        "patch_rows": len(patches),
        "positionwise_patch_rows": len(positionwise_patches),
        "pooled_mean_delta_exploratory_rows": len(pooled_mean_delta_patches),
        "primary_original_rescue_high_rows": len(primary_rescue_patches),
        "mirrored_prompt_control_rows": len(mirrored_control_patches),
        "stable_both_correct_control_rows": len(stable_control_patches),
        "swapped_independent_rescue_rows": len(swapped_rescue_patches),
        "expected_patch_rows": len(expected_patch_keys),
        "missing_patch_rows": len(missing_patch_keys),
        "complete_patch_matrix": not missing_patch_keys,
        "divergence_failures": len(divergence_errors),
        "patch_failures": len(patch_errors),
        "divergence_error_manifest": divergence_error_path,
        "patch_error_manifest": patch_error_path,
        "matched_pairs": len({row.get("phase3_pair_id") for row in divergence}),
        "categorical_flips": sum(
            bool(row.get("categorical_flip")) for row in positionwise_patches
        ),
        "flips_toward_correct": sum(
            bool(row.get("flip_toward_correct")) for row in positionwise_patches
        ),
        "flips_away_from_correct": sum(
            bool(row.get("flip_away_from_correct")) for row in positionwise_patches
        ),
        "mean_source_aligned_patch_effect": mean(
            row.get("source_aligned_patch_effect") for row in positionwise_patches
        ),
        "analysis_stratum_counts": {
            stratum: sum(
                row.get("analysis_stratum") == stratum
                for row in positionwise_patches
            )
            for stratum in sorted({
                row.get("analysis_stratum") for row in positionwise_patches
            })
        },
        "correlations": correlation_rows,
        "interpretation_note": (
            "Primary causal summaries use only position-aligned replacement. Pooled "
            "mean-delta interventions, if present, are exported separately and are not "
            "treated as standard activation patching. High-divergence original-rescue "
            "cases are primary; medium/low candidates support only exploratory "
            "divergence-effect assessment, and mirrored/stable cases are controls."
        ),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if args.require_complete and (
        missing_patch_keys or divergence_errors or patch_errors
    ):
        raise RuntimeError(
            "Phase 3 analysis found incomplete causal pairs or unresolved GPU "
            "failures. Details were saved in summary.json and "
            "missing_patch_locations.csv."
        )


if __name__ == "__main__":
    main()
