"""Merge Phase 3B shards and summarize all-layer divergence and patch effects."""

import argparse
import hashlib
import json
from collections import defaultdict
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .activation_patching_core import atomic_write_json, atomic_write_jsonl
    from .analyze_activation_patching import spearman, write_csv
    from .common import read_jsonl
    from .phase3b_core import GROUPS, PATCH_LAYERS, SCHEMA
except ImportError:
    from activation_patching_core import atomic_write_json, atomic_write_jsonl
    from analyze_activation_patching import spearman, write_csv
    from common import read_jsonl
    from phase3b_core import GROUPS, PATCH_LAYERS, SCHEMA


DIRECTIONS = ("temporal_to_low", "low_to_temporal")
ROLES = (
    "first_mover_own_event", "first_mover_during_event_2",
    "second_mover_during_event_1", "second_mover_own_event",
)


def read_shards(root):
    root = Path(root)
    divergence, patches, captures, configs, technical_controls = [], [], {}, [], {}
    for shard in sorted(root.glob("shard_*")):
        config_path = shard / "run_config.json"
        if not config_path.exists():
            continue
        configs.append(json.loads(config_path.read_text(encoding="utf-8")))
        fingerprint = configs[-1]["run_fingerprint"]
        for path in sorted((shard / "divergence").glob("*.jsonl")):
            items = read_jsonl(path)
            if any(row.get("run_fingerprint") != fingerprint for row in items):
                raise ValueError(f"Divergence rows have wrong shard fingerprint: {path}.")
            divergence.extend(items)
        for path in sorted((shard / "patches").glob("*/*.jsonl")):
            items = read_jsonl(path)
            if any(row.get("run_fingerprint") != fingerprint for row in items):
                raise ValueError(f"Patch rows have wrong shard fingerprint: {path}.")
            patches.extend(items)
        for path in sorted((shard / "activations").glob("*/*/index.json")):
            info = json.loads(path.read_text(encoding="utf-8"))
            if info.get("run_fingerprint") != fingerprint:
                raise ValueError(f"Capture index has wrong shard fingerprint: {path}.")
            key = (path.parent.parent.name, path.parent.name)
            if key in captures:
                raise ValueError(f"Duplicate capture {key}.")
            captures[key] = info
        for path in sorted((shard / "technical_controls").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("run_fingerprint") != configs[-1].get("run_fingerprint"):
                raise ValueError(f"Technical control has wrong shard fingerprint: {path}.")
            pair_id = payload["phase3b_pair_id"]
            if pair_id in technical_controls:
                raise ValueError(f"Duplicate Phase 3B technical control: {pair_id}.")
            technical_controls[pair_id] = payload
    if not configs:
        raise ValueError("No Phase 3B shard configs found.")
    shared = (
        "schema", "repo_commit", "patching_code_sha256", "manifest_sha256", "mapping_sha256", "shard_size",
        "model_name", "model_revision", "transformers_version", "qwen_vl_utils_version",
        "torch_version", "seed", "validate_controls",
        "video_fps", "video_num_frames", "video_max_pixels", "roi_padding", "attn_implementation",
        "gpu_hardware", "single_gpu", "path_map", "verify_standard_generation",
        "model_parallel", "model_device_map_strategy", "gpu_weight_budget_gib", "model_device_map",
    )
    for config in configs[1:]:
        if any(config.get(key) != configs[0].get(key) for key in shared):
            raise ValueError("Phase 3B shards have incompatible model/processor settings.")
    shard_indices = [config["shard_index"] for config in configs]
    if len(shard_indices) != len(set(shard_indices)):
        raise ValueError("Duplicate Phase 3B shard indices.")
    configured_pairs = [pair for config in configs for pair in config["pair_ids"]]
    if len(configured_pairs) != len(set(configured_pairs)):
        raise ValueError("A Phase 3B pair appears in multiple shards.")
    return divergence, patches, captures, configs, technical_controls


def validate_completeness(divergence, patches, captures, require_complete, expected_pair_ids=None):
    expected_pairs = set(expected_pair_ids) if expected_pair_ids is not None else {pair for pair, _ in captures}
    missing_captures = sorted(
        (pair, condition) for pair in expected_pairs
        for condition in ("low_boundary", "temporal_boundary")
        if (pair, condition) not in captures
    )
    expected = {
        (pair, direction, group, layer)
        for pair in expected_pairs for direction in DIRECTIONS for group in GROUPS
        for layer in PATCH_LAYERS + ((35,) if group == "decision_position" else ())
    }
    observed = [(row["phase3b_pair_id"], row["patch_direction"], row["token_group"], int(row["layer"])) for row in patches]
    if len(observed) != len(set(observed)):
        raise ValueError("Duplicate Phase 3B patch keys across shards.")
    missing = sorted(expected - set(observed))
    divergence_keys = [(row["phase3b_pair_id"], row["token_group"], int(row["layer"])) for row in divergence]
    if len(divergence_keys) != len(set(divergence_keys)):
        raise ValueError("Duplicate Phase 3B divergence keys across shards.")
    expected_divergence = {
        (pair, group, layer) for pair in expected_pairs for group in GROUPS for layer in range(36)
    }
    missing_divergence = sorted(expected_divergence - set(divergence_keys))
    if require_complete and (missing or missing_divergence or missing_captures):
        raise ValueError(
            f"Incomplete Phase 3B: {len(missing_captures)} captures, "
            f"{len(missing_divergence)} divergence rows, {len(missing)} patches missing."
        )
    return missing, missing_divergence, missing_captures


def bootstrap_interval(values, rng, repeats=2000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None, None
    samples = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(samples, (0.025, 0.975)))


def aggregate_patches(rows, seed=42):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("analysis_stratum"), row["patch_direction"], row["token_group"], int(row["layer"]))].append(row)
    rng = np.random.default_rng(seed)
    result = []
    for (stratum, direction, group, layer), items in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        if len({row["base_sample_id"] for row in items}) != len(items):
            raise ValueError(f"Repeated base sample in aggregate {direction}, {group}, {layer}.")
        effects = [float(row["source_aligned_patch_effect"]) for row in items]
        lower, upper = bootstrap_interval(effects, rng)
        recoveries = [row["recovery"] for row in items if row.get("recovery") is not None]
        result.append({
            "analysis_stratum": stratum,
            "patch_direction": direction, "token_group": group, "layer": layer,
            "eligible_n": len(items),
            "mean_aligned_margin_effect": float(np.mean(effects)),
            "median_aligned_margin_effect": float(np.median(effects)),
            "bootstrap_ci_low": lower, "bootstrap_ci_high": upper,
            "categorical_flip_rate": float(np.mean([row["categorical_flip"] for row in items])),
            "flip_toward_correct_rate": float(np.mean([row["flip_toward_correct"] for row in items])),
            "flip_away_from_correct_rate": float(np.mean([row["flip_away_from_correct"] for row in items])),
            "median_recovery": float(np.median(recoveries)) if recoveries else None,
            "mean_mapping_source_coverage": float(np.mean([
                row.get("mapping_source_coverage", 1.0) for row in items
            ])),
            "mean_mapping_target_coverage": float(np.mean([
                row.get("mapping_target_coverage", 1.0) for row in items
            ])),
        })
    return result


def aggregate_divergence(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("analysis_stratum"), row["token_group"], int(row["layer"]))].append(row)
    result = []
    for (stratum, group, layer), items in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        result.append({
            "analysis_stratum": stratum,
            "token_group": group, "layer": layer, "eligible_n": len(items),
            "mean_cosine_distance": float(np.mean([row["cosine_distance"] for row in items])),
            "median_cosine_distance": float(np.median([row["cosine_distance"] for row in items])),
            "mean_relative_l2": float(np.mean([row["relative_l2"] for row in items])),
            "mean_mapping_source_coverage": float(np.mean([
                row.get("mapping_source_coverage", 1.0) for row in items
            ])),
            "mean_mapping_target_coverage": float(np.mean([
                row.get("mapping_target_coverage", 1.0) for row in items
            ])),
        })
    return result


def representative_pairs(captures, manifest_path):
    by_mover = defaultdict(list)
    for pair_id, condition in captures:
        if condition != "low_boundary" or (pair_id, "temporal_boundary") not in captures:
            continue
        low = captures[(pair_id, "low_boundary")]
        if low.get("analysis_stratum") != "primary_rescue":
            continue
        by_mover[int(low["first_object_id"])].append(pair_id)
    summary_path = Path(manifest_path).parent / "case_selection_summary.json"
    if summary_path.is_file():
        frozen = json.loads(summary_path.read_text(encoding="utf-8")).get("representative_pair_ids", {})
        if Path(manifest_path).name != "preflight_case_manifest.jsonl":
            if not frozen:
                raise ValueError(f"Frozen representative case IDs are missing from {summary_path}.")
            missing = {label: pair for label, pair in frozen.items() if (pair, "low_boundary") not in captures}
            if missing:
                raise ValueError(f"Frozen representative cases absent from complete analysis: {missing}.")
            return frozen
    elif Path(manifest_path).name in {"case_manifest.jsonl", "analysis_case_manifest.jsonl"}:
        raise ValueError(f"Frozen representative selection summary is missing: {summary_path}.")
    return {
        f"target_{mover}_first": sorted(by_mover[mover])[0]
        for mover in (1, 2) if by_mover[mover]
    }


def case_correlations(divergence, patches):
    index = {(row["phase3b_pair_id"], row["token_group"], int(row["layer"])): row for row in divergence}
    by_case = defaultdict(list)
    for row in patches:
        match = index.get((row["phase3b_pair_id"], row["token_group"], int(row["layer"])))
        if match:
            by_case[(row["phase3b_pair_id"], row["patch_direction"])].append((match, row))
    output = []
    for (pair_id, direction), items in sorted(by_case.items()):
        output.append({
            "phase3b_pair_id": pair_id, "patch_direction": direction, "locations": len(items),
            "spearman_cosine_vs_effect": spearman(
                [item[0]["cosine_distance"] for item in items],
                [item[1]["source_aligned_patch_effect"] for item in items],
            ),
            "spearman_l2_vs_effect": spearman(
                [item[0]["relative_l2"] for item in items],
                [item[1]["source_aligned_patch_effect"] for item in items],
            ),
        })
    return output


def plot_heatmap(rows, groups, layers, metric, title, path, centered=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup = {(row["token_group"], int(row["layer"])): row.get(metric) for row in rows}
    matrix = np.asarray([[lookup.get((group, layer), np.nan) for layer in layers] for group in groups], dtype=float)
    fig, ax = plt.subplots(figsize=(13, max(5, len(groups) * 0.43)), layout="constrained")
    if centered:
        limit = max(float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1, 1e-6)
        image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    else:
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(groups)), [group.replace("_", " ") for group in groups])
    ax.set_xticks(range(len(layers)), layers)
    ax.set_xlabel("Decoder layer")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=metric.replace("_", " "))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_plots(output, divergence, patches, divergence_table, patch_table, captures, representatives):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir = Path(output) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    divergence = [row for row in divergence if row.get("analysis_stratum") == "primary_rescue"]
    patches = [row for row in patches if row.get("analysis_stratum") == "primary_rescue"]
    divergence_table = [row for row in divergence_table if row.get("analysis_stratum") == "primary_rescue"]
    patch_table = [row for row in patch_table if row.get("analysis_stratum") == "primary_rescue"]
    all_layers = list(range(36))
    plot_heatmap(divergence_table, GROUPS, all_layers, "mean_cosine_distance",
                 "All-layer low/temporal divergence", plot_dir / "aggregate_divergence_heatmap.png")
    for direction in DIRECTIONS:
        selected = [row for row in patch_table if row["patch_direction"] == direction]
        plot_heatmap(selected, GROUPS, list(PATCH_LAYERS) + [35],
                     "mean_aligned_margin_effect", f"Aligned margin effect | {direction}",
                     plot_dir / f"causal_effect_heatmap_{direction}.png", True)
        plot_heatmap(selected, GROUPS, list(PATCH_LAYERS) + [35],
                     "categorical_flip_rate", f"Categorical flip rate | {direction}",
                     plot_dir / f"flip_rate_heatmap_{direction}.png")
    forward = [row for row in patch_table if row["patch_direction"] == "temporal_to_low"]
    plot_heatmap(forward, GROUPS, list(PATCH_LAYERS) + [35], "median_recovery",
                 "Median temporal-margin recovery", plot_dir / "recovery_heatmap.png", True)
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for role in ROLES:
        data = defaultdict(list)
        for row in patches:
            if row["patch_direction"] == "temporal_to_low" and row.get("mover_role") == role:
                data[int(row["layer"])].append(row["source_aligned_patch_effect"])
        if data:
            layers = sorted(data)
            ax.plot(layers, [np.mean(data[layer]) for layer in layers], marker="o", label=role.replace("_", " "))
    ax.axhline(0, color="#666", linewidth=0.8)
    ax.set(xlabel="Decoder layer", ylabel="Aligned correct-option margin change", title="Tracking and binding role patches")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(plot_dir / "tracking_binding_layer_curves.png", dpi=160)
    plt.close(fig)
    lookup = {
        (row["phase3b_pair_id"], row["token_group"], int(row["layer"])): row
        for row in divergence
    }
    joined = [
        (lookup[(row["phase3b_pair_id"], row["token_group"], int(row["layer"]))]["cosine_distance"],
         row["source_aligned_patch_effect"])
        for row in patches
        if row["patch_direction"] == "temporal_to_low"
        and (row["phase3b_pair_id"], row["token_group"], int(row["layer"])) in lookup
    ]
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    if joined:
        x, y = zip(*joined)
        ax.scatter(x, y, alpha=0.45, s=18, color="#276f9a")
    ax.axhline(0, color="#666", linewidth=0.8)
    ax.set(xlabel="Low/temporal cosine distance", ylabel="Temporal-to-low aligned margin effect",
           title="Divergence versus patch effect | exploratory locations")
    fig.savefig(plot_dir / "divergence_vs_patch_effect.png", dpi=160)
    plt.close(fig)
    for label, pair_id in representatives.items():
        case = [row for row in divergence if row["phase3b_pair_id"] == pair_id]
        table = aggregate_divergence(case)
        for metric, suffix in (("mean_cosine_distance", "cosine"), ("mean_relative_l2", "l2")):
            plot_heatmap(table, GROUPS, all_layers, metric,
                         f"{label.replace('_', ' ').title()} | {pair_id}",
                         plot_dir / f"representative_{label}_{suffix}_heatmap.png")
        low_index = captures[(pair_id, "low_boundary")]
        temporal_index = captures[(pair_id, "temporal_boundary")]
        role_groups = [
            ("First mover, own event", low_index["mover_roles"]["first_mover_own_event"]),
            ("Second mover, own event", low_index["mover_roles"]["second_mover_own_event"]),
            ("Options", "options_all"),
            ("Decision position", "decision_position"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(11, 6), layout="constrained", sharex=True)
        for ax, (title, group) in zip(axes.flat, role_groups):
            for condition, index, color in (
                ("low", low_index, "#b77721"),
                ("temporal", temporal_index, "#248c48"),
            ):
                trajectory = index["group_mean_norms_by_layer"]
                ax.plot(all_layers, [trajectory[str(layer)][group] for layer in all_layers],
                        color=color, label=condition)
            ax.set_title(title)
            ax.set_ylabel("Pooled state L2 norm")
            ax.set_xlabel("Decoder layer")
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(f"Residual-state trajectories | {pair_id}")
        fig.savefig(plot_dir / f"representative_{label}_activation_trajectories.png", dpi=160)
        plt.close(fig)
    return representatives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards_root", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--allow_incomplete", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    divergence, patches, captures, configs, technical_controls = read_shards(args.shards_root)
    digest = hashlib.sha256(Path(args.manifest_path).read_bytes()).hexdigest()
    if any(config["manifest_sha256"] != digest for config in configs):
        raise ValueError("Frozen manifest content differs from shard provenance.")
    manifest_rows = read_jsonl(args.manifest_path)
    manifest_pairs = {row["phase3b_pair_id"] for row in manifest_rows}
    if len(manifest_rows) != 2 * len(manifest_pairs):
        raise ValueError("Frozen Phase 3B manifest lacks complete low/temporal pairs.")
    configured_pairs = {pair for config in configs for pair in config["pair_ids"]}
    if not args.allow_incomplete and configured_pairs != manifest_pairs:
        raise ValueError(
            f"Shard coverage differs from frozen manifest: "
            f"{len(manifest_pairs - configured_pairs)} missing, "
            f"{len(configured_pairs - manifest_pairs)} unexpected pairs."
        )
    missing_controls = sorted(manifest_pairs - set(technical_controls))
    if not args.allow_incomplete and configs[0].get("validate_controls") and missing_controls:
        raise ValueError(f"Missing {len(missing_controls)} Phase 3B technical control results.")
    missing, missing_divergence, missing_captures = validate_completeness(
        divergence, patches, captures, not args.allow_incomplete, manifest_pairs
    )
    output = Path(args.output_dir)
    cohort = [
        {
            "phase3b_pair_id": row["phase3b_pair_id"],
            "base_sample_id": row["base_sample_id"],
            "prompt_variant": row["prompt_variant"],
            "correct_option": row["correct_option"],
            "analysis_stratum": row["phase3b_analysis_stratum"],
            "prompt_pair_behavior": row["phase3b_prompt_pair_behavior"],
            "first_object_id": row["first_object_id"],
        }
        for row in manifest_rows if row["condition"] == "low_boundary"
    ]
    write_csv(output / "cohort_composition.csv", cohort)
    atomic_write_jsonl(output / "divergence_all_layers.jsonl", divergence)
    atomic_write_jsonl(output / "patch_results.jsonl", patches)
    divergence_table = aggregate_divergence(divergence)
    patch_table = aggregate_patches(patches, args.seed)
    write_csv(output / "divergence_by_layer_group.csv", divergence_table)
    write_csv(output / "patch_by_layer_group_direction.csv", patch_table)
    correlations = case_correlations(divergence, patches)
    write_csv(output / "divergence_patch_correlations_by_case.csv", correlations)
    representatives = representative_pairs(captures, args.manifest_path)
    if not args.no_plots:
        write_plots(output, divergence, patches, divergence_table, patch_table, captures, representatives)
    valid_correlations = [row["spearman_cosine_vs_effect"] for row in correlations if row["spearman_cosine_vs_effect"] is not None]
    summary = {
        "schema": SCHEMA,
        "manifest_path": args.manifest_path,
        "manifest_pair_count": len(manifest_pairs),
        "case_count": len({pair for pair, condition in captures if condition == "low_boundary"}),
        "case_count_by_stratum": dict(Counter(row["analysis_stratum"] for row in cohort)),
        "primary_correct_option_counts": dict(Counter(
            row["correct_option"] for row in cohort if row["analysis_stratum"] == "primary_rescue"
        )),
        "capture_conditions": len(captures), "divergence_rows": len(divergence),
        "patch_rows": len(patches), "missing_patch_count": len(missing),
        "missing_divergence_count": len(missing_divergence),
        "missing_capture_count": len(missing_captures),
        "technical_control_pairs": len(technical_controls),
        "missing_technical_control_count": len(missing_controls),
        "first_missing_patches": missing[:20],
        "representative_pairs": representatives,
        "shard_fingerprints": [config["run_fingerprint"] for config in configs],
        "mean_case_spearman_cosine": float(np.mean(valid_correlations)) if valid_correlations else None,
    }
    atomic_write_json(output / "aggregate_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
