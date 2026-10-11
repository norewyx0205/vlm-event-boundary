"""CPU-only aligned residual-difference footprint; not an attribution measure."""

import argparse
import math
import os
from pathlib import Path

try:
    from .analyze_phase3c import aggregate, case_metadata, verify_inputs, write_csv
    from .phase3c_core import CONDITIONS, atomic_write, digest, event_layout, file_hash, frozen_write
    from .phase3c_execution import load_selection
    from .phase3c_support import audit_supports
except ImportError:
    from analyze_phase3c import aggregate, case_metadata, verify_inputs, write_csv
    from phase3c_core import CONDITIONS, atomic_write, digest, event_layout, file_hash, frozen_write
    from phase3c_execution import load_selection
    from phase3c_support import audit_supports


def energy_partition(energies, target_indices):
    values = [float(value) for value in energies]
    indices = list(target_indices)
    if (not values or len(indices) != len(set(indices)) or
            any(index < 0 or index >= len(values) for index in indices) or
            any(not math.isfinite(value) or value < 0 for value in values)):
        raise ValueError("Invalid aligned per-token difference energies/indices.")
    total = math.fsum(values)
    target = math.fsum(values[index] for index in indices)
    fraction = len(indices) / len(values)
    return {"whole_token_count": len(values), "target_union_token_count": len(indices),
        "target_position_share": fraction, "whole_delta_energy": total,
        "target_delta_energy": target, "context_delta_energy": total - target,
        "target_delta_energy_share": target / total if total else None,
        "context_delta_energy_share": (total - target) / total if total else None,
        "target_energy_per_token": target / len(indices) if indices else None,
        "context_energy_per_token": (total - target) / (len(values) - len(indices)) if len(indices) < len(values) else None,
        "target_energy_enrichment": target / total / fraction if total and fraction else None}


def geometry_summary(weights, declared):
    declared = set(declared)
    if not declared <= set(weights) or any(not math.isfinite(value) or not 0 <= value <= 1.000001 for value in weights.values()):
        raise ValueError("Invalid padded-target overlap/declared support.")
    geometric = {position for position, value in weights.items() if value > 0}
    total = math.fsum(weights.values())
    retained = math.fsum(weights[position] for position in declared)
    return {"geometric_target_cell_count": len(geometric), "declared_union_cell_count": len(declared),
        "omitted_target_touching_cell_count": len(geometric - declared),
        "declared_union_retained_padded_target_overlap_mass_fraction": retained / total if total else None,
        "definition": "rasterized padded target overlap, not receptive-field or information coverage"}


def geometry_audit(pair, mapping, padding=8):
    try:
        from .probe_attention_roi import token_descriptors, video_shape
    except ImportError:
        from probe_attention_roi import token_descriptors, video_shape
    result = {}
    for condition, side in zip(CONDITIONS, ("low_positions", "temporal_positions")):
        record = mapping["processor_records"][condition]
        width, height, _, _ = video_shape(record["video_path"])
        meta = record["video_metadata"]
        grid = meta["merged_video_grid_thw"]
        layout = event_layout(pair[condition], meta, record["visual_positions"], record["prompt_token_count"])
        descriptors = token_descriptors(pair[condition], *grid, width, height, meta["source_frame_groups"], padding, "overlap")
        if len(descriptors) != len(record["visual_positions"]):
            raise ValueError("Geometry descriptor/actual-position correspondence failed.")
        weights = {position: sum(descriptor["spatial_roi_weights"].get(label, 0) for label in ("target_1", "target_2"))
                   for position, descriptor in zip(record["visual_positions"], descriptors) if position in layout["lookup"]}
        declared = mapping["support_audit"]["supports"]["both_targets_event2"][side]
        result[condition] = geometry_summary(weights, declared)
    return result


def run(plan_dir, execution_dir, output_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(min(8, torch.get_num_threads()))
    try:
        from .phase3c_core import read_json
        from .phase3c_interventions import selected_state
    except ImportError:
        from phase3c_core import read_json
        from phase3c_interventions import selected_state
    output = Path(output_dir).resolve()
    for source in (Path(plan_dir).resolve(), Path(execution_dir).resolve()):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Keep source evidence read-only; use a separate diagnostic output root.")
    frozen, pairs, execution, baselines, _ = verify_inputs(plan_dir, execution_dir)
    mappings = load_selection(plan_dir)[2]
    binding = {"schema": "phase3c_delta_footprint_v1", "artifact_type": "real", "device": "cpu",
        "source_selection_fingerprint": frozen["selection_fingerprint"],
        "source_execution_fingerprint": execution["execution_fingerprint"],
        "code_sha256": file_hash(__file__), "layers": list(range(36)),
        "alignment": "frozen_event_relative_same_spatial_cell_correspondence",
        "interpretation": "Descriptive transplant magnitude, not causal importance or attention.",
        "used_for_replication_selection": False}
    binding["fingerprint"] = digest(binding)
    frozen_write(output / "config.json", binding)
    rows, geometry_rows = [], []
    for number, (pair_id, pair) in enumerate(pairs.items(), 1):
        mapping = mappings[pair_id]
        audit_supports(pair_id, pair, mapping)
        supports = mapping["support_audit"]["supports"]
        whole, target = supports["whole_event2"], supports["both_targets_event2"]
        whole_pairs = list(zip(whole["low_positions"], whole["temporal_positions"]))
        lookup = {value: index for index, value in enumerate(whole_pairs)}
        target_pairs = list(zip(target["low_positions"], target["temporal_positions"]))
        if not set(target_pairs) <= set(whole_pairs):
            raise ValueError("Target union does not share whole-event correspondence.")
        indices = [lookup[value] for value in target_pairs]
        for condition, record in geometry_audit(pair, mapping, frozen["settings"]["roi_padding"]).items():
            geometry_rows.append({**case_metadata(pair), "pair_id": pair_id, "condition": condition, **record})
        captured = []
        for condition in CONDITIONS:
            index = read_json(baselines[pair[condition]["eval_id"]]["capture_index_path"])
            if file_hash(index["vectors_path"]) != index["vectors_sha256"]:
                raise ValueError("Source capture bytes changed.")
            captured.append(torch.load(index["vectors_path"], map_location="cpu", weights_only=True))
        for layer in range(36):
            low = selected_state(captured[0]["states"], layer, "block_output", whole["low_positions"])["vectors"].float()
            temporal = selected_state(captured[1]["states"], layer, "block_output", whole["temporal_positions"])["vectors"].float()
            if low.shape != temporal.shape or not bool(torch.isfinite(low).all() and torch.isfinite(temporal).all()):
                raise ValueError("Invalid observed, position-mapped source vectors.")
            energy = ((temporal - low).square().sum(dim=1, dtype=torch.float64)).tolist()
            rows.append({**case_metadata(pair), "pair_id": pair_id, "layer": layer,
                "location": "block_output", **energy_partition(energy, indices)})
        del captured
        print(f"CPU footprint {number}/{len(pairs)}: {pair_id}; source tensors read-only", flush=True)
    summary = aggregate(rows, ("analysis_stratum", "layer"),
        ("target_position_share", "target_delta_energy_share", "context_delta_energy_share",
         "target_energy_enrichment", "target_energy_per_token", "context_energy_per_token"))
    atomic_write(output / "case_footprints.json", rows)
    write_csv(output / "case_footprints.csv", rows)
    atomic_write(output / "summary.json", summary)
    write_csv(output / "summary.csv", summary)
    atomic_write(output / "geometry_audit.json", geometry_rows)
    write_csv(output / "geometry_audit.csv", geometry_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, stratum in zip(axes, ("rescue", "stable")):
        items = sorted([row for row in summary if row["analysis_stratum"] == stratum], key=lambda row: row["layer"])
        x = [row["layer"] for row in items]
        ax.plot(x, [row["target_delta_energy_share_mean"] for row in items], color="#BC365D", label="Target-union delta-energy share")
        ax.plot(x, [row["target_position_share_mean"] for row in items], color="#555555", linestyle="--", label="Target-union token share")
        ax.set(title=f"{stratum} | descriptive only", xlabel="Decoder layer (0-based)", ylabel="Fraction", ylim=(0, 1))
    axes[0].legend(fontsize=8)
    fig.suptitle("Aligned donor-recipient residual difference | Not causal importance or attention")
    fig.savefig(output / "delta_footprint.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    atomic_write(output / "complete.json", {"complete": True, "fingerprint": binding["fingerprint"],
        "case_count": len(pairs), "rows": len(rows), "model_loaded": False,
        "output_sha256": {path.name: file_hash(path) for path in output.iterdir() if path.is_file() and path.name != "complete.json"}})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan_dir", required=True)
    parser.add_argument("--execution_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    run(args.plan_dir, args.execution_dir, args.output_dir)


if __name__ == "__main__":
    main()
