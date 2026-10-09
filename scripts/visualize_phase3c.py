"""Static figures for the fixed Phase 3C pilot; no outcome-driven case/window selection."""

from pathlib import Path

try:
    from .phase3c_core import CONDITIONS, read_json
except ImportError:
    from phase3c_core import CONDITIONS, read_json


SUPPORT_LABELS = {"video_t1_e2": "T1 ROI reference", "video_t2_e2": "T2 ROI reference",
                  "both_targets_event2": "Both-target observed union", "whole_event2": "Whole matched Event 2"}


def figures(output, tables, frozen, pairs, baselines):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    root = Path(output) / "plots"
    root.mkdir(parents=True, exist_ok=True)
    saved = []

    def save(fig, name):
        path = root / name
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(str(path.resolve()))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for row, stratum in enumerate(("rescue", "stable")):
        for column, direction in enumerate(("temporal_to_low", "low_to_temporal")):
            ax = axes[row, column]
            for support, label in SUPPORT_LABELS.items():
                items = sorted([item for item in tables["patch_summary"] if item["analysis_stratum"] == stratum and
                    item["direction"] == direction and item["support"] == support and item["location"] == "block_output" and
                    item["layer"] in frozen["settings"]["patch_layers"]], key=lambda item: item["layer"])
                x = [item["layer"] for item in items]
                y = [item["margin_delta_mean"] for item in items]
                line, = ax.plot(x, y, marker="o", markersize=3, label=label)
                ax.fill_between(x, [item["margin_delta_ci_low"] for item in items],
                    [item["margin_delta_ci_high"] for item in items], color=line.get_color(), alpha=0.10)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set(title=f"{stratum.title()} | {direction.replace('_', ' ')}", xlabel="Decoder layer (0-based)",
                   ylabel="Raw correct-option margin change", xticks=frozen["settings"]["patch_layers"])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Observed visual-state transplantation | Case-bootstrap 95% pilot intervals")
    save(fig, "visual_support_patch_effects.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for row, stratum in enumerate(("rescue", "stable")):
        for column, direction in enumerate(("temporal_to_low", "low_to_temporal")):
            ax = axes[row, column]
            for location, label in (("block_output", "Before DeepStack addition"), ("post_deepstack", "After DeepStack addition")):
                items = sorted([item for item in tables["patch_summary"] if item["analysis_stratum"] == stratum and
                    item["direction"] == direction and item["support"] == "whole_event2" and
                    item["location"] == location and item["layer"] in (0, 1, 2)], key=lambda item: item["layer"])
                x = [item["layer"] for item in items]
                line, = ax.plot(x, [item["margin_delta_mean"] for item in items], marker="o", label=label)
                ax.fill_between(x, [item["margin_delta_ci_low"] for item in items],
                                [item["margin_delta_ci_high"] for item in items], color=line.get_color(), alpha=0.10)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set(title=f"{stratum.title()} | {direction.replace('_', ' ')}", xlabel="Decoder injection layer", ylabel="Raw margin change", xticks=[0, 1, 2])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Timing control | Distinct, location-matched donor states | Whole Event-2 support")
    save(fig, "deepstack_timing_controls.png")

    for stratum in ("rescue", "stable"):
        for query in ("options_all", "query_all"):
            fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
            metrics = (("delta_M_temporal", "Temporal margin change"), ("delta_M_low", "Low margin change"),
                       ("compression", "Boundary-advantage compression"))
            route_labels, matrices = [], []
            for metric, _ in metrics:
                matrix = []
                route_labels = []
                for key in ("target_1", "target_2", "both_targets"):
                    for control in ("target", "background"):
                        items = sorted([item for item in tables["knockout_summary"] if item["analysis_stratum"] == stratum and
                            item["query_group"] == query and item["key_group"] == key and item["control"] == control],
                            key=lambda item: item["window_start"])
                        matrix.append([item[f"{metric}_mean"] for item in items])
                        route_labels.append(f"{key.replace('_', ' ')} | {control}")
                matrices.append(np.array(matrix))
            limit = max(max(float(np.abs(matrix).max()) for matrix in matrices), 0.001)
            for ax, (metric, label), matrix in zip(axes, metrics, matrices):
                plot = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
                ax.set(title=label, xlabel="Knocked-out decoder window", yticks=range(6), yticklabels=route_labels,
                       xticks=range(9), xticklabels=[f"{start}-{start + 3}" for start in range(0, 36, 4)])
                ax.tick_params(axis="x", rotation=45)
            fig.colorbar(plot, ax=axes, shrink=0.8, label="Mean paired-case margin units")
            fig.suptitle(f"{stratum.title()} | {query} to Event-2 visual keys | All heads")
            save(fig, f"knockout_{stratum}_{query}_decomposition.png")

    for pair_id in frozen["technical_preflight_case_ids"]:
        pair = pairs[pair_id]
        mover = pair[CONDITIONS[0]]["first_object_id"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for ax, group in zip(axes, ("whole_event2", "options_all")):
            for condition in CONDITIONS:
                index = read_json(baselines[pair[condition]["eval_id"]]["capture_index_path"])
                y = [index["group_mean_norms_by_site"][f"block_output:L{layer}"][group] for layer in range(36)]
                ax.plot(range(36), y, label=condition.replace("_boundary", ""))
            ax.set(title=group.replace("_", " "), xlabel="Decoder layer (0-based)", ylabel="Norm of group-mean residual")
            ax.legend()
        fig.suptitle(f"Frozen representative: {pair_id} | T{mover} first | Descriptive norms, not tokenwise alignment")
        save(fig, f"representative_t{mover}_first_activation_norms.png")
    return saved
