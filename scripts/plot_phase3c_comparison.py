"""Reporting-only figures; never overwrite frozen measurement artifacts."""

import argparse
from pathlib import Path

try:
    from .analyze_phase3c import aggregate
    from .phase3c_core import atomic_write, file_hash, read_json
except ImportError:
    from analyze_phase3c import aggregate
    from phase3c_core import atomic_write, file_hash, read_json


def verified_table(directory, name):
    directory = Path(directory)
    summary = read_json(directory / "aggregate_summary.json")
    if (summary.get("complete") is not True or summary.get("artifact_type") != "real" or
            summary["output_sha256"].get(name) != file_hash(directory / name)):
        raise ValueError("Reporting requires complete, checksummed scientific outputs.")
    return read_json(directory / name)


def patch_figure(pilot, replication, output, scale=None):
    import matplotlib.pyplot as plt
    import numpy as np
    sources = [("Original pilot", pilot, 8), ("Independent replication", replication, 8)]
    if scale is not None:
        sources.extend([( "Newly tested", [r for r in scale if r["cohort_group"] == "newly_tested"], 34),
                        ("All rescues (descriptive)", scale, 50)])
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), constrained_layout=True)
    colors = ("#167F86", "#B74063")
    labels = ("Whole matched Event 2", "Both-target ROI union")
    source_labels = [f"{name}\nn = {n} independent bases" for name, _, n in sources]
    x = np.arange(len(sources))
    for j, support in enumerate(("whole_event2", "both_targets_event2")):
        values, lows, highs, rates, counts = [], [], [], [], []
        offsets = x + (j - .5) * .32
        for _, rows, n in sources:
            rows = [r for r in rows if r["analysis_stratum"] == "rescue" and
                    r["direction"] == "temporal_to_low" and r["layer"] == 16 and
                    r["location"] == "block_output" and r["support"] == support]
            if not rows and support == "both_targets_event2":
                values.append(float("nan")); lows.append(0); highs.append(0)
                rates.append(float("nan")); counts.append(None)
                continue
            if len(rows) != n:
                raise ValueError("Incomplete fixed-layer cohort in comparison figure.")
            summary = aggregate(rows, (), ("margin_delta",))[0]
            mean = summary["margin_delta_mean"]
            values.append(mean)
            lows.append(mean - summary["margin_delta_ci_low"])
            highs.append(summary["margin_delta_ci_high"] - mean)
            count = sum(r["baseline_margin"] < 0 < r["patched_margin"] and
                        r["baseline_prediction"] != r["correct_option"] and
                        r["patched_prediction"] == r["correct_option"] for r in rows)
            rates.append(count/n); counts.append(count)
        axes[0].bar(offsets, values, width=.29, color=colors[j], label=labels[j])
        axes[0].errorbar(offsets, values, yerr=[lows, highs], color="black", fmt="none", capsize=4)
        for pos, mean in zip(offsets, values):
            if np.isfinite(mean):
                axes[0].text(pos, mean + .05, f"{mean:+.2f}", ha="center", va="bottom", fontsize=9)
        axes[1].bar(offsets, rates, width=.29, color=colors[j])
        for pos, rate, count, (_, _, n) in zip(offsets, rates, counts, sources):
            if count is not None:
                axes[1].text(pos, rate + .035, f"{count}/{n}", ha="center", fontsize=10)
    for ax in axes:
        ax.set_xticks(x, source_labels, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set(ylabel="Correct-option margin change", title="Continuous effect (case-bootstrap 95% intervals)")
    axes[0].axhline(0, color="black", linewidth=.7)
    axes[0].legend(fontsize=9, loc="upper right")
    axes[1].set(ylim=(0, 1.1), ylabel="Fraction of wrong answers strictly rescued", title="Actual answer flips + strict margin crossing")
    fig.suptitle("Video hidden-state patch: temporal donor -> low recipient | fixed L16 block output", fontsize=13)
    fig.supxlabel("ROI selects positions, not object-exclusive information. No ROI rerun in the 50-case expansion.", fontsize=9)
    fig.savefig(output / "figure1_patch_replication.png", dpi=180)
    plt.close(fig)


def knockout_figure(rows, output):
    import matplotlib.pyplot as plt
    import numpy as np
    rows = [r for r in rows if r["analysis_stratum"] == "rescue" and r["query_group"] == "options_all"]
    windows = sorted({(r["window_start"], r["window_end"]) for r in rows})
    keys = [(key, control) for key in ("target_1", "target_2", "both_targets") for control in ("target", "background")]
    lookup = {(r["key_group"], r["control"], (r["window_start"], r["window_end"])): r for r in rows}
    if len(lookup) != len(rows) or len(lookup) != len(keys) * len(windows) or any(r["case_count"] != 8 for r in rows):
        raise ValueError("Attention-blocking figure requires the complete original 8-case grid.")
    matrices = [np.array([[lookup[(key, control, window)][metric + "_mean"] for window in windows]
                         for key, control in keys]) for metric in ("delta_M_temporal", "delta_M_low", "compression")]
    limit = max(.35, max(float(np.max(np.abs(matrix))) for matrix in matrices))
    labels = [f"{name}: {suffix}" for name in ("Target 1", "Target 2", "Both targets")
              for suffix in ("ROI keys blocked", "matched background blocked")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, sharey=True)
    for ax, data, title in zip(axes, matrices, ("Temporal margin change\nafter attention blocking",
            "Low margin change\nafter the same attention blocking", "Gap reduction\nlow change minus temporal change")):
        im = ax.imshow(data, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set(title=title, xticks=range(len(windows)), xticklabels=[f"{a}-{b}" for a, b in windows],
               yticks=range(len(labels)), yticklabels=labels, xlabel="Blocked decoder layers (all heads)")
        ax.tick_params(axis="x", labelsize=8)
        for y in range(len(keys)):
            for x in range(len(windows)):
                ax.text(x, y, f"{data[y,x]:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(data[y,x]) > limit*.6 else "black")
    fig.colorbar(im, ax=axes, shrink=.8, label="Mean logit-margin change, not attention weights")
    fig.suptitle("Attention blocked: option-text queries -> Event-2 visual keys | original pilot only, n = 8", fontsize=13)
    fig.supxlabel("Inputs are unchanged. A smaller temporal-low gap does not by itself mean temporal performance was impaired.", fontsize=10)
    fig.savefig(output / "figure2_attention_blocking.png", dpi=180)
    plt.close(fig)


def generate(pilot_analysis, replication_analysis, output_dir, scale_analysis=None):
    import matplotlib
    matplotlib.use("Agg")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sources = [(Path(pilot_analysis), "case_patch_effects.json"), (Path(replication_analysis), "anchor_effects.json")]
    pilot, replication = [verified_table(path, name) for path, name in sources]
    scale = verified_table(scale_analysis, "case_patch_effects.json") if scale_analysis else None
    patch_figure(pilot, replication, output, scale)
    knockout_figure(verified_table(pilot_analysis, "knockout_summary.json"), output)
    if scale_analysis:
        sources.append((Path(scale_analysis), "case_patch_effects.json"))
    sources.append((Path(pilot_analysis), "knockout_summary.json"))
    manifest = {"reporting_only": True, "no_new_knockout": True,
        "source_sha256": {str(path/name): file_hash(path/name) for path, name in sources},
        "scale50_included": scale_analysis is not None,
        "output_sha256": {p.name: file_hash(p) for p in output.glob("figure*.png")}}
    atomic_write(output / "figure_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot_analysis", "replication_analysis", "output_dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--scale_analysis")
    args = parser.parse_args()
    generate(**vars(args))


if __name__ == "__main__":
    main()
