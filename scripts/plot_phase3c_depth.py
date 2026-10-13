"""Complete every-four-layer depth profiles, not selected-layer significance plots."""

from pathlib import Path


def plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .analyze_phase3c import aggregate
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, direction in enumerate(("temporal_to_low", "low_to_temporal")):
        for group, label, color in ((None, "All 50 rescues (descriptive)", "#137F86"),
                                   ("newly_tested", "34 newly tested bases", "#B43C62")):
            selected = [r for r in rows if r["direction"] == direction and (group is None or r["cohort_group"] == group)]
            summary = sorted(aggregate(selected, ("layer",), ("margin_delta", "strict_incorrect_to_correct", "strict_correct_to_incorrect")),
                             key=lambda r: r["layer"])
            x = [r["layer"] for r in summary]
            axes[0,column].plot(x, [r["margin_delta_mean"] for r in summary], marker="o", label=label, color=color)
            axes[0,column].fill_between(x, [r["margin_delta_ci_low"] for r in summary],
                                       [r["margin_delta_ci_high"] for r in summary], alpha=.13, color=color)
            metric = "strict_incorrect_to_correct" if column == 0 else "strict_correct_to_incorrect"
            axes[1,column].plot(x, [r[metric+"_mean"] for r in summary], marker="o", label=label, color=color)
        axes[0,column].set(title=direction.replace("_", " "), ylabel="Mean correct-option margin change")
        axes[0,column].axhline(0, linewidth=.7, color="black")
        axes[1,column].set(ylim=(-.03,1.03), ylabel="Strict rescue fraction" if column == 0 else "Strict correct-to-wrong fraction")
        for ax in axes[:,column]:
            ax.set(xticks=list(range(0,36,4)), xlabel="Patched decoder block (zero-based)")
            ax.spines[["top","right"]].set_visible(False)
    axes[0,0].legend(fontsize=9)
    fig.suptitle("Whole matched Event-2 video-state patch | 50 fixed rescue bases | every four layers", fontsize=14)
    fig.supxlabel("L0 is before recipient DeepStack addition. Intervals are case-bootstrap diagnostics, not independent layer tests.", fontsize=10)
    fig.savefig(output/"figure1_every_four_layer_profile.png", dpi=170)
    plt.close(fig)
