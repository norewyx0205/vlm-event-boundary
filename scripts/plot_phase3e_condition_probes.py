"""Plot a completed condition-generalization summary without fitting models."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.phase3c_core import atomic_write, file_hash, read_json
from scripts.phase3e_condition_probes import CONDITIONS, READOUTS, REGIMES, SCHEMA


def plot(summary_path):
    path = Path(summary_path).resolve()
    summary = read_json(path)
    if (summary.get("schema") != SCHEMA or not summary.get("complete") or
            set(summary["results"]) != set(READOUTS)):
        raise ValueError("Plot only the completed, fixed four-readout follow-up.")
    outputs = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    labels = ("Low only (512)", "Temporal only (512)", "Mixed, matched (512)", "Mixed, full (1,024; reused)")
    for key, title, filename in (("accuracy", "Held-out statement-truth accuracy (%)", "condition_accuracy"),
            ("mean_within_fold_roc_auc", "Mean within-fold ROC-AUC", "condition_auc")):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
        fig.subplots_adjust(left=.24, right=.90, bottom=.15, top=.83, wspace=.80, hspace=.52)
        scale = 100 if key == "accuracy" else 1
        for axis, name in zip(axes.flat, READOUTS):
            if set(summary["results"][name]) != set(REGIMES):
                raise ValueError("Missing source-condition probe regime.")
            values = []
            for regime in REGIMES:
                measured = summary["results"][name][regime]["metrics"]
                overall = {row["condition"]: row for row in measured if row["group_by"] is None}
                values.append([scale*overall[condition][key] for condition in CONDITIONS])
            values = np.asarray(values)
            artist = axis.imshow(values, cmap="YlGnBu", vmin=0, vmax=scale, aspect="auto")
            axis.set_xticks([0, 1], ["Test Low", "Test Temporal"])
            axis.set_yticks(np.arange(4), labels)
            layer = name.split(":")[1].removesuffix("_block_output")
            axis.set_title(f"{layer} block output", fontsize=13)
            axis.tick_params(length=0)
            axis.set_xticks(np.arange(-.5, 2), minor=True)
            axis.set_yticks(np.arange(-.5, 4), minor=True)
            axis.grid(which="minor", color="white", linewidth=2)
            axis.tick_params(which="minor", length=0)
            for row in range(4):
                for column in range(2):
                    value = values[row, column]
                    axis.text(column, row, f"{value:.2f}%" if scale == 100 else f"{value:.3f}",
                              ha="center", va="center", color="white" if value > .67*scale else "black", fontsize=12)
        color_axis = fig.add_axes([.93, .23, .015, .49])
        fig.colorbar(artist, cax=color_axis)
        fig.suptitle("Source-condition training and cross-condition generalization", fontsize=16, y=.96)
        fig.text(.57, .89, title, ha="center", fontsize=13)
        fig.text(.56, .06, "Training rows per outer fit are shown in parentheses; every regime uses the same 32 training families.\n"
            "Both test conditions come from the same 8 held-out families per fold; independent unit: 40 scene families.\n"
            "No target-condition scaler fitting, calibration or hyperparameter selection for source-only probes.",
            ha="center", fontsize=10)
        for suffix in ("png", "pdf"):
            output = path.parent / (filename+"."+suffix)
            fig.savefig(output, dpi=180, facecolor="white")
            outputs.append(output)
        plt.close(fig)
    receipt = {"schema": SCHEMA+"_figures", "summary_sha256": file_hash(path),
        "plotter_sha256": file_hash(__file__), "figure_sha256": {p.name: file_hash(p) for p in outputs},
        "new_model_fits": 0}
    atomic_write(path.parent / "figure_manifest.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_path", required=True)
    args = parser.parse_args()
    print(plot(args.summary_path))
