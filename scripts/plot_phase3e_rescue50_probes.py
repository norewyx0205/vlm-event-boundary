"""Plot archived-prompt condition-generalization accuracy from verified probes."""

import argparse
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.phase3c_core import atomic_write, file_hash, read_json
from scripts.phase3e_rescue50_probes import CONDITIONS, READOUTS, REGIMES, SCHEMA


def render(summary_path, output_root):
    source, root = Path(summary_path).resolve(), Path(output_root).resolve()
    summary = read_json(source)
    if summary.get("schema") != SCHEMA or summary.get("complete") is not True:
        raise ValueError("Need a completed rescue50 summary.")
    labels = {"low_only": "Low only (160)", "temporal_only": "Temporal only (160)",
              "mixed_matched": "Mixed matched (160)", "mixed_full": "Mixed full (320)"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, name in zip(axes.flat, READOUTS):
        matrix = []
        for regime in REGIMES:
            rows = {r["condition"]: r for r in summary["results"][name][regime]["evaluation"]["archived"]["metrics"]
                    if r["group_by"] is None}
            matrix.append([rows[c]["accuracy"] for c in CONDITIONS])
        matrix = np.asarray(matrix)
        display = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks([0, 1], ["Test Low", "Test Temporal"])
        ax.set_yticks(range(4), [labels[k] for k in REGIMES])
        ax.set_title(name.split(":")[1].replace("_block_output", " block output"), fontsize=14, pad=10)
        for i in range(4):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center", fontsize=15,
                        color="black" if matrix[i, j] > .55 else "white")
        ax.tick_params(axis="both", length=0, labelsize=11)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Rescue50: training condition and held-out statement-truth decoding", fontsize=17, y=.98)
    fig.text(.5, .925, "Same original prompts as KO/patching; last reference-object option token", ha="center", fontsize=12)
    fig.subplots_adjust(left=.21, right=.88, top=.86, bottom=.14, hspace=.43, wspace=.65)
    colorbar = fig.colorbar(display, cax=fig.add_axes([.915, .24, .02, .5]))
    colorbar.set_label("Statement-truth probe accuracy")
    fig.text(.5, .055, "50 held-out bases across five folds; 100 statements per test condition.\n"
        "Numbers in training labels are statement rows per outer fit (40 bases).\n"
        "Training includes pure A/B views; test cells above use archived prompts only.", ha="center", fontsize=11)
    root.mkdir(parents=True, exist_ok=True)
    files = []
    for suffix in ("png", "pdf"):
        path = root/("rescue50_condition_generalization."+suffix)
        if path.exists():
            raise ValueError("Use a new figure directory; do not overwrite old figures.")
        fig.savefig(path, dpi=180, facecolor="white")
        files.append(path)
    plt.close(fig)
    atomic_write(root/"provenance.json", {"summary_sha256": file_hash(source), "script_sha256": file_hash(__file__),
        "evaluation_view": "archived", "files": {p.name: file_hash(p) for p in files}})
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_path", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()
    render(args.summary_path, args.output_root)
