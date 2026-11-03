"""English figures and key-number tables for the 2 November 2026 analysis report.

Reads only frozen local outputs: direct-D v3 (probes_v2, audits, feature store),
Phase 3F verified analysis, and the Phase 3G layerwise summary when present.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
V3 = ROOT / "analysis/phase3e/direct_difference_20261102"
F3 = ROOT / "analysis/phase3f/layout_controls_20261102/verified_analysis_v2/results.json"
G3 = ROOT / "analysis/phase3g/alllayer_20261102/layerwise_v1/summary.json"
H0 = ROOT / "analysis/phase3g/alllayer_20261102/h_null_v1/summary.json"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
VIEWS = (("archived", "Original layout (46/50 correct = A)", BLUE, "o"),
         ("permuted", "Swapped layout (46/50 correct = B)", ORANGE, "s"),
         ("symmetric", "Layout average (letter bias cancelled)", AQUA, "D"))
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "axes.unicode_minus": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": .8, "axes.axisbelow": True, "font.size": 10.5, "axes.spines.top": False,
    "axes.spines.right": False})
NUMBERS = {}


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def dots(ax, labels, cells, offset=.24, legend=True):
    for i, (view, name, color, marker) in enumerate(VIEWS):
        for x, label in enumerate(labels):
            c = cells[label][view]
            xx = x + (i-1)*offset
            ax.plot([xx, xx], [c["ci_low"], c["ci_high"]], color=color, lw=2, solid_capstyle="round")
            ax.plot(xx, c["mean"], marker, color=color, ms=8, mec=SURFACE, mew=1.5,
                    label=name if (x == 0 and legend) else None)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(len(labels)))


# ------------------------------------------------------------------ direct-D v3

def v3_predictions(layer, regime):
    path = V3 / f"probes_v2/reference_last__L{layer}_block_output/{regime}/predictions.json"
    return [p for p in json.loads(path.read_text()) if p["direction"] == "original_minus_reversed"]


def v3_section():
    from scripts import phase3e_order_probes as core
    from scripts.phase3g_layerwise_probes import auc
    manifest, plan, rows, D, left, right = core.load_store(V3 / "completed_capture_v3/feature_store/manifest.json")
    layers = ("16", "17", "18", "20")
    out = {"pairwise_vs_single": {}, "norms": {}, "minority": {}, "two_by_two": {}}
    for li, layer in enumerate(layers):
        P = v3_predictions(layer, "mixed_matched")
        pair = float(np.mean([p["probe_prediction"] == p["y_statement_true"] for p in P]))
        aucs = []
        for fold in sorted({p["model_id"] for p in P}):
            Q = [p for p in P if p["model_id"] == fold]
            aucs.append(auc([p["y_statement_true"] for p in Q] + [1-p["y_statement_true"] for p in Q],
                            [p["left_endpoint_projection"] for p in Q] + [p["right_endpoint_projection"] for p in Q]))
        out["pairwise_vs_single"][layer] = {"pairwise_accuracy": pair, "single_video_auc": float(np.mean(aucs))}
        canonical = [i for i, r in enumerate(rows) if r["direction"] == "original_minus_reversed"]
        norms = {}
        for cond in ("low_boundary", "temporal_boundary"):
            idx = [i for i in canonical if rows[i]["condition"] == cond]
            norms[cond] = {"mean_norm_D": float(np.linalg.norm(D[idx, li], axis=1).mean()),
                           "relative_norm": float((np.linalg.norm(D[idx, li], axis=1) /
                                                   np.linalg.norm(left[idx, li], axis=1)).mean())}
        lookup = {r["example_id"]: i for i, r in enumerate(rows)}
        for cond in ("low_boundary", "temporal_boundary"):
            Q = [p for p in P if p["condition"] == cond]
            norms[cond]["evidence"] = float(np.mean([p["true_statement_evidence"] for p in Q]))
            norms[cond]["evidence_per_norm"] = float(np.mean([p["true_statement_evidence"] /
                np.linalg.norm(D[lookup[p["example_id"]], li]) for p in Q]))
        out["norms"][layer] = norms
        for regime in ("low_only", "temporal_only"):
            P2 = v3_predictions(layer, regime)
            cells = {}
            for cond in ("low_boundary", "temporal_boundary"):
                Q = [p for p in P2 if p["condition"] == cond]
                majority = [p for p in Q if (p["view"] == "archived") == ((p["option"] == "A") == (p["y_statement_true"] == 1))]
                minority = [p for p in Q if p not in majority]
                acc = lambda rows_: [sum(p["probe_prediction"] == p["y_statement_true"] for p in rows_), len(rows_)]
                cells[cond] = {"majority": acc(majority), "minority": acc(minority),
                               "pairwise_accuracy": float(np.mean([p["probe_prediction"] == p["y_statement_true"] for p in Q]))}
            out["two_by_two"].setdefault(layer, {})[regime] = cells
    audit = json.loads((V3 / "high_accuracy_audit_v1/summary.json").read_text())
    out["audit"] = {L: {"reference_all": audit["reference"][L]["all"],
                        "label_flip_mean": audit["controls"][L]["base_label_flip"]["archived_summary"],
                        "norm_only": audit["controls"][L]["endpoint_norm_difference"]["all"],
                        "train_archived_only": audit["controls"][L]["train_only_archived"]["all"],
                        "train_permuted_only": audit["controls"][L]["train_only_option_order_permuted"]["all"]}
                    for L in ("L16", "L18")}
    out["audit"]["zero_features"] = audit["zero_features"]["all"]
    out["behaviour"] = behaviour()
    NUMBERS["direct_D_v3"] = out

    # Figure A: pairwise vs single-video
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = range(len(layers))
    ax.plot(xs, [out["pairwise_vs_single"][l]["pairwise_accuracy"] for l in layers], "-o", color=BLUE, lw=2, ms=8,
            label="Pairwise D accuracy (two videos, same statement)")
    ax.plot(xs, [out["pairwise_vs_single"][l]["single_video_auc"] for l in layers], "-s", color=ORANGE, lw=2, ms=8,
            label="Same probe, single-video AUC across bases")
    ax.axhline(.5, color=INK2, lw=1, ls="--")
    ax.text(-.15, .515, "chance", fontsize=9, color=INK2)
    for x, l in zip(xs, layers):
        for key, dy in (("pairwise_accuracy", .025), ("single_video_auc", -.045)):
            v = out["pairwise_vs_single"][l][key]
            ax.text(x, v+dy, f"{v:.3f}", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(list(xs), [f"L{l}" for l in layers])
    ax.set_ylim(.4, 1.05)
    ax.set_ylabel("Accuracy / AUC")
    ax.set_title("Direct-D probe (mixed-matched training): pairwise vs single-video readout", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    save(fig, "figA_directD_pairwise_vs_single.png")

    # Figure B: ||D|| and probe margin per unit norm
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cond, color, marker, name in (("low_boundary", BLUE, "o", "Low"), ("temporal_boundary", ORANGE, "s", "Temporal")):
        axes[0].plot(xs, [out["norms"][l][cond]["relative_norm"] for l in layers], "-"+marker, color=color, lw=2, ms=8, label=name)
        axes[1].plot(xs, [out["norms"][l][cond]["evidence_per_norm"] for l in layers], "-"+marker, color=color, lw=2, ms=8, label=name)
    axes[0].set_title("Size of the order-reversal effect: ||D|| / ||h||", fontsize=10.5)
    axes[1].set_title("Probe margin per unit ||D||", fontsize=10.5)
    for ax in axes:
        ax.set_xticks(list(xs), [f"L{l}" for l in layers])
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Temporal enlarges D at every layer; its cosine with the probe direction is not higher", x=.02, ha="left", fontsize=12)
    save(fig, "figB_directD_norm_vs_alignment.png")

    # Figure C: behaviour decomposition
    b = out["behaviour"]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    labels = ["margin, correct = A", "margin, correct = B", "layout-invariant S", "B preference b"]
    keys = ["correct_A", "correct_B", "S", "b"]
    for j, (order, shade) in enumerate((("original", 1.0), ("reversed", .55))):
        for k, (cond, color, name) in enumerate((("low", BLUE, "Low"), ("temporal", ORANGE, "Temporal"))):
            vals = [b[order][key][cond] for key in keys]
            ax.bar(np.arange(4) + (j*2+k-1.5)*.19, vals, width=.17, color=color, alpha=shade,
                   label=f"{name}, {'original (rescue-selected)' if order == 'original' else 'reversed (not selected)'} videos")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(4), labels)
    ax.set_ylabel("Model margin (mean over 50 bases)")
    ax.legend(fontsize=8.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.12))
    ax.set_title("Model answers: S = (M_correctA + M_correctB)/2, b = (M_correctB - M_correctA)/2", fontsize=10.5)
    save(fig, "figC_behaviour_letter_bias.png")


def behaviour():
    import re as _re
    d = json.loads((V3 / "completed_capture_v3/feature_store/decisions.json").read_text())
    V = {}
    for k, x in d.items():
        m = _re.match(r"l5_full_sample_(\d+)_(low|temporal)_boundary_(original|swapped)(__order_reversed)?(__pure_option_permutation)?$", k)
        V.setdefault((int(m.group(1)), m.group(2), "reversed" if m.group(4) else "original"), {})[x["correct_option"]] = x
    rng = np.random.default_rng(0)
    result = {}
    for order in ("original", "reversed"):
        bases = sorted({k[0] for k in V if k[2] == order})
        cell = {key: {} for key in ("correct_A", "correct_B", "S", "b", "S_gap_ci", "b_gap_ci", "correct_counts")}
        per = {}
        for cond in ("low", "temporal"):
            A = np.array([V[(x, cond, order)]["A"]["margin"] for x in bases])
            B = np.array([V[(x, cond, order)]["B"]["margin"] for x in bases])
            per[cond] = ((A+B)/2, (B-A)/2)
            cell["correct_A"][cond], cell["correct_B"][cond] = float(A.mean()), float(B.mean())
            cell["S"][cond], cell["b"][cond] = float(((A+B)/2).mean()), float(((B-A)/2).mean())
            cell["correct_counts"][cond] = {
                "correct_A_prompts": int(sum(V[(x, cond, order)]["A"]["is_correct"] for x in bases)),
                "correct_B_prompts": int(sum(V[(x, cond, order)]["B"]["is_correct"] for x in bases)),
                "S_positive": int(sum(((A+B)/2) > 0))}
        for name, idx in (("S_gap_ci", 0), ("b_gap_ci", 1)):
            diff = per["temporal"][idx] - per["low"][idx]
            boots = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
            cell[name] = [float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)),
                          int(sum(diff > 0))]
        result[order] = cell
    return result


# ------------------------------------------------------------------ Phase 3F

def f3_section():
    R = json.loads(F3.read_text())
    NUMBERS["phase3F"] = R
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (cond, title) in zip(axes, (("low_boundary", "Recipient Low (receives Temporal L16 Event-2 state)"),
                                        ("temporal_boundary", "Recipient Temporal (receives Low state)"))):
        cell = R["patch"][cond]
        for i, (view, name, color, marker) in enumerate(VIEWS):
            b, a, d = cell[view]["before"]["mean"], cell[view]["after"]["mean"], cell[view]["change"]
            ax.annotate("", xy=(i, a), xytext=(i, b), arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=14))
            ax.plot(i, b, marker, color=color, ms=8, mfc=SURFACE, mew=2)
            ax.plot(i, a, marker, color=color, ms=8, mec=SURFACE, mew=1.5)
            ax.text(i+.12, (a+b)/2, f"{b:+.2f} → {a:+.2f}\nchange {d['mean']:+.2f}\n[{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]",
                    va="center", fontsize=9, color=INK2, bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"))
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(range(3), ["Original", "Swapped", "Average"])
        ax.set_xlim(-.4, 2.9)
        ax.set_title(title, fontsize=10.5)
    axes[0].set_ylabel("Model margin (correct − incorrect logit)")
    fig.suptitle("L16 whole-Event-2 patch: before (open) → after (filled), mean of 50 bases", x=.02, ha="left", fontsize=12)
    fig.text(.02, -.04, "Original layout Low: 0/50 → 34/50 correct. Swapped layout Low is already 46/50 correct and stays 46/50 "
             "(no flips), yet its margin still rises.", fontsize=9, color=INK2)
    save(fig, "figD_3F_patch_layouts.png")

    windows = ["16", "17", "18", "19", "16-19"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (key, title) in zip(axes, (("temporal_boundary", "Temporal margin change"), ("low_boundary", "Low margin change"),
                                       ("compression", "Advantage compression (Low change − Temporal change)"))):
        cells = {w: (R["event2_ko"][w][key] if key == "compression" else
                     {v: R["event2_ko"][w][key][v]["change"] for v in ("archived", "permuted", "symmetric")}) for w in windows}
        dots(ax, windows, cells)
        ax.set_xticklabels(["L16", "L17", "L18", "L19", "L16–19\njoint"])
        ax.set_title(title, fontsize=10.5)
    axes[0].set_ylabel("Margin change (95% case bootstrap)")
    axes[0].legend(loc="lower left", fontsize=8.5, frameon=False)
    fig.suptitle("Options → Event-2 attention knockout by layer window", x=.02, ha="left", fontsize=12)
    save(fig, "figE_3F_event2_ko_windows.png")

    routes = [("event2", "Event 2"), ("event1", "Event 1"), ("random", "Random 512\nnon-event")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), sharex=True)
    for row, window in enumerate(("16-19", "18")):
        for ax, (key, title) in zip(axes[row], (("temporal_boundary", "Temporal margin change"),
                                                 ("low_boundary", "Low margin change"), ("compression", "Advantage compression"))):
            cells = {}
            for route, _ in routes:
                src = R["event2_ko"][window] if route == "event2" else R["controls"][f"{route}:{window}"]
                cells[route] = src[key] if key == "compression" else {v: src[key][v]["change"] for v in ("archived", "permuted", "symmetric")}
            dots(ax, [r for r, _ in routes], cells, legend=(row == 0 and key == "temporal_boundary"))
            ax.set_title(f"{'L16–19 joint' if window == '16-19' else 'L18 single'}: {title}", fontsize=10.5)
            ax.set_xticklabels([n for _, n in routes])
    axes[0][0].legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.suptitle("Equal-budget knockout controls (294,912 option→visual edges removed per layer in every route)",
                 x=.02, ha="left", fontsize=12)
    save(fig, "figF_3F_route_controls.png")

    order = [("none", "No KO"), ("17", "L17"), ("18", "L18"), ("17-18", "L17–18"), ("17-23", "L17–23")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    cells = {"none": {v: R["patch"]["low_boundary"][v]["change"] for v in ("archived", "permuted", "symmetric")}}
    for w, _ in order[1:]:
        cells[w] = {v: R["patch_x_ko"][w]["patch_under_ko"][v]["change"] for v in ("archived", "permuted", "symmetric")}
    dots(axes[0], [w for w, _ in order], cells)
    axes[0].set_xticklabels([n for _, n in order])
    base = cells["none"]["symmetric"]["mean"]
    for x, (w, _) in enumerate(order[1:], 1):
        axes[0].text(x+.3, cells[w]["symmetric"]["mean"], f"{cells[w]['symmetric']['mean']/base:.0%}", fontsize=9, color=INK2, va="center")
    axes[0].set_ylabel("Margin gain from the patch\n(vs. the same KO without patch)")
    axes[0].set_title("Patch effect remaining under KO (% = layout average vs. no KO)", fontsize=10.5)
    axes[0].legend(loc="upper right", fontsize=8.5, frameon=False)
    ax = axes[1]
    ko_alone = [0] + [R["patch_x_ko"][w]["ko_alone"]["archived"]["correct_after"] for w, _ in order[1:]]
    patched = [R["patch"]["low_boundary"]["archived"]["correct_after"]] + [
        R["patch_x_ko"][w]["patch_plus_ko_vs_base"]["archived"]["correct_after"] for w, _ in order[1:]]
    xs = range(len(order))
    ax.bar([x-.19 for x in xs], ko_alone, width=.36, color=GRAY, label="KO only")
    ax.bar([x+.19 for x in xs], patched, width=.36, color=BLUE, label="Patch + KO")
    for x, (a, b) in enumerate(zip(ko_alone, patched)):
        ax.text(x-.19, a+.6, str(a), ha="center", fontsize=9, color=INK2)
        ax.text(x+.19, b+.6, str(b), ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(xs), [n for _, n in order])
    ax.set_ylim(0, 50)
    ax.set_ylabel("Low correct answers (of 50)")
    ax.set_title("Actual answers, original layout (Low baseline 0/50)", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    fig.suptitle("Temporal→Low L16 patch combined with later options→Event-2 knockout", x=.02, ha="left", fontsize=12)
    save(fig, "figG_3F_patch_x_ko.png")


# ------------------------------------------------------------------ Phase 3G

def g3_section():
    if not G3.exists():
        print("Phase 3G summary not found; skipping")
        return
    S = json.loads(G3.read_text())["results"]
    layers = sorted(S, key=int)
    xs = [int(l) for l in layers]
    NUMBERS["phase3G_layers"] = len(layers)
    get = lambda l, cell, cond, metric: S[l]["cells"][cell][cond][metric]
    # Hue family = training source; shade, line style and marker = test condition (four distinct lines).
    styles = {("low_only", "low_boundary"): (BLUE, "-", "o", "Low-trained → test Low"),
              ("low_only", "temporal_boundary"): ("#0e3f7e", "--", "^", "Low-trained → test Temporal"),
              ("temporal_only", "low_boundary"): (ORANGE, "-", "s", "Temporal-trained → test Low"),
              ("temporal_only", "temporal_boundary"): ("#8f3410", "--", "D", "Temporal-trained → test Temporal")}
    panels = [("D", "accuracy", "Direct-D pairwise accuracy"), ("D", "single_video_auc", "Direct-D probe, single-video AUC"),
              ("h", "accuracy", "Single-video h-probe accuracy"), ("h", "single_video_auc", "Single-video h-probe AUC")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), sharex=True)
    for ax, (kind, metric, title) in zip(axes.flat, panels):
        for (regime, cond), (color, ls, marker, name) in styles.items():
            ax.plot(xs, [get(l, f"{kind}:{regime}", cond, metric) for l in layers], ls, color=color, lw=1.6,
                    marker=marker, ms=4.5, mfc=color if ls == "-" else SURFACE, mew=1.3, label=name)
        ax.axhline(.5, color=INK2, lw=1, ls=":")
        if kind == "h" and metric == "accuracy" and H0.exists():
            # Fold-composition-preserving permutation null (95% upper bound over the four cells), Section 5.
            null = json.loads(H0.read_text())["null"]
            tested = sorted({int(k.split(":")[1][1:]) for k in null if k.startswith("within_fold:")})
            upper = [max(null[f"within_fold:L{l}:{r}"][c]["null_q975"] for r in ("low_only", "temporal_only")
                         for c in ("low_boundary", "temporal_boundary")) for l in tested]
            ax.plot(tested, upper, "_", color=INK, ms=9, mew=2)
            ax.text(-.5, max(upper)+.045, "▬ within-fold permutation null, 97.5% (Figure K)",
                    fontsize=8, color=INK, ha="left")
        for marked in (16, 17, 18, 20):
            ax.axvline(marked, color=GRID, lw=1)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylim(.2 if kind == "h" else .3, 1.02)
    for ax in axes[1]:
        ax.set_xlabel("Decoder block (zero-based)")
    axes[0][0].legend(fontsize=8.5, frameon=False, loc="lower right", handlelength=3.2, markerscale=1.3)
    fig.suptitle("Layerwise 2×2: training condition × test condition (50 bases, held-out folds)", x=.02, ha="left", fontsize=12)
    save(fig, "figH_3G_layerwise_2x2.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=False)
    for ax, (key, gaps, title) in zip(axes, (
            ("test_gap", [("h:low_only", "accuracy", BLUE, "h probe, Low-trained (accuracy)"),
                          ("h:temporal_only", "accuracy", ORANGE, "h probe, Temporal-trained (accuracy)"),
                          ("D:low_only", "single_video_auc", AQUA, "D probe, Low-trained (single-video AUC)")],
             "Test-condition gap: test Temporal − test Low (same probe)"),
            ("train_gap", [("h:low_boundary", "accuracy", BLUE, "h probe, tested on Low (accuracy)"),
                           ("h:temporal_boundary", "accuracy", ORANGE, "h probe, tested on Temporal (accuracy)"),
                           ("D:low_boundary", "single_video_auc", AQUA, "D probe, tested on Low (single-video AUC)")],
             "Training-source gap: Temporal-trained − Low-trained (same test)"))):
        for cell, metric, color, name in gaps:
            mean = [S[l][key][cell][metric]["difference"] for l in layers]
            low = [S[l][key][cell][metric]["ci_low"] for l in layers]
            high = [S[l][key][cell][metric]["ci_high"] for l in layers]
            ax.fill_between(xs, low, high, color=color, alpha=.15, lw=0)
            ax.plot(xs, mean, color=color, lw=1.8, label=name)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("Decoder block (zero-based)")
        ax.legend(fontsize=8.5, frameon=False)
    axes[0].set_ylabel("Difference (95% base bootstrap)")
    fig.suptitle("Where do Low and Temporal differ? Contrasts by layer", x=.02, ha="left", fontsize=12)
    save(fig, "figI_3G_gaps_by_layer.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for cond, color, marker, name in (("low_boundary", BLUE, "o", "Low"), ("temporal_boundary", ORANGE, "s", "Temporal")):
        axes[0].plot(xs, [get(l, "D:mixed_matched", cond, "evidence") for l in layers], "-"+marker, color=color, ms=3.5,
                     lw=1.8, label=name)
    late = [l for l in layers if int(l) >= 11]
    ratio = [get(l, "D:mixed_matched", "temporal_boundary", "unit_evidence") /
             get(l, "D:mixed_matched", "low_boundary", "unit_evidence") for l in late]
    axes[1].plot([int(l) for l in late], ratio, "-o", color=AQUA, ms=3.5, lw=1.8)
    axes[1].axhline(1, color=INK2, lw=1)
    axes[0].set_title("Probe margin, mixed-matched D probe", fontsize=10.5)
    axes[1].set_title("Probe margin per unit ||D||, Temporal / Low (L11 onwards)", fontsize=10.5)
    axes[0].legend(fontsize=9, frameon=False)
    for ax in axes:
        ax.set_xlabel("Decoder block (zero-based)")
    fig.suptitle("Temporal raises the probe margin at every informative layer; margin per unit ||D|| is not higher (ratio 0.73–1.03)",
                 x=.02, ha="left", fontsize=12)
    save(fig, "figJ_3G_probe_margin_by_layer.png")


def h_null_section():
    if not H0.exists():
        print("h-probe null summary not found; skipping")
        return
    H = json.loads(H0.read_text())
    NUMBERS["h_null"] = {k: {c: {m: v[m] for m in ("observed", "null_mean", "null_q975", "p_upper")}
                             for c, v in cells.items()} for k, cells in H["null"].items()}
    NUMBERS["h_null_controls"] = {k: {"observed_low": v["accuracy"]["low_boundary"],
                                      **{n: v[f"{n}_null_low"]["null_mean"] for n in ("within_fold", "free")}}
                                  for k, v in H["controls"].items()}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    cells = (("low_only", "low_boundary", BLUE, "o", "Low-trained → Low"),
             ("low_only", "temporal_boundary", BLUE, "^", "Low-trained → Temporal"),
             ("temporal_only", "low_boundary", ORANGE, "s", "Temporal-trained → Low"),
             ("temporal_only", "temporal_boundary", ORANGE, "D", "Temporal-trained → Temporal"))
    for ax, (null, title) in zip(axes, (("within_fold", "Within-fold null: each fold keeps its before/after-true mix"),
                                         ("free", "Free null: every base flipped at random"))):
        layers = sorted({int(k.split(":")[1][1:]) for k in H["null"] if k.startswith(null + ":")})
        for i, l in enumerate(layers):
            for j, (regime, cond, color, marker, name) in enumerate(cells):
                v = H["null"][f"{null}:L{l}:{regime}"][cond]
                x = i + (j - 1.5) * .16
                ax.plot([x, x], [v["null_q025"], v["null_q975"]], color=GRAY, lw=5, solid_capstyle="butt")
                ax.plot(x, v["null_mean"], "_", color=INK2, ms=9, mew=1.5)
                ax.plot(x, v["observed"], marker, color=color, ms=6, label=name if i == 0 else None)
        ax.axhline(.5, color=INK2, lw=1, ls=":")
        ax.set_xticks(range(len(layers)), [f"L{l}" for l in layers])
        ax.set_title(title, fontsize=10.5)
    axes[0].set_ylabel("h-probe accuracy (held-out)")
    axes[0].legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.suptitle("Single-video h-probe against permutation nulls (grey bar: 95% of 100 draws; dash: null mean; marker: observed)",
                 x=.02, ha="left", fontsize=12)
    save(fig, "figK_3G_h_probe_permutation_null.png")


if __name__ == "__main__":
    v3_section()
    f3_section()
    g3_section()
    h_null_section()
    (HERE / "key_numbers.json").write_text(json.dumps(NUMBERS, indent=1, default=float))
    print(sorted(p.name for p in FIG.iterdir()))
