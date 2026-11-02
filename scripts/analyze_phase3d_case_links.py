"""CPU-only, exploratory joins of frozen patch, probe and singleton-KO exports.

No inference, model loading, probe fitting or modification of source artifacts.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


COMPARISONS = (
    ("patch_vs_KO_damage", "patch_margin_gain", "ko_temporal_damage"),
    ("probe_vs_KO_damage", "probe_evidence_gain", "ko_temporal_damage"),
    ("patch_vs_probe", "patch_margin_gain", "probe_evidence_gain"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def unique(rows, key, label):
    result = {}
    for row in rows:
        value = row[key]
        require(value not in result, f"Duplicate {label}: {value}")
        result[value] = row
    return result


def sealed_inputs(root, manifest_path, names):
    manifest = read(manifest_path)
    refs = unique(manifest["files"], "path", "backup member")
    result = []
    for name in names:
        path = root / name
        expected = refs["run/" + name]["sha256"]
        require(sha(path) == expected, f"Backup hash mismatch: {path}")
        result.append({"path": str(path.resolve()), "sha256": expected})
    return result


def load_inputs(source, singleton):
    source_names = ["plan.json", "case_metadata.json", "outer_folds.json",
                    "feature_store/rows.json", "probes/aggregate_summary.json",
                    "probes/option_pair__L20_output/result.json",
                    "probes/option_pair__L20_output/case_evidence.json",
                    "probes/option_pair__L20_output/models.json"]
    single_names = ["plan.json", "analysis/aggregate_summary.json",
                    "analysis/case_decomposition.json"]
    seals = sealed_inputs(source, source / "lifecycle/backup_manifest.json", source_names)
    seals += sealed_inputs(singleton, singleton / "backup_metadata/backup_manifest.json", single_names)
    plan, single_plan = read(source / "plan.json"), read(singleton / "plan.json")
    aggregate = read(singleton / "analysis/aggregate_summary.json")
    require(single_plan["source_plan_fingerprint"] == plan["fingerprint"], "Different source runs")
    require(aggregate["source_plan_fingerprint"] == plan["fingerprint"], "Different analysis source")
    require(aggregate["plan_fingerprint"] == single_plan["fingerprint"], "Different singleton plan")
    require(aggregate["complete"] and aggregate["base_count"] == 50, "Incomplete singleton cohort")
    require(set(single_plan["case_ids"]) == set(plan["case_ids"]), "Different frozen cohorts")
    require(sha(singleton / "analysis/case_decomposition.json") ==
            aggregate["output_sha256"]["case_decomposition.json"], "Unsealed singleton rows")
    probe_root = source / "probes/option_pair__L20_output"
    result = read(probe_root / "result.json")
    require(result["complete"] and result["is_primary_readout"] and
            result["readout_name"] == "option_pair:L20_output", "Primary readout changed")
    for name in ("case_evidence.json", "models.json"):
        require(sha(probe_root / name) == result["output_sha256"][name], f"Unsealed {name}")
    probe_aggregate = read(source / "probes/aggregate_summary.json")
    require(sha(probe_root / "result.json") ==
            probe_aggregate["results"]["option_pair:L20_output"]["sha256"], "Unsealed primary result")
    rows = join_cases(read(source / "case_metadata.json"), read(source / "feature_store/rows.json"),
                      read(probe_root / "case_evidence.json"),
                      read(singleton / "analysis/case_decomposition.json"),
                      read(source / "outer_folds.json"), read(probe_root / "models.json"))
    require({r["pair_id"] for r in rows} == set(plan["case_ids"]), "Unexpected case IDs")
    return rows, {"source_plan_fingerprint": plan["fingerprint"],
                  "singleton_plan_fingerprint": single_plan["fingerprint"],
                  "checked_inputs": seals,
                  "verification_scope": "Downloaded exports against saved backup manifests; not a fresh raw-tensor audit.",
                  "complete_off_vm_backup_confirmed": False}


def join_cases(metadata, observations, evidence, knockout, folds, models, expected_count=50):
    cases = unique(metadata, "base_sample_id", "metadata base")
    require(len(cases) == expected_count, "Unexpected number of bases")
    unique(metadata, "pair_id", "metadata pair")
    probe = unique([r for r in evidence if r["view"] == "archived"], "base_sample_id", "probe base")
    ko = unique([r for r in knockout if r["layer_count"] == 1 and r["window_start"] == 18],
                "base_sample_id", "L18 singleton base")
    observed = {}
    for row in observations:
        if row["view"] != "archived":
            continue
        key = (row["base_sample_id"], row["condition"], row["intervention"])
        require(key not in observed, f"Duplicate observation: {key}")
        observed[key] = row
    require(set(cases) == set(probe) == set(ko), "Missing or extra bases in join")
    test_folds = {}
    for fold in folds:
        require(not set(fold["test_base_ids"]) & set(fold["train_base_ids"]), "Fold leakage")
        for base in fold["test_base_ids"]:
            require(base not in test_folds, "Base appears in multiple test folds")
            test_folds[base] = fold["outer_fold"]
    require(set(test_folds) == set(cases), "Test-fold coverage mismatch")
    model_map = unique(models, "model_id", "probe model")
    joined = []
    for base, meta in sorted(cases.items()):
        p, k = probe[base], ko[base]
        require(k["pair_id"] == meta["pair_id"], "Pair ID mismatch")
        require(k["window_end"] == 18 and k["query_group"] == "options_all" and
                k["key_group"] == "whole_event2", "KO route changed")
        require(k["low_key_count"] == k["temporal_key_count"] == 512, "KO support changed")
        require(meta["y"] == p["y"] and meta["first_object_id"] == k["first_object_id"], "Label mismatch")
        require(meta["correct_option"] == k["correct_option"], "Answer-label mismatch")
        model = model_map[p["model_id"]]
        require(p["model_id"] == f"outer_{test_folds[base]}", "Wrong held-out probe")
        require(base in model["test_base_ids"] and base not in model["train_base_ids"], "Probe leakage")
        low = observed[base, "low_boundary", "baseline"]
        temporal = observed[base, "temporal_boundary", "baseline"]
        patched = observed[base, "low_boundary", "temporal_to_low_L16"]
        for row in (low, temporal, patched):
            require(row["pair_id"] == meta["pair_id"] and row["y"] == meta["y"] and
                    row["correct_option"] == meta["correct_option"], "Observation identity mismatch")
        require(low["input_fingerprint"] == patched["input_fingerprint"], "Different patch recipient input")
        require(low["answer_margin"] == k["low_base_margin"] and
                temporal["answer_margin"] == k["temporal_base_margin"], "Baseline replay mismatch")
        require(low["answer_margin"] < 0 < temporal["answer_margin"], "Not a strict baseline rescue")
        delta_patch = patched["answer_margin"] - low["answer_margin"]
        require(delta_patch == p["delta_M_temporal_to_low_L16"], "Patch margin mismatch")
        require(k["temporal_ko_margin"] - k["temporal_base_margin"] == k["delta_M_temporal"],
                "KO margin arithmetic mismatch")
        require(k["low_ko_margin"] - k["low_base_margin"] == k["delta_M_low"] and
                k["delta_M_low"] - k["delta_M_temporal"] == k["compression"],
                "KO low/compression arithmetic mismatch")
        ko_flip = k["temporal_ko_margin"] < 0
        require(ko_flip == k["temporal_strict_correct_to_incorrect"], "KO flip mismatch")
        require(not k["temporal_non_ab_first_token"], "Non-A/B KO outcome requires separate handling")
        require(math.isclose(p["S_temporal"] - p["S_low"], p["delta_S_boundary"], abs_tol=1e-12),
                "Probe boundary arithmetic mismatch")
        row = {"base_sample_id": base, "pair_id": meta["pair_id"], "outer_fold": test_folds[base],
               "first_object_id": meta["first_object_id"], "correct_option": meta["correct_option"],
               "y": meta["y"], "probe_model_id": p["model_id"],
               "low_baseline_margin": low["answer_margin"], "temporal_baseline_margin": temporal["answer_margin"],
               "baseline_gap": temporal["answer_margin"] - low["answer_margin"],
               "patch_margin": patched["answer_margin"], "patch_margin_gain": delta_patch,
               "probe_evidence_gain": p["delta_S_temporal_to_low_L16"],
               "probe_boundary_gain": p["delta_S_boundary"], "ko_temporal_margin": k["temporal_ko_margin"],
               "ko_temporal_margin_change": k["delta_M_temporal"],
               "ko_temporal_damage": -k["delta_M_temporal"], "ko_low_margin_change": k["delta_M_low"],
               "ko_advantage_compression": k["compression"],
               "patch_strict_rescue": patched["answer_margin"] > 0,
               "patch_zero_margin_tie": patched["answer_margin"] == 0,
               "probe_evidence_increased": p["delta_S_temporal_to_low_L16"] > 0,
               "ko_strict_temporal_flip": ko_flip, "ko_zero_margin_tie": k["temporal_ko_margin"] == 0}
        require(all(math.isfinite(v) for v in row.values() if isinstance(v, (int, float))), "Non-finite result")
        joined.append(row)
    return joined


def rank_correlation(x, y, axis=0):
    a, b = rankdata(x, axis=axis), rankdata(y, axis=axis)
    a = a - np.mean(a, axis=axis, keepdims=True)
    b = b - np.mean(b, axis=axis, keepdims=True)
    denominator = np.sqrt(np.sum(a * a, axis=axis) * np.sum(b * b, axis=axis))
    return np.divide(np.sum(a * b, axis=axis), denominator,
                     out=np.full(np.shape(denominator), np.nan), where=denominator > 0)


def safe_float(value):
    return float(value) if np.isfinite(value) else None


def within_fold_rho(x, y, groups):
    a, b = np.zeros(len(x)), np.zeros(len(y))
    for group in sorted(set(groups)):
        selected = groups == group
        for values, dest in ((x, a), (y, b)):
            ranks = rankdata(values[selected])
            std = np.std(ranks)
            if std == 0:
                return None
            dest[selected] = (ranks - np.mean(ranks)) / std
    return float(np.mean(a * b))


def summarize(rows, resamples=5000, seed=20261026):
    require(resamples >= 100, "At least 100 diagnostic resamples required")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(resamples, len(rows)))
    groups = np.array([r["outer_fold"] for r in rows])
    associations = []
    for name, x_name, y_name in COMPARISONS:
        x, y = (np.array([r[key] for r in rows], dtype=float) for key in (x_name, y_name))
        boots = rank_correlation(x[indices], y[indices], axis=1)
        finite = boots[np.isfinite(boots)]
        require(len(finite) > .99 * resamples, "Too many constant bootstrap resamples")
        associations.append({"comparison": name, "x": x_name, "y": y_name,
            "case_count": len(rows), "spearman_rho": safe_float(rank_correlation(x, y)),
            "diagnostic_95_interval": np.quantile(finite, [.025, .975]).tolist(),
            "valid_bootstrap_resamples": len(finite), "undefined_bootstrap_resamples": int(resamples-len(finite)),
            "within_fold_standardized_rank_correlation": within_fold_rho(x, y, groups),
            "foldwise_rho": {str(g): safe_float(rank_correlation(x[groups == g], y[groups == g])) for g in sorted(set(groups))},
            "leave_one_fold_out_rho": {str(g): safe_float(rank_correlation(x[groups != g], y[groups != g])) for g in sorted(set(groups))},
            "leave_one_case_out_range": [float(v) for v in np.nanquantile([
                rank_correlation(np.delete(x, i), np.delete(y, i)) for i in range(len(rows))], [0, 1])],
            "by_first_mover_rho": {str(mover): safe_float(rank_correlation(
                x[[r["first_object_id"] == mover for r in rows]],
                y[[r["first_object_id"] == mover for r in rows]])) for mover in (1, 2)}})
    transitions = [{"patch_strict_rescue": rescue, "ko_strict_temporal_flip": flip,
                    "case_count": sum(r["patch_strict_rescue"] == rescue and
                                      r["ko_strict_temporal_flip"] == flip for r in rows)}
                   for rescue in (True, False) for flip in (True, False)]
    return {"schema": "phase3d_case_links_v1", "analysis_status": "exploratory_descriptive_same_cohort",
            "base_count": len(rows), "probe_refitting": False, "gpu_inference": False,
            "readout": "Frozen primary option_pair:L20_output; archived view only",
            "ko": "All-head options_all -> 512 matched whole_event2 keys, L18 only",
            "seed": seed, "bootstrap_resamples": resamples, "associations": associations,
            "counts": {"strict_patch_rescues": sum(r["patch_strict_rescue"] for r in rows),
                       "positive_probe_gains": sum(r["probe_evidence_increased"] for r in rows),
                       "strict_temporal_KO_flips": sum(r["ko_strict_temporal_flip"] for r in rows),
                       "patch_zero_margin_ties": sum(r["patch_zero_margin_tie"] for r in rows),
                       "KO_zero_margin_ties": sum(r["ko_zero_margin_tie"] for r in rows)},
            "transitions": transitions,
            "fold_counts": dict(Counter(str(r["outer_fold"]) for r in rows)),
            "first_mover_counts": dict(Counter(str(r["first_object_id"]) for r in rows)),
            "uncertainty": "Diagnostic case bootstrap conditional on saved fitted probes; no refitting, selection or multiplicity adjustment.",
            "limitations": ["All observations are from the same rescue-selected cohort, not independent confirmation.",
                "Probe units differ between outer-fold models; pooled rho has a fold-scale sensitivity analysis, not global calibration.",
                "Overlapping training folds create dependence not captured by a conditional case bootstrap.",
                "Whole-event KO lacks an equal-budget same-event background control; lexical confounding remains in the probe.",
                "Flip overlap depends on baseline margins and is not component attribution.",
                "Correlations do not establish mediation, a shared mechanism, or path-specific dependence."]}


def figures(rows, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    labels = {"patch_margin_gain": "L16 patch: correct-option margin gain",
              "probe_evidence_gain": "L20 patch-induced signed probe-score gain",
              "ko_temporal_damage": "L18 KO: temporal margin loss (positive = damage)"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), layout="constrained")
    for ax, association in zip(axes, summary["associations"]):
        for mover, color, marker in ((1, "#167b85", "o"), (2, "#aa3863", "^")):
            selected = [r for r in rows if r["first_object_id"] == mover]
            ax.scatter([r[association["x"]] for r in selected], [r[association["y"]] for r in selected],
                       c=color, marker=marker, alpha=.8, s=39, label=f"T{mover} first (n={len(selected)})")
        ax.axhline(0, color=".75", lw=.8)
        ax.axvline(0, color=".75", lw=.8)
        ax.set(xlabel=labels[association["x"]], ylabel=labels[association["y"]])
        ax.set_title(f"Spearman rho = {association['spearman_rho']:.3f}", fontsize=13)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=10)
    fig.suptitle("Cross-intervention case correspondence: 50 fixed rescue bases", fontsize=17)
    fig.supxlabel("Exploratory CPU-only join of frozen exports; no probe refitting. Probe units vary across folds.\n"
                  "Case correspondence does not establish an L16 -> L18 pathway or mediation.", fontsize=11)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"case_correspondence.{extension}", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 8.8))
    fig.subplots_adjust(left=.025, right=.975, top=.91, bottom=.04)
    ax.set(xlim=(0, 16), ylim=(0, 8))
    ax.axis("off")
    colors = ["#167b85", "#9b3b66", "#46658a", "#44744b"]
    headings = ["Early visual readout", "L16 visual block output", "Unpatched option readouts", "Answer selection"]
    bodies = ["Event-order label is decodable\nfrom L0 (100% held-out).\n\nAnnotation-guided Event-2 ROIs\nmay reveal the second mover;\nnot proof of abstract reasoning.",
              "Replace 512 matched Event-2\npositions with temporal states.\n\nIntervention occurs AFTER\nblock 16 attention and MLP;\noption states are not replaced.",
              "Changes begin at L17.\nFixed L20 probe gain: +0.391.\n\nFrozen held-out readout;\n92% lexical-control accuracy\nlimits semantic interpretation.",
              "L16 transplant increases\ncorrect-option margin by +1.581.\n\n34/50 low runs cross\nfrom negative to positive\nanswer margin."]
    for i, (heading, body, color) in enumerate(zip(headings, bodies, colors)):
        x = .25 + i * 4
        ax.plot([x, x + 3.45], [6.95, 6.95], color=color, lw=4)
        ax.text(x, 6.7, heading, fontsize=13, weight="bold", color=color, va="top")
        ax.text(x, 6.25, body, fontsize=11, va="top", linespacing=1.45)
        if i < 3:
            ax.add_patch(FancyArrowPatch((x + 3.4, 4.85), (x + 3.85, 4.85),
                                        arrowstyle="->", linestyle="--", color=".4", mutation_scale=14))
    ax.text(8, 3.72, "Dashed arrows: proposed explanatory ordering, not established causal pathways",
            ha="center", fontsize=11, color=".35")
    ax.axhline(3.38, color=".82", lw=1)
    ax.text(.25, 3.05, "SEPARATE ATTENTION INTERVENTION", color="#9b3b66", weight="bold", fontsize=12)
    ax.text(.25, 2.57, "Inside L18 attention: option-text queries cannot attend matched Event-2 visual keys (all heads).",
            fontsize=12)
    ax.text(.25, 2.05, "Strongest tested singleton temporal damage: -1.138 margin; 24/50 correct-to-wrong.\n"
            "Joint L16-19: -2.039; 30/50. L16-only: no temporal flips. These are different intervention sites.",
            fontsize=11, linespacing=1.6)
    ax.axhline(1.15, color=".82", lw=1)
    ax.text(.25, .83, "SUPPORTED SYNTHESIS", fontsize=11, weight="bold", color="#167b85")
    ax.text(.25, .38, "Evidence consistent with a causally sensitive middle-decoder integration stage; not the first emergence of order information,\n"
            "not a demonstrated probe mediator, and not an identified complete before/after circuit.",
            fontsize=11, linespacing=1.4)
    fig.suptitle("Candidate middle-decoder integration account", fontsize=21, y=.98)
    ax.text(8, 7.7, "50 equally weighted rescue bases; all decoder indices are zero-based", ha="center", fontsize=12, color=".4")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"middle_decoder_synthesis.{extension}", dpi=180)
    plt.close(fig)


def report(summary):
    lines = ["# Cross-Intervention Case Analysis", "", "Exploratory, descriptive, CPU-only; no inference or probe refitting.", "",
             "## Frozen Scope", "", "Join by base ID and exact pair ID: 50 archived L16 temporal-to-low patches, the frozen primary L20 option probe, and L18 singleton whole-Event-2 knockout. Baseline margins must match exactly between runs. All cases have equal weight.", "",
             "KO damage is **minus** the temporal KO margin change, so positive means greater damage. Probe scores are not answer-model logits.", "",
             "## Continuous Correspondence", "",
             "| Comparison | Spearman rho | Conditional diagnostic 95% interval | Within-fold standardized-rank correlation |", "| --- | ---: | --- | ---: |"]
    for a in summary["associations"]:
        lo, hi = a["diagnostic_95_interval"]
        lines.append(f"| {a['comparison']} | {a['spearman_rho']:.3f} | [{lo:.3f}, {hi:.3f}] | {a['within_fold_standardized_rank_correlation']:.3f} |")
    lines += ["", "All three requested descriptive comparisons are retained, regardless of magnitude. No p-values or mediation models are fitted. Fold-wise, leave-one-fold-out, leave-one-case-out and first-mover summaries are in `summary.json`; they are sensitivity diagnostics, not additional independent experiments.", "",
              "## Behavioral Transitions", "", "| L16 patch strictly rescues low | L18 KO strictly breaks temporal | Cases |", "| --- | --- | ---: |"]
    lines += [f"| {r['patch_strict_rescue']} | {r['ko_strict_temporal_flip']} | {r['case_count']} |" for r in summary["transitions"]]
    lines += ["", "A strict crossing excludes zero-margin ties. The interventions act on different recipients (patch on low; KO on temporal). Flip overlap is threshold-dependent and cannot identify the fraction of a mechanism explained.", "",
              "## Interpretation Boundaries", ""] + ["- " + x for x in summary["limitations"]]
    lines += ["", "## Reproducibility", "", "`case_links.csv` contains one row per base. `input_provenance.json` records checked source hashes and frozen run fingerprints; `output_manifest.json` records derived outputs. Frozen exports are read-only. This package does not include the full VM backups.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_evidence", type=Path, required=True)
    parser.add_argument("--singleton_evidence", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    for root in (args.source_evidence.resolve(), args.singleton_evidence.resolve()):
        require(output != root and root not in output.parents and output not in root.parents,
                "Output must be separate from frozen evidence")
    rows, provenance = load_inputs(args.source_evidence, args.singleton_evidence)
    summary = summarize(rows, args.resamples, args.seed)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "case_links.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "summary.json", summary)
    write_json(output / "input_provenance.json", {**provenance, "analysis_script_sha256": sha(__file__)})
    (output / "report.md").write_text(report(summary), encoding="utf-8")
    figures(rows, summary, output)
    for ref in provenance["checked_inputs"]:
        require(sha(ref["path"]) == ref["sha256"], "Frozen source changed during analysis")
    names = ["case_links.csv", "summary.json", "input_provenance.json", "report.md",
             "case_correspondence.png", "case_correspondence.pdf", "middle_decoder_synthesis.png",
             "middle_decoder_synthesis.pdf", "middle_decoder_synthesis.svg"]
    write_json(output / "output_manifest.json", {"files": [{"path": name, "sha256": sha(output / name)} for name in names],
        "frozen_sources_modified": False, "gpu_inference": False, "probe_refitting": False})
    print(json.dumps({"base_count": len(rows), "counts": summary["counts"],
                      "correlations": {a["comparison"]: a["spearman_rho"] for a in summary["associations"]}}, indent=2))


if __name__ == "__main__":
    main()
