"""Layerwise 2x2 (training source x test condition) direct-D and single-video probes on the all-layer store."""

import argparse
import json
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts import phase3e_order_probes as dprobes
from scripts.phase3c_core import atomic_write, digest, file_hash, frozen_write, read_json
from scripts.phase3e_option_probes import SETTINGS

SCHEMA = "phase3g_layerwise_probes_v1"
LOW, TEMPORAL = "low_boundary", "temporal_boundary"
D_REGIMES = ("low_only", "temporal_only", "mixed_matched")
H_REGIMES = ("low_only", "temporal_only")
DRAWS = 1000


def pairing(preparation, endpoints):
    """Endpoint indices of every directed D row, recovered through the frozen pairing function."""
    index = np.repeat(np.arange(len(endpoints), dtype=np.float64)[:, None, None], 4, axis=1)
    rows, _, left, right = dprobes.paired_dataset(preparation, endpoints, index)
    return rows, left[:, 0, 0].astype(int), right[:, 0, 0].astype(int)


def auc(labels, scores):
    """Mann-Whitney AUC with average ranks for ties."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    positive, negative = labels == 1, labels == 0
    if not positive.any() or not negative.any():
        return float("nan")
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start
        while end+1 < len(scores) and sorted_scores[end+1] == sorted_scores[start]:
            end += 1
        ranks[order[start:end+1]] = (start+end)/2+1
        start = end+1
    return float((ranks[positive].sum() - positive.sum()*(positive.sum()+1)/2) / (positive.sum()*negative.sum()))


def d_table(predictions, norms):
    """One canonical row per statement pair: base, fold, condition, correct, evidence, norm, endpoint scores."""
    rows = []
    for p in predictions:
        if p["direction"] != "original_minus_reversed":
            continue
        rows.append({"base": p["base_sample_id"], "fold": p["model_id"], "condition": p["condition"],
            "correct": int(p["probe_prediction"] == p["y_statement_true"]), "evidence": p["true_statement_evidence"],
            "unit_evidence": p["true_statement_evidence"]/norms[p["example_id"]],
            "single": [(p["left_endpoint_projection"], p["y_statement_true"]),
                       (p["right_endpoint_projection"], 1-p["y_statement_true"])]})
    return rows


def h_table(predictions):
    return [{"base": p["base_sample_id"], "fold": p["model_id"], "condition": p["condition"],
             "correct": int(p["probe_prediction"] == p["y_statement_true"]),
             "single": [(p["probe_logit"], p["y_statement_true"])]} for p in predictions]


def summarize(rows, condition, bases=None):
    """Metrics for one test condition, optionally on a resampled list of bases (with repeats)."""
    by_base = {}
    for r in rows:
        if r["condition"] == condition:
            by_base.setdefault(r["base"], []).append(r)
    chosen = [b for b in (bases if bases is not None else sorted(by_base))]
    selected = [r for b in chosen for r in by_base[b]]
    result = {"accuracy": float(np.mean([r["correct"] for r in selected]))}
    if "evidence" in selected[0]:
        result["evidence"] = float(np.mean([r["evidence"] for r in selected]))
        result["unit_evidence"] = float(np.mean([r["unit_evidence"] for r in selected]))
    folds = {}
    for r in selected:
        folds.setdefault(r["fold"], []).extend(r["single"])
    result["single_video_auc"] = float(np.nanmean([auc([y for _, y in v], [s for s, _ in v]) for v in folds.values()]))
    return result


def contrast(left_rows, left_condition, right_rows, right_condition, seed):
    """right - left with a shared base resample; returns point estimate and 95% interval per metric."""
    point_left, point_right = summarize(left_rows, left_condition), summarize(right_rows, right_condition)
    bases = sorted({r["base"] for r in left_rows})
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in point_left}
    for _ in range(DRAWS):
        sample = list(rng.choice(bases, len(bases)))
        a, b = summarize(left_rows, left_condition, sample), summarize(right_rows, right_condition, sample)
        for k in draws:
            draws[k].append(b[k]-a[k])
    return {k: {"left": point_left[k], "right": point_right[k], "difference": point_right[k]-point_left[k],
                "ci_low": float(np.percentile(draws[k], 2.5)), "ci_high": float(np.percentile(draws[k], 97.5))}
            for k in point_left}


def layer_analysis(layer, values, rows, left, right, folds, selection, examples, h_values, h_folds, h_selection, settings):
    # Subtract in float64, as the frozen direct-D loader does; crossfit requires D == left - right exactly.
    a, b = values[left, layer].astype(np.float64), values[right, layer].astype(np.float64)
    D = a-b
    norms = {r["example_id"]: float(np.linalg.norm(d)) for r, d in zip(rows, D)}
    tables = {}
    for regime in D_REGIMES:
        fitted = dprobes.crossfit(rows, D, a, b, folds, selection, regime, settings)
        tables[("D", regime)] = d_table(fitted["predictions"], norms)
    for regime in H_REGIMES:
        fitted = dprobes.previous.fit_source(examples, h_values[:, layer], h_folds, h_selection, regime, settings=settings)
        tables[("h", regime)] = h_table(fitted["predictions"])
    result = {"cells": {}, "test_gap": {}, "train_gap": {}}
    for (kind, regime), table in tables.items():
        result["cells"][f"{kind}:{regime}"] = {c: summarize(table, c) for c in (LOW, TEMPORAL)}
        result["test_gap"][f"{kind}:{regime}"] = contrast(table, LOW, table, TEMPORAL, seed=layer)
    for kind in ("D", "h"):
        for condition in (LOW, TEMPORAL):
            result["train_gap"][f"{kind}:{condition}"] = contrast(
                tables[(kind, "low_only")], condition, tables[(kind, "temporal_only")], condition, seed=layer+100)
    return result


def run(store_path, output_root, layers=None):
    from scripts.run_phase3g_alllayer_capture import LAYERS, load_store
    store_path, output = Path(store_path).resolve(), Path(output_root).resolve()
    if output == store_path.parent or store_path.parent in output.parents:
        raise ValueError("Keep layerwise analysis outside the frozen store.")
    manifest, preparation, endpoints, values = load_store(store_path)
    rows, left, right = pairing(preparation, endpoints)
    folds, selection = dprobes.previous.folds_for(preparation), dprobes.selections(rows, preparation)
    original = [i for i, e in enumerate(endpoints) if e["order_variant"] == "original"]
    examples = [endpoints[i] for i in original]
    h_values = values[original]
    h_selection = dprobes.previous.selections(examples, preparation)
    settings = dict(SETTINGS)
    output.mkdir(parents=True, exist_ok=True)
    config = {"schema": SCHEMA, "store_manifest_sha256": file_hash(store_path), "settings": settings,
              "d_regimes": list(D_REGIMES), "h_regimes": list(H_REGIMES), "bootstrap_draws": DRAWS,
              "code_sha256": {n: file_hash(Path(__file__).with_name(n)) for n in
                              ("phase3g_layerwise_probes.py", "phase3e_order_probes.py", "phase3e_rescue50_probes.py",
                               "phase3e_option_probes.py")}}
    if (output/"config.json").exists() and read_json(output/"config.json") != config:
        raise ValueError("Existing layerwise output was produced by a different store/code.")
    frozen_write(output/"config.json", config)
    results = {}
    for layer in (layers if layers is not None else LAYERS):
        path = output/"layers"/f"L{layer:02d}.json"
        if not path.exists():
            frozen_write(path, layer_analysis(layer, values, rows, left, right, folds, selection,
                                              examples, h_values, folds, h_selection, settings))
        results[str(layer)] = read_json(path)
        print("LAYERWISE", layer, flush=True)
    summary = {"schema": SCHEMA, "complete": layers is None, "layers": sorted(results, key=int),
               "results": results, "store_fingerprint": manifest["fingerprint"]}
    atomic_write(output/"summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--layers", help="comma-separated subset for a quick look; omit for all 36")
    args = parser.parse_args()
    chosen = [int(x) for x in args.layers.split(",")] if args.layers else None
    print(json.dumps({k: v for k, v in run(args.store, args.output_root, chosen).items() if k != "results"}, indent=2))
