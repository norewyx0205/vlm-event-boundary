"""CPU analysis of Phase 3H: layout-symmetric S and B preference per timing cell, planned contrasts and reading rule.

Contrasts follow docs/phase3h_timing_dissociation.md exactly; the fresh cohort is primary, the rescue cohort
secondary (only G90 is selection-free there). Intervals are 95% base bootstrap intervals (2,000 draws).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.phase3c_core import atomic_write, digest, file_hash, read_json

SCHEMA = "phase3h_timing_analysis_v1"
CELLS = ("low", "low_mid", "low_late", "temporal_early", "temporal")
CONTRASTS = {"G105": ("temporal", "low_late"), "G90": ("temporal_early", "low_mid"),
             "T0": ("low_late", "low"), "T45": ("temporal", "temporal_early"), "Total": ("temporal", "low")}
SELECTION_FREE = {"fresh": tuple(CONTRASTS), "rescue": ("G90",)}
DRAWS, SEED = 2000, 20261103


def per_base(rows, cohort):
    """{base: {cell: {"S", "b", "correct_A", "correct_B"}}} from both layouts of every cell."""
    table = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["cohort"] == cohort:
            table[r["base_sample_id"]][r["timing_cell"]][r["correct_option"]] = r
    result = {}
    for base, cells in sorted(table.items()):
        if set(cells) != set(CELLS) or any(set(v) != {"A", "B"} for v in cells.values()):
            raise ValueError(f"Base {base} lacks a cell or one of the two answer layouts.")
        result[base] = {cell: {"S": (v["A"]["margin"]+v["B"]["margin"])/2, "b": (v["B"]["margin"]-v["A"]["margin"])/2,
                               "correct_A": bool(v["A"]["is_correct"]), "correct_B": bool(v["B"]["is_correct"])}
                        for cell, v in cells.items()}
    return result


def interval(values, rng):
    values = np.asarray(values, dtype=float)
    draws = values[rng.integers(0, len(values), (DRAWS, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5)), "bases_above_zero": int((values > 0).sum()), "n": len(values)}


def contrasts(table, seed=SEED):
    rng = np.random.default_rng(seed)
    bases = sorted(table)
    result = {name: interval([table[b][left]["S"]-table[b][right]["S"] for b in bases], rng)
              for name, (left, right) in CONTRASTS.items()}
    identity = max(abs(table[b]["temporal"]["S"]-table[b]["low"]["S"]-
                       (table[b]["temporal"]["S"]-table[b]["low_late"]["S"])-(table[b]["low_late"]["S"]-table[b]["low"]["S"]))
                   for b in bases)
    if identity > 1e-9:
        raise ValueError("Total != G105 + T0 per base.")
    return result


def cell_summary(table, seed=SEED):
    rng = np.random.default_rng(seed+1)
    bases = sorted(table)
    return {cell: {"S": interval([table[b][cell]["S"] for b in bases], rng),
                   "b": interval([table[b][cell]["b"] for b in bases], rng),
                   "correct_A_layout": sum(table[b][cell]["correct_A"] for b in bases),
                   "correct_B_layout": sum(table[b][cell]["correct_B"] for b in bases),
                   "S_above_zero": sum(table[b][cell]["S"] > 0 for b in bases)} for cell in CELLS}


def reading(result):
    """Pre-specified reading rule (primary cohort)."""
    g105, g90, t0, total = (result[k] for k in ("G105", "G90", "T0", "Total"))
    excludes = lambda c: c["ci_low"] > 0 or c["ci_high"] < 0
    if excludes(g105) and g105["mean"] >= total["mean"]/2 and np.sign(g90["mean"]) == np.sign(g105["mean"]):
        return "gap_dominant"
    if excludes(t0) and not excludes(g105):
        return "timing_dominant"
    return "mixed_or_inconclusive"


def analyze(results_path):
    results = read_json(results_path)
    if results.get("complete") is not True or results["fingerprint"] != digest({k: v for k, v in results.items() if k != "fingerprint"}):
        raise ValueError("Need the complete, unaltered Phase 3H results.")
    output = {"schema": SCHEMA, "results_sha256": file_hash(results_path), "draws": DRAWS, "seed": SEED, "cohorts": {}}
    for cohort in ("fresh", "rescue"):
        table = per_base(results["rows"], cohort)
        found = contrasts(table)
        output["cohorts"][cohort] = {"bases": len(table), "cells": cell_summary(table), "contrasts": found,
                                     "selection_free_contrasts": list(SELECTION_FREE[cohort])}
    output["primary_reading"] = reading(output["cohorts"]["fresh"]["contrasts"])
    return output


def report(analysis):
    lines = ["# Phase 3H timing dissociation: results", "",
             f"Primary reading (fresh cohort, pre-specified rule): **{analysis['primary_reading']}**", ""]
    for cohort, data in analysis["cohorts"].items():
        lines += [f"## {cohort} cohort ({data['bases']} bases)", "",
                  "| Cell | S [95%] | b [95%] | S > 0 | correct, A layout / B layout |", "| --- | --- | --- | --- | --- |"]
        for cell, v in data["cells"].items():
            lines.append(f"| {cell} | {v['S']['mean']:+.3f} [{v['S']['ci_low']:+.3f}, {v['S']['ci_high']:+.3f}] | "
                         f"{v['b']['mean']:+.3f} [{v['b']['ci_low']:+.3f}, {v['b']['ci_high']:+.3f}] | "
                         f"{v['S_above_zero']}/{data['bases']} | {v['correct_A_layout']} / {v['correct_B_layout']} |")
        lines += ["", "| Contrast | Mean [95%] | Bases > 0 | Selection-free |", "| --- | --- | --- | --- |"]
        for name, v in data["contrasts"].items():
            lines.append(f"| {name} | {v['mean']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}] | {v['bases_above_zero']}/{v['n']} | "
                         f"{'yes' if name in data['selection_free_contrasts'] else 'no'} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    analysis = analyze(args.results)
    atomic_write(output/"analysis.json", analysis)
    (output/"report.md").write_text(report(analysis))
    print(report(analysis))
