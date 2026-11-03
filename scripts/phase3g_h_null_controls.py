"""Structure-preserving permutation null and mechanism check for the below-chance single-video h-probe.

The null flips the truth of whole bases (both statements, both layouts, both conditions together), so every
base keeps one true and one false statement per prompt, the same text, folds and selection; only the
association between a base's representation and its true order is broken. Two nulls are drawn:
"free" flips each base independently; "within_fold" permutes which bases have 'before' true inside each
outer test fold, so every fold keeps its count of before-true bases (the lexical composition that a
statement-wording rule can exploit) while individual bases lose their link to the video. Each draw re-runs
the frozen h-probe procedure (inner C selection included) on the all-layer store.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts import phase3e_order_probes as dprobes
from scripts.phase3c_core import atomic_write, file_hash, frozen_write, read_json
from scripts.phase3e_option_probes import SETTINGS

SCHEMA = "phase3g_h_null_controls_v1"
LOW, TEMPORAL = "low_boundary", "temporal_boundary"
REGIMES = ("low_only", "temporal_only")
SEED = 20261103
STATE = {}


def flipped(examples, bases):
    return [{**e, "y_statement_true": 1-e["y_statement_true"]} if e["base_sample_id"] in bases else e
            for e in examples]


def flip_sets(bases, count, seed=SEED):
    rng = np.random.default_rng(seed)
    return [sorted(int(b) for b in np.asarray(bases)[rng.random(len(bases)) < 0.5]) for _ in range(count)]


def within_fold_flip_sets(examples, folds, count, seed=SEED):
    """Flip sets that keep each outer test fold's number of before-true bases fixed."""
    kind = {e["base_sample_id"]: e["relation_word"] == "before" for e in examples if e["y_statement_true"]}
    rng, result = np.random.default_rng(seed), []
    for _ in range(count):
        flips = []
        for fold in folds:
            bases = sorted(fold["test_scene_seeds"])
            shuffled = rng.permutation([kind[b] for b in bases])
            flips += [int(b) for b, new in zip(bases, shuffled) if new != kind[b]]
        result.append(sorted(flips))
    return result


def accuracies(predictions):
    hits = defaultdict(list)
    for p in predictions:
        hits[p["condition"]].append(p["probe_prediction"] == p["y_statement_true"])
    return {c: float(np.mean(hits[c])) for c in (LOW, TEMPORAL)}


def fit(examples, values, regime, categorical=False):
    fitted = dprobes.previous.fit_source(examples, values, STATE["folds"], STATE["selection"], regime,
                                         categorical=categorical, settings=dict(SETTINGS))
    return fitted["predictions"]


def initialise(examples, folds, selection, values, layers, lexical):
    STATE.update(examples=examples, folds=folds, selection=selection, values=values, layers=layers, lexical=lexical)


def permuted(task):
    """One null draw: a layer index (real features) or a lexical-control name (categorical features)."""
    source, regime, bases = task
    examples = flipped(STATE["examples"], set(bases))
    if isinstance(source, str):
        return accuracies(fit(examples, STATE["lexical"][source], regime, categorical=True))
    return accuracies(fit(examples, STATE["values"][:, STATE["layers"].index(source)], regime))


def mechanism(examples, folds, predictions):
    """Per held-out fold: how many bases have 'before' true, which relation word the probe calls true, accuracy."""
    kind = {e["base_sample_id"]: e["relation_word"] for e in examples if e["y_statement_true"]}
    result = []
    for fold in folds:
        test = set(fold["test_scene_seeds"])
        rows = [p for p in predictions if p["base_sample_id"] in test]
        called = Counter((p["relation_word"], p["probe_prediction"]) for p in rows)
        result.append({"outer_fold": fold["outer_fold"], "test_bases": len(test),
            "test_before_true": sum(kind[b] == "before" for b in test),
            "train_before_true": sum(kind[b] == "before" for b in kind if b not in test),
            "share_called_true": {r: called[(r, 1)]/max(1, called[(r, 0)]+called[(r, 1)]) for r in ("before", "after")},
            "accuracy": float(np.mean([p["probe_prediction"] == p["y_statement_true"] for p in rows]))})
    return result


def null_summary(observed, draws):
    draws = np.asarray(draws)
    return {"observed": observed, "null_mean": float(draws.mean()), "null_sd": float(draws.std(ddof=1)),
            "null_q025": float(np.percentile(draws, 2.5)), "null_q975": float(np.percentile(draws, 97.5)),
            "p_upper": float(((draws >= observed).sum()+1)/(len(draws)+1)),
            "p_lower": float(((draws <= observed).sum()+1)/(len(draws)+1)), "draws": len(draws)}


def run(store_path, layerwise_summary, output_root, null_layers, permutations, lexical_permutations, workers):
    from scripts.run_phase3g_alllayer_capture import load_store
    output = Path(output_root).resolve()
    manifest, preparation, endpoints, values = load_store(store_path)
    original = [i for i, e in enumerate(endpoints) if e["order_variant"] == "original"]
    examples = [endpoints[i] for i in original]
    folds = dprobes.previous.folds_for(preparation)
    selection = dprobes.previous.selections(examples, preparation)
    bases = sorted({e["base_sample_id"] for e in examples})
    nulls = {"within_fold": (within_fold_flip_sets(examples, folds, permutations, SEED+2),
                             within_fold_flip_sets(examples, folds, lexical_permutations, SEED+3)),
             "free": (flip_sets(bases, permutations), flip_sets(bases, lexical_permutations, SEED+1))}
    layers = sorted({layer for chosen in null_layers.values() for layer in chosen})
    config = {"schema": SCHEMA, "store_manifest_sha256": file_hash(store_path),
              "layerwise_summary_sha256": file_hash(layerwise_summary), "settings": dict(SETTINGS),
              "null_layers": null_layers, "regimes": list(REGIMES), "permutations": permutations,
              "lexical_permutations": lexical_permutations, "seed": SEED,
              "rules": {"free": "flip y for all rows of each base independently with probability 0.5",
                        "within_fold": "permute before/after-true base types within each outer test fold"},
              "code_sha256": file_hash(Path(__file__))}
    output.mkdir(parents=True, exist_ok=True)
    if (output/"config.json").exists() and read_json(output/"config.json") != config:
        raise ValueError("Existing null output was produced by a different store/code/config.")
    frozen_write(output/"config.json", config)
    lexical = dprobes.previous.lexical_features(examples, preparation)
    controls_features = {"lexical_full": lexical, "relation_only": [{"relation": r["relation"]} for r in lexical]}
    initialise(examples, folds, selection, None, layers, controls_features)
    observed_summary = read_json(layerwise_summary)["results"]

    def cell(name, tasks):
        path = output/"cells"/f"{name}.json"
        if not path.exists():
            frozen_write(path, {"cell": name, "draws": list(pool.map(permuted, tasks))})
        return read_json(path)["draws"]

    controls, results = {}, {}
    with ProcessPoolExecutor(workers, initializer=initialise,
                             initargs=(examples, folds, selection, values[original][:, layers], layers,
                                       controls_features)) as pool:
        # Text-only controls: the statement's lexical content alone, observed and under both nulls.
        for name, features in controls_features.items():
            predictions = fit(examples, features, "low_only", categorical=True)
            observed = accuracies(predictions)
            controls[name] = {"accuracy": observed, "mechanism": mechanism(examples, folds, predictions)}
            for null, (_, lexical_flips) in nulls.items():
                draws = cell(f"{null}_lexical_{name}", [(name, "low_only", f) for f in lexical_flips])
                controls[name][f"{null}_null_low"] = null_summary(observed[LOW], [d[LOW] for d in draws])
            print("CONTROL", name, observed[LOW], {n: round(controls[name][f"{n}_null_low"]["null_mean"], 3) for n in nulls}, flush=True)
        for null, (flips, _) in nulls.items():
            for layer in null_layers[null]:
                for regime in REGIMES:
                    draws = cell(f"{null}_L{layer:02d}_{regime}", [(layer, regime, f) for f in flips])
                    observed = observed_summary[str(layer)]["cells"][f"h:{regime}"]
                    results[f"{null}:L{layer}:{regime}"] = {c: null_summary(observed[c]["accuracy"], [d[c] for d in draws])
                                                            for c in (LOW, TEMPORAL)}
                    print("NULL", null, layer, regime, {c: (v["observed"], round(v["null_mean"], 3), round(v["p_upper"], 3))
                                                        for c, v in results[f"{null}:L{layer}:{regime}"].items()}, flush=True)
    observed_l5 = fit(examples, values[original][:, 5], "low_only") if 5 in layers else None
    summary = {"schema": SCHEMA, "store_fingerprint": manifest["fingerprint"], "controls": controls, "null": results,
               "mechanism_L5_low_only": mechanism(examples, folds, observed_l5) if observed_l5 else None}
    atomic_write(output/"summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--layerwise_summary", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--free_layers", default="5,16,18")
    parser.add_argument("--within_fold_layers", default="5,10,16,17,18")
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--lexical_permutations", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    chosen = {"free": [int(x) for x in args.free_layers.split(",")],
              "within_fold": [int(x) for x in args.within_fold_layers.split(",")]}
    summary = run(args.store, args.layerwise_summary, args.output_root, chosen,
                  args.permutations, args.lexical_permutations, args.workers)
    print(json.dumps({k: v for k, v in summary.items() if k in ("controls", "null")}, indent=1)[:4000])
