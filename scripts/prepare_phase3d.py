"""Audit Phase 3D metadata without Torch, model loading, training or GPU work."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .phase3c_core import atomic_write, checked_positions, file_hash, read_json
    from .phase3d_core import SCHEMA, CONDITIONS, audit_pairs, base_folds, declared_saved_sites
except ImportError:
    from phase3c_core import atomic_write, checked_positions, file_hash, read_json
    from phase3d_core import SCHEMA, CONDITIONS, audit_pairs, base_folds, declared_saved_sites


def capture_coverage(pairs, mappings, indexes):
    by_eval = defaultdict(list)
    for index in indexes:
        if index.get("artifact_type") != "real":
            raise ValueError("Mock/untyped capture metadata is not a reuse candidate.")
        by_eval[index["eval_id"]].append(index)
    rows = []
    for pair_id, pair in pairs.items():
        mapping = mappings[pair_id]
        for condition in CONDITIONS:
            eval_id = pair[condition]["eval_id"]
            record = mapping["processor_records"][condition]
            side = "low_positions" if condition == CONDITIONS[0] else "temporal_positions"
            visual = set(checked_positions(mapping["support_audit"]["supports"]["whole_event2"][side],
                                           "whole_event2"))
            text = set(checked_positions(record["group_positions"]["option_target_1_mentions"], "T1_mentions"))
            text.update(checked_positions(record["group_positions"]["option_target_2_mentions"], "T2_mentions"))
            required = {"L16_input_visual_via_L15_output": ((15, "block_output"), visual),
                        "L16_output_visual": ((16, "block_output"), visual),
                        **{f"L{layer}_option_mentions": ((layer, "block_output"), text)
                           for layer in range(17, 25)}}
            available = {}
            for name, (site, positions) in required.items():
                available[name] = any(site in (declared_saved_sites(index) or set()) and
                    positions <= set(index.get("positions", [])) for index in by_eval[eval_id])
            rows.append({"pair_id": pair_id, "condition": condition, "eval_id": eval_id,
                         "metadata_candidate_coverage": available,
                         "ambiguous_saved_site_indexes": sum(declared_saved_sites(index) is None
                             for index in by_eval[eval_id]),
                         "tensor_hash_and_live_replay_verified": False,
                         "patched_downstream_capture_verified": False})
    counts = {name: sum(row["metadata_candidate_coverage"][name] for row in rows)
              for name in rows[0]["metadata_candidate_coverage"]}
    return {"conditions": len(rows), "candidate_condition_counts": counts, "rows": rows,
            "scope": "Metadata candidates only; no tensor bytes or live parity verified. "
                     "L15-to-L16-input equivalence still requires an architecture gate. "
                     "Observed-but-not-declared-saved sites are not assumed present."}


def prepare(args):
    pairs_path = Path(args.pairs_path).resolve()
    pairs = read_json(pairs_path)
    cases = audit_pairs(pairs, args.expected_cases)
    folds = base_folds(cases, args.folds, args.seed)
    inputs = {str(pairs_path): file_hash(pairs_path)}
    coverage = None
    if args.mappings_path:
        mappings_path = Path(args.mappings_path).resolve()
        mappings = read_json(mappings_path)
        inputs[str(mappings_path)] = file_hash(mappings_path)
        indexes = []
        if args.source_capture_refs:
            path = Path(args.source_capture_refs).resolve()
            inputs[str(path)] = file_hash(path)
            indexes.extend(value["index"] for value in read_json(path).values())
        for root in args.capture_root:
            root = Path(root).resolve()
            paths = sorted(root.rglob("index.json"))
            if not paths:
                raise ValueError(f"No capture indexes under explicitly supplied root: {root}")
            for path in paths:
                inputs[str(path)] = file_hash(path)
                indexes.append(read_json(path))
        coverage = capture_coverage(pairs, mappings, indexes)
    elif args.capture_root or args.source_capture_refs:
        raise ValueError("Capture coverage needs mappings; do not infer positions from tensor shapes.")
    summary = {"schema": SCHEMA, "artifact_type": "metadata_audit", "metadata_audit_complete": True,
               "gpu_preflight_passed": False, "probe_trained": False, "experiment_complete": False,
               "base_count": len(cases), "condition_rows": 2*len(cases),
               "primary_reporting": f"All {len(cases)} bases equally; provenance is retained, not a primary stratum.",
               "canonical_relation_counts": dict(Counter(row["canonical_relation"] for row in cases)),
               "sentence_relation_counts": dict(Counter(row["sentence_relation"] for row in cases)),
               "correct_option_counts": dict(Counter(row["correct_option"] for row in cases)),
               "archived_prompt_variant_counts": dict(Counter(row["prompt_variant"] for row in cases)),
               "answer_majority_accuracy": max(Counter(row["correct_option"] for row in cases).values())/len(cases),
               "fold_seed": args.seed, "outer_folds": args.folds,
               "source_sha256": inputs,
               "protocol_sha256": file_hash(Path(__file__).resolve().parents[1]/"docs/phase3d_protocol.md"),
               "preparation_code_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in
                                           ("prepare_phase3d.py", "phase3d_core.py")},
               "warnings": ["Canonical T1/T2 relation is not necessarily sentence correct_relation.",
                            "Archived swapped is not necessarily a pure A/B permutation.",
                            "No new view is screened for rescue or counted as an independent case.",
                            "Folds bind every condition/view/patch of the same base.",
                            "This audit does not freeze a launch-ready experiment or verify model outputs."]}
    output = Path(args.output_dir).resolve()
    sources = [pairs_path.parent]
    sources.extend(Path(root).resolve() for root in args.capture_root)
    sources.extend(Path(path).resolve().parent for path in (args.mappings_path, args.source_capture_refs) if path)
    if any(output == root or output in root.parents or root in output.parents for root in sources):
        raise ValueError("Use an isolated Phase 3D output root, outside all source data roots.")
    outputs = {"canonical_case_metadata.json": cases, "outer_folds.json": folds,
               "option_order_controls.json": [{"pair_id": row["pair_id"], **row["option_order_control"]}
                                               for row in cases]}
    if coverage is not None:
        outputs["capture_coverage.json"] = coverage
    # Check all existing preparation files before creating any additional file.
    for name, value in outputs.items():
        if (output/name).exists() and read_json(output/name) != value:
            raise ValueError("Frozen Phase 3D metadata differs; use a new output directory.")
    audit_path = output/"metadata_audit.json"
    if audit_path.exists():
        prior = read_json(audit_path)
        if {key: value for key, value in prior.items() if key != "output_sha256"} != summary:
            raise ValueError("Phase 3D preparation sources/settings/code changed; use a new output directory.")
    for name, value in outputs.items():
        if not (output/name).exists():
            atomic_write(output/name, value)
    summary["output_sha256"] = {name: file_hash(output/name) for name in outputs}
    if audit_path.exists() and read_json(audit_path) != summary:
        raise ValueError("Frozen Phase 3D output bytes changed; inspect before reuse.")
    if not audit_path.exists():
        atomic_write(audit_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_cases", type=int, default=50)
    parser.add_argument("--mappings_path")
    parser.add_argument("--source_capture_refs")
    parser.add_argument("--capture_root", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    result = prepare(parser.parse_args())
    print(f"Phase 3D CPU metadata audit: {result['base_count']} bases; "
          f"canonical labels={result['canonical_relation_counts']}; "
          f"answer labels={result['correct_option_counts']}; "
          "GPU not started, probe not trained.", flush=True)


if __name__ == "__main__":
    main()
