"""Audit every frozen VM baseline before allowing any full-run patches."""

import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

try:
    from .common import read_jsonl
    from .phase3b_baseline import BaselineEngine, CONDITIONS, fingerprint, measurement_hashes, pair_failures, runtime_signature, sha256
    from .run_phase3b_vm import manifest_pairs, run_lock, write_json
except ImportError:
    from common import read_jsonl
    from phase3b_baseline import BaselineEngine, CONDITIONS, fingerprint, measurement_hashes, pair_failures, runtime_signature, sha256
    from run_phase3b_vm import manifest_pairs, run_lock, write_json


def require_gate(root, plan):
    root = Path(root)
    path = root / "baseline_audit/summary.json"
    if not path.is_file():
        raise RuntimeError("Missing cohort-wide VM baseline gate; run --stage baseline first.")
    summary = json.loads(path.read_text())
    config = json.loads((path.parent / "config.json").read_text())
    rows = path.parent / "rows.jsonl"
    if (not summary.get("passed") or summary.get("pipeline_fingerprint") != plan["pipeline_fingerprint"]
            or config.get("pipeline_fingerprint") != plan["pipeline_fingerprint"]
            or summary.get("rows_sha256") != sha256(rows)
            or summary.get("config_sha256") != sha256(path.parent / "config.json")
            or summary.get("pair_count") != plan["pair_count"]
            or config.get("measurement_code_sha256") != measurement_hashes()):
        raise RuntimeError("Baseline gate failed, incomplete, stale or incompatible; patching is blocked.")
    pairs = manifest_pairs(root / "selection/analysis_case_manifest.jsonl")
    saved = read_jsonl(rows)
    by_id = {row["eval_id"]: row for row in saved}
    if len(by_id) != len(saved) or set(by_id) != {row["eval_id"] for pair in pairs.values() for row in pair.values()}:
        raise RuntimeError("Baseline gate row coverage is invalid.")
    if any(pair_failures(pair, by_id) for pair in pairs.values()):
        raise RuntimeError("Baseline gate rows do not satisfy the strict behavioral/parity criteria.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--hardware_config", required=True)
    args = parser.parse_args()
    import torch
    from run_phase3b_patching import load_mappings, prepare_pair, validate_prepared_mapping
    root = Path(args.run_root)
    plan = json.loads((root / "vm_run_config.json").read_text())
    hardware = json.loads(Path(args.hardware_config).read_text())
    settings = SimpleNamespace(
        model_name=plan["model_name"], model_revision=plan["model_revision"],
        model_parallel=plan["execution_mode"] == "model_parallel",
        single_gpu=plan["execution_mode"] != "model_parallel",
        gpu_weight_budget_gib=plan["gpu_weight_budget_gib"], seed=plan["seed"],
        attn_implementation=plan["attn_implementation"],
        expected_gpu_hardware_path=args.hardware_config,
        project_root=Path(plan["path_map"]["/content/vlm-event-boundary"]),
        path_map=plan["path_map"], roi_padding=plan["roi_padding"],
        video_fps=plan["video_fps"], video_num_frames=plan["video_num_frames"],
        video_max_pixels=plan["video_max_pixels"],
    )
    environment = runtime_signature()
    if (environment["torch"].split("+")[0] != plan["expected_torch_version"]
            or environment["packages"]["transformers"] != plan["expected_transformers_version"]
            or environment["packages"]["qwen-vl-utils"] != plan["expected_qwen_vl_utils_version"]):
        raise RuntimeError("Baseline runtime differs from the frozen plan.")
    binding = {
        "schema": "phase3b_baseline_gate_v1", "pipeline_fingerprint": plan["pipeline_fingerprint"],
        "runtime": environment, "measurement_code_sha256": measurement_hashes(),
        "audit_code_sha256": {name: sha256(Path(__file__).parent / name) for name in (
            "run_phase3b_baseline.py", "phase3b_baseline.py",
        )},
        "hardware_config_sha256": sha256(args.hardware_config),
        "model_device_map": hardware["model_device_map"],
    }
    config = {**binding, "audit_fingerprint": fingerprint(binding)}
    output = root / "baseline_audit"
    pairs = manifest_pairs(root / "selection/analysis_case_manifest.jsonl")
    mappings = load_mappings(root / "selection/selected_video_mappings.jsonl")
    output.mkdir(exist_ok=True)
    with run_lock(output):
        config_path = output / "config.json"
        if config_path.is_file() and json.loads(config_path.read_text()) != config:
            raise RuntimeError("Baseline resume environment or inputs changed; preserve this run and use a new root.")
        write_json(config_path, config)
        rows = read_jsonl(output / "rows.jsonl") if (output / "rows.jsonl").is_file() else []
        saved = {row["eval_id"]: row for row in rows}
        expected = {row["eval_id"] for pair in pairs.values() for row in pair.values()}
        if len(saved) != len(rows) or not set(saved) <= expected:
            raise RuntimeError("Invalid baseline resume row coverage.")
        for item in rows:
            if item.get("audit_fingerprint") != config["audit_fingerprint"]:
                raise RuntimeError("Baseline row fingerprint mismatch.")
        reused = {}
        if (root / "checkpoint_reuse.json").is_file():
            for path in (root / "primary/checkpoints").glob("shard_*/activations/*/*/index.json"):
                index = json.loads(path.read_text())
                reused[index["eval_id"]] = index
        engine = None
        started = time.perf_counter()
        failures = []
        for number, (pair_id, pair) in enumerate(pairs.items(), 1):
            if not all(row["eval_id"] in saved for row in pair.values()):
                if engine is None:
                    engine = BaselineEngine(settings, hardware["model_device_map"])
                prepared = prepare_pair(pair, engine.processor, settings, engine.model.device)
                for condition in CONDITIONS:
                    row = pair[condition]
                    if row["eval_id"] in saved:
                        continue
                    validate_prepared_mapping(prepared[condition], mappings[pair_id], condition)
                    item = engine.audit(prepared[condition])
                    video = prepared[condition]["video_path"]
                    if item["video_sha256"] != plan["video_sha256"][video]:
                        raise RuntimeError(f"Frozen video changed: {row['eval_id']}.")
                    item.update({"pair_id": pair_id, "audit_fingerprint": config["audit_fingerprint"]})
                    rows.append(item)
                    saved[row["eval_id"]] = item
                    from activation_patching_core import atomic_write_jsonl
                    atomic_write_jsonl(output / "rows.jsonl", rows)
                del prepared
                gc.collect()
                torch.cuda.empty_cache()
            reasons = pair_failures(pair, saved)
            for row in pair.values():
                if row["eval_id"] in reused and saved[row["eval_id"]]["capture_decision"] != reused[row["eval_id"]]["decision"]:
                    reasons.append(f"{row['condition']}: fresh VM baseline differs from the reusable checkpoint")
            if reasons:
                failures.append({"pair_id": pair_id, "base_sample_id": pair["low_boundary"]["base_sample_id"], "reasons": reasons})
            write_json(output / "summary.json", {
                "schema": "phase3b_baseline_gate_v1", "passed": False,
                "pipeline_fingerprint": plan["pipeline_fingerprint"],
                "completed_pairs": number, "pair_count": len(pairs), "completed_rows": len(rows),
                "failures": failures, "elapsed_sec": time.perf_counter() - started,
            })
            print(f"BASELINE {number}/{len(pairs)}: {pair_id} {'PASS' if not reasons else 'BLOCKED'}; "
                  f"{len(rows)}/{len(expected)} rows; elapsed={(time.perf_counter()-started)/60:.1f} min; checkpoint={output}", flush=True)
        write_json(output / "summary.json", {
            "schema": "phase3b_baseline_gate_v1", "passed": not failures,
            "pipeline_fingerprint": plan["pipeline_fingerprint"], "pair_count": len(pairs),
            "completed_rows": len(rows), "failures": failures,
            "rows_sha256": sha256(output / "rows.jsonl"), "config_sha256": sha256(config_path),
            "elapsed_sec": time.perf_counter() - started, "patches_performed": 0,
        })
        require_gate(root, plan)
        print("Cohort-wide VM baseline gate PASSED. No patches performed.", flush=True)


if __name__ == "__main__":
    main()
