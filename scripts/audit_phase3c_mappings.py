"""Complete Phase 3C mapping/control eligibility using the CPU processor only."""

import argparse
import ast
import json
import os
import time
from importlib.metadata import version
from pathlib import Path

try:
    from .phase3b_paths import load_path_map
    from .phase3c_core import (
        CONDITIONS, SCHEMA, atomic_write, digest, file_hash, frozen_write,
        input_ids_hash, prepare_support_audit, read_json, read_jsonl, select_balanced,
    )
except ImportError:
    from phase3b_paths import load_path_map
    from phase3c_core import (
        CONDITIONS, SCHEMA, atomic_write, digest, file_hash, frozen_write,
        input_ids_hash, prepare_support_audit, read_json, read_jsonl, select_balanced,
    )


def load_processor(settings):
    # This entry point never needs GPU visibility or model weights.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor

    runtime = {"transformers": transformers.__version__, "torch": torch.__version__,
               "qwen_vl_utils": version("qwen-vl-utils"), "device": "cpu",
               "model_weights_loaded": False}
    for name, expected in (("transformers", settings["expected_transformers_version"]),
                           ("qwen_vl_utils", settings["expected_qwen_vl_utils_version"])):
        if runtime[name] != expected:
            raise ValueError(f"CPU processor runtime mismatch: {name}={runtime[name]}, expected {expected}.")
    if runtime["torch"].split("+")[0] != settings["expected_torch_version"]:
        raise ValueError("CPU processor Torch version differs from the reference runtime.")
    kwargs = {"revision": settings["model_revision"]}
    config = AutoConfig.from_pretrained(settings["model_name"], **kwargs)
    if (config.text_config.num_hidden_layers != 36 or config.text_config.num_attention_heads != 32 or
            config.vision_config.deepstack_visual_indexes != [8, 16, 24] or config.video_token_id != 151656):
        raise ValueError("Pinned architecture does not match the DeepStack/position audit assumptions.")
    runtime["deepstack_decoder_injection_layers"] = list(range(len(config.vision_config.deepstack_visual_indexes)))
    return AutoProcessor.from_pretrained(settings["model_name"], **kwargs), runtime


def processor_condition(candidate, condition, processor, settings, project_root, path_map):
    try:
        from .phase3b_core import prepare_example
        from .probe_attention_roi import token_descriptors, video_shape, visual_positions
    except ImportError:
        from phase3b_core import prepare_example
        from probe_attention_roi import token_descriptors, video_shape, visual_positions

    prepared = prepare_example(candidate["rows"][condition], processor, project_root,
        settings["video_fps"], settings["video_num_frames"], settings["video_max_pixels"],
        settings["roi_padding"], path_map=path_map)
    actual_path = prepared["video_path"]
    expected = candidate["video_provenance"][condition]["sha256"]
    if file_hash(actual_path) != expected:
        raise ValueError("Video bytes differ from the baseline-verified Phase 3B evidence.")
    inputs, metadata = prepared["inputs"], prepared["video_metadata"]
    ids = inputs.input_ids[0].detach().cpu().tolist()
    positions, _ = visual_positions(inputs, processor)
    sampled_metadata = prepared["input_metadata"].get("video_metadata")
    if isinstance(sampled_metadata, str):
        sampled_metadata = ast.literal_eval(sampled_metadata)
    if isinstance(sampled_metadata, list) and len(sampled_metadata) == 1:
        sampled_metadata = sampled_metadata[0]
    if not isinstance(sampled_metadata, dict) or not sampled_metadata.get("frames_indices"):
        raise ValueError("Actual sampled-frame indices are unavailable; inferred frame groups cannot freeze.")
    width, height, _, _ = video_shape(actual_path)
    temporal, merged_h, merged_w = metadata["merged_video_grid_thw"]
    descriptors = token_descriptors(candidate["rows"][condition], temporal, merged_h, merged_w,
        width, height, metadata["source_frame_groups"], settings["roi_padding"], "overlap")
    if len(descriptors) != len(positions):
        raise ValueError("Processor visual positions and ROI descriptors differ in length.")
    # Tied/ambiguous object cells are not safe background controls either.
    background = [position for position, descriptor in zip(positions, descriptors)
        if descriptor["temporal_phase_weights"].get("event_2", 0) > 0.5 and
        all(descriptor["spatial_roi_weights"].get(label, 0) < metadata["minimum_primary_roi_overlap"]
            for label in ("target_1", "target_2", "distractors"))]
    return {
        "input_ids": ids, "attention_mask": inputs.attention_mask[0].detach().cpu().tolist(),
        "prompt_token_count": len(ids), "prompt_input_ids_sha256": input_ids_hash(ids),
        "visual_positions": positions, "group_positions": prepared["groups"],
        "sampled_frame_indices": sampled_metadata["frames_indices"],
        "video_metadata": metadata, "input_metadata": prepared["input_metadata"],
        "background_event2_positions": background,
        "distractors_event2_positions": prepared["groups"]["video_distractors_e2"],
        "video_path": actual_path, "video_sha256": expected,
    }


def validate_plan(plan_dir):
    plan_dir = Path(plan_dir)
    plan = read_json(plan_dir / "plan_config.json")
    if plan.get("artifact_type") != "real" or plan.get("plan_fingerprint") != digest({
        key: value for key, value in plan.items() if key != "plan_fingerprint"
    }):
        raise ValueError("Invalid Phase 3C plan fingerprint/type.")
    for name, expected in plan["preparation_code_sha256"].items():
        if file_hash(Path(__file__).parent / name) != expected:
            raise ValueError("Preparation code changed after plan creation; use a new Phase 3C root.")
    if file_hash(Path(__file__).resolve().parents[1] / "docs/phase3c_protocol.md") != plan["protocol_sha256"]:
        raise ValueError("Protocol changed after plan creation; use a new Phase 3C root.")
    for relative, expected in plan["source_evidence_sha256"].items():
        if file_hash(Path(plan["source_run_root"]) / relative) != expected:
            raise ValueError(f"Read-only source evidence changed: {relative}.")
    if digest(read_jsonl(plan_dir / "candidate_manifest.jsonl")) != plan["candidate_manifest_payload_sha256"]:
        raise ValueError("Candidate manifest changed after archive audit.")
    return plan


def run_audit(plan_dir, processor, runtime, project_root, path_map, max_pairs=None, retry_failed=False):
    plan_dir = Path(plan_dir).resolve()
    plan = validate_plan(plan_dir)
    candidates = read_jsonl(plan_dir / "candidate_manifest.jsonl")
    output = plan_dir / "processor_audit"
    config = {"schema": SCHEMA, "artifact_type": "real", "plan_fingerprint": plan["plan_fingerprint"],
        "candidate_manifest_sha256": file_hash(plan_dir / "candidate_manifest.jsonl"),
        "runtime": runtime, "project_root": str(Path(project_root).resolve()), "path_map": path_map}
    frozen_write(output / "config.json", config)
    path = output / "mapping_audit.jsonl"
    prior = read_jsonl(path) if path.exists() else []
    records = {}
    lookup = {row["pair_id"]: row for row in candidates}
    for row in prior:
        pair_id = row["pair_id"]
        if pair_id in records or pair_id not in lookup:
            raise ValueError("Duplicate/foreign processor checkpoint key.")
        if row["plan_fingerprint"] != plan["plan_fingerprint"] or row["candidate_sha256"] != digest(lookup[pair_id]):
            raise ValueError("Processor checkpoint belongs to a different candidate or plan.")
        records[pair_id] = row
    started, attempted, reused = time.perf_counter(), 0, 0
    for index, candidate in enumerate(candidates):
        pair_id = candidate["pair_id"]
        old = records.get(pair_id)
        if old is not None and (old["eligible"] or not retry_failed):
            if old["eligible"]:
                for condition in CONDITIONS:
                    item = old["processor_records"][condition]
                    if file_hash(item["video_path"]) != item["video_sha256"]:
                        raise ValueError("Cached processor audit video bytes changed.")
            reused += 1
        else:
            if max_pairs is not None and attempted >= max_pairs:
                break
            print(f"Phase 3C CPU mapping {index + 1}/{len(candidates)}: {pair_id}; "
                  f"elapsed={time.perf_counter() - started:.1f}s; checkpoint={path}", flush=True)
            row = {"schema": SCHEMA, "pair_id": pair_id, "plan_fingerprint": plan["plan_fingerprint"],
                   "candidate_sha256": digest(candidate), "eligible": False}
            try:
                processed = {condition: processor_condition(candidate, condition, processor, plan["settings"],
                    project_root, path_map) for condition in CONDITIONS}
                support = prepare_support_audit(candidate, candidate["archived_mapping"], processed,
                    plan["settings"]["max_event_progress_error"], plan["settings"]["seed"])
                row.update({"eligible": True, "processor_records": processed, "support_audit": support})
            except Exception as exc:
                row.update({"failure_type": type(exc).__name__, "failure_message": str(exc)})
                print(f"  mapping ineligible: {type(exc).__name__}: {str(exc)[:500]}", flush=True)
            records[pair_id] = row
            attempted += 1
            atomic_write(path, [records[item["pair_id"]] for item in candidates if item["pair_id"] in records], jsonl=True)
        eligible = {key for key, record in records.items() if record["eligible"]}
        selected, missing = select_balanced(candidates, eligible)
        status = {"schema": SCHEMA, "plan_fingerprint": plan["plan_fingerprint"],
            "elapsed_sec": time.perf_counter() - started, "newly_audited": attempted, "reused": reused,
            "candidate_pool_count": len(candidates), "audited_count": len(records),
            "eligible_count": len(eligible), "missing_quotas": missing,
            "cohort_can_freeze": not missing and all(
                item["pair_id"] in records for item in candidates[:max(
                    (candidates.index(item) for item in selected), default=-1) + 1]),
            "gpu_experiment_started": False, "checkpoint": str(path)}
        atomic_write(output / "progress_status.json", status)
        print(f"  eligible={records[pair_id]['eligible']}; completed={len(records)}/{len(candidates)}; "
              f"missing_quotas={missing}; elapsed={status['elapsed_sec']:.1f}s", flush=True)
        if status["cohort_can_freeze"]:
            break
    eligible = {key for key, record in records.items() if record["eligible"]}
    selected, missing = select_balanced(candidates, eligible)
    summary = {"schema": SCHEMA, "plan_fingerprint": plan["plan_fingerprint"],
        "audited_count": len(records), "eligible_count": len(eligible), "missing_quotas": missing,
        "selected_preview_pair_ids": [item["pair_id"] for item in selected],
        "newly_audited": attempted, "reused": reused, "elapsed_sec": time.perf_counter() - started,
        "gpu_experiment_started": False}
    atomic_write(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan_dir", required=True)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--path_map", help="JSON file overriding archived video-root path mappings.")
    parser.add_argument("--max_pairs", type=int, help="Limit newly processed pairs; resume retains successful records.")
    parser.add_argument("--retry_failed", action="store_true")
    args = parser.parse_args()
    if args.max_pairs is not None and args.max_pairs < 1:
        parser.error("--max_pairs must be positive.")
    try:
        plan = validate_plan(args.plan_dir)
        source = read_json(Path(plan["source_run_root"]) / "vm_run_config.json")
        path_map = dict(source.get("path_map", {}))
        path_map.update(load_path_map(args.path_map))
        processor, runtime = load_processor(plan["settings"])
        summary = run_audit(args.plan_dir, processor, runtime, args.project_root, path_map,
                            args.max_pairs, args.retry_failed)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Phase 3C CPU processor audit is not ready: {exc}\nGPU experiment not started.\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["missing_quotas"]:
        print("12-case freeze is not ready; inspect failures or resume the CPU audit. GPU work not started.")


if __name__ == "__main__":
    main()
