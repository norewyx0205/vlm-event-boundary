"""Prospectively replace baseline-invalid rescues without re-screening the old pool."""

import argparse
import gc
import json
import subprocess
import sys
import time
from collections import Counter
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


def load_source_audit(source, diagnostic):
    source, diagnostic = Path(source), Path(diagnostic)
    config = json.loads((source / "vm_run_config.json").read_text())
    if (sha256(source / "selection/analysis_case_manifest.jsonl") != config["manifest_sha256"]["full"]
            or sha256(source / "selection/selected_video_mappings.jsonl") != config["mapping_sha256"]):
        raise RuntimeError("Source frozen manifest or mappings changed.")
    for path, expected in config["video_sha256"].items():
        if sha256(path) != expected:
            raise RuntimeError("Source frozen video bytes changed.")
    audit_plan = json.loads((diagnostic / "audit_plan.json").read_text())
    summary = json.loads((diagnostic / "audit_summary.json").read_text())
    if (audit_plan["pipeline_fingerprint"] != config["pipeline_fingerprint"] or not summary.get("complete")
            or summary.get("failure_count") or measurement_hashes() != {
                name: config["source_code_sha256"][name] for name in measurement_hashes()
            }):
        raise RuntimeError("Source baseline audit is incomplete or scientific code changed.")
    checksums = json.loads((diagnostic / "file_checksums.json").read_text())
    for name in ("audit_plan.json", "audit_summary.json", "baseline_rows.jsonl", "environment.json", "diagnostic_script.py"):
        if sha256(diagnostic / name) != checksums[name]:
            raise RuntimeError(f"Source audit checksum mismatch: {name}.")
    pairs = manifest_pairs(source / "selection/analysis_case_manifest.jsonl")
    rows = {}
    for path in sorted((source / "primary/checkpoints").glob("shard_*/activations/*/*/index.json")):
        item = json.loads(path.read_text())
        shard = json.loads((path.parents[3] / "run_config.json").read_text())
        if item.get("run_fingerprint") != shard["run_fingerprint"] or item["eval_id"] in rows:
            raise RuntimeError("Source capture index fingerprint/coverage is invalid.")
        rows[item["eval_id"]] = item
    diagnostics = read_jsonl(diagnostic / "baseline_rows.jsonl")
    if len({item["eval_id"] for item in diagnostics}) != len(diagnostics):
        raise RuntimeError("Duplicate source diagnostic baselines.")
    rows.update({item["eval_id"]: item for item in diagnostics})
    if set(rows) != {row["eval_id"] for pair in pairs.values() for row in pair.values()}:
        raise RuntimeError("Source audit does not cover the whole frozen cohort.")
    exclusions = []
    for pair_id, pair in pairs.items():
        reasons = pair_failures(pair, rows)
        if reasons:
            if pair["low_boundary"]["phase3b_analysis_stratum"] != "primary_rescue":
                raise RuntimeError("A control baseline failed; do not silently repair controls.")
            exclusions.append({
                "pair_id": pair_id, "base_sample_id": pair["low_boundary"]["base_sample_id"],
                "reasons": reasons,
                "baselines": {condition: rows[pair[condition]["eval_id"]] for condition in CONDITIONS},
            })
    if len(exclusions) != 3:
        raise RuntimeError(f"This protocol amendment expects exactly 3 invalid primary cases, found {len(exclusions)}.")
    return config, pairs, exclusions


def pick_replacements(candidates, count=3):
    by_base = {}
    for item in sorted(candidates, key=lambda item: (int(item["base_sample_id"]), item["prompt_variant"] != "original")):
        if item.get("eligible"):
            by_base.setdefault(int(item["base_sample_id"]), item)
    return [by_base[key] for key in sorted(by_base)[:count]]


def freeze_amendment(source, output, pairs, exclusions, replacements, protocol):
    try:
        from .activation_patching_core import atomic_write_jsonl
        from .select_phase3b_cases import write_frozen_rows
    except ImportError:
        from activation_patching_core import atomic_write_jsonl
        from select_phase3b_cases import write_frozen_rows
    excluded = {item["pair_id"] for item in exclusions}
    keep = {key: pair for key, pair in pairs.items() if key not in excluded}
    if len(replacements) != 3 or len({item["base_sample_id"] for item in replacements}) != 3:
        raise RuntimeError("Need three independent, prospectively eligible replacements.")
    old_bases = {pair["low_boundary"]["base_sample_id"] for pair in pairs.values()}
    for item in replacements:
        if item["base_sample_id"] in old_bases or item["pair_id"] in keep or not item.get("eligible"):
            raise RuntimeError("Replacement duplicates an existing case or is ineligible.")
        pair = {row["condition"]: row for row in item["manifest_rows"]}
        if pair_failures(pair, {row["eval_id"]: row for row in item["baseline_rows"]}):
            raise RuntimeError("Replacement lacks a strict verified rescue baseline.")
        keep[item["pair_id"]] = pair
    primary = [pair["low_boundary"] for pair in keep.values() if pair["low_boundary"]["phase3b_analysis_stratum"] == "primary_rescue"]
    if len(primary) != 50 or len({row["base_sample_id"] for row in primary}) != 50:
        raise RuntimeError("Amended cohort does not contain 50 independent primary rescues.")
    output = Path(output)
    summary = json.loads((Path(source) / "selection/case_selection_summary.json").read_text())
    protected = set(summary["representative_pair_ids"].values()) | set(manifest_pairs(Path(source) / "selection/preflight_case_manifest.jsonl"))
    if not protected <= set(keep):
        raise RuntimeError("Frozen representatives/preflight changed; a separate protocol is required.")
    all_rows = [row for key in sorted(keep) for row in keep[key].values()]
    mappings = {item["pair_id"]: item for item in read_jsonl(Path(source) / "selection/selected_video_mappings.jsonl")}
    mappings.update({item["pair_id"]: item["mapping"] for item in replacements})
    names = {
        "case_manifest.jsonl": "primary_rescue", "control_case_manifest.jsonl": "stable_both_correct_control",
        "mirrored_case_manifest.jsonl": "mirrored_prompt_control",
        "independent_mirrored_rescue_manifest.jsonl": "independent_mirrored_rescue",
    }
    for name, stratum in names.items():
        write_frozen_rows(output / name, [row for row in all_rows if row["phase3b_analysis_stratum"] == stratum])
    write_frozen_rows(output / "analysis_case_manifest.jsonl", all_rows)
    write_frozen_rows(output / "preflight_case_manifest.jsonl", read_jsonl(Path(source) / "selection/preflight_case_manifest.jsonl"))
    write_frozen_rows(output / "selected_video_mappings.jsonl", [mappings[key] for key in sorted(keep)])
    write_frozen_rows(output / "reserve_case_manifest.jsonl", [])
    summary.update({
        "primary_bases": [row["base_sample_id"] for row in primary], "reserve_bases": [],
        "primary_prompt_variants": dict(Counter(row["prompt_variant"] for row in primary)),
        "primary_first_mover": dict(Counter(row["first_object_id"] for row in primary)),
        "vm_eligible_primary_count": 50,
        "cohort_amendment": {
            "schema": "phase3b_vm_baseline_amendment_v1", "protocol": protocol,
            "excluded_pair_ids": sorted(excluded), "replacement_pair_ids": [item["pair_id"] for item in replacements],
            "old_selection_sha256": sha256(Path(source) / "selection/case_selection_summary.json"),
            "interpretation": "VM-baseline-eligible rescue cohort; post-screening eligibility amendment, not a new random sample.",
            "patch_effects_used_for_selection": False,
        },
    })
    summary = json.loads(json.dumps(summary))
    summary_path = output / "case_selection_summary.json"
    if summary_path.is_file() and json.loads(summary_path.read_text()) != summary:
        raise RuntimeError("Frozen amended selection differs; use a new directory.")
    write_json(output / "exclusions.json", exclusions)
    atomic_write_jsonl(output / "replacement_evidence.jsonl", replacements)
    write_json(summary_path, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_run_root", required=True)
    parser.add_argument("--diagnostic_dir", required=True)
    parser.add_argument("--rescue_pool_root", required=True)
    parser.add_argument("--output_dir", required=True, help="Separate supplement checkpoint directory.")
    parser.add_argument("--selection_dir", required=True, help="New amended frozen selection; never the old selection.")
    parser.add_argument("--max_additional_bases", type=int, default=50)
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--plan_only", action="store_true")
    args = parser.parse_args()
    if args.max_additional_bases < 5 or args.max_additional_bases % 5:
        parser.error("Use a positive multiple of the fixed five-base batch size.")
    source, diagnostic, output = Path(args.source_run_root), Path(args.diagnostic_dir), Path(args.output_dir)
    if output.resolve() == source.resolve() or output.resolve().is_relative_to(source.resolve()):
        parser.error("Supplement output must not modify the old run.")
    if Path(args.selection_dir).resolve() == (source / "selection").resolve():
        parser.error("Use a NEW selection directory.")
    config, pairs, exclusions = load_source_audit(source, diagnostic)
    old_annotations = read_jsonl(Path(args.rescue_pool_root) / "L5_full/annotations.jsonl")
    generation = json.loads((Path(args.rescue_pool_root) / "phase3b_generation_config.json").read_text())
    for name, expected in generation["generator_code_sha256"].items():
        if sha256(Path(__file__).parent / name) != expected:
            raise RuntimeError("Rescue-pool generator changed; amendment must keep stimulus parameters fixed.")
    start_id = max(int(row["base_sample_id"]) for row in old_annotations) + 1
    protocol = {
        "schema": "phase3b_prospective_supplement_v1", "source_run_root": str(source.resolve()),
        "source_pipeline_fingerprint": config["pipeline_fingerprint"],
        "source_audit_rows_sha256": sha256(diagnostic / "baseline_rows.jsonl"),
        "source_audit_environment": json.loads((diagnostic / "environment.json").read_text()),
        "source_pool_annotations_sha256": sha256(Path(args.rescue_pool_root) / "L5_full/annotations.jsonl"),
        "generation_config": generation, "first_unscreened_base_id": start_id,
        "max_additional_bases": args.max_additional_bases, "batch_size": 5,
        "replacement_count": 3, "order": "ascending base ID; original preferred if both prompts eligible",
        "measurement_code_sha256": measurement_hashes(),
        "repair_code_sha256": {name: sha256(Path(__file__).parent / name) for name in (
            "repair_phase3b_cohort.py", "phase3b_baseline.py", "audit_phase3b_mappings.py",
        )},
        "selection_dir": str(Path(args.selection_dir).resolve()), "patching_allowed": False,
    }
    print(f"SUPPLEMENT: exclude {[item['base_sample_id'] for item in exclusions]}; "
          f"start={start_id}; fixed budget={args.max_additional_bases} new bases; need=3. No old videos re-screened.", flush=True)
    if args.plan_only:
        print(json.dumps(protocol, indent=2))
        return
    import torch
    from activation_patching_core import atomic_write_jsonl
    from audit_phase3b_mappings import audit_pair
    from phase3b_core import prepare_example
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        protocol_path = output / "protocol.json"
        if protocol_path.is_file() and json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError("Supplement protocol changed; existing checkpoint cannot be reused.")
        write_json(protocol_path, protocol)
        write_json(output / "exclusions.json", exclusions)
        environment = runtime_signature()
        source_environment = protocol["source_audit_environment"]
        if environment["python"] != source_environment["python_version"] or any(
            environment["packages"][package] != source_environment[key]
            for package, key in (("numpy", "numpy_version"), ("accelerate", "accelerate_version"),
                                 ("av", "av_version"), ("transformers", "transformers_version"),
                                 ("qwen-vl-utils", "qwen_vl_utils_version"))
        ) or environment["torch"] != source_environment["torch_version"]:
            raise RuntimeError("Supplement numerical/preprocessing environment differs from the source VM audit.")
        environment_path = output / "environment.json"
        if environment_path.is_file() and json.loads(environment_path.read_text()) != environment:
            raise RuntimeError("Supplement runtime changed; refusing mixed screening checkpoints.")
        write_json(environment_path, environment)
        hardware_path = source / "preflight/model_parallel/checkpoints/shard_00/run_config.json"
        hardware = json.loads(hardware_path.read_text())
        settings = SimpleNamespace(
            **{key: config[key] for key in ("model_name", "model_revision", "seed", "video_fps", "video_num_frames", "video_max_pixels", "roi_padding", "gpu_weight_budget_gib", "attn_implementation", "path_map")},
            project_root=Path(args.project_root), model_parallel=True, single_gpu=False,
            expected_gpu_hardware_path=str(hardware_path), min_mapping_coverage=0.5, max_progress_error=0.35,
        )
        if (environment["packages"]["transformers"] != config["expected_transformers_version"]
                or environment["torch"].split("+")[0] != config["expected_torch_version"]
                or environment["packages"]["qwen-vl-utils"] != config["expected_qwen_vl_utils_version"]):
            raise RuntimeError("Supplement runtime differs from source model settings.")
        candidates = read_jsonl(output / "candidates.jsonl") if (output / "candidates.jsonl").is_file() else []
        engine, started = None, time.perf_counter()
        for batch_start in range(start_id, start_id + args.max_additional_bases, 5):
            replacements = pick_replacements(candidates)
            if len(replacements) == 3:
                break
            pool = output / "supplement_pool"
            command = [sys.executable, "-u", str(Path(args.project_root) / "scripts/generate_phase3b_rescue_pool.py"),
                       "--output_root", str(pool), "--start_base_id", str(batch_start), "--count", "5",
                       "--max_new_bases", str(start_id - 31 + args.max_additional_bases)]
            for name in ("seed", "fps", "event_duration_sec", "temporal_gap_sec", "visual_marker_sec", "audio_beep_duration_sec", "static_distractors", "moving_distractors"):
                command.extend([f"--{name}", str(generation[name])])
            command.extend(["--level_durations", ",".join(map(str, generation["level_durations"]))])
            subprocess.run(command, check=True)
            batch = output / f"batch_{batch_start:03d}"
            annotations = pool / "batches" / f"batch_{batch_start:03d}_{batch_start+4:03d}/annotations.jsonl"
            annotations_rows = read_jsonl(annotations)
            binding = {
                "protocol_fingerprint": fingerprint(protocol), "annotation_sha256": sha256(annotations),
                "runtime": environment,
                "video_sha256": {row["video_path"]: sha256(row["video_path"]) for row in annotations_rows},
            }
            batch.mkdir(exist_ok=True)
            if (batch / "config.json").is_file() and json.loads((batch / "config.json").read_text()) != binding:
                raise RuntimeError("Supplement batch inputs changed.")
            write_json(batch / "config.json", binding)
            baselines = read_jsonl(batch / "baselines.jsonl") if (batch / "baselines.jsonl").is_file() else []
            saved = {row["eval_id"]: row for row in baselines}
            if len(saved) != len(baselines) or not set(saved) <= {row["eval_id"] for row in annotations_rows}:
                raise RuntimeError("Invalid supplement baseline checkpoint coverage.")
            if any(row.get("screening_fingerprint") != fingerprint(binding) for row in baselines):
                raise RuntimeError("Supplement baseline fingerprint mismatch.")
            for number, row in enumerate(annotations_rows, 1):
                if row["eval_id"] in saved:
                    print(f"REUSE SCREEN {row['eval_id']}", flush=True)
                    continue
                if engine is None:
                    engine = BaselineEngine(settings, hardware["model_device_map"])
                prepared = prepare_example(row, engine.processor, settings.project_root, settings.video_fps,
                                           settings.video_num_frames, settings.video_max_pixels, settings.roi_padding,
                                           device=engine.model.device)
                row["archived_input_metadata"] = prepared["input_metadata"]
                item = engine.audit(prepared)
                item["screening_fingerprint"] = fingerprint(binding)
                saved[row["eval_id"]] = item
                baselines.append(item)
                atomic_write_jsonl(batch / "baselines.jsonl", baselines)
                print(f"SCREEN batch={batch_start} row={number}/20; {row['eval_id']} "
                      f"prediction={item['capture_decision']['prediction']} margin={item['capture_decision']['margin']:.6f}; "
                      f"elapsed={(time.perf_counter()-started)/60:.1f} min; checkpoint={batch}", flush=True)
                del prepared
                gc.collect()
                torch.cuda.empty_cache()
            for base in range(batch_start, batch_start + 5):
                for prompt in ("original", "swapped"):
                    pair_id = f"phase3b_base_{base:03d}_{prompt}"
                    if any(item["pair_id"] == pair_id for item in candidates):
                        continue
                    rows = [dict(row) for row in annotations_rows if int(row["base_sample_id"]) == base and row["prompt_variant"] == prompt]
                    for row in rows:
                        item = saved[row["eval_id"]]
                        row.update({
                            "phase3b_pair_id": pair_id, "phase3b_analysis_stratum": "primary_rescue",
                            "phase3b_prompt_pair_behavior": "temporal_rescue",
                            "archived_prediction": item["capture_decision"]["prediction"],
                            "archived_is_correct": item["capture_decision"]["is_correct"],
                            "archived_input_metadata": item["input_metadata"],
                            "archived_raw_response": item["capture_decision"]["predicted_first_token_text"],
                            "phase3b_screening_source": str(batch),
                        })
                    pair = {row["condition"]: row for row in rows}
                    reasons = pair_failures(pair, saved)
                    mapping = None
                    if not reasons:
                        if engine is None:
                            engine = BaselineEngine(settings, hardware["model_device_map"])
                        mapping = audit_pair(pair["low_boundary"], pair["temporal_boundary"], engine.processor, settings)
                        if not mapping["eligible"]:
                            reasons.append("processor-only event-relative mapping ineligible")
                    candidates.append({
                        "pair_id": pair_id, "base_sample_id": base, "prompt_variant": prompt,
                        "eligible": not reasons, "reasons": reasons, "mapping": mapping,
                        "manifest_rows": rows, "baseline_rows": [saved[row["eval_id"]] for row in rows],
                    })
                    atomic_write_jsonl(output / "candidates.jsonl", candidates)
            write_json(output / "progress.json", {
                "last_screened_base_id": batch_start + 4, "additional_bases_screened": batch_start + 5 - start_id,
                "eligible_replacement_count": len(pick_replacements(candidates)),
                "eligible_b_rescue_bases": sorted({item["base_sample_id"] for item in candidates if item["eligible"] and item["prompt_variant"] == "swapped"}),
                "elapsed_sec": time.perf_counter() - started, "full_run_started": False,
            })
            print(f"SUPPLEMENT CHECKPOINT: through base {batch_start+4}; "
                  f"eligible replacements={len(pick_replacements(candidates))}/3; elapsed={(time.perf_counter()-started)/60:.1f} min", flush=True)
        replacements = pick_replacements(candidates)
        if len(replacements) != 3:
            raise RuntimeError("Fixed supplement budget exhausted before finding 3 eligible rescues. No formal cohort frozen; review the amendment before extending the budget.")
        freeze_amendment(source, args.selection_dir, pairs, exclusions, replacements, protocol)
        from select_phase3b_cases import write_frozen_rows
        write_frozen_rows(Path(args.selection_dir) / "supplement_screening_audit.jsonl", candidates)
        write_json(Path(args.selection_dir) / "supplement_progress.json", json.loads((output / "progress.json").read_text()))
        print(f"Frozen 50-case VM-eligible amended cohort: {args.selection_dir}; "
              f"replacements={[item['base_sample_id'] for item in replacements]}. Full patching NOT started.", flush=True)


if __name__ == "__main__":
    main()
