"""Copy byte-identical, explicitly certified completed shards into an amended run."""

import json
import shutil
from pathlib import Path

try:
    from .phase3b_baseline import measurement_hashes, sha256
    from .run_phase3b_vm import manifest_pairs, write_json
except ImportError:
    from phase3b_baseline import measurement_hashes, sha256
    from run_phase3b_vm import manifest_pairs, write_json


METHOD_KEYS = (
    "schema", "patching_code_sha256", "shard_size", "model_name", "model_revision",
    "transformers_version", "qwen_vl_utils_version", "torch_version", "seed", "validate_controls",
    "video_fps", "video_num_frames", "video_max_pixels", "roi_padding", "attn_implementation",
    "gpu_hardware", "single_gpu", "path_map", "verify_standard_generation", "model_parallel",
    "model_device_map_strategy", "gpu_weight_budget_gib", "model_device_map",
)


def copy_verified(source, target, records, root):
    source, target = Path(source), Path(target)
    if source.is_symlink():
        raise RuntimeError(f"Reuse refuses symlinks: {source}.")
    if source.is_dir():
        for child in sorted(source.iterdir()):
            if child.name != ".pipeline.lock" and child.suffix != ".tmp":
                copy_verified(child, target / child.name, records, root)
    else:
        expected = sha256(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) != expected:
            raise RuntimeError(f"Reuse would overwrite a different artifact: {target}.")
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copy2(source, temporary)
            if sha256(temporary) != expected:
                raise RuntimeError("Copy checksum failed.")
            temporary.replace(target)
        records[str(target.relative_to(root))] = expected


def prepare_reuse(args, plan):
    try:
        from .analyze_phase3b import read_shards, validate_completeness
        from .common import read_jsonl
    except ImportError:
        from analyze_phase3b import read_shards, validate_completeness
        from common import read_jsonl
    source, root = Path(args.reuse_completed_from).resolve(), Path(args.output_root).resolve()
    if source == root or source in root.parents or root in source.parents:
        raise RuntimeError("Use a separate amended run root; old evidence must remain untouched.")
    old = json.loads((source / "vm_run_config.json").read_text())
    if any(old["source_code_sha256"].get(name) != value for name, value in measurement_hashes().items()):
        raise RuntimeError("Measurement code changed; old patches cannot be reused.")
    settings = (
        "model_name", "model_revision", "expected_transformers_version", "expected_torch_version",
        "expected_qwen_vl_utils_version", "seed", "dtype", "attn_implementation", "video_fps",
        "video_num_frames", "video_max_pixels", "roi_padding", "shard_size", "execution_mode",
        "gpu_weight_budget_gib", "gpus", "path_map",
    )
    if any(old.get(key) != plan.get(key) for key in settings):
        raise RuntimeError("Reuse model/processor/placement settings differ from the source run.")
    old_pairs = manifest_pairs(source / "selection/analysis_case_manifest.jsonl")
    new_pairs = manifest_pairs(root / "selection/analysis_case_manifest.jsonl")
    old_mappings = {item["pair_id"]: item for item in read_jsonl(source / "selection/selected_video_mappings.jsonl")}
    new_mappings = {item["pair_id"]: item for item in read_jsonl(root / "selection/selected_video_mappings.jsonl")}
    records, shards = {}, {}
    old_preflight = source / "preflight/model_parallel/checkpoints/shard_00/run_config.json"
    hardware = json.loads(old_preflight.read_text())
    if args.execution_mode != "model_parallel":
        raise RuntimeError("Amended-cohort reuse currently supports the validated model-parallel VM path only.")
    old_preflight_pairs = manifest_pairs(source / "selection/preflight_case_manifest.jsonl")
    if old_preflight_pairs != manifest_pairs(root / "selection/preflight_case_manifest.jsonl"):
        raise RuntimeError("Preflight cases changed; run a new preflight without checkpoint reuse.")
    preflight = json.loads((source / "vm_preflight_summary.json").read_text())
    relocation = json.loads((source / "relocation_control/relocation_summary.json").read_text())
    if (not preflight.get("complete") or preflight["pipeline_fingerprint"] != old["pipeline_fingerprint"]
            or relocation.get("case_count") != 2 or not relocation.get("patch_rows")
            or relocation.get("reencode_prediction_matches") != 2
            or sha256(Path(__file__).parent / "run_phase3b_relocation_control.py") != old["source_code_sha256"]["run_phase3b_relocation_control.py"]):
        raise RuntimeError("Source VM preflight/relocation control is incomplete.")
    for folder in ("preflight", "relocation_control"):
        copy_verified(source / folder, root / folder, records, root)
    for filename in ("vm_preflight_summary.json", "vm_run_config.json"):
        copy_verified(source / filename, root / "reuse_source" / filename, records, root)
    copy_verified(source / "selection", root / "reuse_source/selection", records, root)
    for shard in sorted((source / "primary/checkpoints").glob("shard_*")):
        config_path = shard / "run_config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        index = config["shard_index"]
        ids = list(new_pairs)[index * 5:(index + 1) * 5]
        if ids != config["pair_ids"]:
            continue
        if any(old_pairs.get(key) != new_pairs[key] or old_mappings.get(key) != new_mappings.get(key) for key in ids):
            raise RuntimeError("A reusable pair's annotations or mapping changed.")
        if any(config.get(key) != hardware.get(key) for key in METHOD_KEYS if key not in {"shard_size"}):
            raise RuntimeError("Source shard methodology/hardware differs from its preflight.")
        for key in ids:
            for row in new_pairs[key].values():
                video = plan["video_paths_by_eval_id"][row["eval_id"]]
                if config["video_sha256_by_eval_id"].get(row["eval_id"]) != plan["video_sha256"][video]:
                    raise RuntimeError("Reusable video bytes changed.")
        # Incomplete shards are left in the old run, never promoted as reusable.
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temporary:
            Path(temporary, shard.name).symlink_to(shard, target_is_directory=True)
            divergence, patches, captures, configs, controls = read_shards(temporary)
        missing = validate_completeness(divergence, patches, captures, False, ids)
        if any(missing) or set(controls) != set(ids):
            continue
        for info in captures.values():
            parity = info.get("standard_parity") or {}
            if not parity.get("first_token_match") or not parity.get("logits_allclose"):
                raise RuntimeError("Reusable capture has no passing standard parity.")
        for pair_id, condition in captures:
            activation = shard / "activations" / pair_id / condition
            if not all((activation / filename).is_file() for filename in ["baseline.pt"] + [f"layer_{layer:02d}.pt" for layer in range(36)]):
                raise RuntimeError("Reusable activation tensors are incomplete.")
        copy_verified(shard, root / "primary/checkpoints" / shard.name, records, root)
        shards[shard.name] = {"run_fingerprint": config["run_fingerprint"], "pair_ids": ids}
    certificate = {
        "schema": "phase3b_checkpoint_reuse_v1", "source_run_root": str(source),
        "source_pipeline_fingerprint": old["pipeline_fingerprint"],
        "target_pipeline_fingerprint": plan["pipeline_fingerprint"],
        "manifest_sha256": plan["manifest_sha256"]["full"], "mapping_sha256": plan["mapping_sha256"],
        "measurement_code_sha256": measurement_hashes(), "shards": shards,
        "files_sha256": records, "preflight_reused": True,
        "old_fingerprints_rewritten": False,
    }
    write_json(root / "checkpoint_reuse.json", certificate)
    print(f"Verified byte-identical reuse: {len(shards)} complete shards / {sum(len(item['pair_ids']) for item in shards.values())} pairs. Partial shards not reused.", flush=True)
    return certificate


def validate_reuse(root, plan=None, manifest=None):
    root = Path(root)
    path = root / "checkpoint_reuse.json"
    if not path.is_file():
        return None
    certificate = json.loads(path.read_text())
    if certificate.get("schema") != "phase3b_checkpoint_reuse_v1" or certificate.get("old_fingerprints_rewritten") is not False:
        raise RuntimeError("Invalid checkpoint reuse certificate.")
    if certificate["measurement_code_sha256"] != measurement_hashes():
        raise RuntimeError("Reused checkpoint measurement code changed.")
    if plan is not None and certificate["target_pipeline_fingerprint"] != plan["pipeline_fingerprint"]:
        raise RuntimeError("Reuse certificate belongs to another amended cohort.")
    if manifest is not None and certificate["manifest_sha256"] != sha256(manifest):
        raise RuntimeError("Reuse certificate does not match the analysis manifest.")
    for relative, expected in certificate["files_sha256"].items():
        file = (root / relative).resolve()
        if not file.is_relative_to(root.resolve()) or file.is_symlink() or not file.is_file() or sha256(file) != expected:
            raise RuntimeError(f"Reused artifact changed or missing: {relative}.")
    return certificate
