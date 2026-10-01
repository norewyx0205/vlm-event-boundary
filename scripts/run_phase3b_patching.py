"""Capture all-layer states and run fixed-grid Phase 3B activation patches."""

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from collections import defaultdict
from contextlib import ExitStack
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import torch
import transformers

try:
    from .activation_patching_core import (
        atomic_write_json, atomic_write_jsonl, decision_from_logits,
        hidden_from_layer_output, locate_decoder_layers, replace_layer_hidden,
        validate_archived_input_metadata, validate_archived_prediction,
    )
    from .common import PROJECT_ROOT, read_jsonl
    from .phase3b_core import (
        SCHEMA, GROUPS, PATCH_LAYERS, TEXT_GROUPS, VIDEO_GROUPS,
        mover_roles, prepare_example,
    )
    from .probe_attention_roi import model_forward, standard_first_token
    from .run_eval import configure_reproducibility, environment_metadata, load_model
except ImportError:
    from activation_patching_core import (
        atomic_write_json, atomic_write_jsonl, decision_from_logits,
        hidden_from_layer_output, locate_decoder_layers, replace_layer_hidden,
        validate_archived_input_metadata, validate_archived_prediction,
    )
    from common import PROJECT_ROOT, read_jsonl
    from phase3b_core import SCHEMA, GROUPS, PATCH_LAYERS, TEXT_GROUPS, VIDEO_GROUPS, mover_roles, prepare_example
    from probe_attention_roi import model_forward, standard_first_token
    from run_eval import configure_reproducibility, environment_metadata, load_model


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def case_pairs(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key = row["phase3b_pair_id"]
        condition = row["condition"]
        if condition in grouped[key]:
            raise ValueError(f"Duplicate {condition} for {key}.")
        grouped[key][condition] = row
    for key, pair in grouped.items():
        if set(pair) != {"low_boundary", "temporal_boundary"}:
            raise ValueError(f"Incomplete Phase 3B pair: {key}.")
    return dict(sorted(grouped.items()))


def load_mappings(path):
    output = {}
    for row in read_jsonl(path):
        if row["pair_id"] in output:
            raise ValueError(f"Duplicate mapping {row['pair_id']}.")
        output[row["pair_id"]] = row
    return output


def activation_root(output_dir, pair_id, condition):
    return Path(output_dir) / "activations" / pair_id / condition


def capture_complete(root, fingerprint, layer_count=36):
    root = Path(root)
    if not (root / "index.json").is_file() or not (root / "baseline.pt").is_file():
        return False
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    return index.get("run_fingerprint") == fingerprint and all(
        (root / f"layer_{layer:02d}.pt").is_file() for layer in range(layer_count)
    )


def prepare_shard_run_config(root, fingerprint, payload, *, allow_failed_recovery):
    """Preserve incompatible failed setup; never replace captured research data."""
    root = Path(root)
    config_path = root / "run_config.json"
    quarantined = None
    if config_path.is_file():
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        if prior.get("run_fingerprint") == fingerprint:
            return prior
        files = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*") if path.is_file() or path.is_symlink()
        }
        if not allow_failed_recovery or not files <= {"run_config.json", "capture_errors.json"}:
            raise RuntimeError(
                "Existing shard has a different model/code/data fingerprint. "
                "Saved or unrecognized artifacts will not be overwritten; preserve this "
                "checkpoint and use a separate --output_dir for the changed run. "
                f"Checkpoint: {root}."
            )
        quarantine_root = root.parent / "incompatible_failed_shards"
        quarantine_root.mkdir(parents=True, exist_ok=True)
        quarantined = quarantine_root / f"{root.name}_{time.time_ns()}"
        root.rename(quarantined)
        print(
            f"Preserved incompatible setup-only shard at {quarantined}; "
            "no activation or result files were present. Restarting capture setup.",
            flush=True,
        )
    elif root.exists() and any(root.iterdir()):
        raise RuntimeError(
            f"Non-empty shard has no run_config.json: {root}. "
            "Preserve it and use a separate --output_dir."
        )
    config = {**payload, "run_fingerprint": fingerprint}
    if quarantined is not None:
        config["quarantined_failed_checkpoint"] = str(quarantined)
    atomic_write_json(config_path, config)
    return config


def _capture_all_layers(model, processor, prepared, root, fingerprint, verify_standard=True):
    layers, layer_path = locate_decoder_layers(model)
    if len(layers) != 36:
        raise RuntimeError(f"Phase 3B expects 36 decoder layers, got {len(layers)}.")
    groups = prepared["groups"]
    union = sorted({position for name in GROUPS for position in groups[name]})
    if not union or any(not groups[name] for name in GROUPS):
        missing = [name for name in GROUPS if not groups[name]]
        raise RuntimeError(f"Empty planned Phase 3B token groups: {missing}.")
    archived_metadata = validate_archived_input_metadata(prepared)
    index_lookup = {position: i for i, position in enumerate(union)}
    snapshots = {}
    def capture_hook(layer):
        def hook(_module, _inputs, output):
            hidden = hidden_from_layer_output(output)
            index = torch.as_tensor(union, device=hidden.device, dtype=torch.long)
            snapshots[layer] = hidden[0].index_select(0, index).detach().cpu()
            return output
        return hook

    with ExitStack() as stack:
        for layer, module in enumerate(layers):
            handle = module.register_forward_hook(capture_hook(layer))
            stack.callback(handle.remove)
        kwargs = dict(prepared["inputs"])
        kwargs.update({"use_cache": False, "return_dict": True})
        with torch.inference_mode():
            output = model_forward(model, kwargs)
    logits = output.logits[0, -1, :].float().detach().cpu()
    decision = decision_from_logits(logits, processor, prepared["row"]["correct_option"])
    validate_archived_prediction(prepared, decision)
    standard = None
    if verify_standard:
        standard_id, standard_logits = standard_first_token(model, prepared["inputs"])
        finite = torch.isfinite(logits) & torch.isfinite(standard_logits)
        diff = (logits[finite] - standard_logits[finite]).abs()
        standard = {
            "first_token_match": standard_id == decision["predicted_first_token_id"],
            "logits_allclose": bool(torch.allclose(logits[finite], standard_logits[finite], rtol=0.001, atol=0.25)),
            "max_abs_diff": float(diff.max()) if diff.numel() else None,
        }
        if not standard["first_token_match"] or not standard["logits_allclose"]:
            raise RuntimeError(f"Standard-generation parity failed: {standard}.")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    norm_trajectories = {}
    for layer, vectors in snapshots.items():
        path = root / f"layer_{layer:02d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        group_means = {
            name: vectors[[index_lookup[position] for position in groups[name]]].float().mean(dim=0)
            for name in GROUPS
        }
        norm_trajectories[str(layer)] = {
            name: float(value.norm()) for name, value in group_means.items()
        }
        torch.save({"vectors": vectors, "positions": union, "group_means": group_means}, temporary)
        temporary.replace(path)
    baseline_path = root / "baseline.pt"
    temporary = baseline_path.with_suffix(".pt.tmp")
    torch.save({"logits": logits}, temporary)
    temporary.replace(baseline_path)
    info = {
        "schema": SCHEMA, "run_fingerprint": fingerprint,
        "eval_id": prepared["row"]["eval_id"],
        "positions": union,
        "groups": {name: [index_lookup[position] for position in groups[name]] for name in GROUPS},
        "group_token_counts": {name: len(groups[name]) for name in GROUPS},
        "group_mean_norms_by_layer": norm_trajectories,
        "group_positions": groups,
        "video_metadata": prepared["video_metadata"],
        "input_metadata": prepared["input_metadata"],
        "decision": decision, "standard_parity": standard,
        "archived_input_parity": archived_metadata,
        "decoder_layer_path": layer_path,
        "hidden_dtype": str(snapshots[0].dtype),
        "first_object_id": prepared["row"]["first_object_id"],
        "analysis_stratum": prepared["row"].get("phase3b_analysis_stratum"),
        "mover_roles": mover_roles(prepared["row"]["first_object_id"]),
    }
    atomic_write_json(root / "index.json", info)
    return info


def load_capture(root, layer, positions):
    payload = _load_layer(str(root), layer)
    union = payload["positions"]
    lookup = {position: i for i, position in enumerate(union)}
    missing = set(positions) - set(lookup)
    if missing:
        raise RuntimeError(f"Activation shard lacks mapped positions: {sorted(missing)[:8]}.")
    indices = torch.as_tensor([lookup[position] for position in positions], dtype=torch.long)
    return payload["vectors"].index_select(0, indices)


@lru_cache(maxsize=72)
def _load_layer(root, layer):
    return torch.load(Path(root) / f"layer_{layer:02d}.pt", map_location="cpu", weights_only=True)


def pooled_divergence(source, target, eps=1e-12):
    source_mean = source.float().mean(dim=0)
    target_mean = target.float().mean(dim=0)
    cosine = 1 - torch.nn.functional.cosine_similarity(source_mean[None], target_mean[None], dim=-1).item()
    relative = (target_mean - source_mean).norm().item() / max(source_mean.norm().item(), eps)
    tokenwise = 1 - torch.nn.functional.cosine_similarity(source.float(), target.float(), dim=-1)
    return {
        "cosine_distance": float(cosine), "relative_l2": float(relative),
        "tokenwise_cosine_mean": float(tokenwise.mean()),
        "tokenwise_relative_l2_mean": float(
            ((target.float() - source.float()).norm(dim=-1) / source.float().norm(dim=-1).clamp_min(eps)).mean()
        ),
    }


def aligned_positions(group, low_index, temporal_index, mapping):
    if group in VIDEO_GROUPS:
        item = mapping["groups"][group]
        if not item["eligible"]:
            raise RuntimeError(f"Mapping ineligible for {group}: {item['failure_reason']}.")
        return item["source_positions"], item["target_positions"], "event_relative_replace"
    low_positions = low_index["group_positions"][group]
    temporal_positions = temporal_index["group_positions"][group]
    if low_positions != temporal_positions or not low_positions:
        raise RuntimeError(f"Text positions differ across conditions for {group}.")
    return low_positions, temporal_positions, "positionwise_replace"


def divergence_for_case(pair_id, root, mapping, fingerprint):
    low_root = activation_root(root, pair_id, "low_boundary")
    temporal_root = activation_root(root, pair_id, "temporal_boundary")
    low_index = json.loads((low_root / "index.json").read_text(encoding="utf-8"))
    temporal_index = json.loads((temporal_root / "index.json").read_text(encoding="utf-8"))
    rows = []
    for layer in range(36):
        for group in GROUPS:
            low_positions, temporal_positions, method = aligned_positions(group, low_index, temporal_index, mapping)
            low = load_capture(low_root, layer, low_positions)
            temporal = load_capture(temporal_root, layer, temporal_positions)
            rows.append({
                "schema": SCHEMA, "run_fingerprint": fingerprint,
                "phase3b_pair_id": pair_id,
                "base_sample_id": mapping["base_sample_id"],
                "prompt_variant": mapping["prompt_variant"],
                "analysis_stratum": low_index.get("analysis_stratum"),
                "first_object_id": mapping["first_object_id"],
                "mover_role": next((name for name, literal in low_index["mover_roles"].items() if literal == group), None),
                "layer": layer, "token_group": group,
                "alignment_method": method, "mapped_token_count": len(low_positions),
                "mapping_source_coverage": mapping["groups"][group]["source_coverage"] if group in VIDEO_GROUPS else 1.0,
                "mapping_target_coverage": mapping["groups"][group]["target_coverage"] if group in VIDEO_GROUPS else 1.0,
                **pooled_divergence(low, temporal),
            })
    return rows


def patch_forward(model, processor, prepared_target, layer, target_positions, source_values):
    layers, _ = locate_decoder_layers(model)
    if layer == 35 and target_positions != prepared_target["groups"]["decision_position"]:
        raise ValueError("Residual-post layer 35 permits decision-position patching only.")
    if len(target_positions) != len(source_values):
        raise ValueError("Patch position/vector count mismatch.")

    def hook(_module, _inputs, output):
        hidden = hidden_from_layer_output(output)
        patched = hidden.clone()
        index = torch.as_tensor(target_positions, device=hidden.device, dtype=torch.long)
        values = source_values.to(device=hidden.device, dtype=hidden.dtype)
        patched[0].index_copy_(0, index, values)
        return replace_layer_hidden(output, patched)

    handle = layers[layer].register_forward_hook(hook)
    try:
        kwargs = dict(prepared_target["inputs"])
        kwargs.update({"use_cache": False, "return_dict": True})
        with torch.inference_mode():
            output = model_forward(model, kwargs)
    finally:
        handle.remove()
    logits = output.logits[0, -1, :].float().detach().cpu()
    return logits, decision_from_logits(logits, processor, prepared_target["row"]["correct_option"])


def no_patch_forward(model, prepared):
    kwargs = dict(prepared["inputs"])
    kwargs.update({"use_cache": False, "return_dict": True})
    with torch.inference_mode():
        return model_forward(model, kwargs).logits[0, -1, :].float().detach().cpu()


def validate_patch_controls(model, processor, prepared, root, atol=0.001):
    baseline = torch.load(Path(root) / "baseline.pt", map_location="cpu", weights_only=True)["logits"]
    rerun = no_patch_forward(model, prepared)
    if float((baseline - rerun).abs().max()) > atol:
        raise RuntimeError("No-patch baseline logits changed between capture and patch stages.")
    checks = {"no_patch_max_abs_diff": float((baseline - rerun).abs().max())}
    for group in ("query_all", "video_t1_e1"):
        positions = prepared["groups"][group]
        if not positions:
            continue
        values = load_capture(root, 0, positions)
        patched, _ = patch_forward(model, processor, prepared, 0, positions, values)
        max_diff = float((baseline - patched).abs().max())
        if max_diff > atol:
            raise RuntimeError(f"Same-state identity patch changed logits for {group}: {max_diff}.")
        checks[f"identity_{group}_max_abs_diff"] = max_diff
    return checks


def validate_prepared_mapping(prepared, mapping, condition):
    index_key = "low_input_metadata" if condition == "low_boundary" else "temporal_input_metadata"
    audited = mapping.get(index_key) or {}
    current = prepared["input_metadata"]
    keys = ("video_grid_thw", "visual_token_count_from_grid_thw", "input_ids")
    mismatches = [key for key in keys if audited.get(key) != current.get(key)]
    if mismatches:
        raise RuntimeError(f"Processor-only mapping audit differs on {mismatches}.")
    live_ids = prepared["inputs"].input_ids[0].detach().cpu().numpy().tobytes()
    if hashlib.sha256(live_ids).hexdigest() != mapping.get("prompt_input_ids_sha256"):
        raise RuntimeError("Prompt token IDs differ from the processor-only mapping audit.")
    mapping_side = "source_positions" if condition == "low_boundary" else "target_positions"
    for group in VIDEO_GROUPS:
        unknown = set(mapping["groups"][group][mapping_side]) - set(prepared["groups"][group])
        if unknown:
            raise RuntimeError(f"Mapping for {group} includes tokens outside the live group.")


def direction_patch_rows(pair_id, direction, pair, mapping, root, model, processor,
                         prepared_target, fingerprint, continue_on_error=False):
    source_condition = "temporal_boundary" if direction == "temporal_to_low" else "low_boundary"
    target_condition = "low_boundary" if direction == "temporal_to_low" else "temporal_boundary"
    source_root = activation_root(root, pair_id, source_condition)
    target_root = activation_root(root, pair_id, target_condition)
    source_index = json.loads((source_root / "index.json").read_text(encoding="utf-8"))
    target_index = json.loads((target_root / "index.json").read_text(encoding="utf-8"))
    source_decision, target_decision = source_index["decision"], target_index["decision"]
    result_path = Path(root) / "patches" / pair_id / f"{direction}.jsonl"
    errors_path = result_path.with_suffix(".errors.json")
    results = read_jsonl(result_path) if result_path.exists() else []
    completed = {(int(row["layer"]), row["token_group"], row["patch_method"]) for row in results}
    errors = json.loads(errors_path.read_text(encoding="utf-8")) if errors_path.exists() else []
    low_margin = (target_decision if target_condition == "low_boundary" else source_decision)["margin"]
    temporal_margin = (target_decision if target_condition == "temporal_boundary" else source_decision)["margin"]
    started = time.perf_counter()
    planned_count = len(GROUPS) * len(PATCH_LAYERS) + 1
    for group_index, group in enumerate(GROUPS, 1):
        low_positions, temporal_positions, method = aligned_positions(
            group,
            target_index if target_condition == "low_boundary" else source_index,
            source_index if source_condition == "temporal_boundary" else target_index,
            mapping,
        )
        if direction == "temporal_to_low":
            source_positions, target_positions = temporal_positions, low_positions
        else:
            source_positions, target_positions = low_positions, temporal_positions
        layers = PATCH_LAYERS + ((35,) if group == "decision_position" else ())
        for layer in layers:
            key = (layer, group, method)
            if key in completed:
                continue
            try:
                source_values = load_capture(source_root, layer, source_positions)
                patched_logits, patched = patch_forward(
                    model, processor, prepared_target, layer, target_positions, source_values
                )
                change = patched["margin"] - target_decision["margin"]
                aligned = change if direction == "temporal_to_low" else -change
                denominator = temporal_margin - low_margin
                role = next((name for name, literal in mover_roles(pair["low_boundary"]["first_object_id"]).items() if literal == group), None)
                row = {
                    "schema": SCHEMA, "run_fingerprint": fingerprint,
                    "phase3b_pair_id": pair_id,
                    "base_sample_id": pair["low_boundary"]["base_sample_id"],
                    "prompt_variant": pair["low_boundary"]["prompt_variant"],
                    "correct_option": pair["low_boundary"]["correct_option"],
                    "analysis_stratum": pair["low_boundary"]["phase3b_analysis_stratum"],
                    "first_object_id": pair["low_boundary"]["first_object_id"],
                    "mover_role": role, "layer": layer, "token_group": group,
                    "patch_method": method, "patch_direction": direction,
                    "source_condition": source_condition, "target_condition": target_condition,
                    "source_positions": source_positions, "target_positions": target_positions,
                    "mapped_token_count": len(target_positions),
                    "mapping_source_coverage": mapping["groups"][group]["source_coverage"] if group in VIDEO_GROUPS else 1.0,
                    "mapping_target_coverage": mapping["groups"][group]["target_coverage"] if group in VIDEO_GROUPS else 1.0,
                    "baseline_source_margin": source_decision["margin"],
                    "baseline_target_margin": target_decision["margin"],
                    "unpatched_low_margin": low_margin,
                    "unpatched_temporal_margin": temporal_margin,
                    "patched_margin": patched["margin"],
                    "margin_change": change,
                    "source_aligned_patch_effect": aligned,
                    "recovery": change / denominator if direction == "temporal_to_low" and abs(denominator) > 1e-12 else None,
                    "baseline_target_prediction": target_decision["prediction"],
                    "patched_prediction": patched["prediction"],
                    "categorical_flip": patched["prediction"] != target_decision["prediction"],
                    "flip_toward_correct": not target_decision["is_correct"] and patched["is_correct"],
                    "flip_away_from_correct": target_decision["is_correct"] and not patched["is_correct"],
                    "status": "ok",
                }
                results.append(row)
                completed.add(key)
                atomic_write_jsonl(result_path, results)
            except Exception as exc:
                errors.append({"pair_id": pair_id, "direction": direction, "group": group,
                               "layer": layer, "error": str(exc), "traceback": traceback.format_exc()})
                atomic_write_json(errors_path, errors)
                if not continue_on_error:
                    raise
        if group_index % 3 == 0 or group_index == len(GROUPS):
            print(
                f"  {pair_id} {direction}: {len(completed)}/{planned_count} patches checkpointed "
                f"after {(time.perf_counter() - started) / 60:.1f} min | {result_path}",
                flush=True,
            )
    unresolved = [
        error for error in errors
        if not any(layer == error["layer"] and group == error["group"] for layer, group, _ in completed)
    ]
    atomic_write_json(errors_path, unresolved)
    return len(results)


def select_shard(pairs, shard_index, shard_size):
    if shard_index < 0 or shard_size < 1:
        raise ValueError("Invalid Phase 3B shard index/size.")
    items = list(pairs.items())
    return dict(items[shard_index * shard_size:(shard_index + 1) * shard_size])


def prepare_pair(pair, processor, args, device):
    prepared = {
        condition: prepare_example(
            row, processor, args.project_root, args.video_fps, args.video_num_frames,
            args.video_max_pixels, args.roi_padding, device,
        )
        for condition, row in pair.items()
    }
    low_ids = prepared["low_boundary"]["inputs"].input_ids
    temporal_ids = prepared["temporal_boundary"]["inputs"].input_ids
    if low_ids.shape != temporal_ids.shape or not torch.equal(low_ids, temporal_ids):
        raise RuntimeError("Matched low/temporal prompt IDs differ.")
    return prepared


def validate_behavioral_category(pair, low_index, temporal_index):
    category = pair["low_boundary"]["phase3b_prompt_pair_behavior"]
    expected = {
        "temporal_rescue": (False, True),
        "stable_both_correct": (True, True),
        "temporal_degradation": (True, False),
        "both_wrong": (False, False),
    }
    if category not in expected:
        raise RuntimeError(f"Unsupported archived prompt-pair behavior: {category}.")
    actual = (low_index["decision"]["margin"] > 0, temporal_index["decision"]["margin"] > 0)
    if actual != expected[category] or any(
        item["decision"]["margin"] == 0 for item in (low_index, temporal_index)
    ):
        raise RuntimeError(f"Prompt-pair behavior changed: archived={category}, live={actual}.")


def run_capture_stage(args, model, processor, pairs, mappings, root, fingerprint):
    errors = []
    completed = 0
    for index, (pair_id, pair) in enumerate(pairs.items(), 1):
        print(f"Capture {index}/{len(pairs)}: {pair_id}", flush=True)
        pair_started = time.perf_counter()
        try:
            mapping = mappings[pair_id]
            if not mapping["eligible"]:
                raise RuntimeError(f"Mapping audit found ineligible groups: {mapping['ineligible_groups']}.")
            roots = {condition: activation_root(root, pair_id, condition) for condition in pair}
            divergence_path = Path(root) / "divergence" / f"{pair_id}.jsonl"
            if all(capture_complete(path, fingerprint) for path in roots.values()) and divergence_path.is_file():
                completed += 1
                print(f"  Reused capture checkpoint: {divergence_path}", flush=True)
                continue
            prepared = prepare_pair(pair, processor, args, model.device)
            for condition in ("low_boundary", "temporal_boundary"):
                validate_prepared_mapping(prepared[condition], mapping, condition)
                if not capture_complete(roots[condition], fingerprint):
                    _capture_all_layers(
                        model, processor, prepared[condition], roots[condition], fingerprint,
                        args.verify_standard_generation,
                    )
            low_index = json.loads((roots["low_boundary"] / "index.json").read_text(encoding="utf-8"))
            temporal_index = json.loads((roots["temporal_boundary"] / "index.json").read_text(encoding="utf-8"))
            validate_behavioral_category(pair, low_index, temporal_index)
            atomic_write_jsonl(divergence_path, divergence_for_case(pair_id, root, mapping, fingerprint))
            completed += 1
            print(
                f"  Capture checkpoint {completed}/{len(pairs)} saved after "
                f"{(time.perf_counter() - pair_started) / 60:.1f} min | {divergence_path}",
                flush=True,
            )
            _load_layer.cache_clear()
            if args.empty_cache_each_pair and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            errors.append({"pair_id": pair_id, "error": str(exc), "traceback": traceback.format_exc()})
            atomic_write_json(Path(root) / "capture_errors.json", errors)
            if not args.continue_on_error:
                raise
    atomic_write_json(Path(root) / "capture_summary.json", {
        "schema": SCHEMA, "run_fingerprint": fingerprint, "selected_pairs": len(pairs),
        "completed_pairs": completed, "failures": errors,
    })
    atomic_write_json(Path(root) / "capture_errors.json", errors)
    if errors and args.require_complete:
        raise RuntimeError(f"Capture has {len(errors)} unresolved failures.")


def run_patch_stage(args, model, processor, pairs, mappings, root, fingerprint):
    errors = []
    directions = (
        ("temporal_to_low", "low_boundary"),
        ("low_to_temporal", "temporal_boundary"),
    )
    selected_directions = [item for item in directions if args.direction in ("both", item[0])]
    for index, (pair_id, pair) in enumerate(pairs.items(), 1):
        print(f"Patch {index}/{len(pairs)}: {pair_id}", flush=True)
        pair_started = time.perf_counter()
        try:
            mapping = mappings[pair_id]
            roots = {condition: activation_root(root, pair_id, condition) for condition in pair}
            if not all(capture_complete(path, fingerprint) for path in roots.values()):
                raise RuntimeError("Capture shards are incomplete or have a different fingerprint.")
            prepared = prepare_pair(pair, processor, args, model.device)
            for condition in ("low_boundary", "temporal_boundary"):
                validate_prepared_mapping(prepared[condition], mapping, condition)
            if args.validate_controls:
                control_path = Path(root) / "technical_controls" / f"{pair_id}.json"
                saved_control = (
                    json.loads(control_path.read_text(encoding="utf-8"))
                    if control_path.is_file() else None
                )
                if saved_control is None or saved_control.get("run_fingerprint") != fingerprint:
                    controls = {
                        condition: validate_patch_controls(model, processor, prepared[condition], roots[condition])
                        for condition in ("low_boundary", "temporal_boundary")
                    }
                    atomic_write_json(control_path, {
                        "phase3b_pair_id": pair_id,
                        "run_fingerprint": fingerprint,
                        "conditions": controls,
                    })
            for direction, target in selected_directions:
                count = direction_patch_rows(
                    pair_id, direction, pair, mapping, root, model, processor,
                    prepared[target], fingerprint, args.continue_on_error,
                )
                print(
                    f"  {direction}: {count} completed patches after "
                    f"{(time.perf_counter() - pair_started) / 60:.1f} min | "
                    f"{Path(root) / 'patches' / pair_id / (direction + '.jsonl')}",
                    flush=True,
                )
                unresolved_path = Path(root) / "patches" / pair_id / f"{direction}.errors.json"
                unresolved = json.loads(unresolved_path.read_text(encoding="utf-8"))
                if unresolved and args.require_complete:
                    raise RuntimeError(f"{len(unresolved)} unresolved {direction} patch failures for {pair_id}.")
            _load_layer.cache_clear()
            if args.empty_cache_each_pair and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            errors.append({"pair_id": pair_id, "error": str(exc), "traceback": traceback.format_exc()})
            atomic_write_json(Path(root) / "patch_errors.json", errors)
            if not args.continue_on_error:
                raise
    atomic_write_json(Path(root) / "patch_summary.json", {
        "schema": SCHEMA, "run_fingerprint": fingerprint,
        "selected_pairs": len(pairs), "direction": args.direction, "failures": errors,
    })
    atomic_write_json(Path(root) / "patch_errors.json", errors)
    if errors and args.require_complete:
        raise RuntimeError(f"Patch stage has {len(errors)} unresolved failures.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("capture", "patch"), required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--mapping_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=5)
    parser.add_argument("--direction", choices=("both", "temporal_to_low", "low_to_temporal"), default="both")
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--model_revision", default="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
    parser.add_argument("--expected_transformers_version", default="5.9.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video_fps", type=float, default=None)
    parser.add_argument("--video_num_frames", type=int, default=None)
    parser.add_argument("--video_max_pixels", type=int, default=None)
    parser.add_argument("--roi_padding", type=int, default=8)
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--no_verify_standard_generation", dest="verify_standard_generation", action="store_false")
    parser.add_argument("--no_validate_controls", dest="validate_controls", action="store_false")
    parser.add_argument("--empty_cache_each_pair", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--no_require_complete", dest="require_complete", action="store_false")
    args = parser.parse_args()
    if args.video_fps is not None and args.video_num_frames is not None:
        parser.error("Use only one temporal sampling control.")
    if args.expected_transformers_version and transformers.__version__ != args.expected_transformers_version:
        parser.error(f"Expected transformers {args.expected_transformers_version}, got {transformers.__version__}.")
    configure_reproducibility(args.seed, deterministic=True)
    pairs = select_shard(case_pairs(read_jsonl(args.manifest_path)), args.shard_index, args.shard_size)
    if not pairs:
        parser.error("Selected shard is empty.")
    mappings = load_mappings(args.mapping_path)
    missing = set(pairs) - set(mappings)
    if missing:
        parser.error(f"Mapping audit is missing pairs: {sorted(missing)}")
    root = Path(args.output_dir) / f"shard_{args.shard_index:02d}"
    fingerprint_payload = {
        "schema": SCHEMA, "repo_commit": repo_commit(),
        "patching_code_sha256": {
            name: file_digest(Path(__file__).parent / name)
            for name in (
                "run_phase3b_patching.py", "phase3b_core.py", "probe_attention_roi.py",
                "run_eval.py", "activation_patching_core.py",
            )
        },
        "manifest_sha256": file_digest(args.manifest_path),
        "mapping_sha256": file_digest(args.mapping_path),
        "model_name": args.model_name, "model_revision": args.model_revision,
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "qwen_vl_utils_version": version("qwen-vl-utils"),
        "seed": args.seed, "video_fps": args.video_fps,
        "video_num_frames": args.video_num_frames, "video_max_pixels": args.video_max_pixels,
        "roi_padding": args.roi_padding, "attn_implementation": args.attn_implementation,
        "validate_controls": args.validate_controls,
        "shard_index": args.shard_index, "shard_size": args.shard_size,
        "pair_ids": list(pairs),
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    prepare_shard_run_config(
        root, fingerprint, fingerprint_payload,
        allow_failed_recovery=args.stage == "capture",
    )
    print(f"Phase 3B {args.stage} shard {args.shard_index}: {len(pairs)} pairs", flush=True)
    started = time.perf_counter()
    model, processor = load_model(
        args.model_name, model_revision=args.model_revision,
        attn_implementation=args.attn_implementation,
    )
    if args.stage == "capture":
        run_capture_stage(args, model, processor, pairs, mappings, root, fingerprint)
    else:
        run_patch_stage(args, model, processor, pairs, mappings, root, fingerprint)
    print(f"Finished in {(time.perf_counter() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
