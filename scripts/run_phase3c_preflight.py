"""Run resumable Phase 3C cohort baselines and technical preflight, not the primary grid."""

import argparse
import gc
import hashlib
import inspect
import json
import os
import platform
import shutil
import tempfile
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

try:
    from .phase3c_core import (
        CONDITIONS, SCHEMA, atomic_write, baseline_failures, boundary_outcomes,
        digest, file_hash, frozen_write, input_ids_hash, read_json,
    )
    from .phase3c_execution import (
        checkpoint_rows, execution_hashes, load_selection, require_stage, save_stage_summary, technical_tasks,
        technical_failures,
    )
    from .phase3b_paths import load_path_map
    from .run_phase3b_vm import parse_gpus, run_lock, worker_environment
except ImportError:
    from phase3c_core import (
        CONDITIONS, SCHEMA, atomic_write, baseline_failures, boundary_outcomes,
        digest, file_hash, frozen_write, input_ids_hash, read_json,
    )
    from phase3c_execution import (
        checkpoint_rows, execution_hashes, load_selection, require_stage, save_stage_summary, technical_tasks,
        technical_failures,
    )
    from phase3b_paths import load_path_map
    from run_phase3b_vm import parse_gpus, run_lock, worker_environment


def logits_parity(left, right, exact=False):
    import torch
    left, right = left.float().cpu(), right.float().cpu()
    if left.shape != right.shape:
        raise ValueError("Logit shapes differ during parity validation.")
    finite = torch.isfinite(left) & torch.isfinite(right)
    patterns = torch.equal(torch.isfinite(left), torch.isfinite(right))
    difference = (left[finite] - right[finite]).abs()
    exact_match = patterns and torch.equal(left, right)
    close = (patterns and bool(finite.any()) and not bool(torch.isnan(left).any()) and
             not bool(torch.isnan(right).any()) and bool(torch.allclose(left, right, rtol=0.001, atol=0.25)))
    return {"exact_match": exact_match, "logits_allclose": close,
            "max_abs_diff": float(difference.max()) if difference.numel() else None,
            "passed": exact_match if exact else close}


def save_tensors(path, payload):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Engine:
    def __init__(self, settings, mode, weight_budget):
        import torch
        import transformers
        try:
            from .run_eval import configure_reproducibility
            from .run_phase3b_patching import gpu_hardware_metadata, load_phase3b_model
        except ImportError:
            from run_eval import configure_reproducibility
            from run_phase3b_patching import gpu_hardware_metadata, load_phase3b_model
        self.torch = torch
        self.settings = settings
        packages = {name: version(name) for name in ("qwen-vl-utils", "accelerate", "numpy", "torchcodec", "av")}
        if (transformers.__version__ != settings["expected_transformers_version"] or
                torch.__version__.split("+")[0] != settings["expected_torch_version"] or
                packages["qwen-vl-utils"] != settings["expected_qwen_vl_utils_version"]):
            raise ValueError("Runtime differs from the frozen Phase 3C package versions; model not loaded.")
        if not torch.cuda.is_available():
            raise ValueError("Real baseline/preflight requires CUDA; CPU tests are not scientific evidence.")
        configure_reproducibility(settings["seed"], deterministic=True)
        args = SimpleNamespace(model_name=settings["model_name"], model_revision=settings["model_revision"],
            model_parallel=mode == "model_parallel", single_gpu=mode == "single_gpu",
            gpu_weight_budget_gib=weight_budget, attn_implementation=settings["attn_implementation"])
        started = time.perf_counter()
        self.model, self.processor = load_phase3b_model(args)
        self.load_sec = time.perf_counter() - started
        self.text = self.model.model.language_model
        architecture = self.model.config
        if (len(self.text.layers) != 36 or architecture.text_config.num_attention_heads != 32 or
                architecture.vision_config.deepstack_visual_indexes != [8, 16, 24] or
                architecture.video_token_id != 151656 or architecture._commit_hash != settings["model_revision"] or
                self.model.training or any(parameter.dtype != torch.float16 for parameter in self.model.parameters())):
            raise ValueError("Loaded model architecture/revision/precision differs from the frozen plan.")
        self.runtime = {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "packages": packages,
            "gpu_hardware": gpu_hardware_metadata(), "execution_mode": mode,
            "gpu_weight_budget_gib": weight_budget,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "hf_home": os.environ.get("HF_HOME"), "torch_home": os.environ.get("TORCH_HOME"),
            "model_device_map": {key: str(value) for key, value in self.model.hf_device_map.items()},
            "model_implementation_sha256": file_hash(inspect.getfile(type(self.text)))}

    def prepare(self, row, mapping):
        try:
            from .phase3b_core import prepare_example
            from .activation_patching_core import validate_archived_input_metadata
        except ImportError:
            from phase3b_core import prepare_example
            from activation_patching_core import validate_archived_input_metadata
        settings = self.settings
        prepared = prepare_example(row, self.processor, self.project_root,
            settings["video_fps"], settings["video_num_frames"], settings["video_max_pixels"],
            settings["roi_padding"], path_map=self.path_map)
        record = mapping["processor_records"][row["condition"]]
        inputs = prepared["inputs"]
        ids = inputs.input_ids[0].cpu().tolist()
        if (ids != record["input_ids"] or input_ids_hash(ids) != record["prompt_input_ids_sha256"] or
                inputs.attention_mask[0].cpu().tolist() != record["attention_mask"] or
                prepared["groups"] != record["group_positions"] or
                prepared["video_metadata"] != record["video_metadata"] or
                file_hash(prepared["video_path"]) != row["phase3c_video_sha256"]):
            raise ValueError("Live GPU-run processor inputs/video differ from the frozen CPU audit.")
        metadata = validate_archived_input_metadata(prepared)
        if metadata.get("matches") is not True:
            raise ValueError("Missing archived processor parity evidence.")
        prepared["archived_input_parity"] = metadata
        prepared["input_tensor_sha256"] = {
            name: {"dtype": str(value.dtype), "shape": list(value.shape),
                   "sha256": hashlib.sha256(value.detach().cpu().contiguous().view(self.torch.uint8).numpy().tobytes()).hexdigest()}
            for name, value in inputs.items() if self.torch.is_tensor(value)}
        prepared["inputs"] = inputs.to(self.model.device)
        return prepared

    def forward(self, prepared):
        try:
            from .probe_attention_roi import model_forward
        except ImportError:
            from probe_attention_roi import model_forward
        with self.torch.inference_mode():
            output = model_forward(self.model, {**dict(prepared["inputs"]), "use_cache": False,
                                                "return_dict": True, "output_attentions": False})
        logits = output.logits[0, -1].float().detach().cpu()
        if not bool(self.torch.isfinite(logits).all()):
            raise ValueError("Non-finite raw first-token logits.")
        return logits

    def decide(self, logits, row):
        try:
            from .activation_patching_core import decision_from_logits
        except ImportError:
            from activation_patching_core import decision_from_logits
        return decision_from_logits(logits, self.processor, row["correct_option"])

    def baseline(self, prepared, mapping, root, fingerprint):
        try:
            from .phase3c_interventions import ResidualSites
            from .probe_attention_roi import standard_first_token
        except ImportError:
            from phase3c_interventions import ResidualSites
            from probe_attention_roi import standard_first_token
        row = prepared["row"]
        record = mapping["processor_records"][row["condition"]]
        side = "low_positions" if row["condition"] == CONDITIONS[0] else "temporal_positions"
        positions = sorted(set(mapping["support_audit"]["supports"]["whole_event2"][side]) |
                           {position for group in prepared["groups"].values() for position in group})
        plain = self.forward(prepared)
        controller = ResidualSites(self.text, len(record["input_ids"]), record["visual_positions"], positions)
        with controller.installed():
            logits = self.forward(prepared)
        hook_audit = controller.validate()
        capture_parity = logits_parity(plain, logits, exact=True)
        standard_id, standard_logits = standard_first_token(self.model, prepared["inputs"])
        decision = self.decide(logits, row)
        parity = logits_parity(logits, standard_logits)
        parity["first_token_match"] = standard_id == decision["predicted_first_token_id"]
        tensors = (Path(root) / "captures" / row["phase3c_pair_id"] / row["condition"] /
                   f"attempt_{time.time_ns()}" / "states.pt")
        capture_bytes = sum(state["vectors"].numel() * state["vectors"].element_size() for state in controller.states.values())
        if shutil.disk_usage(root).free < capture_bytes * 1.1 + 512 * 1024 ** 2:
            raise RuntimeError("Persistent disk is too full for this capture; preserve and back up checkpoints before retrying.")
        save_tensors(tensors, {"states": controller.states, "logits": logits})
        index_path = tensors.parent / "index.json"
        groups = {**prepared["groups"], **{name: mapping["support_audit"]["supports"][name][side]
                  for name in ("both_targets_event2", "whole_event2")}}
        lookup = {position: index for index, position in enumerate(positions)}
        group_indices = {name: [lookup[position] for position in values] for name, values in groups.items()}
        norms = {f"{location}:L{layer}": {
            name: float(state["vectors"][indices].float().mean(0).norm()) if indices else None
            for name, indices in group_indices.items()}
            for (layer, location), state in controller.states.items()}
        atomic_write(index_path, {"schema": SCHEMA, "artifact_type": "real",
            "execution_fingerprint": fingerprint, "eval_id": row["eval_id"], "positions": positions,
            "vectors_path": str(tensors.resolve()), "vectors_sha256": file_hash(tensors),
            "site_count": len(controller.states), "hook_audit": hook_audit,
            "input_tensor_sha256": prepared["input_tensor_sha256"], "group_positions": groups,
            "group_indices": group_indices, "group_mean_norms_by_site": norms,
            "first_object_id": row["first_object_id"], "analysis_stratum": row["phase3c_analysis_stratum"]})
        result = {"eval_id": row["eval_id"], "decision": decision, "standard_parity": parity,
            "archived_input_parity": prepared["archived_input_parity"], "hook_audit": hook_audit,
            "capture_noop_parity": capture_parity, "capture_index_path": str(index_path.resolve()),
            "capture_index_sha256": file_hash(index_path), "input_metadata": prepared["input_metadata"],
            "input_tensor_sha256": prepared["input_tensor_sha256"]}
        result["passed"] = not baseline_failures(row, result) and capture_parity["passed"]
        return result

    def technical(self, prepared, mapping, task, captures):
        try:
            from .phase3c_interventions import AttentionKnockout, ResidualSites, selected_state
        except ImportError:
            from phase3c_interventions import AttentionKnockout, ResidualSites, selected_state
        condition = task["condition"]
        record = mapping["processor_records"][condition]
        baseline = captures[condition]["logits"]
        length = len(record["input_ids"])
        result = {"spec": task, "is_primary_effect_estimate": False, "prompt_token_count": length}
        if task["kind"] in ("identity", "transplant_smoke"):
            donor_condition = (condition if task["kind"] == "identity" else
                               CONDITIONS[1] if condition == CONDITIONS[0] else CONDITIONS[0])
            support = mapping["support_audit"]["supports"][task["support"]]
            side = "low_positions" if condition == CONDITIONS[0] else "temporal_positions"
            donor_side = "low_positions" if donor_condition == CONDITIONS[0] else "temporal_positions"
            donor = selected_state(captures[donor_condition]["states"], task["layer"], task["location"], support[donor_side])
            patch = {"layer": task["layer"], "location": task["location"], "recipient_positions": support[side], "donor": donor}
            controller = ResidualSites(self.text, length, record["visual_positions"], patch=patch)
            with controller.installed():
                logits = self.forward(prepared)
            result.update({"hook_audit": controller.validate(), "donor_condition": donor_condition,
                "recipient_condition": condition, "donor_positions": support[donor_side],
                "recipient_positions": support[side], "support_audit": support,
                "donor_capture_location": task["location"], "recipient_patch_location": task["location"]})
        else:
            route = mapping["support_audit"]["knockout_controls"][condition][task["query_group"]][task["key_group"]]["background"]
            keys = route["target_key_positions"] if task["control"] == "target" else route["control_key_positions"]
            controller = AttentionKnockout(self.text.layers, task["window"], route["query_positions"], keys,
                length, enabled=task["kind"] != "disabled_knockout")
            with controller.installed():
                logits = self.forward(prepared)
            result.update({"mask_audit": controller.validate(), "edge_budget": route["target_budget"],
                           "query_positions": route["query_positions"], "key_positions": keys})
        decision = self.decide(logits, prepared["row"])
        before = self.decide(baseline, prepared["row"])
        result.update({"baseline_decision": before, "decision": decision,
                       "margin_delta": decision["margin"] - before["margin"], "passed": True})
        if task["kind"] in ("identity", "disabled_knockout"):
            result["noop_parity"] = logits_parity(baseline, logits, exact=True)
            result["passed"] = result["noop_parity"]["passed"]
        return result


def bind_execution(root, frozen, runtime, project_root, path_map):
    tasks_binding = {"technical_noop_max_abs_diff": 0.0, "standard_logit_rtol": 0.001,
        "standard_logit_atol": 0.25, "attention_row_sum_atol": 0.005,
        "technical_smoke_only": True, "primary_grid_runner_implemented": False}
    config = {"schema": SCHEMA, "artifact_type": "real", "selection_fingerprint": frozen["selection_fingerprint"],
        "runtime": runtime, "execution_code_sha256": execution_hashes(), "technical_settings": tasks_binding,
        "project_root": str(Path(project_root).resolve()), "path_map": path_map}
    config["execution_fingerprint"] = digest(config)
    frozen_write(Path(root) / "execution_config.json", config)
    return config


def check_execution_request(root, frozen, project_root, path_map, mode, weight_budget, gpus):
    path = Path(root) / "execution_config.json"
    if not path.exists():
        return
    prior = read_json(path)
    if prior.get("execution_fingerprint") != digest({key: value for key, value in prior.items()
                                                    if key != "execution_fingerprint"}):
        raise ValueError("Execution configuration fingerprint is invalid.")
    runtime = prior["runtime"]
    if (prior.get("artifact_type") != "real" or prior["selection_fingerprint"] != frozen["selection_fingerprint"] or
            prior["execution_code_sha256"] != execution_hashes() or
            prior["project_root"] != str(Path(project_root).resolve()) or prior["path_map"] != path_map or
            runtime["execution_mode"] != mode or runtime["gpu_weight_budget_gib"] != weight_budget or
            runtime["visible_devices"] != ",".join(gpus)):
        raise ValueError("Incompatible execution request; preserve this checkpoint and use a new execution root.")


def progress(output, stage, completed, total, new, started, load_sec):
    elapsed = time.perf_counter() - started
    remaining = total - completed
    status = {"stage": stage, "completed": completed, "expected": total,
        "newly_computed": new, "elapsed_sec": elapsed, "model_load_sec": load_sec,
        "remaining": remaining, "eta_sec": elapsed / new * remaining if new else None,
        "checkpoint": str(Path(output) / "rows.jsonl"), "primary_grid_complete": False}
    atomic_write(Path(output) / "progress_status.json", status)
    print(f"Phase 3C {stage}: {completed}/{total}; new={new}; elapsed={elapsed / 60:.1f} min; "
          f"ETA={status['eta_sec'] / 60:.1f} min; checkpoint={status['checkpoint']}" if new else
          f"Phase 3C {stage}: reused {completed}/{total}; checkpoint={status['checkpoint']}", flush=True)


def run_stage(root, frozen, pairs, mappings, engine, execution, stage, max_tasks=None, retry_failed=False):
    root = Path(root)
    tasks = ([{"task_id": row["eval_id"], "pair_id": pair_id, "condition": condition, "kind": "baseline"}
              for pair_id, pair in pairs.items() for condition, row in pair.items()]
             if stage == "baseline" else technical_tasks(frozen, mappings))
    expected = {task["task_id"] for task in tasks}
    output = root / stage
    frozen_write(output / "task_manifest.jsonl", tasks, jsonl=True)
    saved = checkpoint_rows(output / "rows.jsonl", execution["execution_fingerprint"], expected)
    if stage == "preflight":
        baseline_ids = {row["eval_id"] for pair in pairs.values() for row in pair.values()}
        baselines = require_stage(root, execution, pairs, "baseline", baseline_ids)
    else:
        baselines = {}
        for item in saved.values():
            if item.get("passed"):
                index = read_json(item["capture_index_path"])
                if (file_hash(item["capture_index_path"]) != item["capture_index_sha256"] or
                        file_hash(index["vectors_path"]) != index["vectors_sha256"]):
                    raise ValueError("Resumed baseline capture bytes changed.")
    save_stage_summary(output, execution, pairs, saved, expected, stage)
    if not retry_failed and any(item.get("passed") is not True for item in saved.values()):
        print("A failed task is checkpointed. No new tasks will run without an explicit --retry_failed.", flush=True)
        return save_stage_summary(output, execution, pairs, saved, expected, stage)
    started, computed, prepared, current, captures = time.perf_counter(), 0, None, None, {}
    for task in tasks:
        task_id = task["task_id"]
        if task_id in saved and (saved[task_id].get("passed") or not retry_failed):
            continue
        if max_tasks is not None and computed >= max_tasks:
            break
        pair_id, condition = task["pair_id"], task["condition"]
        row = pairs[pair_id][condition]
        print(f"Starting {stage} {task['kind']}: {pair_id} {condition}; completed={len(saved)}/{len(tasks)}", flush=True)
        result = {"task_id": task_id, "execution_fingerprint": execution["execution_fingerprint"], "passed": False}
        task_started = time.perf_counter()
        failure_traceback = None
        try:
            if current != (pair_id, condition):
                prepared = None
                gc.collect()
                engine.torch.cuda.empty_cache()
                prepared = engine.prepare(row, mappings[pair_id])
                if stage == "preflight":
                    if prepared["input_tensor_sha256"] != baselines[row["eval_id"]].get("input_tensor_sha256"):
                        raise ValueError("Live processor tensor bytes differ from the passed GPU baseline.")
                    captures = {}
                    for side in CONDITIONS:
                        index = read_json(baselines[pairs[pair_id][side]["eval_id"]]["capture_index_path"])
                        captures[side] = engine.torch.load(index["vectors_path"], map_location="cpu", weights_only=True)
                current = (pair_id, condition)
            result.update(engine.baseline(prepared, mappings[pair_id], root, execution["execution_fingerprint"])
                          if stage == "baseline" else engine.technical(prepared, mappings[pair_id], task, captures))
        except Exception as exc:
            result.update({"failure_type": type(exc).__name__, "failure_message": str(exc), "spec": task})
            failure_traceback = traceback.format_exc()
            print(failure_traceback, flush=True)
        result["elapsed_sec"] = time.perf_counter() - task_started
        if not result["passed"]:
            errors_path = output / "errors.json"
            errors = read_json(errors_path) if errors_path.exists() else []
            errors.append({**result, "attempted_at": datetime.now(timezone.utc).isoformat(),
                "traceback": failure_traceback, "gate_failures": baseline_failures(row, result)
                if stage == "baseline" else technical_failures(result)})
            atomic_write(errors_path, errors)
        saved[task_id] = result
        atomic_write(output / "rows.jsonl", [saved[item["task_id"]] for item in tasks if item["task_id"] in saved], jsonl=True)
        computed += 1
        save_stage_summary(output, execution, pairs, saved, expected, stage)
        progress(output, stage, len(saved), len(tasks), computed, started, engine.load_sec)
        if not result["passed"]:
            print("Stage stopped at a failed control. Preserve evidence; --retry_failed explicitly retries it.", flush=True)
            break
    summary = save_stage_summary(output, execution, pairs, saved, expected, stage)
    if stage == "preflight":
        # Decomposition is technical smoke output, never the primary pilot estimate.
        comparisons = []
        grouped = {}
        for item in saved.values():
            if item.get("passed") and item.get("spec", {}).get("kind") == "knockout_smoke":
                spec = {key: value for key, value in item["spec"].items() if key not in ("condition", "task_id")}
                grouped.setdefault(digest(spec), {"spec": spec, "conditions": {}})["conditions"][item["spec"]["condition"]] = item
        for group in grouped.values():
            if set(group["conditions"]) != set(CONDITIONS):
                continue
            low, temporal = (group["conditions"][condition] for condition in CONDITIONS)
            comparisons.append({"spec": group["spec"], "is_primary_effect_estimate": False,
                **boundary_outcomes(low["baseline_decision"]["margin"], temporal["baseline_decision"]["margin"],
                                    low["decision"]["margin"], temporal["decision"]["margin"])})
        atomic_write(output / "boundary_smoke_diagnostics.json", comparisons)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("baseline", "preflight"))
    parser.add_argument("--plan_dir", required=True)
    parser.add_argument("--output_dir", help="Separate execution root; default PLAN_DIR/execution_v1.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--storage_root", default="/data/yuxuanstorage", help="Persistent VM volume for outputs and model caches.")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--execution_mode", choices=("model_parallel", "single_gpu"), default="model_parallel")
    parser.add_argument("--gpu_weight_budget_gib", type=float, default=10)
    parser.add_argument("--path_map")
    parser.add_argument("--max_tasks", type=int, help="Limit new tasks and checkpoint; an incomplete stage exits 2.")
    parser.add_argument("--retry_failed", action="store_true")
    args = parser.parse_args()
    if args.max_tasks is not None and args.max_tasks <= 0:
        parser.error("--max_tasks must be positive.")
    try:
        gpus = parse_gpus(args.gpus)
        if len(gpus) != (2 if args.execution_mode == "model_parallel" else 1):
            raise ValueError("GPU count does not match the requested execution mode.")
        frozen, pairs, mappings = load_selection(args.plan_dir)
        plan = read_json(Path(args.plan_dir) / "plan_config.json")
        source = Path(plan["source_run_root"]).resolve()
        root = Path(args.output_dir or str(Path(args.plan_dir) / "execution_v1")).resolve()
        plan_root = Path(args.plan_dir).resolve()
        if (source == root or source in root.parents or root in source.parents or
                root == plan_root or root in plan_root.parents):
            raise ValueError("Execution must not overwrite source evidence or preparation artifacts.")
        storage_root = Path(args.storage_root).resolve()
        if not storage_root.is_dir() or storage_root not in root.parents:
            raise ValueError("Execution output must be beneath an existing persistent storage root.")
        audit_config = read_json(Path(args.plan_dir) / "processor_audit/config.json")
        path_map = dict(audit_config["path_map"])
        path_map.update(load_path_map(args.path_map))
        if args.stage == "preflight":
            execution = read_json(root / "execution_config.json")
            if (execution["execution_code_sha256"] != execution_hashes() or
                    execution["selection_fingerprint"] != frozen["selection_fingerprint"]):
                raise ValueError("Execution code or selection changed; baseline/preflight must use a new root.")
            require_stage(root, execution, pairs, "baseline", {row["eval_id"] for pair in pairs.values() for row in pair.values()})
        environment = worker_environment(",".join(gpus), storage_root)
        for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
            os.environ.pop(name, None)
        os.environ.update(environment)
        with run_lock(root):
            check_execution_request(root, frozen, args.project_root, path_map, args.execution_mode,
                                    args.gpu_weight_budget_gib, gpus)
            engine = Engine(frozen["settings"], args.execution_mode, args.gpu_weight_budget_gib)
            engine.project_root, engine.path_map = args.project_root, path_map
            execution = bind_execution(root, frozen, engine.runtime, args.project_root, path_map)
            summary = run_stage(root, frozen, pairs, mappings, engine, execution, args.stage, args.max_tasks, args.retry_failed)
        print(json.dumps(summary, indent=2, sort_keys=True))
        if not summary["passed"]:
            parser.exit(2, "Stage incomplete or blocked. Check checkpoint/errors; primary intervention grid not started.\n")
        print("Technical stage PASSED; this does not mark the primary Phase 3C experiment complete.", flush=True)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(1, f"Phase 3C execution blocked: {exc}\nNo primary intervention grid was started.\n")


if __name__ == "__main__":
    main()
