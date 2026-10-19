"""Checked Phase 3D plans, processor audits, real captures and feature export."""

import copy
import gc
import hashlib
import math
import os
import time
from pathlib import Path

from .phase3b_paths import resolve_video_path
from .phase3c_core import (CONDITIONS, atomic_write, checked_positions, digest, file_hash,
                           frozen_write, input_ids_hash, read_json, read_jsonl)
from .phase3d_core import audit_pairs, audit_row, base_folds, declared_saved_sites

SCHEMA = "phase3d_execution_v1"
VIEWS = ("archived", "option_order_permuted")
FEATURE_NAMES = sorted([f"{group}:L{layer}_output" for group in
                        ("source_pair", "option_pair", "decision", "query", "option_A_pair", "option_B_pair",
                         "option_A_relation", "option_B_relation") for layer in range(36)] +
                       ["source_pair:L16_input"])


def seal(value):
    return {**value, "record_sha256": digest({k: v for k, v in value.items() if k != "record_sha256"})}


def checked_seal(value):
    if value.get("record_sha256") != seal(value)["record_sha256"]:
        raise ValueError("Altered or unsealed research checkpoint.")
    return value


def code_hashes():
    from .phase3c_execution import execution_hashes
    names = ("phase3d_core.py", "phase3d_features.py", "phase3d_execution.py", "phase3d_knockout.py",
             "run_phase3d_vm.py", "run_phase3d_probes.py", "phase3d_probes.py", "phase3c_knockout25.py",
             "phase3c_budget.py", "run_phase3c_preflight.py", "backup_phase3d.py", "backup_phase3b.py",
             "run_phase3c_vm.py", "run_phase3c_unattended.py", "phase3b_email.py", "surf_workspace.py")
    return {**execution_hashes(), **{name: file_hash(Path(__file__).with_name(name)) for name in names}}


def permuted_row(row):
    """New prompt, not an archived run: discard all stale output/input annotations."""
    result = {key: copy.deepcopy(value) for key, value in row.items()
              if not key.startswith(("archived_", "phase3"))}
    result.update(option_A=row["option_B"], option_B=row["option_A"],
                  correct_option="B" if row["correct_option"] == "A" else "A",
                  eval_id=row["eval_id"] + "__pure_option_permutation")
    result["correct_sentence"] = result["option_" + result["correct_option"]]
    audit_row(result)
    return result


def feature_tasks(plan, stage, pair_ids=None):
    if stage not in ("audit", "capture", "patch"):
        raise ValueError("Unknown Phase 3D feature stage.")
    result = []
    for pair_id in pair_ids or plan["case_ids"]:
        for condition in CONDITIONS:
            for view in VIEWS if stage != "patch" else ("archived",):
                item = {"pair_id": pair_id, "condition": condition, "view": view, "stage": stage}
                item["intervention"] = ("baseline" if stage != "patch" else
                    "temporal_to_low_L16" if condition == CONDITIONS[0] else "low_to_temporal_L16")
                item["task_id"] = digest([plan["fingerprint"], item])
                result.append(item)
    return result


def original_row(context, spec):
    _, pairs, _ = context
    row = pairs[spec["pair_id"]][spec["condition"]]
    return row if spec["view"] == "archived" else permuted_row(row)


def freeze_plan(args):
    source, root = Path(args.source_run).resolve(), Path(args.output_root).resolve()
    if source == root or source in root.parents or root in source.parents:
        raise ValueError("Phase 3B/3C sources must remain read-only in a separate root.")
    if root.exists() and not (root / "plan.json").exists() and any(
            path.name not in {".phase3d.lock", ".pipeline.lock"} for path in root.iterdir()):
        raise ValueError("Use a new empty Phase 3D output root; never adopt a historical run directory.")
    old = read_json(source / "plan.json")
    pairs, mappings = (read_json(source / f"{name}.json") for name in ("pairs", "mappings"))
    cases = audit_pairs(pairs)
    if (old.get("artifact_type") != "real" or old.get("fingerprint") !=
            digest({k: v for k, v in old.items() if k != "fingerprint"}) or
            ("pairs_sha256" in old and old["pairs_sha256"] != digest(pairs)) or
            old.get("mappings_sha256") != digest(mappings) or
            old.get("case_ids") != [case["pair_id"] for case in cases]):
        raise ValueError("Source plan/pairs/mappings are not intact real evidence.")
    summary = read_json(source / "baseline/summary.json")
    rows_path = source / "baseline/rows.jsonl"
    manifest_path = source / "baseline/task_manifest.jsonl"
    if (summary.get("complete") is not True or summary.get("completed") != 100 or summary.get("failed") != 0 or
            summary.get("plan_fingerprint") != old["fingerprint"] or
            summary.get("rows_sha256") != file_hash(rows_path) or
            summary.get("manifest_sha256") != file_hash(manifest_path)):
        raise ValueError("Source baseline gate is incomplete or altered.")
    source_specs = {r["task_id"]: r for r in read_jsonl(manifest_path)}
    baseline_rows = read_jsonl(rows_path)
    refs = {}
    for row in baseline_rows:
        checked_seal(row)
        if (row.get("passed") is not True or row.get("plan_fingerprint") != old["fingerprint"] or
                row.get("spec") != source_specs.get(row.get("task_id")) or row["eval_id"] in refs):
            raise ValueError("Invalid, foreign or duplicated source baseline.")
        refs[row["eval_id"]] = row
    expected = {r["eval_id"] for pair in pairs.values() for r in pair.values()}
    if set(refs) != expected or len(source_specs) != 100:
        raise ValueError("Source baseline grid differs from the frozen 50 bases.")
    videos = {}
    for pair_id, pair in pairs.items():
        mapping = mappings[pair_id]
        support = mapping["support_audit"]["supports"]["whole_event2"]
        if support.get("one_to_one") is not True:
            raise ValueError("Whole Event-2 support is not explicitly one-to-one.")
        for condition, side in zip(CONDITIONS, ("low_positions", "temporal_positions")):
            positions = checked_positions(support[side], side)
            if len(positions) != 512:
                raise ValueError("Phase 3D requires exactly 512 observed donor positions per condition.")
            row = pair[condition]
            path = resolve_video_path(row["video_path"], args.project_root, old.get("source_path_map", {}))
            if file_hash(path) != row["phase3c_video_sha256"]:
                raise ValueError("Selected frozen video bytes changed; do not regenerate.")
            videos[str(path)] = row["phase3c_video_sha256"]
            checked_source(refs[row["eval_id"]])
    reuse = None
    if getattr(args, "ko_source", None):
        from .phase3d_knockout import freeze_reuse_source
        ko_source = Path(args.ko_source).resolve()
        if root == ko_source or root in ko_source.parents or ko_source in root.parents:
            raise ValueError("Keep the historical KO root read-only and separate.")
        reuse = freeze_reuse_source(ko_source, pairs)
    plan = {"schema": SCHEMA, "artifact_type": "real", "source_root": str(source),
            "source_files": {name: file_hash(source / name) for name in
                ("plan.json", "pairs.json", "mappings.json", "baseline/summary.json",
                 "baseline/rows.jsonl", "baseline/task_manifest.jsonl")},
            "settings": old["settings"], "source_runtime": old["source_runtime"],
            "source_path_map": old.get("source_path_map", {}), "project_root": str(Path(args.project_root).resolve()),
            "video_sha256": videos, "case_ids": [c["pair_id"] for c in cases],
            "preflight_ids": [next(c["pair_id"] for c in cases if c["first_object_id"] == mover) for mover in (1, 2)],
            "case_metadata_sha256": digest(cases), "folds_sha256": digest(base_folds(cases)),
            "source_baselines_sha256": digest(refs), "code_sha256": code_hashes(),
            "protocol_sha256": file_hash(Path(__file__).resolve().parents[1] / "docs/phase3d_protocol.md"),
            "feature_names": FEATURE_NAMES, "capture_reuse": "Fresh compact means; exact replay of existing L16 donor states.",
            "ko_legacy_reuse": "Optional semantic matching; full original condition mask/input/runtime/baseline/output chain required. Summary-only inherited rows are recomputed.",
            "ko_reuse_source_sha256": digest(reuse) if reuse else None,
            "case_reporting": "All 50 bases equally, without pilot/new strata."}
    plan["fingerprint"] = digest(plan)
    frozen_write(root / "plan.json", plan)
    for name, value in (("pairs", pairs), ("mappings", mappings), ("case_metadata", cases),
                        ("outer_folds", base_folds(cases)), ("source_baselines", refs)):
        frozen_write(root / f"{name}.json", value)
    if reuse is not None:
        frozen_write(root / "ko_reuse_source.json", reuse)
    return plan, pairs, mappings


def load_context(root):
    root = Path(root)
    plan = read_json(root / "plan.json")
    if (plan.get("schema") != SCHEMA or plan.get("artifact_type") != "real" or
            plan.get("fingerprint") != digest({k: v for k, v in plan.items() if k != "fingerprint"}) or
            plan["code_sha256"] != code_hashes() or
            plan["protocol_sha256"] != file_hash(Path(__file__).resolve().parents[1] / "docs/phase3d_protocol.md")):
        raise ValueError("Phase 3D plan/code/protocol changed; preserve this root and use a new one.")
    for name, sha in plan["source_files"].items():
        if file_hash(Path(plan["source_root"]) / name) != sha:
            raise ValueError("Frozen source metadata changed.")
    pairs, mappings = (read_json(root / f"{name}.json") for name in ("pairs", "mappings"))
    cases = audit_pairs(pairs)
    if (digest(cases) != plan["case_metadata_sha256"] or read_json(root / "case_metadata.json") != cases or
            digest(read_json(root / "outer_folds.json")) != plan["folds_sha256"] or
            digest(read_json(root / "source_baselines.json")) != plan["source_baselines_sha256"] or
            digest(mappings) != read_json(Path(plan["source_root"]) / "plan.json")["mappings_sha256"]):
        raise ValueError("Frozen Phase 3D snapshot changed.")
    if plan.get("ko_reuse_source_sha256"):
        reuse = read_json(root / "ko_reuse_source.json")
        if digest(reuse) != plan["ko_reuse_source_sha256"]:
            raise ValueError("Frozen KO reuse candidate ledger changed.")
        for name, sha in reuse["source_files"].items():
            if file_hash(Path(reuse["source_root"]) / name) != sha:
                raise ValueError("Historical KO source evidence changed.")
    return plan, pairs, mappings


def checked_source(row, tensor=False):
    checked_seal(row)
    path = Path(row["capture_index_path"])
    if file_hash(path) != row["capture_index_sha256"]:
        raise ValueError("Source capture index checksum changed.")
    index = read_json(path)
    if (index.get("artifact_type") != "real" or index["eval_id"] != row["eval_id"] or
            index["input_tensor_sha256"] != row["input_tensor_sha256"] or
            (16, "block_output") not in (declared_saved_sites(index) or set())):
        raise ValueError("Source lacks a declared, input-bound L16 block-output capture.")
    if tensor:
        import torch
        if file_hash(index["vectors_path"]) != index["vectors_sha256"]:
            raise ValueError("Source activation tensor checksum changed.")
        return torch.load(index["vectors_path"], map_location="cpu", weights_only=True)
    return index


def tensor_hashes(inputs):
    import torch
    return {name: {"dtype": str(value.dtype), "shape": list(value.shape),
                   "sha256": hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
            for name, value in inputs.items() if torch.is_tensor(value)}


def processor_record(prepared):
    ids = prepared["inputs"].input_ids[0].cpu().tolist()
    return {"input_ids": ids, "prompt_input_ids_sha256": input_ids_hash(ids),
            "attention_mask": prepared["inputs"].attention_mask[0].cpu().tolist(),
            "visual_positions": [i for i, value in enumerate(ids) if value == 151656],
            "group_positions": prepared["groups"], "video_metadata": prepared["video_metadata"],
            "input_tensor_sha256": tensor_hashes(prepared["inputs"]),
            "video_sha256": file_hash(prepared["video_path"])}


def prepare_live(processor, context, spec):
    from .phase3b_core import prepare_example
    plan, _, mappings = context
    row = original_row(context, spec)
    settings = plan["settings"]
    live = prepare_example(row, processor, plan["project_root"], settings["video_fps"],
        settings["video_num_frames"], settings["video_max_pixels"], settings["roi_padding"], path_map=plan["source_path_map"])
    record = processor_record(live)
    old = mappings[spec["pair_id"]]["processor_records"][spec["condition"]]
    if record["video_sha256"] != old["video_sha256"] or record["video_metadata"] != old["video_metadata"]:
        raise ValueError("Frozen video/processor sampling metadata changed.")
    if spec["view"] == "archived":
        for name in ("input_ids", "prompt_input_ids_sha256", "attention_mask", "visual_positions", "group_positions"):
            if record[name] != old[name]:
                raise ValueError("Original processor mapping differs from frozen audit.")
    else:
        if record["input_ids"] == old["input_ids"] or record["visual_positions"] != old["visual_positions"]:
            raise ValueError("Pure option permutation must change text but preserve visual positions.")
        for group in ("video_t1_e1", "video_t1_e2", "video_t2_e1", "video_t2_e2"):
            if record["group_positions"][group] != old["group_positions"][group]:
                raise ValueError("Pure option permutation changed a visual identity/event group.")
    return live, record


def checkpoint_path(root, spec):
    return Path(root) / spec["stage"] / "task_checkpoints" / (spec["task_id"] + ".json")


def saved_checkpoint(root, plan, spec):
    path = checkpoint_path(root, spec)
    if not path.exists():
        return None
    row = checked_seal(read_json(path))
    if row.get("spec") != spec or row.get("plan_fingerprint") != plan["fingerprint"] or row.get("passed") is not True:
        raise ValueError("Foreign or failed Phase 3D checkpoint; never silently skip it.")
    if "payload" in row:
        payload_path(root, row)
    return row


def payload_path(root, row):
    spec = row["payload"]
    path = (Path(root) / spec["path"]).resolve()
    if not path.is_relative_to(Path(root).resolve()) or path.is_symlink() or file_hash(path) != spec["sha256"]:
        raise ValueError("Payload path/hash changed.")
    return path


def payload(root, row):
    import torch
    return torch.load(payload_path(root, row), map_location="cpu", weights_only=True)


def commit(root, plan, spec, result, tensors=None):
    from .run_phase3c_preflight import save_tensors
    if tensors is not None:
        path = Path(spec["stage"]) / "payloads" / f"{spec['task_id']}_{time.time_ns()}.pt"
        save_tensors(Path(root) / path, tensors)
        result["payload"] = {"path": path.as_posix(), "sha256": file_hash(Path(root) / path)}
    row = seal({**result, "spec": spec, "plan_fingerprint": plan["fingerprint"], "passed": True})
    frozen_write(checkpoint_path(root, spec), row)
    return row


def run_tasks(root, context, specs, execute, retry_failed=False):
    plan = context[0]
    started, reused = time.perf_counter(), 0
    frozen_write(Path(root) / specs[0]["stage"] / ("manifest_" + digest(specs) + ".json"), specs)
    for count, spec in enumerate(specs, 1):
        row = saved_checkpoint(root, plan, spec)
        if row is not None:
            reused += 1
        else:
            failure = Path(root) / "failures" / (spec["task_id"] + ".json")
            if failure.exists() and not retry_failed:
                raise RuntimeError("Earlier failure preserved; inspect logs and use --retry_failed explicitly.")
            if __import__("shutil").disk_usage(root).free < 12 * 1024**3:
                raise RuntimeError("Less than 12 GiB persistent reserve; stop without deleting evidence.")
            try:
                print(f"[Phase 3D {spec['stage']}] starting {count}/{len(specs)}: "
                      f"{spec['pair_id']} {spec['condition']} {spec['view']}", flush=True)
                result, tensors = execute(spec)
                row = commit(root, plan, spec, result, tensors)
            except Exception as exc:
                atomic_write(failure, {"spec": spec, "error_type": type(exc).__name__, "message": str(exc),
                                       "at_ns": time.time_ns()})
                raise
        elapsed = time.perf_counter() - started
        status = {"stage": spec["stage"], "completed": count, "expected": len(specs), "resumed": reused,
                  "elapsed_sec": elapsed, "checkpoint": str(checkpoint_path(root, spec))}
        atomic_write(Path(root) / "progress_status.json", status)
        print(f"[Phase 3D {spec['stage']}] {count}/{len(specs)}; resumed={reused}; "
              f"elapsed={elapsed/60:.1f} min; checkpoint={status['checkpoint']}", flush=True)
    return [saved_checkpoint(root, plan, spec) for spec in specs]


def audit_execute(root, context, processor, spec):
    from .phase3d_features import capture_groups
    live, record = prepare_live(processor, context, spec)
    plan, _, mappings = context
    original = mappings[spec["pair_id"]]["processor_records"][spec["condition"]]
    refs = read_json(Path(root) / "source_baselines.json")
    primary = context[1][spec["pair_id"]][spec["condition"]]
    expected = refs[primary["eval_id"]]["input_tensor_sha256"]
    if spec["view"] == "archived":
        if record["input_tensor_sha256"] != expected:
            raise ValueError("Archived processor tensor bytes did not replay exactly.")
    else:
        for name in ("pixel_values_videos", "video_grid_thw"):
            if record["input_tensor_sha256"].get(name) != expected.get(name):
                raise ValueError("Pure A/B control changed video tensor bytes.")
        if record["prompt_input_ids_sha256"] == original["prompt_input_ids_sha256"]:
            raise ValueError("Permuted prompt reused the archived input hash.")
    side = "low_positions" if spec["condition"] == CONDITIONS[0] else "temporal_positions"
    whole = mappings[spec["pair_id"]]["support_audit"]["supports"]["whole_event2"][side]
    groups = capture_groups(live["row"], record, whole, processor.tokenizer)
    return {"processor_record": record, "groups": groups,
            "input_fingerprint": digest(record["input_tensor_sha256"]), "require_rescue_behavior": spec["view"] == "archived"}, None


def audited_live(root, context, engine, spec):
    audit_spec = next(s for s in feature_tasks(context[0], "audit", [spec["pair_id"]])
                      if s["condition"] == spec["condition"] and s["view"] == spec["view"])
    audit = saved_checkpoint(root, context[0], audit_spec)
    if audit is None:
        raise ValueError("Complete processor-only audit before GPU work.")
    live, record = prepare_live(engine.processor, context, spec)
    if record != audit["processor_record"]:
        raise ValueError("Live GPU input differs from its fresh CPU audit.")
    live["inputs"] = live["inputs"].to(engine.model.device)
    live["input_tensor_sha256"] = record["input_tensor_sha256"]
    return live, audit


def timing_signature(controller):
    from .phase3c_budget import tensor_signature
    return {"upstream": {str(layer): {name: tensor_signature(value) for name, value in groups.items()}
                         for layer, groups in controller.means.items() if layer < 16},
            "same_block_text": {name: tensor_signature(value) for name, value in controller.means[16].items()
                                if not name.startswith("source_") and name != "whole_event2"},
            "text16": tensor_signature(controller.text16),
            "input16": {name: tensor_signature(value) for name, value in controller.input16.items()}}


def observe(engine, live, audit, patch=None):
    from .phase3d_features import CompactSites
    record = audit["processor_record"]
    controller = CompactSites(engine.text, len(record["input_ids"]), record["visual_positions"], audit["groups"], patch)
    with controller.installed():
        logits = engine.forward(live)
    controller.validate()
    features = controller.feature_views()
    if sorted(features) != FEATURE_NAMES:
        raise RuntimeError("Missing compact feature readout.")
    tensors = {"features": engine.torch.stack([features[name] for name in FEATURE_NAMES]), "logits": logits,
               "timing": timing_signature(controller)}
    return controller, tensors


def capture_execute(root, context, engine, spec):
    from .phase3c_interventions import selected_state
    from .phase3d_features import mapped_l16_patch
    from .probe_attention_roi import standard_first_token
    from .run_phase3c_preflight import logits_parity
    live, audit = audited_live(root, context, engine, spec)
    plain = engine.forward(live)
    controller, tensors = observe(engine, live, audit)
    token, standard = standard_first_token(engine.model, live["inputs"])
    decision = engine.decide(tensors["logits"], live["row"])
    gates = {"capture_noop": logits_parity(plain, tensors["logits"], exact=True),
             "standard": {**logits_parity(tensors["logits"], standard),
                          "first_token_match": token == decision["predicted_first_token_id"]}}
    if not gates["capture_noop"]["passed"] or not gates["standard"]["passed"] or not gates["standard"]["first_token_match"]:
        raise RuntimeError("Capture/no-op or standard first-token/full-logit parity failed.")
    tensors.update(plain_logits=plain, standard_logits=standard.detach().float().cpu())
    result = {"input_fingerprint": audit["input_fingerprint"], "audit_record_sha256": audit["record_sha256"],
              "decision": decision, "gates": gates, "hook_audit": controller.validate()}
    if spec["view"] == "archived":
        ref = read_json(Path(root) / "source_baselines.json")[live["row"]["eval_id"]]
        source = checked_source(ref, tensor=True)
        positions = controller.donor16["positions"]
        vectors = selected_state(source["states"], 16, "block_output", positions)["vectors"]
        replay = {"logits": logits_parity(source["logits"], tensors["logits"], exact=True),
                  "states_exact": engine.torch.equal(vectors, controller.donor16["vectors"]),
                  "inputs_exact": ref["input_tensor_sha256"] == live["input_tensor_sha256"],
                  "decision_exact": ref["decision"] == decision}
        correct = decision["prediction"] == live["row"]["correct_option"]
        if (not replay["logits"]["passed"] or not all(replay[name] for name in ("states_exact", "inputs_exact", "decision_exact")) or
                decision["prediction"] not in ("A", "B") or decision["margin"] == 0 or
                correct != (decision["margin"] > 0) or correct != (spec["condition"] == CONDITIONS[1])):
            raise RuntimeError("Frozen rescue/source-state replay failed; do not drop, replace or relax this case.")
        identity = mapped_l16_patch(controller.donor16, positions, positions, positions)
        identity_controller, identity_payload = observe(engine, live, audit, identity)
        gates["identity"] = logits_parity(tensors["logits"], identity_payload["logits"], exact=True)
        if (not gates["identity"]["passed"] or
                not engine.torch.equal(tensors["features"], identity_payload["features"]) or
                tensors["timing"] != identity_payload["timing"]):
            raise RuntimeError("Same-state identity patch changed logits, features or timing.")
        tensors.update(identity_logits=identity_payload["logits"], identity_features=identity_payload["features"],
                       identity_timing=identity_payload["timing"])
        # A disabled real mask hook is checked before the expensive KO grid.
        from .phase3c_interventions import AttentionKnockout
        disabled = AttentionKnockout(engine.text.layers, [0, 1, 2, 3], audit["groups"]["query_all"],
            audit["groups"]["whole_event2"], len(audit["processor_record"]["input_ids"]), enabled=False)
        with disabled.installed():
            disabled_logits = engine.forward(live)
        gates["disabled_ko"] = logits_parity(tensors["logits"], disabled_logits, exact=True)
        result["disabled_ko_mask_audit"] = disabled.validate()
        if not gates["disabled_ko"]["passed"]:
            raise RuntimeError("Disabled attention-mask hook changed baseline logits.")
        tensors["disabled_ko_logits"] = disabled_logits
        result.update(source_replay=replay, source_record_sha256=ref["record_sha256"],
                      source_index_sha256=ref["capture_index_sha256"])
        tensors["donor16"] = controller.donor16
        del source, identity_controller, identity_payload
    gc.collect()
    engine.torch.cuda.empty_cache()
    return result, tensors


def capture_reference(root, context, pair_id, condition):
    spec = next(s for s in feature_tasks(context[0], "capture", [pair_id])
                if s["condition"] == condition and s["view"] == "archived")
    row = saved_checkpoint(root, context[0], spec)
    if row is None:
        raise ValueError("Capture gate is incomplete; patch/KO is blocked.")
    return row, payload(root, row)


def patch_execute(root, context, engine, spec):
    from .phase3d_features import mapped_l16_patch
    from .run_phase3c_preflight import logits_parity
    live, audit = audited_live(root, context, engine, spec)
    condition = spec["condition"]
    opposite = CONDITIONS[1] if condition == CONDITIONS[0] else CONDITIONS[0]
    before, baseline = capture_reference(root, context, spec["pair_id"], condition)
    donor, source = capture_reference(root, context, spec["pair_id"], opposite)
    replay = logits_parity(baseline["logits"], engine.forward(live), exact=True)
    if not replay["passed"] or before["input_fingerprint"] != audit["input_fingerprint"]:
        raise RuntimeError("Patch recipient replay failed.")
    support = context[2][spec["pair_id"]]["support_audit"]["supports"]["whole_event2"]
    recipient_side = "low_positions" if condition == CONDITIONS[0] else "temporal_positions"
    donor_side = "temporal_positions" if condition == CONDITIONS[0] else "low_positions"
    patch = mapped_l16_patch(source["donor16"], support[donor_side], support[recipient_side], audit["groups"]["whole_event2"])
    controller, tensors = observe(engine, live, audit, patch)
    if tensors["timing"] != baseline["timing"]:
        raise RuntimeError("Patch altered L16 input/upstream/same-block text; downstream begins at L17.")
    lookup = {position: index for index, position in enumerate(patch["recipient_positions"])}
    expected = patch["donor"]["vectors"][[lookup[p] for p in controller.donor16["positions"]]]
    if not engine.torch.equal(expected, controller.donor16["vectors"]):
        raise RuntimeError("Observed recipient support was not fully replaced by mapped donor vectors.")
    tensors["patched_visual16"] = controller.donor16
    return {"input_fingerprint": audit["input_fingerprint"], "audit_record_sha256": audit["record_sha256"],
            "baseline_record_sha256": before["record_sha256"], "donor_record_sha256": donor["record_sha256"],
            "decision": engine.decide(tensors["logits"], live["row"]), "recipient_replay": replay,
            "same_block_timing_exact": True, "observed_donor_replacement_exact": True,
            "recipient_positions": support[recipient_side], "donor_positions": support[donor_side],
            "hook_audit": controller.validate()}, tensors


def verify_features(root, context, pair_ids=None, patched=True):
    """Rebuild gates from task data; no hand-filled overall boolean is accepted."""
    import torch
    from .run_phase3c_preflight import logits_parity
    from .phase3c_interventions import selected_state
    plan, pairs, mappings = context
    specs = feature_tasks(plan, "capture", pair_ids) + (feature_tasks(plan, "patch", pair_ids) if patched else [])
    results = []
    started = time.perf_counter()
    refs = read_json(Path(root) / "source_baselines.json")
    for count, spec in enumerate(specs, 1):
        row = saved_checkpoint(root, plan, spec)
        if row is None:
            raise ValueError("Incomplete feature/capture grid.")
        audit_spec = next(s for s in feature_tasks(plan, "audit", [spec["pair_id"]])
            if s["condition"] == spec["condition"] and s["view"] == spec["view"])
        audit = saved_checkpoint(root, plan, audit_spec)
        data = payload(root, row)
        if (audit is None or row["audit_record_sha256"] != audit["record_sha256"] or
                row["input_fingerprint"] != digest(audit["processor_record"]["input_tensor_sha256"]) or
                data["features"].shape[0] != len(FEATURE_NAMES) or data["features"].ndim != 2 or
                data["features"].dtype != torch.float32 or not bool(torch.isfinite(data["features"]).all()) or
                not bool(torch.isfinite(data["logits"]).all())):
            raise ValueError("Feature payload/input/audit binding is invalid.")
        record = audit["processor_record"]
        support = mappings[spec["pair_id"]]["support_audit"]["supports"]["whole_event2"]
        side = "low_positions" if spec["condition"] == CONDITIONS[0] else "temporal_positions"
        if (audit["input_fingerprint"] != digest(record["input_tensor_sha256"]) or
                record["prompt_input_ids_sha256"] != input_ids_hash(record["input_ids"]) or
                audit["groups"]["whole_event2"] != support[side] or len(support[side]) != 512):
            raise ValueError("Audited prompt/matched-support binding changed.")
        hook = row["hook_audit"]
        if (hook["block_output_sites"] != 36 or hook["post_deepstack_layers"] != [0, 1, 2] or
                hook["L16_input_matches_L15_output_at_planned_positions"] is not True or
                hook["patch_applied_count"] != (1 if spec["stage"] == "patch" else 0)):
            raise ValueError("Saved capture hook semantics differ from the protocol.")
        decision = row["decision"]
        if (any(type(decision.get(name)) not in (int, float) or not math.isfinite(decision[name]) for name in
                ("margin", "correct_logit", "incorrect_logit")) or
                not math.isclose(decision["margin"], decision["correct_logit"] - decision["incorrect_logit"], abs_tol=1e-8) or
                decision["correct_option"] != original_row(context, spec)["correct_option"] or
                float(data["logits"][decision["correct_token_id"]]) != decision["correct_logit"] or
                float(data["logits"][decision["incorrect_token_id"]]) != decision["incorrect_logit"] or
                int(data["logits"].argmax()) != decision["predicted_first_token_id"]):
            raise ValueError("Saved decisions differ from measured payload logits.")
        if spec["stage"] == "capture":
            gates = row["gates"]
            if (not torch.equal(data["plain_logits"], data["logits"]) or
                    not logits_parity(data["logits"], data["standard_logits"])["passed"] or
                    int(data["standard_logits"].argmax()) != decision["predicted_first_token_id"] or
                    not gates["capture_noop"]["exact_match"] or gates["capture_noop"]["max_abs_diff"] != 0 or
                    not gates["standard"]["passed"] or not gates["standard"]["first_token_match"]):
                raise ValueError("Saved baseline parity/no-op failed.")
            if spec["view"] == "archived":
                source = row["source_replay"]
                ref = refs[pairs[spec["pair_id"]][spec["condition"]]["eval_id"]]
                old_data = checked_source(ref, tensor=True)
                donor = data["donor16"]
                if (not source["logits"]["exact_match"] or source["logits"]["max_abs_diff"] != 0 or
                        not all(source[k] for k in ("states_exact", "inputs_exact", "decision_exact")) or
                        not gates["identity"]["exact_match"] or gates["identity"]["max_abs_diff"] != 0 or
                        not gates["disabled_ko"]["exact_match"] or
                        row["source_record_sha256"] != ref["record_sha256"] or
                        row["decision"] != ref["decision"] or
                        audit["processor_record"]["input_tensor_sha256"] != ref["input_tensor_sha256"] or
                        not torch.equal(data["logits"], old_data["logits"].float()) or
                        not torch.equal(donor["vectors"], selected_state(old_data["states"], 16, "block_output", donor["positions"])["vectors"]) or
                        not torch.equal(data["identity_logits"], data["logits"]) or
                        not torch.equal(data["disabled_ko_logits"], data["logits"]) or
                        not torch.equal(data["identity_features"], data["features"]) or data["identity_timing"] != data["timing"]):
                    raise ValueError("Saved source/identity/disabled-mask gates failed.")
                del old_data
        else:
            before, baseline = capture_reference(root, context, spec["pair_id"], spec["condition"])
            donor_condition = CONDITIONS[1] if spec["condition"] == CONDITIONS[0] else CONDITIONS[0]
            donor, donor_payload = capture_reference(root, context, spec["pair_id"], donor_condition)
            support = mappings[spec["pair_id"]]["support_audit"]["supports"]["whole_event2"]
            side = "low_positions" if spec["condition"] == CONDITIONS[0] else "temporal_positions"
            other = "temporal_positions" if spec["condition"] == CONDITIONS[0] else "low_positions"
            observed = data["patched_visual16"]
            lookup = {position: i for i, position in enumerate(donor_payload["donor16"]["positions"])}
            expected = donor_payload["donor16"]["vectors"][[lookup[p] for p in support[other]]]
            destination = {position: i for i, position in enumerate(observed["positions"])}
            if (row["baseline_record_sha256"] != before["record_sha256"] or
                    row["donor_record_sha256"] != donor["record_sha256"] or data["timing"] != baseline["timing"] or
                    row["recipient_positions"] != support[side] or row["donor_positions"] != support[other] or
                    not row["recipient_replay"]["exact_match"] or row["recipient_replay"]["max_abs_diff"] != 0 or
                    set(observed["positions"]) != set(support[side]) or
                    not torch.equal(expected, observed["vectors"][[destination[p] for p in support[side]]]) or
                    row["hook_audit"]["patch_applied_count"] != 1 or
                    row.get("same_block_timing_exact") is not True or row.get("observed_donor_replacement_exact") is not True):
                raise ValueError("Saved patch input/donor/timing/replacement gate failed.")
        results.append(row)
        if count % 10 == 0 or count == len(specs):
            print(f"[Phase 3D verification] {count}/{len(specs)} checked; elapsed={(time.perf_counter()-started)/60:.1f} min", flush=True)
    return results


def export_features(root, context):
    import numpy as np
    from .run_phase3d_probes import DATASET_SCHEMA, GATES, load_dataset, save_npz
    records = verify_features(root, context)
    plan = context[0]
    cases = read_json(Path(root) / "case_metadata.json")
    labels = {case["pair_id"]: case for case in cases}
    rows, arrays = [], []
    for row in records:
        spec = row["spec"]
        case = labels[spec["pair_id"]]
        rows.append({"base_sample_id": case["base_sample_id"], "y": case["y"], "pair_id": case["pair_id"],
                     "condition": spec["condition"], "view": spec["view"], "intervention": spec["intervention"],
                     "correct_option": row["decision"]["correct_option"], "answer_margin": row["decision"]["margin"],
                     "input_fingerprint": row["input_fingerprint"], "source_checkpoint_sha256": row["record_sha256"]})
        arrays.append(payload(root, row)["features"].numpy())
    output = Path(root) / "feature_store"
    gates = {"schema": "phase3d_capture_gate_summary_v1", "complete": True, "base_count": 50,
             "case_labels_sha256": digest(sorted((c["base_sample_id"], c["y"]) for c in cases)),
             "plan_fingerprint": plan["fingerprint"], "verified_record_hashes": [r["record_sha256"] for r in records],
             **{name: True for name in GATES}}
    for name, value in (("rows.json", rows), ("case_metadata.json", cases),
                        ("outer_folds.json", read_json(Path(root) / "outer_folds.json")), ("capture_gates.json", gates)):
        frozen_write(output / name, value)
    values = np.stack(arrays)
    feature_path = output / "features.npz"
    if feature_path.exists():
        with np.load(feature_path, allow_pickle=False) as saved:
            if not np.array_equal(saved["features"], values):
                raise ValueError("Existing feature store changed; never overwrite completed evidence.")
    else:
        save_npz(feature_path, features=values)
    manifest = {"schema": DATASET_SCHEMA, "artifact_type": "real", "complete": True,
                "plan_fingerprint": plan["fingerprint"], "feature_names": FEATURE_NAMES}
    for name, filename in (("case_metadata", "case_metadata.json"), ("outer_folds", "outer_folds.json"),
                           ("row_metadata", "rows.json"), ("gate_summary", "capture_gates.json"), ("features", "features.npz")):
        manifest[name] = {"path": filename, "sha256": file_hash(output / filename)}
    frozen_write(output / "manifest.json", manifest)
    load_dataset(output / "manifest.json")
    return manifest
