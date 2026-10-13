"""Queued every-four-layer depth scan followed by a frozen 25-rescue KO grid."""

import argparse
import gc
import importlib.util
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from scripts import phase3c_depth_protocol as protocol
from scripts import phase3c_knockout25 as knockout
from scripts import run_phase3c_scale as prefix_runner
from scripts.analyze_phase3c import control_contrasts
from scripts.analyze_phase3c import verify_inputs as verify_pilot
from scripts.phase3c_core import atomic_write, baseline_failures, digest, file_hash, frozen_write, read_json
from scripts.phase3c_interventions import ResidualSites, selected_state
from scripts.plot_phase3c_depth import plot

# Load an isolated adapter namespace; never alter the running prefix module or its files.
module_spec = importlib.util.spec_from_file_location("scripts._depth50_engine_adapter",ROOT/"scripts/run_phase3c_scale.py")
api = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(api)
api.protocol = protocol
CONDITIONS = protocol.CONDITIONS


class DepthSites(ResidualSites):
    def visit(self, hidden, layer, location):
        positions = self.positions
        if location != "block_output" or layer not in protocol.GRID:
            self.positions = None
        try:
            return super().visit(hidden,layer,location)
        finally:
            self.positions = positions

    def validate(self):
        positions,self.positions = self.positions,None
        try:
            result = super().validate()
        finally:
            self.positions = positions
        if set(self.states) != {(layer,"block_output") for layer in protocol.GRID}:
            raise RuntimeError("Missing every-four-layer capture site.")
        return {**result,"observed_sites":[[layer,"block_output"] for layer in protocol.GRID]}


def capture_extra(engine,live,mapping,reference,root,plan):
    from scripts.probe_attention_roi import standard_first_token
    if reference is None:
        raise ValueError("Every expanded-depth condition requires a verified prefix baseline.")
    before = reference["row"]
    payload = api.checked_payload(before,engine.torch)
    row = live["row"]
    record = mapping["processor_records"][row["condition"]]
    side = "low_positions" if row["condition"] == CONDITIONS[0] else "temporal_positions"
    positions = sorted(mapping["support_audit"]["supports"]["whole_event2"][side])
    plain = engine.forward(live)
    controller = DepthSites(engine.text,len(record["input_ids"]),record["visual_positions"],positions)
    with controller.installed():
        logits = engine.forward(live)
    audit = controller.validate()
    token,standard = standard_first_token(engine.model,live["inputs"])
    decision = engine.decide(logits,row)
    parity = api.logits_parity(logits,standard)
    parity["first_token_match"] = token == decision["predicted_first_token_id"]
    reused = reference["index"]["site_count"] == 39
    replay_layers = protocol.GRID if reused else protocol.prefix.LAYERS
    replay = {"logits_exact":api.logits_parity(payload["logits"],logits,exact=True)["passed"],
        "input_bytes_exact":before["input_tensor_sha256"] == live["input_tensor_sha256"],
        "decision_exact":before["decision"] == decision,
        "states_exact":all(engine.torch.equal(selected_state(payload["states"],layer,"block_output",positions)["vectors"],
            controller.states[(layer,"block_output")]["vectors"]) for layer in replay_layers)}
    result = {"eval_id":row["eval_id"],"decision":decision,"standard_parity":parity,
        "capture_noop_parity":api.logits_parity(plain,logits,exact=True),"hook_audit":audit,
        "input_tensor_sha256":live["input_tensor_sha256"],"input_metadata":live["input_metadata"],
        "archived_input_parity":live["archived_input_parity"],"capture_reused":reused,"source_replay":replay,
        "source_baseline_record_sha256":before["record_sha256"],"exact_replay_layers":list(replay_layers)}
    if baseline_failures(row,result) or not result["capture_noop_parity"]["passed"] or not all(replay.values()):
        raise RuntimeError("Depth baseline or exact prefix-state replay failed; no case may be skipped.")
    if reused:
        result.update({name:before[name] for name in ("capture_index_path","capture_index_sha256")})
    else:
        path = root/"captures"/row["phase3c_pair_id"]/row["condition"]/"states.pt"
        index = path.with_name("index.json")
        if path.exists() or index.exists():
            raise ValueError("Uncheckpointed extra capture exists; preserve and diagnose.")
        states = {key:value for key,value in controller.states.items() if key[0] in protocol.LAYERS}
        api.save_tensors(path,{"states":states,"logits":logits})
        frozen_write(index,{"schema":protocol.SCHEMA,"artifact_type":"real","eval_id":row["eval_id"],
            "plan_fingerprint":plan["fingerprint"],"input_tensor_sha256":live["input_tensor_sha256"],
            "positions":positions,"site_count":len(states),"hook_audit":audit,
            "saved_sites":[[layer,"block_output"] for layer in protocol.LAYERS],
            "vectors_path":str(path.resolve()),"vectors_sha256":file_hash(path)})
        result.update(capture_index_path=str(index.resolve()),capture_index_sha256=file_hash(index))
    result["passed"] = True
    return result


api.capture = capture_extra
original_validate_row = api.validate_row


def validate_depth_row(row,spec,plan,pairs,mappings,baselines,prior):
    original_validate_row(row,spec,plan,pairs,mappings,baselines,prior)
    if spec["kind"] == "baseline":
        if (not all(row["source_replay"].values()) or row["source_baseline_record_sha256"] !=
            plan["source_baseline_record_sha256"][row["eval_id"]] or row["exact_replay_layers"] !=
            list(protocol.GRID if row["capture_reused"] else protocol.prefix.LAYERS)):
            raise ValueError("Depth baseline replay binding changed.")
        index = read_json(row["capture_index_path"])
        if index["site_count"] != (39 if row["capture_reused"] else len(protocol.LAYERS)):
            raise ValueError("Depth capture storage scope changed.")


api.validate_row = validate_depth_row


def saved_ko(args,context,complete=False,baselines=None):
    plan,pairs,mappings,_,_ = context
    baselines = baselines or api.baseline_lookup(args,context)
    originals = knockout.old_rows(plan["knockout"],pairs,mappings)
    specs = knockout.tasks(plan["knockout"],plan["fingerprint"])
    expected = {r["task_id"]:r for r in specs}
    rows = {}
    for path in sorted((Path(args.output_root)/"knockout/task_checkpoints").glob("*.json")):
        row = read_json(path)
        key = row.get("task_id")
        if (key not in expected or key in rows or path.stem != key or row.get("passed") is not True or
            row.get("record_sha256") != api.seal(row)["record_sha256"] or row["spec"] != expected[key] or
            row["plan_fingerprint"] != plan["fingerprint"]):
            raise ValueError("KO checkpoint altered, failed, foreign or duplicated.")
        value = row["paired_outcome"]
        if value["pair_id"] != row["spec"]["pair_id"] or any(not isinstance(value[metric],(int,float)) or
                not math.isfinite(value[metric]) for metric in ("delta_M_low","delta_M_temporal","compression")):
            raise ValueError("Invalid paired KO outcome.")
        if not math.isclose(value["compression"],value["delta_M_low"]-value["delta_M_temporal"],abs_tol=1e-6):
            raise ValueError("KO boundary-advantage decomposition changed.")
        spec = row["spec"]
        if row["reused"]:
            source_key = (spec["pair_id"],spec["query_group"],spec["key_group"],spec["control"],spec["window"][0])
            original = originals.get(source_key)
            if (original != value or row["source_row_sha256"] != digest(original) or
                row["source_table_sha256"] != plan["knockout"]["original_ko_table_sha256"]):
                raise ValueError("Reused KO pair does not match its verified source.")
        else:
            for condition in CONDITIONS:
                actual = row["conditions"][condition]
                baseline = baselines[pairs[spec["pair_id"]][condition]["eval_id"]]
                queries,keys,budget = knockout.positions(mappings[spec["pair_id"]],condition,spec)
                audit = actual["mask_audit"]
                if (actual["query_positions"] != queries or actual["key_positions"] != keys or actual["edge_budget"] != budget or
                    actual["input_tensor_sha256"] != baseline["input_tensor_sha256"] or
                    actual["baseline_record_sha256"] != baseline["record_sha256"] or audit["enabled"] is not True or
                    audit["all_heads"] != 32 or audit["pre_softmax_mask"] != "negative_infinity" or
                    set(audit["layers"]) != {str(layer) for layer in spec["window"]} or
                    any(v["budget"] != budget or v["max_blocked_probability"] != 0 or v["max_row_sum_error"] > .005
                        for v in audit["layers"].values())):
                    raise ValueError("Saved KO mask/input budget does not match its planned intervention.")
            if knockout.paired_row(spec,row["conditions"],baselines,pairs) != value:
                raise ValueError("Saved KO outcome differs from its raw condition decisions.")
        rows[key] = row
    if complete and set(rows) != set(expected):
        raise ValueError("Incomplete 25-case KO grid.")
    return specs,rows


def run_ko(args,context):
    plan,pairs,mappings,_,_ = context
    baselines = api.baseline_lookup(args,context)
    output = Path(args.output_root)/"knockout"
    specs,saved = saved_ko(args,context,baselines=baselines)
    frozen_write(output/"task_manifest.jsonl",specs,jsonl=True)
    old = knockout.old_rows(plan["knockout"],pairs,mappings)
    if len(saved) == len(specs):
        return knockout.analyze(saved,output/"analysis",plan["fingerprint"])
    engine = api.make_engine(args,plan,"knockout")
    replay_path = output/"replay.json"
    replay = read_json(replay_path) if replay_path.exists() else {}
    live_cache = {}
    for pair_id in plan["knockout"]["case_ids"]:
        for condition in CONDITIONS:
            annotation = pairs[pair_id][condition]
            live = engine.prepare(annotation,mappings[pair_id])
            reference = baselines[annotation["eval_id"]]
            if live["input_tensor_sha256"] != reference["input_tensor_sha256"]:
                raise ValueError("KO live input bytes changed.")
            logits = engine.forward(live)
            payload = api.checked_payload(reference,engine.torch)
            if not api.logits_parity(payload["logits"],logits,exact=True)["passed"]:
                raise RuntimeError("KO baseline full-logit replay failed.")
            task = {"query_group":"options_all","key_group":"whole_event2","control":"whole_event_reference","window":[0,1,2,3]}
            control = knockout.condition_result(engine,live,mappings[pair_id],task,logits,False)
            replay[annotation["eval_id"]] = {"passed":True,"input_tensor_sha256":live["input_tensor_sha256"],
                "baseline_record_sha256":reference["record_sha256"],"noop_parity":control["noop_parity"],
                "plan_fingerprint":plan["fingerprint"]}
            atomic_write(replay_path,replay)
            print(f"KO replay/no-op {len(replay)}/50: {annotation['eval_id']}",flush=True)
            del live,payload
            gc.collect(); engine.torch.cuda.empty_cache()
    if len(replay) != 50:
        raise ValueError("All 50 KO conditions require live replay/disabled-hook parity.")
    current_pair,live_cache = None,{}
    live = None
    expected_new = sum(spec["task_id"] not in saved and
        (spec["pair_id"],spec["query_group"],spec["key_group"],spec["control"],spec["window"][0]) not in old for spec in specs)
    started,computed = time.perf_counter(),0
    for spec in specs:
        key = spec["task_id"]
        if key in saved:
            continue
        source_key = (spec["pair_id"],spec["query_group"],spec["key_group"],spec["control"],spec["window"][0])
        source = old.get(source_key)
        result = {"task_id":key,"spec":spec,"plan_fingerprint":plan["fingerprint"],"passed":False,"reused":source is not None,
            "ko_cohort_group":"original_pilot" if spec["pair_id"] in plan["knockout"]["retained_pilot_ids"] else "added_17"}
        try:
            if source:
                for condition,label in zip(CONDITIONS,("low","temporal")):
                    before = baselines[pairs[spec["pair_id"]][condition]["eval_id"]]["decision"]
                    if any(source[f"{label}_base_{name}"] != before[name] for name in
                           ("margin","correct_logit","incorrect_logit","prediction")):
                        raise ValueError("Reused KO baseline differs from the live exact-replayed baseline.")
                result.update(paired_outcome=source,source_row_sha256=digest(source),
                    source_table_sha256=plan["knockout"]["original_ko_table_sha256"])
            else:
                if current_pair != spec["pair_id"]:
                    live_cache = {}; gc.collect(); engine.torch.cuda.empty_cache()
                    live_cache = {condition:engine.prepare(pairs[spec["pair_id"]][condition],mappings[spec["pair_id"]]) for condition in CONDITIONS}
                    current_pair = spec["pair_id"]
                conditions = {}
                for condition in CONDITIONS:
                    live = live_cache[condition]
                    before = baselines[live["row"]["eval_id"]]
                    if before["input_tensor_sha256"] != live["input_tensor_sha256"]:
                        raise ValueError("Intervened input changed after KO replay.")
                    conditions[condition] = knockout.condition_result(engine,live,mappings[spec["pair_id"]],spec,None,True)
                    conditions[condition]["baseline_record_sha256"] = before["record_sha256"]
                    conditions[condition]["input_tensor_sha256"] = live["input_tensor_sha256"]
                result.update(conditions=conditions,paired_outcome=knockout.paired_row(spec,conditions,baselines,pairs))
                computed += 1
            result["passed"] = True
        except Exception as exc:
            result.update(error_type=type(exc).__name__,error_message=str(exc),traceback=traceback.format_exc())
            print(result["traceback"],flush=True)
        result = api.seal(result)
        frozen_write(output/"task_checkpoints"/f"{key}.json",result)
        saved[key] = result
        elapsed = time.perf_counter()-started
        atomic_write(output/"progress_status.json",{"completed_pairs":len(saved),"expected_pairs":len(specs),
            "newly_computed_pairs":computed,"reused_pairs":sum(r["reused"] for r in saved.values()),"elapsed_sec":elapsed,
            "eta_sec":elapsed/computed*(expected_new-computed) if computed else None,
            "failed_pairs":sum(not r["passed"] for r in saved.values())})
        if not source or len(saved)%100 == 0:
            print(f"KO25 {len(saved)}/{len(specs)} paired tasks; new={computed}; elapsed={elapsed/60:.1f} min; checkpoint={output/'task_checkpoints'}",flush=True)
        if not result["passed"]:
            raise RuntimeError("KO failure preserved; do not silently drop/replace a case.")
    live_cache = {}; live = None; del engine
    gc.collect()
    _,saved = saved_ko(args,context,True,baselines)
    atomic_write(output/"rows.jsonl",[saved[spec["task_id"]] for spec in specs],jsonl=True)
    return knockout.analyze(saved,output/"analysis",plan["fingerprint"])


def analyze_depth(args,context):
    plan,pairs,_,_,_ = context
    baselines = api.baseline_lookup(args,context)
    api.saved_rows(args.output_root,"identity",context,baselines,True)
    _,new = api.saved_rows(args.output_root,"patch",context,baselines,True)
    table = api.shared.patch_table(new,pairs)
    original = read_json(Path(args.prefix_run)/"analysis/case_patch_effects.json")
    for row in table:
        row["cohort_group"] = plan["cohort_groups"][row["pair_id"]]
        row["strict_incorrect_to_correct"] = (row["baseline_margin"] < 0 < row["patched_margin"] and
            row["baseline_prediction"] != row["correct_option"] and row["patched_prediction"] == row["correct_option"])
        row["strict_correct_to_incorrect"] = (row["baseline_margin"] > 0 > row["patched_margin"] and
            row["baseline_prediction"] == row["correct_option"] and row["patched_prediction"] in ("A","B") and
            row["patched_prediction"] != row["correct_option"])
    table = original+table
    expected = {(key,direction,layer) for key in plan["case_ids"] for direction in ("temporal_to_low","low_to_temporal") for layer in protocol.GRID}
    actual = [(row["pair_id"],row["direction"],row["layer"]) for row in table]
    if len(actual) != 900 or set(actual) != expected:
        raise ValueError("Missing/duplicate depth location; no completed depth analysis.")
    output = Path(args.output_root)/"analysis"
    metrics = ("margin_delta","strict_incorrect_to_correct","strict_correct_to_incorrect","recipient_coverage","non_ab_first_token","zero_margin_tie")
    summary = api.shared.aggregate(table,("direction","layer"),metrics)
    groups = api.shared.aggregate(table,("cohort_group","direction","layer"),metrics)
    for name,values in (("case_patch_effects",table),("depth_summary",summary),("cohort_summary",groups)):
        atomic_write(output/f"{name}.json",values); api.shared.write_csv(output/f"{name}.csv",values)
    plot(table,output/"plots")
    result = {"artifact_type":"real","complete":True,"plan_fingerprint":plan["fingerprint"],"case_count":50,
        "patch_rows":900,"reused_prefix_rows":200,"extension_rows":700,"layers":list(protocol.GRID),"primary_layer":16,
        "layer0_before_deepstack":True,"new_capture_count":sum(not r["capture_reused"] for r in baselines.values()),
        "output_sha256":{str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file() and p.name != "aggregate_summary.json"}}
    atomic_write(output/"aggregate_summary.json",result)
    return result


def verify(args):
    context = protocol.load(args)
    baselines = api.baseline_lookup(args,context)
    for stage in ("identity","patch"):
        api.saved_rows(args.output_root,stage,context,baselines,True)
    specs,rows = saved_ko(args,context,True,baselines)
    for directory in ("analysis","knockout/analysis"):
        root = Path(args.output_root)/directory
        value = read_json(root/"aggregate_summary.json")
        if value.get("complete") is not True:
            raise ValueError("Follow-up analysis incomplete.")
        field = "plan_fingerprint" if directory == "analysis" else "fingerprint"
        if value.get(field) != context[0]["fingerprint"] or value.get("case_count") != (50 if directory == "analysis" else 25):
            raise ValueError("Follow-up summary cohort/fingerprint changed.")
        if directory == "analysis" and (value.get("patch_rows") != 900 or value.get("layers") != list(protocol.GRID)):
            raise ValueError("Depth summary does not cover the full nine-layer grid.")
        if directory == "knockout/analysis" and value.get("paired_rows") != len(specs):
            raise ValueError("KO summary does not cover the full paired grid.")
        for name,sha in value["output_sha256"].items():
            path = (root/name).resolve()
            if not path.is_relative_to(root) or file_hash(path) != sha:
                raise ValueError("Follow-up analysis checksum changed.")
    if len(rows) != len(specs) or len(baselines) != 100:
        raise ValueError("Follow-up task counts differ.")
    return context


def source_active(args):
    result = subprocess.run(["ps","-eo","pid=,args="],capture_output=True,text=True,check=True,timeout=30)
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2:continue
        pid,command = fields
        try:
            executable = os.readlink(f"/proc/{pid}/exe")
        except OSError:
            executable = None
        if executable == shutil.which("tmux"):
            continue
        try:
            parts = shlex.split(command)
        except ValueError:
            continue
        if "--output_root" in parts and parts[parts.index("--output_root")+1] == args.prefix_run:
            return True
    return False


def wait_source(args,save):
    started = time.perf_counter()
    path = Path(args.prefix_run+"_lifecycle")/"job_status.json"
    while True:
        status = read_json(path) if path.exists() else {}
        active = source_active(args)
        save("waiting_for_verified_prefix",source_state=status.get("state"),source_active=active)
        if not active:
            if status.get("experiment_success") is not True:
                raise RuntimeError("Prefix did not complete successfully; continuation blocked.")
            if status.get("backup_verified") is not True:
                # The immutable prefix wrapper can fail on its nested backup lock.
                # Recover packaging only after workers/locks end and scientific ledgers verify.
                api.assert_idle()
                source_context = protocol.prefix.load(args.prefix_run)
                prefix_runner.verify_analysis(SimpleNamespace(output_root=args.prefix_run),source_context)
                save("recovering_verified_prefix_backup",source_state=status.get("state"))
                bundle = api.shared.create_backup(args.prefix_run,args.backup_dir,bundle_prefix="phase3c_backup",
                    file_provider=prefix_runner.backup_files,metadata={"schema":"phase3c_scale50_backup_v1",
                        "available_activation_file_count":len(list((Path(args.prefix_run)/"captures").rglob('*.pt'))),
                        "selected_videos_included":True,"old_activation_stores_included":False,"recovered_by_depth_handoff":True})
                manifest = api.shared.verify_backup(bundle)
                if not manifest["complete"] or manifest["reports_only"] or manifest["available_activation_file_count"] != 68:
                    raise RuntimeError("Recovered prefix backup is incomplete; continuation blocked.")
                status = {"original_lifecycle_status":status,"experiment_success":True,"backup_verified":True,
                          "backup_dir":str(bundle),"packaging_recovered_after_exit":True}
                atomic_write(Path(args.output_root)/"prefix_backup_handoff.json",status)
            return status
        if time.perf_counter()-started > 3*3600:
            raise RuntimeError("Prefix wait exceeded three hours; inspect status before retrying.")
        time.sleep(30)


def storage_gate(args):
    value = protocol.storage_estimate()
    free = shutil.disk_usage(args.storage_root).free
    atomic_write(Path(args.output_root)/"storage_preflight.json",{**value,"observed_free_bytes":free})
    if free < value["required_free_bytes"]:
        raise RuntimeError("Insufficient persistent storage for depth/KO captures, full backup and reserve; nothing deleted.")


def backup_files(root,reports_only):
    from scripts.backup_phase3b import backup_files as collect
    files = collect(root,reports_only)
    intent = read_json(Path(root)/"pending_plan.json")
    source = Path(intent["prefix_root"])
    for name,path in collect(source,reports_only).items():
        files["prefix/"+name] = path
    if not reports_only:
        mapping = read_json(source/"mappings.json")
        videos = {record["video_path"]:record["video_sha256"] for item in mapping.values() for record in item["processor_records"].values()}
        for index,(name,sha) in enumerate(sorted(videos.items())):
            path = Path(name)
            if path.is_symlink() or file_hash(path) != sha:
                raise ValueError("Selected video changed; full backup blocked.")
            files[f"source_videos/{index:03d}_{path.name}"] = path
    return files


def command(args,action,stage=None):
    result = [sys.executable,"-u",str(Path(__file__).resolve()),"--action",action,"--stage",stage or args.stage]
    for name in ("prefix_run","source_run","pilot_plan","pilot_run","replication_run","output_root","project_root","storage_root","gpus","backup_dir"):
        result += [f"--{name}",getattr(args,name)]
    if args.roi_only:
        result.append("--roi_only")
    if getattr(args,"attempt_id",None):
        result += ["--attempt_id",args.attempt_id]
    if action == "job":
        if args.email_notify:result += ["--email_notify","--email_config",args.email_config]
        if args.pause_after:result += ["--pause_after","--surf_config",args.surf_config,"--confirm_exclusive_workspace"]
    return result


def job(args):
    root,directory = Path(args.output_root),Path(args.output_root+"_lifecycle")
    started = time.perf_counter()
    notifier,client = None,None
    status = {"schema":protocol.SCHEMA,"started_at":datetime.now(timezone.utc).isoformat(),"backup_verified":False}
    def save(state,**values):
        status.update(state=state,elapsed_sec=time.perf_counter()-started,**values)
        atomic_write(directory/"job_status.json",status)
        print(f"Depth+KO25 lifecycle: {state}; elapsed={status['elapsed_sec']/60:.1f} min",flush=True)
    def notify():
        if notifier is None:return
        label = "SUCCESS" if status.get("experiment_success") else "FAILED" if "experiment_success" in status else "NOT STARTED"
        body = (f"Experiment: {label}\nEvery-four-layer whole-Event-2 patch + 25-rescue attention knockout\nRun: {root}\n"
            f"State: {status['state']}\nElapsed including queue: {status['elapsed_sec']/3600:.2f} hours\n"
            "Depth: 50 bases x 9 locations x 2 directions = 900 rows (200 prefix + 700 extension).\n"
            "KO: 25 fixed rescue bases; original ROI/background grid + options-to-whole-event route.\n"
            f"Verified full available-evidence backup: {status['backup_verified']}\nBackup: {status.get('backup_dir','not available')}\n"
            "Download all parts/manifest/SHA256SUMS and verify locally. Retain previous all-layer backups.\n"
            "Pause/billing stop is not confirmed; a request follows this email only after verified backup and idle checks.\n"
            "Technical status only; no credentials/tensors attached; not a mechanistic conclusion.")
        try:status["email"] = notifier.send(f"[Phase 3C Depth50 + KO25] {label}",body)
        except Exception as exc:status["email"] = {"error_type":type(exc).__name__}
        atomic_write(directory/"job_status.json",status)
    with api.shared.lifecycle_lock(directory):
        save("prechecking")
        try:
            notifier,client = api.services(args)
            protocol.pending(args); storage_gate(args)
            status["prefix_completion"] = wait_source(args,save)
            api.assert_idle()
            verify_pilot(args.pilot_plan,args.pilot_run)
            protocol.freeze_complete(args)
            args.attempt_id = uuid.uuid4().hex
            code = 0
            for stage in ("baseline","patch","knockout"):
                save("experiment_running",stage=stage)
                code = api.shared.run_child(args,command(args,"run",stage),f"depth25_{stage}")
                if code:break
            success = False
            if code == 0:
                try:
                    verify(args)
                    attempt = read_json(root/"last_attempt.json")
                    success = attempt["attempt_id"] == args.attempt_id and attempt["complete"] is True
                except Exception as exc:
                    status["verification_error_type"] = type(exc).__name__
            save("experiment_finished",experiment_success=success,experiment_return_code=code)
            with api.shared.run_lock(root):
                atomic_write(root/"unattended_status.json",status)
                shutil.copyfile(directory/"job.log",root/"unattended_job.log")
            save("backup_running")
            bundle = api.shared.create_backup(root,args.backup_dir,bundle_prefix="phase3c_backup",file_provider=backup_files,
                metadata={"schema":"phase3c_depth50_ko25_backup_v1","previous_large_all_layer_stores_included":False,
                    "new_extra_capture_count":len(list((root/"captures").rglob('*.pt'))),
                    "prefix_compact_capture_count":len(list((Path(args.prefix_run)/"captures").rglob('*.pt')))})
            manifest = api.shared.verify_backup(bundle)
            if manifest["reports_only"] or not manifest["complete"] or success and (
                manifest["new_extra_capture_count"] != 68 or manifest["prefix_compact_capture_count"] != 68):
                raise RuntimeError("Complete depth-capture backup coverage failed.")
            save("backup_verified",backup_verified=True,backup_dir=str(bundle))
            if args.pause_after:
                with api.shared.run_lock(root):
                    api.assert_idle(); save("pause_request_pending"); notify(); save("pause_requested",pause_result=client.request_pause())
            else:save("done_without_pause"); notify()
            return 0 if success else 1
        except Exception as exc:
            print(traceback.format_exc(),flush=True)
            save("needs_attention",error_type=type(exc).__name__,pause_not_confirmed=True)
            notify()
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action",required=True,choices=("prepare","launch","job","run","verify"))
    parser.add_argument("--stage",choices=("baseline","patch","knockout"),default="patch")
    for name in ("prefix_run","source_run","pilot_plan","pilot_run","replication_run","output_root"):
        parser.add_argument(f"--{name}",required=True)
    parser.add_argument("--project_root",default=str(ROOT))
    parser.add_argument("--storage_root",default="/data/yuxuanstorage")
    parser.add_argument("--backup_dir",default="/data/yuxuanstorage/backups")
    parser.add_argument("--gpus",default="0,1")
    parser.add_argument("--email_notify",action="store_true")
    parser.add_argument("--email_config",default=api.shared.EMAIL_CONFIG)
    parser.add_argument("--pause_after",action="store_true")
    parser.add_argument("--surf_config",default=api.shared.SURF_CONFIG)
    parser.add_argument("--confirm_exclusive_workspace",action="store_true")
    parser.add_argument("--roi_only",action="store_true")
    parser.add_argument("--attempt_id",help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("prefix_run","source_run","pilot_plan","pilot_run","replication_run","output_root","project_root","storage_root","backup_dir"):
        setattr(args,name,str(Path(getattr(args,name)).resolve()))
    if not Path(args.output_root).is_relative_to(args.storage_root) or Path(args.output_root) == Path(args.storage_root):
        parser.error("Use a separate persistent run directory.")
    if not Path(args.backup_dir).is_relative_to(args.storage_root) or Path(args.backup_dir).is_relative_to(args.output_root):
        parser.error("Use separate persistent backups.")
    root = Path(args.output_root)
    if args.action == "job":sys.exit(job(args))
    if args.action == "launch":
        protocol.pending(args); storage_gate(args); api.services(args)
        directory = Path(str(root)+"_lifecycle")
        if not shutil.which("tmux") or subprocess.run(["tmux","has-session","-t","=phase3c_depth_ko25"],capture_output=True).returncode == 0:
            parser.error("Detached continuation session already exists or tmux is unavailable.")
        with api.shared.lifecycle_lock(directory),api.shared.run_lock(root):
            shell = f"exec {shlex.join(command(args,'job'))} >> {shlex.quote(str(directory/'job.log'))} 2>&1"
            subprocess.run(["tmux","new-session","-d","-s","phase3c_depth_ko25","-c",args.project_root,shell],check=True)
        print(f"Queued depth50 + KO25; CPU-only wait; status={directory/'job_status.json'}",flush=True)
        return
    with api.shared.run_lock(root):
        if args.action == "prepare":
            value = protocol.pending(args,True); storage_gate(args); print(json.dumps(value,indent=2)); return
        if args.action == "verify":verify(args); print("Complete 900-row depth grid and 25-case KO grid verified."); return
        context = protocol.load(args)
        if args.stage == "baseline":api.run_stage(args,context,"baseline")
        elif args.stage == "patch":
            engine = api.run_stage(args,context,"identity")
            engine = api.run_stage(args,context,"patch",engine)
            del engine; gc.collect(); analyze_depth(args,context)
        else:
            run_ko(args,context); verify(args)
            atomic_write(root/"last_attempt.json",{"attempt_id":args.attempt_id or uuid.uuid4().hex,"complete":True})


if __name__ == "__main__":main()
