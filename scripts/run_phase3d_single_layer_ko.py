"""Run the isolated 400-forward singleton KO follow-up, never the old KO grid."""

import argparse
import gc
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import phase3d_execution as execution
from scripts import phase3d_single_layer_ko as followup
from scripts.phase3c_core import atomic_write, file_hash, read_json
from scripts.run_phase3d_vm import lock, make_engine, lifecycle_dir
from scripts.run_phase3c_vm import run_child


def storage_gate(args):
    storage, root, backup = (Path(getattr(args, name)) for name in ("storage_root", "output_root", "backup_dir"))
    if not storage.is_dir() or (sys.platform.startswith("linux") and not os.path.ismount(storage)):
        raise ValueError("Persistent volume must be mounted before preparation/run/backup.")
    if (root / "plan.json").exists():
        source = Path(read_json(root / "plan.json")["source_root"])
        if source == backup or source in backup.parents or backup in source.parents:
            raise ValueError("Keep backups separate from the frozen source.")
    present = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0
    # Raw baseline/no-op/KO logits, JSON proofs, reports, and an uncompressed-size ZIP allowance.
    remaining = max(0, 2*1024**3-present)
    videos = 0
    source_plan = root / "source_plan.json"
    if source_plan.exists():
        videos = sum(Path(p).stat().st_size for p in read_json(source_plan)["video_sha256"])
    required = remaining + int(1.1*(present+remaining+videos)) + 12*1024**3
    free = min(shutil.disk_usage(storage).free, shutil.disk_usage(backup if backup.exists() else storage).free)
    if free < required:
        raise RuntimeError(f"Insufficient persistent/backup space: {free/1024**3:.1f} GiB free, "
                           f"{required/1024**3:.1f} GiB required. No evidence deleted.")
    return {"free_bytes": free, "required_free_bytes": required, "reserve_bytes": 12*1024**3}


def require_preflight(args, context):
    plan = context[0]
    ids = plan["preflight_ids"]
    if len(ids) != 2 or {context[1][key]["low_boundary"]["first_object_id"] for key in ids} != {1, 2}:
        raise ValueError("Preflight requires one frozen rescue of each mover order.")
    if not (Path(args.output_root) / "preflight_summary.json").is_file():
        raise ValueError("Run and inspect --stage preflight first; its completion summary is missing.")
    rows = followup.verify_rows(args.output_root, context, ids)
    summary = read_json(Path(args.output_root) / "preflight_summary.json")
    if (summary.get("complete") is not True or summary.get("plan_fingerprint") != plan["fingerprint"] or
            summary.get("verified_record_hashes") != [r["record_sha256"] for r in rows] or
            summary.get("is_full_experiment") is not False):
        raise ValueError("Two-case singleton preflight is incomplete or stale.")


def run(args):
    with lock(args.output_root):
        context = followup.load_context(args.output_root)
        if args.stage == "verify":
            summary = followup.analyze(args.output_root, context)
        else:
            if args.stage == "full":
                require_preflight(args, context)
            storage_gate(args)
            ids = context[0]["preflight_ids"] if args.stage == "preflight" else None
            gates = followup.specs(context[0], "baseline_gate", ids)
            tasks = followup.specs(context[0], "knockout", ids)
            if any(execution.saved_checkpoint(args.output_root, context[0], s) is None for s in gates+tasks):
                engine = make_engine(args, context)
                execute = followup.Executor(args.output_root, context, engine)
                execution.run_tasks(args.output_root, context, gates, execute, args.retry_failed)
                followup.verify_rows(args.output_root, context, ids, include_knockouts=False)
                execution.run_tasks(args.output_root, context, tasks, execute, args.retry_failed)
                del execute, engine
                gc.collect()
                import torch
                torch.cuda.empty_cache()
            rows = followup.verify_rows(args.output_root, context, ids)
            if args.stage == "preflight":
                summary = {"complete": True, "is_full_experiment": False, "base_count": 2,
                    "plan_fingerprint": context[0]["fingerprint"], "new_condition_rows": 16,
                    "verified_record_hashes": [r["record_sha256"] for r in rows]}
                atomic_write(Path(args.output_root) / "preflight_summary.json", summary)
                require_preflight(args, context)
            else:
                summary = followup.analyze(args.output_root, context)
        atomic_write(Path(args.output_root) / "last_attempt.json", {"complete": True, "stage": args.stage,
            "attempt_id": args.attempt_id, "plan_fingerprint": context[0]["fingerprint"], "summary": summary})
        return summary


def backup_files(root, reports_only):
    root = Path(root).resolve()
    directories = {"baseline_gate", "knockout", "runtime_observations", "logs", "failures", "analysis"}
    names = {"plan.json", "source_plan.json", "source_runtime.json", "source_records.json", "source_logits.pt",
        "pairs.json", "mappings.json", "runtime.json", "preflight_summary.json", "last_attempt.json",
        "progress_status.json", "unattended_status.json", "unattended_job.log"}
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Backups refuse symlinks.")
        if not path.is_file() or path.name in {".phase3d.lock", ".pipeline.lock"}:
            continue
        relative = path.relative_to(root)
        if relative.parts[0] not in directories and relative.as_posix() not in names:
            raise ValueError("Unexpected artifact; keep private credentials outside research/backup roots.")
        if not reports_only or path.suffix != ".pt":
            files["run/"+relative.as_posix()] = path
    if not reports_only:
        for number, (name, sha) in enumerate(sorted(read_json(root / "source_plan.json")["video_sha256"].items())):
            path = Path(name)
            if path.is_symlink() or file_hash(path) != sha:
                raise ValueError("Selected source video changed; full backup/Pause blocked.")
            files[f"source_videos/{number:03d}_{path.name}"] = path
    return files


def assert_idle():
    from scripts.run_phase3d_vm import assert_idle as old_guard
    old_guard()
    result = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True, timeout=30)
    pattern = re.compile(r"(?:^|[ /])run_phase3d_single_layer_ko\.py(?:\s|$)")
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or int(parts[0]) == os.getpid() or not pattern.search(parts[1]):
            continue
        try:
            executable = Path(os.readlink(f"/proc/{parts[0]}/exe")).name
        except OSError:
            raise RuntimeError("Cannot prove workspace quiescence; launch/Pause blocked.") from None
        if executable != "tmux":
            raise RuntimeError("Another singleton KO process is active; launch/Pause blocked.")


def services(args):
    from scripts.surf_workspace import SurfClient, load_private_config, check_private_location
    from scripts.phase3b_email import EmailNotifier, load_private_config as email_config
    forbidden = [args.project_root, args.output_root, args.backup_dir, str(lifecycle_dir(args)),
                 read_json(Path(args.output_root) / "plan.json")["source_root"]]
    notifier = client = None
    if args.email_notify:
        check_private_location(args.email_config, forbidden)
        notifier = EmailNotifier(email_config(args.email_config, forbidden))
        notifier.check()
    if args.pause_after:
        if not args.confirm_exclusive_workspace:
            raise ValueError("Whole-VM Pause requires exclusive-workspace confirmation.")
        check_private_location(args.surf_config, forbidden)
        client = SurfClient(load_private_config(args.surf_config, forbidden))
        client.check(require_pause=True)
    return notifier, client


def child_command(args, action="run"):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--action", action,
        "--stage", args.stage, "--output_root", args.output_root, "--storage_root", args.storage_root,
        "--backup_dir", args.backup_dir, "--project_root", args.project_root,
        "--gpus", args.gpus, "--attempt_id", args.attempt_id]
    if args.retry_failed:
        command.append("--retry_failed")
    if action == "job":
        if args.email_notify:
            command += ["--email_notify", "--email_config", args.email_config]
        if args.pause_after:
            command += ["--pause_after", "--surf_config", args.surf_config, "--confirm_exclusive_workspace"]
    return command


def job(args, child=run_child, pack=None, verify=None, idle=assert_idle, service_factory=services):
    from scripts.backup_phase3b import create_backup, verify_backup
    pack, verify = pack or create_backup, verify or verify_backup
    root, directory = Path(args.output_root), lifecycle_dir(args)
    started, notifier = time.perf_counter(), None
    status = {"schema": followup.SCHEMA, "stage": args.stage, "attempt_id": args.attempt_id,
        "started_at": datetime.now(timezone.utc).isoformat(), "output_root": str(root),
        "backup_verified": False, "local_backup_confirmed": False, "pause_not_confirmed": True}
    def save(state, **fields):
        status.update(state=state, elapsed_sec=time.perf_counter()-started, **fields)
        atomic_write(directory / "job_status.json", status)
        print(f"[singleton KO lifecycle] {state}; elapsed={status['elapsed_sec']/60:.1f} min", flush=True)
    def notify():
        if notifier is None:
            return
        label = "SUCCESS" if status.get("experiment_success") else "FAILED" if "experiment_success" in status else "NOT STARTED"
        body = (f"Experiment: {label}\nStage: {args.stage}\nRun: {root}\nState: {status['state']}\n"
            f"Elapsed: {status['elapsed_sec']/3600:.2f} hours\nLog: {directory/'job.log'}\n"
            f"Status: {directory/'job_status.json'}\nVerified VM backup: {status['backup_verified']}\n"
            f"Backup: {status.get('backup_dir', 'not available')}\n"
            "Scope: same 50 bases; four singleton layers; 400 new interventional condition forwards at full completion.\n"
            "Preflight is NOT the full experiment. Historical four-layer reference is not rerun.\n"
            "Retain frozen Phase 3D backup separately. No full historical activation stores duplicated.\n"
            "Off-VM backup is NOT confirmed; download every part, manifest and SHA256SUMS and verify.\n"
            "Pause is requested only after verified full backup and idle check. Billing stop is NOT confirmed.\n"
            "Technical completion only; no scientific interpretation or credentials attached.")
        try:
            status["email"] = notifier.send(f"[Phase 3D singleton KO] {args.stage} {label}", body)
        except Exception as exc:
            status["email"] = {"error_type": type(exc).__name__}
        atomic_write(directory / "job_status.json", status)
    with lock(directory, ".lifecycle.lock"):
        save("prechecking")
        try:
            context = followup.load_context(root)
            notifier, client = service_factory(args)
            storage_gate(args)
            idle()
            if args.stage == "full":
                require_preflight(args, context)
            save("experiment_running")
            try:
                code = child(args, child_command(args), "singleton_ko_"+args.stage)
            except Exception as exc:
                code = 1
                status["child_error_type"] = type(exc).__name__
            last = read_json(root / "last_attempt.json") if (root / "last_attempt.json").exists() else {}
            success = (code == 0 and last.get("complete") is True and last.get("stage") == args.stage and
                last.get("attempt_id") == args.attempt_id and last.get("plan_fingerprint") == context[0]["fingerprint"])
            if success:
                try:
                    with lock(root):
                        followup.load_context(root)
                        if args.stage == "preflight":
                            require_preflight(args, context)
                        else:
                            followup.analyze(root, context)
                except Exception as exc:
                    success = False
                    status["verification_error_type"] = type(exc).__name__
            save("experiment_finished", experiment_success=success, experiment_return_code=code)
            with lock(root):
                atomic_write(root / "unattended_status.json", status)
                if (directory / "job.log").exists():
                    shutil.copyfile(directory / "job.log", root / "unattended_job.log")
                save("backup_running")
                bundle = pack(root, args.backup_dir, file_provider=backup_files, bundle_prefix="phase3c_backup",
                    metadata={"schema": followup.SCHEMA+"_backup", "plan_fingerprint": context[0]["fingerprint"],
                              "historical_activation_stores_included": False, "selected_videos_included": True,
                              "new_activation_capture": False, "tensor_payload_contents": "Full-vocabulary logits only."})
                manifest = verify(bundle)
            if (manifest.get("schema") != followup.SCHEMA+"_backup" or manifest.get("complete") is not True or
                    manifest.get("reports_only") is not False or manifest.get("activation_tensors_included") is not True):
                raise RuntimeError("Verified full backup is required before automatic Pause.")
            save("backup_verified", backup_verified=True, backup_dir=str(bundle))
            with lock(root):
                if args.pause_after:
                    idle()
                    save("pause_request_pending")
                    notify()
                    save("pause_requested", pause_result=client.request_pause())
                else:
                    save("done_without_pause")
                    notify()
            return 0 if success else max(1, code)
        except Exception as exc:
            traceback.print_exc()
            save("needs_attention", error_type=type(exc).__name__)
            notify()
            return 1


def main():
    from scripts.phase3b_email import DEFAULT_CONFIG as EMAIL_CONFIG
    from scripts.surf_workspace import DEFAULT_CONFIG as SURF_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "plan", "run", "launch", "job", "backup"), default="plan")
    parser.add_argument("--stage", choices=("preflight", "full", "verify"), default="preflight")
    parser.add_argument("--source_run")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--storage_root", default="/data/yuxuanstorage")
    parser.add_argument("--backup_dir", default="/data/yuxuanstorage/backups")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--retry_failed", action="store_true")
    parser.add_argument("--session_name", default="phase3d-single-layer")
    parser.add_argument("--email_notify", action="store_true")
    parser.add_argument("--email_config", default=EMAIL_CONFIG)
    parser.add_argument("--pause_after", action="store_true")
    parser.add_argument("--surf_config", default=SURF_CONFIG)
    parser.add_argument("--confirm_exclusive_workspace", action="store_true")
    parser.add_argument("--attempt_id", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("output_root", "storage_root", "backup_dir", "project_root"):
        setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    root, storage, backup = (Path(getattr(args, name)) for name in ("output_root", "storage_root", "backup_dir"))
    if (root == storage or backup == storage or not root.is_relative_to(storage) or not backup.is_relative_to(storage) or
            root == backup or root in backup.parents or backup in root.parents):
        parser.error("Use separate run/backup directories on the persistent mount.")
    if args.pause_after and not args.confirm_exclusive_workspace:
        parser.error("--pause_after requires --confirm_exclusive_workspace.")
    if (args.email_notify or args.pause_after) and args.action not in ("launch", "job", "plan"):
        parser.error("Email/Pause flags require the detached lifecycle, not a bare run.")
    if args.action in ("launch", "job") and args.stage == "verify":
        parser.error("Detached lifecycle is for preflight/full; verification is CPU-only.")
    args.attempt_id = args.attempt_id or uuid.uuid4().hex
    if args.action == "prepare":
        if not args.source_run:
            parser.error("--source_run must name the frozen complete Phase 3D run.")
        source = Path(args.source_run).expanduser().resolve()
        if source == root or source in root.parents or root in source.parents:
            parser.error("Source and follow-up must be separate; do not create a lock inside frozen evidence.")
        if source == backup or source in backup.parents or backup in source.parents:
            parser.error("Keep backups separate from the frozen source.")
        storage_gate(args)
        with lock(root):
            followup.freeze_plan(args.source_run, root)
        print("Singleton plan frozen. No model/GPU/SURF action started.")
    elif args.action == "run":
        run(args)
    elif args.action == "job":
        sys.exit(job(args))
    else:
        context = followup.load_context(root)
        source = Path(context[0]["source_root"])
        if source == backup or source in backup.parents or backup in source.parents:
            parser.error("Keep backups separate from the frozen source.")
        if args.stage == "full" and args.action != "backup":
            require_preflight(args, context)
        if args.action == "backup":
            from scripts.backup_phase3b import create_backup, verify_backup
            storage_gate(args)
            with lock(root):
                bundle = create_backup(root, backup, file_provider=backup_files, bundle_prefix="phase3c_backup",
                    metadata={"schema": followup.SCHEMA+"_backup", "plan_fingerprint": context[0]["fingerprint"],
                              "historical_activation_stores_included": False, "selected_videos_included": True,
                              "new_activation_capture": False, "tensor_payload_contents": "Full-vocabulary logits only."})
                verify_backup(bundle)
            print(f"Verified full VM backup: {bundle}; off-VM copy still required.")
        elif args.action == "plan":
            print(json.dumps({"plan": context[0]["fingerprint"], "base_count": 50,
                "new_condition_forwards": 400, "paired_singleton_rows": 200, "historical_joint_pairs": 50,
                "preflight_new_condition_forwards": 16, "technical_forwards_additional": True,
                "layers_zero_based": list(followup.LAYERS), "route": followup.ROUTE,
                "storage": storage_gate(args), "gpu_started": False}, indent=2))
        else:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.session_name) or not shutil.which("tmux"):
                parser.error("Install tmux and use a simple unique session name.")
            if subprocess.run(["tmux", "has-session", "-t", "="+args.session_name], capture_output=True).returncode == 0:
                parser.error("tmux session exists; attach instead of launching a duplicate.")
            with lock(lifecycle_dir(args), ".lifecycle.lock"), lock(root):
                storage_gate(args)
                services(args)
                assert_idle()
                command = child_command(args, "job")
                atomic_write(lifecycle_dir(args) / "launch_plan.json", {"command": command, "attempt_id": args.attempt_id})
            shell = f"exec {shlex.join(command)} >> {shlex.quote(str(lifecycle_dir(args)/'job.log'))} 2>&1"
            subprocess.run(["tmux", "new-session", "-d", "-s", args.session_name, "-c", args.project_root, shell], check=True)
            print(f"Detached job started: {args.session_name}; log={lifecycle_dir(args)/'job.log'}")


if __name__ == "__main__":
    main()
