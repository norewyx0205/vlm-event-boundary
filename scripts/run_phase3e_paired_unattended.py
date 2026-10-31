"""Explicit tmux lifecycle for paired-option capture, failure mail and verified Pause."""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_phase3e_paired_capture as capture
from scripts.backup_phase3b import create_backup, verify_backup
from scripts.phase3c_core import atomic_write, digest, file_hash, read_json
from scripts.run_phase3c_vm import run_child
from scripts.run_phase3d_vm import lock

SCHEMA = "phase3e_paired_option_lifecycle_v1"
BACKUP_SCHEMA = "phase3e_paired_option_job_backup_v1"


def lifecycle_dir(args):
    root = Path(args.output_root)
    return root.parent / (root.name+"_lifecycle")


def validate_args(args):
    for name in ("output_root", "storage_root", "backup_dir", "project_root", "source_plan"):
        setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    root, storage, backup = (Path(getattr(args, name)) for name in ("output_root", "storage_root", "backup_dir"))
    capture.persistent_volume(root, storage)
    if (backup == storage or not backup.is_relative_to(storage) or root == backup or
            root in backup.parents or backup in root.parents or
            backup == lifecycle_dir(args) or backup in lifecycle_dir(args).parents or lifecycle_dir(args) in backup.parents):
        raise ValueError("Use separate sibling run, lifecycle and backup directories on persistent storage.")
    if not Path(args.project_root, "scripts/run_phase3e_paired_capture.py").is_file():
        raise ValueError("Project root must contain the paired-capture runner.")
    if args.pause_after and not args.confirm_exclusive_workspace:
        raise ValueError("Whole-VM Pause requires exclusive-workspace confirmation.")
    if (not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.session_name) or
            not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", args.gpus) or
            len(set(args.gpus.split(","))) != len(args.gpus.split(","))):
        raise ValueError("Use a simple session name and distinct numeric GPU indices.")
    if args.stage not in ("preflight", "full") or not re.fullmatch(r"[0-9a-f]{32}", args.attempt_id):
        raise ValueError("Invalid lifecycle stage/attempt identity.")
    return args


def stages(stage):
    if stage == "preflight":
        return ("preflight",)
    if stage == "full":
        return ("capture", "export", "verify")
    raise ValueError("Only an explicit preflight or full stage can launch.")


def command(args, stage):
    return [sys.executable, "-u", str(Path(args.project_root) / "scripts/run_phase3e_paired_capture.py"),
        "--stage", stage, "--output_root", args.output_root, "--storage_root", args.storage_root,
        "--source_plan", args.source_plan]


def assert_idle():
    from scripts.run_phase3d_single_layer_ko import assert_idle as previous_guard
    previous_guard()
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True, timeout=30)
    if gpu.stdout.strip():
        raise RuntimeError("GPU compute remains active; launch/Pause blocked.")
    processes = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True, timeout=30)
    pattern = re.compile(r"(?:^|[ /])(?:scripts\.)?(?:run_phase3e_paired_capture|run_phase3e_paired_unattended|"
                         r"run_phase3e_preflight|run_phase3e_option_probes|prepare_phase3e_paired_design|"
                         r"audit_phase3e_interactions)(?:\.py)?(?:\s|$)")
    for line in processes.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or int(parts[0]) == os.getpid() or not pattern.search(parts[1]):
            continue
        # Launchers never execute a model; the shared workspace lock serializes their jobs.
        if "--action launch" in parts[1]:
            continue
        try:
            executable = Path(os.readlink(f"/proc/{parts[0]}/exe")).name
        except OSError:
            raise RuntimeError("Cannot establish workspace quiescence; launch/Pause blocked.") from None
        if executable != "tmux":
            raise RuntimeError("Another Phase 3E process is active; launch/Pause blocked.")


def forbidden(args):
    return (args.project_root, args.output_root, args.backup_dir, str(lifecycle_dir(args)), args.source_plan)


def make_notifier(args):
    if not args.email_notify:
        return None
    from scripts.phase3b_email import EmailNotifier, load_private_config
    notifier = EmailNotifier(load_private_config(args.email_config, forbidden(args)))
    notifier.check()
    return notifier


def make_client(args):
    if not args.pause_after:
        return None
    if not args.confirm_exclusive_workspace:
        raise ValueError("Automatic Pause requires exclusive-workspace confirmation.")
    from scripts.surf_workspace import SurfClient, load_private_config
    client = SurfClient(load_private_config(args.surf_config, forbidden(args)))
    client.check(require_pause=True)
    return client


def precheck(args):
    plan = capture.load_plan(args.output_root)
    config = capture.config_for(args.output_root, plan, args.source_plan)
    capture.space(args.output_root)
    if args.stage == "full":
        proof = capture.verify(args.output_root, preflight=True)
        if read_json(Path(args.output_root) / "preflight_summary.json") != proof:
            raise ValueError("Full launch requires the same passed two-family compact preflight.")
        projection = capture.storage_projection(args.output_root)
        capture.space(args.output_root, projection["required_free_bytes"])
    return plan, config


def verify_stage(args):
    result = capture.verify(args.output_root, preflight=args.stage == "preflight")
    name = "preflight_summary.json" if args.stage == "preflight" else "capture_summary.json"
    if read_json(Path(args.output_root) / name) != result:
        raise ValueError("Stage summary differs from independent raw checkpoint verification.")
    if args.stage == "full":
        from scripts.phase3e_option_store import load_store
        manifest, _, _, _, _ = load_store(Path(args.output_root) / "feature_store/manifest.json")
        if manifest["plan_fingerprint"] != result["plan_fingerprint"]:
            raise ValueError("Exported store belongs to another capture plan.")
    return result


def snapshot_code(args):
    root = Path(args.output_root) / "code_snapshot"
    sources = [(path, Path("scripts") / path.name) for path in sorted(Path(args.project_root, "scripts").glob("*.py"))]
    sources += [(Path(args.project_root) / "docs/phase3e_b_paired_option_protocol.md",
                 Path("docs/phase3e_b_paired_option_protocol.md"))]
    for source, relative in sources:
        destination = root / relative
        if source.is_symlink() or destination.is_symlink():
            raise ValueError("Code snapshots refuse symlinks.")
        if destination.exists():
            if file_hash(destination) != file_hash(source):
                raise ValueError("Saved code snapshot differs; use a new run root.")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if file_hash(source) != file_hash(destination):
            raise ValueError("Code changed during snapshot.")


def completion_message(status, args):
    label = ("SUCCESS" if status.get("experiment_success") else
             "FAILED" if "experiment_success" in status else "NOT STARTED")
    body = (f"Experiment: {label}\nStage: {args.stage}\nLifecycle state: {status['state']}\n"
        f"Run: {args.output_root}\nLog: {lifecycle_dir(args)/'job.log'}\n"
        f"Status: {lifecycle_dir(args)/'job_status.json'}\n"
        f"Elapsed: {status['elapsed_sec']/3600:.2f} hours\n"
        f"Verified VM backup: {status['backup_verified']}\nBackup: {status.get('backup_dir', 'not available')}\n"
        f"Full 40-family capture complete: {status.get('full_experiment_complete', False)}\n"
        "Scope: normal-forward option capture only. No new KO/patch or probe fitting.\n"
        "Preflight = 2 families / 32 prompts; full = 40 families / 640 prompts.\n"
        "A verified backup of a failed run preserves available partial artifacts, not missing results.\n"
        "Off-VM backup is NOT confirmed. Download all parts, manifest and SHA256SUMS and verify.\n"
        "Pause is requested only after a verified full snapshot and a quiescence check.\n"
        "Pause/billing stop is NOT confirmed by this email; check the portal.\n")
    if status.get("error_type"):
        body += f"Lifecycle error type: {status['error_type']}; inspect public logs.\n"
    return f"[Phase 3E-B] {args.stage} {label}", body


def job(args, child=run_child, pack=create_backup, verify=verify_backup, idle=assert_idle,
        notifier_factory=make_notifier, client_factory=make_client):
    root, directory = Path(args.output_root), lifecycle_dir(args)
    started = time.perf_counter()
    status = {"schema": SCHEMA, "attempt_id": args.attempt_id, "stage": args.stage,
        "started_at": datetime.now(timezone.utc).isoformat(), "backup_verified": False,
        "local_backup_confirmed": False, "pause_not_confirmed": True, "output_root": str(root)}
    notifier = client = None

    def save(state, **fields):
        status.update(state=state, elapsed_sec=time.perf_counter()-started, **fields)
        atomic_write(directory / "job_status.json", status)
        print(f"[Phase 3E-B lifecycle] {state}", flush=True)

    def notify():
        if notifier is not None:
            try:
                status["email"] = notifier.send(*completion_message(status, args))
            except Exception as exc:
                status["email"] = {"error_type": type(exc).__name__}
            atomic_write(directory / "job_status.json", status)

    with lock(args.storage_root, ".phase3e_workspace.lock"), lock(directory, ".lifecycle.lock"):
        save("prechecking")
        try:
            notifier = notifier_factory(args)
            client = client_factory(args)
            with lock(root, ".phase3e_option.lock"):
                plan, config = precheck(args)
                idle()
                snapshot_code(args)
            save("experiment_running", plan_fingerprint=plan["fingerprint"], config_sha256=digest(config))
            code = 0
            for stage in stages(args.stage):
                save("experiment_running", active_stage=stage)
                try:
                    code = child(args, command(args, stage), "phase3e_"+stage)
                except Exception as exc:
                    code = 1
                    status["child_error_type"] = type(exc).__name__
                if code != 0:
                    break
            success = code == 0
            if success:
                try:
                    with lock(root, ".phase3e_option.lock"):
                        proof = verify_stage(args)
                        if proof["plan_fingerprint"] != plan["fingerprint"]:
                            raise ValueError("Completed stage belongs to another plan.")
                except Exception as exc:
                    success = False
                    status["verification_error_type"] = type(exc).__name__
            save("experiment_finished", experiment_success=success, experiment_return_code=code,
                 full_experiment_complete=success and args.stage == "full")
            with lock(root, ".phase3e_option.lock"):
                idle()
                if not success:
                    atomic_write(root / "failure.json", {"attempt_id": args.attempt_id, "stage": args.stage,
                        "exit_code": code, "child_error_type": status.get("child_error_type"),
                        "verification_error_type": status.get("verification_error_type")})
                atomic_write(root / "unattended_status.json", status)
                if (directory / "job.log").exists():
                    shutil.copyfile(directory / "job.log", root / "unattended_job.log")
                files = capture.public_files(root)
                capture.space(root, capture.RESERVE+int(1.1*sum(p.stat().st_size for p in files.values())))
                metadata = {"schema": BACKUP_SCHEMA, "attempt_id": args.attempt_id, "requested_stage": args.stage,
                    "capture_plan_fingerprint": plan["fingerprint"], "stage_verified": success,
                    "full_experiment_complete": status["full_experiment_complete"],
                    "snapshot_scope": "all available public files; failed runs may contain partial captures"}
                save("backup_running")
                bundle = pack(root, args.backup_dir, file_provider=capture.public_files,
                    bundle_prefix="phase3c_backup", metadata=metadata)
                manifest = verify(bundle)
                if (manifest.get("schema") != BACKUP_SCHEMA or manifest.get("attempt_id") != args.attempt_id or
                        manifest.get("source_run_root") != str(root.resolve()) or manifest.get("complete") is not True or
                        manifest.get("reports_only") is not False or manifest.get("activation_tensors_included") is not True or
                        any(manifest.get(k) != v for k, v in metadata.items()) or
                        {r["path"] for r in manifest["files"]} != set(files)):
                    raise RuntimeError("Backup does not cover this attempt's full public snapshot.")
                save("backup_verified", backup_verified=True, backup_dir=str(bundle))
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
            save("needs_attention", error_type=type(exc).__name__)
            notify()
            return 1


def launch(args):
    if not shutil.which("tmux"):
        raise RuntimeError("tmux must be installed before detached launch.")
    if subprocess.run(["tmux", "has-session", "-t", "="+args.session_name], capture_output=True).returncode == 0:
        raise RuntimeError("tmux session already exists; attach instead of launching a duplicate.")
    with lock(args.storage_root, ".phase3e_workspace.lock"), lock(lifecycle_dir(args), ".lifecycle.lock"), \
            lock(args.output_root, ".phase3e_option.lock"):
        precheck(args)
        make_notifier(args)
        make_client(args)
        assert_idle()
        snapshot_code(args)
        invocation = [sys.executable, "-u", str(Path(__file__).resolve()), "--action", "job", "--stage", args.stage,
            "--output_root", args.output_root, "--storage_root", args.storage_root, "--project_root", args.project_root,
            "--backup_dir", args.backup_dir, "--source_plan", args.source_plan, "--gpus", args.gpus,
            "--attempt_id", args.attempt_id, "--session_name", args.session_name]
        if args.email_notify:
            invocation += ["--email_notify", "--email_config", args.email_config]
        if args.pause_after:
            invocation += ["--pause_after", "--surf_config", args.surf_config, "--confirm_exclusive_workspace"]
        atomic_write(lifecycle_dir(args) / "launch_plan.json", {"attempt_id": args.attempt_id, "command": invocation})
    shell = f"exec {shlex.join(invocation)} >> {shlex.quote(str(lifecycle_dir(args)/'job.log'))} 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", args.session_name, "-c", args.project_root, shell], check=True)
    return {"status": "detached_job_started", "stage": args.stage, "session_name": args.session_name,
            "log": str(lifecycle_dir(args) / "job.log")}


def main():
    from scripts.phase3b_email import DEFAULT_CONFIG as EMAIL
    from scripts.surf_workspace import DEFAULT_CONFIG as SURF
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("plan", "launch", "job"), default="plan")
    parser.add_argument("--stage", choices=("preflight", "full"), default="preflight")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--source_plan", required=True)
    parser.add_argument("--storage_root", default="/data/yuxuanstorage")
    parser.add_argument("--backup_dir", default="/data/yuxuanstorage/backups")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--session_name", default="phase3e-paired-options")
    parser.add_argument("--email_notify", action="store_true")
    parser.add_argument("--email_config", default=EMAIL)
    parser.add_argument("--pause_after", action="store_true")
    parser.add_argument("--surf_config", default=SURF)
    parser.add_argument("--confirm_exclusive_workspace", action="store_true")
    parser.add_argument("--attempt_id", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.attempt_id = args.attempt_id or uuid.uuid4().hex
    validate_args(args)
    if args.action == "plan":
        plan, config = precheck(args)
        print(json.dumps({"stage": args.stage, "child_stages": stages(args.stage), "GPU_started": False,
            "plan_fingerprint": plan["fingerprint"], "config_sha256": digest(config),
            "email_enabled": args.email_notify, "pause_enabled": args.pause_after}, indent=2))
    elif args.action == "launch":
        print(json.dumps(launch(args), indent=2))
    else:
        sys.exit(job(args))


if __name__ == "__main__":
    main()
