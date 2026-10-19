"""Explicit Phase 3D CPU audit, GPU gates, KO50, probes and detached lifecycle."""

import argparse
import fcntl
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
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import phase3d_execution as execution
from scripts.phase3c_core import atomic_write, digest, file_hash, frozen_write, read_json

STAGES = ("audit", "preflight", "capture", "patch", "export", "knockout", "probes", "verify", "full")


@contextmanager
def lock(root, name=".phase3d.lock"):
    Path(root).mkdir(parents=True, exist_ok=True)
    with (Path(root) / name).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another Phase 3D job owns this root; do not launch a duplicate.") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def storage_gate(args, context):
    root = Path(args.output_root)
    if not Path(args.storage_root).is_dir():
        raise ValueError("Persistent storage is not mounted.")
    if sys.platform.startswith("linux") and not os.path.ismount(args.storage_root):
        raise ValueError("Persistent --storage_root is not a mount point; do not write results to an unmounted directory.")
    # Compact features + identity evidence + export, CPU probe products and a full ZIP copy.
    hidden = 4096
    features = 300 * len(execution.FEATURE_NAMES) * hidden * 4
    captures = features + 100 * (512 * hidden * 2 + len(execution.FEATURE_NAMES) * hidden * 4)
    videos = sum(Path(path).stat().st_size for path in context[0]["video_sha256"])
    present = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0
    remaining = max(0, int(1.3 * (captures + features + 2 * 1024**3)) - present)
    required = remaining + int((present + remaining + videos) * 1.1) + 12 * 1024**3
    free = shutil.disk_usage(args.storage_root).free
    backup_free = shutil.disk_usage(Path(args.backup_dir) if Path(args.backup_dir).exists() else args.storage_root).free
    if free < required or backup_free < required:
        raise RuntimeError(f"Persistent storage/backup space insufficient: free={free/1024**3:.1f} GiB, "
                           f"conservative requirement={required/1024**3:.1f} GiB. Nothing deleted.")
    return {"free_bytes": free, "required_free_bytes": required, "reserve_bytes": 12 * 1024**3}


def require_preflight(args, context):
    summary = read_json(Path(args.output_root) / "preflight_summary.json")
    records = execution.verify_features(args.output_root, context, context[0]["preflight_ids"])
    if (summary.get("complete") is not True or summary.get("plan_fingerprint") != context[0]["fingerprint"] or
            summary.get("verified_record_hashes") != [row["record_sha256"] for row in records]):
        raise ValueError("Two-case preflight is not complete for this exact plan.")


def check_cpu_dependencies(args):
    # Export verification imports the loader in the pinned inference environment;
    # fitting may use a separate CPU Python, but neither environment is auto-upgraded.
    import scripts.run_phase3d_probes  # noqa: F401
    subprocess.run([args.probe_python, "-c", "import numpy, scipy, sklearn; "
                    "print('Phase 3D CPU dependencies:', numpy.__version__, scipy.__version__, sklearn.__version__)"],
                   check=True, cwd=args.project_root)


def make_engine(args, context):
    from scripts.run_phase3b_vm import worker_environment
    from scripts.run_phase3c_preflight import Engine
    from scripts.phase3c_budget import runtime_compatibility
    plan = context[0]
    env = worker_environment(args.gpus, args.storage_root)
    for name in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "HF_HOME", "TORCH_HOME"):
        os.environ[name] = env[name]
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "RESEARCH_CLOUD_TOKEN", "SURF_API_TOKEN"):
        os.environ.pop(name, None)
    before = plan["source_runtime"]
    engine = Engine(plan["settings"], before["execution_mode"], before["gpu_weight_budget_gib"])
    compatibility = runtime_compatibility(before, engine.runtime)
    frozen = Path(args.output_root) / "runtime.json"
    if frozen.exists():
        runtime_compatibility(read_json(frozen)["runtime"], engine.runtime)
    else:
        frozen_write(frozen, {"runtime": engine.runtime, "source_compatibility": compatibility,
                              "plan_fingerprint": plan["fingerprint"]})
    atomic_write(Path(args.output_root) / "runtime_observations" / f"{time.time_ns()}.json", engine.runtime)
    engine.project_root, engine.path_map = plan["project_root"], plan["source_path_map"]
    return engine


def stage(args, context, name):
    plan, pairs, _ = context
    root = Path(args.output_root)
    ids = plan["preflight_ids"] if args.preflight_subset else None
    if name == "audit":
        specs = execution.feature_tasks(plan, name, ids)
        if all(execution.saved_checkpoint(root, plan, spec) for spec in specs):
            print(f"Processor audit: reusing {len(specs)} verified checkpoints.", flush=True)
            return
        from importlib.metadata import version
        import transformers
        if (transformers.__version__ != plan["settings"]["expected_transformers_version"] or
                version("qwen-vl-utils") != plan["settings"]["expected_qwen_vl_utils_version"]):
            raise ValueError("Processor-only audit requires the pinned GPU processor packages.")
        processor = transformers.AutoProcessor.from_pretrained(plan["settings"]["model_name"],
                                                               revision=plan["settings"]["model_revision"])
        execution.run_tasks(root, context, specs,
            lambda spec: execution.audit_execute(root, context, processor, spec), args.retry_failed)
    elif name in ("capture", "patch", "knockout"):
        if name == "patch":
            execution.verify_features(root, context, ids, patched=False)
        if name == "knockout":
            execution.verify_features(root, context)
            from scripts.phase3d_knockout import condition_specs, execute_condition, analyze_complete
            specs = condition_specs(plan, pairs)
        else:
            specs = execution.feature_tasks(plan, name, ids)
        if any(execution.saved_checkpoint(root, plan, spec) is None for spec in specs):
            engine = make_engine(args, context)
            cache = {}
            function = (lambda spec: execute_condition(root, context, engine, spec, cache)) if name == "knockout" else (
                lambda spec: getattr(execution, name + "_execute")(root, context, engine, spec))
            execution.run_tasks(root, context, specs, function, args.retry_failed)
        if name == "knockout":
            analyze_complete(root, context)
        else:
            execution.verify_features(root, context, ids, patched=name == "patch")
    elif name == "export":
        execution.export_features(root, context)
    elif name == "verify":
        verify_complete(args, context)
    else:
        raise ValueError("CPU probes run in an isolated child, not in a loaded GPU process.")


def child_command(args, stage_name, subset=False):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--action", "run", "--stage", stage_name,
               "--output_root", args.output_root, "--storage_root", args.storage_root,
               "--project_root", args.project_root, "--gpus", args.gpus, "--backup_dir", args.backup_dir,
               "--attempt_id", args.attempt_id, "--probe_python", args.probe_python]
    if subset:
        command.append("--preflight_subset")
    if args.retry_failed:
        command.append("--retry_failed")
    return command


def run_child(args, command, label):
    from scripts.run_phase3c_vm import run_child as stream_child
    # Existing streamer forwards signals, saves complete stdout/stderr and sanitizes API tokens.
    return stream_child(args, command, "phase3d_" + label)


def probe_command(args):
    return [args.probe_python, "-u", str(Path(args.project_root) / "scripts/run_phase3d_probes.py"),
            "--manifest_path", str(Path(args.output_root) / "feature_store/manifest.json"),
            "--output_dir", str(Path(args.output_root) / "probes")]


def verify_complete(args, context):
    from scripts.phase3d_knockout import analyze_complete
    from scripts.run_phase3d_probes import DEFAULT_READOUTS, SETTINGS, load_dataset, saved_result
    root = Path(args.output_root)
    execution.verify_features(root, context)
    manifest = root / "feature_store/manifest.json"
    load_dataset(manifest)
    ko = analyze_complete(root, context)
    subprocess.run([*probe_command(args), "--verify_only"], cwd=args.project_root, check=True)
    config = read_json(root / "probes/run_config.json")
    summary = read_json(root / "probes/aggregate_summary.json")
    if (config.get("fingerprint") != digest({k: v for k, v in config.items() if k != "fingerprint"}) or
            config["manifest_sha256"] != file_hash(manifest) or summary.get("complete") is not True or
            summary.get("fingerprint") != config["fingerprint"] or summary.get("base_count") != 50 or
            config["feature_names"] != DEFAULT_READOUTS or config["settings"] != SETTINGS):
        raise ValueError("Probe analysis is incomplete or bound to another dataset.")
    expected = set(config["feature_names"]) | {"condition__source_pair:L16_output", "condition__option_pair:L20_output",
                 "static_attributes", "option_lexical"} | {
                 f"label_permutation_{i:03d}" for i in range(config["settings"]["label_permutations"])}
    if set(summary["results"]) != expected:
        raise ValueError("Probe aggregate omitted a requested readout/control.")
    for name, ref in summary["results"].items():
        path = (root / "probes" / ref["result_path"]).resolve()
        if not path.is_relative_to((root / "probes").resolve()) or file_hash(path) != ref["sha256"]:
            raise ValueError("Probe checkpoint resource/hash changed.")
        result = read_json(path)
        if saved_result(path.parent, result["fingerprint"]) is None:
            raise ValueError("Incomplete per-readout probe checkpoint.")
    value = {"schema": execution.SCHEMA, "artifact_type": "real", "complete": True,
             "plan_fingerprint": context[0]["fingerprint"], "base_count": 50, "feature_rows": 300,
             "ko_paired_rows": ko["paired_rows"], "ko_condition_rows": ko["condition_rows"],
             "feature_manifest_sha256": file_hash(manifest), "probe_summary_sha256": file_hash(root / "probes/aggregate_summary.json"),
             "knockout_summary_sha256": file_hash(root / "knockout/analysis/aggregate_summary.json"),
             "interpretation": "Technical completion; no claim of mediation or a uniquely attentional mechanism."}
    atomic_write(root / "aggregate_summary.json", value)
    return value


def run(args):
    nested = os.environ.get("PHASE3D_PARENT_ATTEMPT") == args.attempt_id
    with nullcontext() if nested else lock(args.output_root, ".orchestrator.lock"):
        previous = os.environ.get("PHASE3D_PARENT_ATTEMPT")
        os.environ["PHASE3D_PARENT_ATTEMPT"] = args.attempt_id
        try:
            _run(args)
        finally:
            if previous is None:
                os.environ.pop("PHASE3D_PARENT_ATTEMPT", None)
            else:
                os.environ["PHASE3D_PARENT_ATTEMPT"] = previous


def _run(args):
    context = execution.load_context(args.output_root)
    if args.stage not in ("audit", "preflight") and not args.preflight_subset:
        require_preflight(args, context)
    if args.stage in ("preflight", "full"):
        if args.preflight_subset:
            raise ValueError("Subset mode cannot certify a full run.")
        steps = ("audit", "capture", "patch") if args.stage == "preflight" else (
            "audit", "capture", "patch", "export", "knockout", "probes", "verify")
        storage_gate(args, context)
        check_cpu_dependencies(args)
        for name in steps:
            command = probe_command(args) if name == "probes" else child_command(args, name, args.stage == "preflight")
            if run_child(args, command, name) != 0:
                raise RuntimeError(f"Phase 3D {name} failed; inspect logs and preserve checkpoints.")
        if args.stage == "preflight":
            rows = execution.verify_features(args.output_root, context, context[0]["preflight_ids"])
            atomic_write(Path(args.output_root) / "preflight_summary.json", {"complete": True,
                "plan_fingerprint": context[0]["fingerprint"], "independent_bases": 2,
                "verified_record_hashes": [r["record_sha256"] for r in rows], "is_full_experiment": False})
        else:
            verify_complete(args, context)
    elif args.stage == "probes":
        if run_child(args, probe_command(args), "probes") != 0:
            raise RuntimeError("CPU probe runner failed.")
    else:
        with lock(args.output_root):
            stage(args, context, args.stage)
    atomic_write(Path(args.output_root) / "last_attempt.json", {"attempt_id": args.attempt_id,
        "stage": args.stage, "complete": True, "plan_fingerprint": context[0]["fingerprint"]})


def backup_files(root, reports_only):
    root = Path(root).resolve()
    plan = read_json(root / "plan.json")
    if (plan.get("schema") != execution.SCHEMA or plan.get("artifact_type") != "real" or
            plan.get("fingerprint") != digest({k: v for k, v in plan.items() if k != "fingerprint"})):
        raise ValueError("Backup requires an intact Phase 3D plan.")
    directories = {"audit", "capture", "patch", "knockout", "feature_store", "probes", "runtime_observations", "logs", "failures"}
    filenames = {"plan.json", "pairs.json", "mappings.json", "case_metadata.json", "outer_folds.json", "source_baselines.json",
                 "runtime.json", "progress_status.json", "preflight_summary.json", "last_attempt.json", "aggregate_summary.json",
                 "unattended_status.json", "unattended_job.log", "ko_reuse_audit.json"}
    filenames.add("ko_reuse_source.json")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Phase 3D backups refuse symlinks.")
        if not path.is_file() or path.name in {".phase3d.lock", ".orchestrator.lock", ".pipeline.lock", ".probe.lock"}:
            continue
        relative = path.relative_to(root)
        if relative.parts[0] not in directories and relative.as_posix() not in filenames:
            raise ValueError("Unexpected file in research root; keep private configs outside artifacts.")
        if not reports_only or path.suffix not in (".pt", ".npz", ".mp4"):
            files["run/" + relative.as_posix()] = path
    if not reports_only:
        for number, (path, sha) in enumerate(sorted(plan["video_sha256"].items())):
            if Path(path).is_symlink() or file_hash(path) != sha:
                raise ValueError("Frozen video changed; full backup/Pause blocked.")
            files[f"source_videos/{number:03d}_{Path(path).name}"] = Path(path)
    return files


def assert_idle():
    from scripts.run_phase3c_unattended import assert_quiescent
    assert_quiescent()
    result = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=30, check=True)
    pattern = re.compile(r"(?:^|[ /])(?:run_phase3d_vm|run_phase3d_probes|run_phase3c_scale|run_phase3c_support|run_phase3c_replication)\.py(?:\s|$)")
    own = {os.getpid()}
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or int(parts[0]) in own or not pattern.search(parts[1]):
            continue
        # tmux command strings are not themselves executing research jobs.
        try:
            executable = Path(os.readlink(f"/proc/{parts[0]}/exe")).name
        except OSError:
            raise RuntimeError("Cannot establish workspace quiescence; Pause blocked.") from None
        if executable != "tmux":
            raise RuntimeError("Another research process is active; launch/Pause blocked.")


def services(args):
    from scripts.surf_workspace import SurfClient, load_private_config, check_private_location
    from scripts.phase3b_email import EmailNotifier, load_private_config as email_config
    forbidden = [args.project_root, args.output_root, args.backup_dir, str(lifecycle_dir(args))]
    forbidden.append(execution.load_context(args.output_root)[0]["source_root"])
    notifier = client = None
    if args.email_notify:
        check_private_location(args.email_config, forbidden)
        notifier = EmailNotifier(email_config(args.email_config, forbidden))
        notifier.check()
    if args.pause_after:
        if not args.confirm_exclusive_workspace:
            raise ValueError("Automatic whole-VM Pause needs explicit exclusive-workspace confirmation.")
        check_private_location(args.surf_config, forbidden)
        client = SurfClient(load_private_config(args.surf_config, forbidden))
        client.check(require_pause=True)
    return notifier, client


def lifecycle_dir(args):
    root = Path(args.output_root)
    return root.parent / (root.name + "_lifecycle")


def job(args, child=run_child, pack=None, verify=None, idle=assert_idle, service_factory=services):
    from scripts.backup_phase3b import create_backup, verify_backup
    pack, verify = pack or create_backup, verify or verify_backup
    directory, root = lifecycle_dir(args), Path(args.output_root)
    started, notifier, client = time.perf_counter(), None, None
    status = {"schema": execution.SCHEMA, "stage": args.stage, "attempt_id": args.attempt_id,
              "started_at": datetime.now(timezone.utc).isoformat(), "backup_verified": False,
              "local_backup_confirmed": False, "pause_not_confirmed": True, "output_root": str(root)}
    def save(state, **fields):
        status.update(state=state, elapsed_sec=time.perf_counter()-started, **fields)
        atomic_write(directory / "job_status.json", status)
        print(f"[Phase 3D lifecycle] {state}; elapsed={status['elapsed_sec']/60:.1f} min", flush=True)
    def notify():
        if notifier is None:
            return
        label = "SUCCESS" if status.get("experiment_success") else "FAILED" if "experiment_success" in status else "NOT STARTED"
        body = (f"Experiment: {label}\nPhase 3D stage: {args.stage}\n50 independent frozen rescue bases.\n"
                f"Run: {root}\nState: {status['state']}\nElapsed: {status['elapsed_sec']/3600:.2f} hours\n"
                f"Log: {directory/'job.log'}\nStatus: {directory/'job_status.json'}\n"
                f"Verified full VM backup: {status['backup_verified']}\nBackup: {status.get('backup_dir', 'not available')}\n"
                "Download all parts/manifest/SHA256SUMS and verify off-VM; local backup is NOT confirmed.\n"
                "Retain frozen Phase 3B/3C backups separately. No old activation stores are duplicated here.\n"
                "A successful preflight is NOT a full experiment. This is a technical summary, not a scientific interpretation.\n"
                "Pause is requested only after verified full backup and idle check; billing stop is NOT confirmed.\n"
                "No credentials, tensors or videos attached.")
        try:
            status["email"] = notifier.send(f"[Phase 3D] {args.stage} {label}", body)
        except Exception as exc:
            status["email"] = {"error_type": type(exc).__name__}
        atomic_write(directory / "job_status.json", status)
    with lock(directory, ".lifecycle.lock"):
        save("prechecking")
        try:
            context = execution.load_context(root)
            notifier, client = service_factory(args)
            storage_gate(args, context)
            idle()
            command = child_command(args, args.stage)
            save("experiment_running")
            try:
                code = child(args, command, args.stage)
            except Exception as exc:
                code = 1
                status["child_error_type"] = type(exc).__name__
            last = read_json(root / "last_attempt.json") if (root / "last_attempt.json").exists() else {}
            success = code == 0 and last.get("complete") is True and last.get("stage") == args.stage and last.get("attempt_id") == args.attempt_id
            if success:
                try:
                    if args.stage == "full":
                        verify_complete(args, context)
                    elif args.stage == "preflight":
                        require_preflight(args, context)
                except Exception as exc:
                    success = False
                    status["verification_error_type"] = type(exc).__name__
            save("experiment_finished", experiment_success=success, experiment_return_code=code)
            with lock(root, ".orchestrator.lock"), lock(root):
                atomic_write(root / "unattended_status.json", status)
                if (directory / "job.log").exists():
                    shutil.copyfile(directory / "job.log", root / "unattended_job.log")
                save("backup_running")
                bundle = pack(root, args.backup_dir, file_provider=backup_files, bundle_prefix="phase3c_backup",
                    metadata={"schema": "phase3d_backup_v1", "phase3d_plan_fingerprint": context[0]["fingerprint"],
                              "selected_videos_included": True, "old_activation_stores_included": False})
                manifest = verify(bundle)
            if (manifest.get("schema") != "phase3d_backup_v1" or manifest.get("complete") is not True or
                    manifest.get("reports_only") or manifest.get("activation_tensors_included") is not True):
                raise RuntimeError("Verified full backup required before Pause.")
            save("backup_verified", backup_verified=True, backup_dir=str(bundle))
            if args.pause_after:
                with lock(root, ".orchestrator.lock"), lock(root):
                    idle()
                    save("pause_request_pending")
                    notify()
                    result = client.request_pause()
                    save("pause_requested", pause_result=result)
            else:
                save("done_without_pause")
                notify()
            return 0 if success else code if code > 0 else 1
        except Exception as exc:
            traceback.print_exc()
            save("needs_attention", error_type=type(exc).__name__)
            notify()
            return 1


def launch_command(args):
    command = child_command(args, args.stage)
    command[command.index("--action") + 1] = "job"
    if args.email_notify:
        command += ["--email_notify", "--email_config", args.email_config]
    if args.pause_after:
        command += ["--pause_after", "--surf_config", args.surf_config, "--confirm_exclusive_workspace"]
    return command


def normalize_paths(args):
    for name in ("output_root", "storage_root", "project_root", "backup_dir"):
        setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    # Resolving a venv's Python symlink selects the base interpreter, losing its packages.
    args.probe_python = os.path.abspath(os.path.expanduser(args.probe_python))


def main():
    from scripts.phase3b_email import DEFAULT_CONFIG as EMAIL_CONFIG
    from scripts.surf_workspace import DEFAULT_CONFIG as SURF_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "plan", "run", "launch", "job"), default="plan")
    parser.add_argument("--stage", choices=STAGES, default="preflight")
    parser.add_argument("--source_run", help="Complete Phase 3C scale50 prefix containing all 100 L16 source captures.")
    parser.add_argument("--ko_source", help="Optional complete KO25 root; inherited summary-only masks will not be reused.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--storage_root", default="/data/yuxuanstorage")
    parser.add_argument("--backup_dir", default="/data/yuxuanstorage/backups")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--probe_python", default=sys.executable)
    parser.add_argument("--retry_failed", action="store_true")
    parser.add_argument("--preflight_subset", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attempt_id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--session_name", default="phase3d")
    parser.add_argument("--email_notify", action="store_true")
    parser.add_argument("--email_config", default=EMAIL_CONFIG)
    parser.add_argument("--pause_after", action="store_true")
    parser.add_argument("--surf_config", default=SURF_CONFIG)
    parser.add_argument("--confirm_exclusive_workspace", action="store_true")
    args = parser.parse_args()
    normalize_paths(args)
    if (not Path(args.storage_root).is_dir() or not Path(args.output_root).is_relative_to(args.storage_root) or
            not Path(args.backup_dir).is_relative_to(args.storage_root) or
            Path(args.backup_dir).is_relative_to(args.output_root) or Path(args.output_root).is_relative_to(args.backup_dir)):
        parser.error("Use separate run and backup directories on the mounted persistent volume.")
    if args.pause_after and not args.confirm_exclusive_workspace:
        parser.error("--pause_after requires --confirm_exclusive_workspace.")
    if (args.pause_after or args.email_notify) and args.action not in ("launch", "job", "plan"):
        parser.error("Email/Pause flags belong to --action launch or job, not an unwrapped stage.")
    if args.action in ("launch", "job") and args.stage not in ("preflight", "full"):
        parser.error("Detached lifecycle supports only complete preflight or full pipelines.")
    args.attempt_id = args.attempt_id or uuid.uuid4().hex
    if args.action == "prepare":
        if not args.source_run:
            parser.error("--source_run is required for preparation.")
        with lock(args.output_root):
            execution.freeze_plan(args)
        print("Phase 3D metadata plan frozen; no processor/model/GPU or SURF action started.")
    elif args.action == "plan":
        context = execution.load_context(args.output_root)
        if args.stage == "full":
            require_preflight(args, context)
        print(json.dumps({"plan": context[0]["fingerprint"], "stage": args.stage, "base_count": 50,
                          "feature_rows": 300, "ko_paired_tasks": 5850, "storage": storage_gate(args, context),
                          "gpu_started": False, "automatic_pause": args.pause_after}, indent=2))
    elif args.action == "run":
        run(args)
    elif args.action == "job":
        sys.exit(job(args))
    else:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.session_name):
            parser.error("Use a simple tmux session name.")
        context = execution.load_context(args.output_root)
        if args.stage == "full":
            require_preflight(args, context)
        storage_gate(args, context)
        if not shutil.which("tmux"):
            parser.error("Install tmux before detached launch.")
        if subprocess.run(["tmux", "has-session", "-t", "=" + args.session_name], capture_output=True).returncode == 0:
            parser.error("Session exists; attach instead of launching a duplicate.")
        with lock(lifecycle_dir(args), ".lifecycle.lock"), lock(args.output_root):
            check_cpu_dependencies(args)
            services(args)
            assert_idle()
            command = launch_command(args)
            atomic_write(lifecycle_dir(args) / "launch_plan.json", {"command": command, "attempt_id": args.attempt_id})
        shell = f"exec {shlex.join(command)} >> {shlex.quote(str(lifecycle_dir(args) / 'job.log'))} 2>&1"
        subprocess.run(["tmux", "new-session", "-d", "-s", args.session_name, "-c", args.project_root, shell], check=True)
        print(f"Detached job started. Attach: tmux attach -t {args.session_name}\nLog: {lifecycle_dir(args)/'job.log'}")
        print("Close the browser/computer, not the VM. Pause requires verified full backup and an idle workspace.")


if __name__ == "__main__":
    main()
