"""Back up isolated Phase 3C evidence, including preparation and selected videos."""

import argparse
import math
from pathlib import Path

try:
    from .backup_phase3b import create_backup as package_files, verify_backup as verify_parts
    from .phase3c_core import digest, file_hash, read_json
    from .run_phase3c_vm import vm_lock
except ImportError:
    from backup_phase3b import create_backup as package_files, verify_backup as verify_parts
    from phase3c_core import digest, file_hash, read_json
    from run_phase3c_vm import vm_lock


PUBLIC_DIRS = {"preparation_snapshot", "captures", "baseline", "preflight", "patch", "routing", "knockout", "analysis", "logs"}
PUBLIC_FILES = {"phase3c_vm_config.json", "execution_config.json", "vm_last_status.json",
                "unattended_run_status.json", "unattended_job.log"}


def backup_files(root, reports_only):
    root = Path(root).resolve()
    config = read_json(root / "phase3c_vm_config.json")
    if (config.get("schema") != "phase3c_vm_execution_v1" or config.get("artifact_type") != "real" or
            config.get("vm_fingerprint") != digest({k: v for k, v in config.items() if k != "vm_fingerprint"}) or
            config.get("output_root") != str(root)):
        raise ValueError("Phase 3C backup requires an intact VM configuration.")
    for name, expected in config["preparation_snapshot_sha256"].items():
        path = root / "preparation_snapshot" / name
        if not path.resolve().is_relative_to(root / "preparation_snapshot") or file_hash(path) != expected:
            raise ValueError("Preparation snapshot is missing or changed; backup cannot authorize Pause.")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Phase 3C backup refuses symlinks.")
        if path.name in {".pipeline.lock", ".phase3c_vm.lock"} or path.suffix == ".tmp":
            continue
        relative = path.relative_to(root)
        if relative.parts[0] not in PUBLIC_DIRS and relative.as_posix() not in PUBLIC_FILES:
            raise ValueError(f"Unexpected artifact in Phase 3C output: {relative}. Keep private configurations outside this root.")
        if path.is_file() and not (reports_only and path.suffix in {".pt", ".mp4"}):
            files[f"run/{relative.as_posix()}"] = path
    if not reports_only:
        for index, (source, expected) in enumerate(sorted(config["video_sha256"].items())):
            path = Path(source)
            if path.is_symlink() or file_hash(path) != expected:
                raise ValueError("A frozen selected video is missing or changed; full backup aborted.")
            files[f"source_videos/{index:03d}_{path.name}"] = path
    return files


def create_backup(run_root, backup_dir, reports_only=False, part_bytes=4 * 1024**3):
    root = Path(run_root).resolve()
    with vm_lock(root):
        config = read_json(root / "phase3c_vm_config.json")
        return package_files(root, backup_dir, reports_only, part_bytes,
            file_provider=backup_files, bundle_prefix="phase3c_backup", metadata={
                "schema": "phase3c_local_backup_v1", "vm_fingerprint": config["vm_fingerprint"],
                "selection_fingerprint": config["selection_fingerprint"],
                "available_activation_file_count": len(list((root / "captures").rglob("*.pt"))),
                "backup_code_sha256": file_hash(__file__),
                "phase3b_source_backup_included": False,
                "scope": "All available Phase 3C artifacts and selected videos; retain the frozen Phase 3B backup separately. "
                         "Complete packaging is not evidence that the experiment completed."})


def verify_backup(bundle):
    manifest = read_json(Path(bundle) / "backup_manifest.json")
    if manifest.get("schema") != "phase3c_local_backup_v1" or manifest.get("artifact_type") != "real":
        raise ValueError("Not a Phase 3C backup; use the Phase 3B verifier for legacy packages.")
    return verify_parts(bundle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--run_root")
    choice.add_argument("--verify", help="Verify all downloaded parts on your computer; no model or extraction is needed.")
    parser.add_argument("--backup_dir", default="/data/yuxuanstorage/backups")
    parser.add_argument("--part_gib", type=float, default=4)
    parser.add_argument("--reports_only", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.part_gib) or args.part_gib <= 0:
        parser.error("Part size must be finite and positive.")
    if args.verify:
        verify_backup(args.verify)
    else:
        create_backup(args.run_root, args.backup_dir, args.reports_only, int(args.part_gib * 1024**3))


if __name__ == "__main__":
    main()
