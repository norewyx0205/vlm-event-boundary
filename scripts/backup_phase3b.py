"""Create checksummed, timestamped Phase 3B backup parts on persistent storage."""

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    from .run_phase3b_vm import digest, run_lock, write_json
except ImportError:
    from run_phase3b_vm import digest, run_lock, write_json


def backup_files(root, reports_only):
    root = Path(root).resolve()
    files = {}
    for path in sorted(root.rglob("*")):
        if path.name == ".pipeline.lock" or path.suffix == ".tmp":
            continue
        if path.is_symlink():
            raise ValueError(f"Backup refuses symlinks: {path}.")
        if path.is_file() and not (reports_only and path.suffix in {".pt", ".mp4"}):
            files[f"run/{path.relative_to(root).as_posix()}"] = path
    config_path = root / "vm_run_config.json"
    if not reports_only and config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for index, (source, expected) in enumerate(sorted(config["video_sha256"].items())):
            path = Path(source)
            if not path.is_file() or digest(path) != expected:
                raise RuntimeError(f"Frozen video is missing or changed: {source}. Full backup aborted.")
            files[f"source_videos/{index:03d}_{path.name}"] = path
    return files


def create_backup(run_root, backup_dir, reports_only=False, part_bytes=4 * 1024**3):
    root, destination = Path(run_root).resolve(), Path(backup_dir).resolve()
    if not root.is_dir() or destination.is_relative_to(root):
        raise ValueError("Use an existing run root and a backup directory outside it.")
    if part_bytes < 1:
        raise ValueError("Backup part size must be positive.")
    destination.mkdir(parents=True, exist_ok=True)
    with run_lock(root):
        files = backup_files(root, reports_only)
        if not files:
            raise ValueError("No research artifacts found to back up.")
        total = sum(path.stat().st_size for path in files.values())
        if shutil.disk_usage(destination).free < total * 1.02 + 512 * 1024**2:
            raise RuntimeError(
                f"Insufficient space to duplicate {total / 1024**3:.1f} GiB for a full backup. "
                "Transfer the persistent run directory and frozen videos directly to your computer with rsync/scp; "
                "a reports-only ZIP does NOT back up activations. Do not delete checkpoints to make room."
            )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        bundle = destination / f"phase3b_backup_{stamp}"
        bundle.mkdir()
        manifest = {
            "schema": "phase3b_local_backup_v1", "artifact_type": "real",
            "created_at": datetime.now(timezone.utc).isoformat(), "source_run_root": str(root),
            "complete": False, "expected_file_count": len(files), "expected_total_bytes": total,
            "reports_only": reports_only, "activation_tensors_included": not reports_only,
            "local_backup_confirmed": False,
            "status": "VM package created; download and verify on your computer",
            "run_status": json.loads((root / "vm_last_status.json").read_text()) if (root / "vm_last_status.json").is_file() else None,
            "files": [], "archives": [],
        }
        started, copied, part_index, part_size, archive = time.perf_counter(), 0, 0, 0, None
        archive_path = None
        def finish_part():
            if archive is not None:
                archive.close()
                sha = digest(archive_path)
                manifest["archives"].append({"name": archive_path.name, "size_bytes": archive_path.stat().st_size, "sha256": sha})
                write_json(bundle / "backup_manifest.json", manifest)
        try:
            for name, path in files.items():
                stat = path.stat()
                if archive is None or (part_size and part_size + stat.st_size > part_bytes):
                    finish_part()
                    part_index += 1
                    archive_path = bundle / f"part_{part_index:03d}.zip"
                    archive = zipfile.ZipFile(archive_path, "w", allowZip64=True)
                    part_size = 0
                info = zipfile.ZipInfo.from_file(path, arcname=name)
                info.compress_type = zipfile.ZIP_STORED if path.suffix in {".pt", ".mp4"} else zipfile.ZIP_DEFLATED
                sha = hashlib.sha256()
                with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(block)
                        sha.update(block)
                if (path.stat().st_size, path.stat().st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                    raise RuntimeError(f"File changed during backup: {path}. Stop writers before packaging.")
                manifest["files"].append({"path": name, "source_path": str(path), "part": archive_path.name,
                                          "size_bytes": stat.st_size, "sha256": sha.hexdigest()})
                copied += stat.st_size
                part_size += stat.st_size
                if len(manifest["files"]) % 100 == 0:
                    print(f"Backup: {len(manifest['files'])}/{len(files)} files, {copied/1024**3:.1f}/{total/1024**3:.1f} GiB; elapsed={(time.perf_counter()-started)/60:.1f} min", flush=True)
            finish_part()
            archive = None
        except BaseException:
            if archive is not None:
                archive.close()
            manifest["status"] = "Incomplete packaging; do not treat as a verified backup"
            write_json(bundle / "backup_manifest.json", manifest)
            raise
        manifest["elapsed_sec"] = time.perf_counter() - started
        (bundle / "SHA256SUMS").write_text("".join(f"{part['sha256']}  {part['name']}\n" for part in manifest["archives"]), encoding="ascii")
        manifest["complete"] = True
        write_json(bundle / "backup_manifest.json", manifest)
        print(f"Backup package: {bundle}. Download ALL parts, backup_manifest.json and SHA256SUMS to your computer.")
        if reports_only:
            print("REPORTS ONLY: .pt activations and videos are NOT backed up.")
        return bundle


def verify_backup(bundle):
    bundle = Path(bundle)
    manifest = json.loads((bundle / "backup_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete") or not manifest["archives"] or manifest["status"].startswith("Incomplete"):
        raise RuntimeError("Backup manifest is incomplete.")
    if len(manifest["files"]) != manifest["expected_file_count"] or sum(entry["size_bytes"] for entry in manifest["files"]) != manifest["expected_total_bytes"]:
        raise RuntimeError("Backup file coverage is incomplete.")
    expected_by_part = {}
    for entry in manifest["files"]:
        expected_by_part.setdefault(entry["part"], {})[entry["path"]] = entry
    for part in manifest["archives"]:
        path = bundle / part["name"]
        if digest(path) != part["sha256"]:
            raise RuntimeError(f"Backup checksum mismatch: {path}.")
        with zipfile.ZipFile(path) as archive:
            expected = expected_by_part[part["name"]]
            if len(archive.namelist()) != len(expected) or set(archive.namelist()) != set(expected):
                raise RuntimeError(f"Backup member list differs: {path}.")
            for name, entry in expected.items():
                sha = hashlib.sha256()
                with archive.open(name) as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        sha.update(block)
                if sha.hexdigest() != entry["sha256"] or archive.getinfo(name).file_size != entry["size_bytes"]:
                    raise RuntimeError(f"Backup file checksum mismatch: {name}.")
    print("All backup parts and file checksums verified on this machine.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root")
    parser.add_argument("--backup_dir", default="/data/yuxuanstorage/backups")
    parser.add_argument("--reports_only", action="store_true")
    parser.add_argument("--part_gib", type=float, default=4)
    parser.add_argument("--verify", help="Verify a downloaded backup directory, without loading a model.")
    args = parser.parse_args()
    if args.verify:
        verify_backup(args.verify)
    elif args.run_root:
        create_backup(args.run_root, args.backup_dir, args.reports_only, int(args.part_gib * 1024**3))
    else:
        parser.error("Use --run_root to create a package or --verify to verify downloaded parts.")


if __name__ == "__main__":
    main()
