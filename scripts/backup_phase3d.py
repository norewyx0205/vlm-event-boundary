"""Package or verify isolated Phase 3D evidence without loading a model."""

import argparse
import math
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.backup_phase3b import create_backup, verify_backup
from scripts.phase3c_core import read_json
from scripts.run_phase3d_vm import backup_files, lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--run_root")
    choice.add_argument("--verify")
    parser.add_argument("--backup_dir", default="/data/yuxuanstorage/backups")
    parser.add_argument("--part_gib", type=float, default=4)
    parser.add_argument("--reports_only", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.part_gib) or args.part_gib <= 0:
        parser.error("Part size must be finite and positive.")
    if args.verify:
        manifest = verify_backup(args.verify)
        if manifest.get("schema") != "phase3d_backup_v1":
            parser.error("Not a Phase 3D backup. Use the original verifier for older packages.")
    else:
        root = Path(args.run_root).resolve()
        with lock(root, ".orchestrator.lock"), lock(root):
            plan = read_json(root / "plan.json")
            create_backup(root, args.backup_dir, args.reports_only, int(args.part_gib*1024**3),
                file_provider=backup_files, bundle_prefix="phase3c_backup", metadata={
                    "schema": "phase3d_backup_v1", "phase3d_plan_fingerprint": plan["fingerprint"],
                    "selected_videos_included": not args.reports_only, "old_activation_stores_included": False})


if __name__ == "__main__":
    main()
