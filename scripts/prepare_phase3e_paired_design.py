"""Freeze CPU-only paired-video materials; this command cannot launch a model."""

import argparse
import json
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import phase3e_paired_design as design
from scripts.phase3c_core import atomic_write, file_hash


def prepare(root, render_families=0):
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new directory; never overwrite old captures/plans.")
    if render_families not in (0, 1, 2):
        raise ValueError("Only zero, one or two CPU example families may be rendered here.")
    root.mkdir(parents=True, exist_ok=True)
    plan = design.build_plan(root)
    atomic_write(root / "plan.json", plan)
    atomic_write(root / "prompt_manifest.jsonl", plan["rows"], jsonl=True)
    atomic_write(root / "statement_manifest.jsonl", plan["statement_examples"], jsonl=True)
    atomic_write(root / "order_pairs.jsonl", plan["order_pairs"], jsonl=True)
    atomic_write(root / "boundary_pairs.jsonl", plan["boundary_pairs"], jsonl=True)
    atomic_write(root / "scene_folds.json", plan["folds"])
    rendered = design.render_examples(root, plan, render_families) if render_families else {
        "status": "not_requested", "videos": [], "model_forward_count": 0}
    atomic_write(root / "render_audit.json", rendered)
    summary = {"schema": design.SCHEMA, "status": "cpu_design_audit_passed",
        "plan_fingerprint": plan["fingerprint"], "counts": plan["counts"],
        "rendered_example_videos": len(rendered["videos"]),
        "material_audit": design.validate_plan(plan), "new_ko_scheduled": False,
        "new_gpu_inference_performed": False,
        "code_sha256": {Path(source).name: file_hash(source) for source in
            (__file__, design.__file__, design.fixture.__file__, design.fixture.generator.__file__,
             Path(design.__file__).with_name("phase3d_core.py"),
             Path(design.__file__).with_name("phase3c_core.py"))}}
    atomic_write(root / "summary.json", summary)
    atomic_write(root / "output_manifest.json", {"files": [
        {"path": path.relative_to(root).as_posix(), "sha256": file_hash(path)}
        for path in sorted(root.rglob("*")) if path.is_file()]})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--render_example_families", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_root, args.render_example_families), indent=2))
