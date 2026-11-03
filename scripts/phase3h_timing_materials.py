"""Phase 3H timing cells: re-time one base's target events (distractors follow the generator) and nothing else.

Materials are CPU-only. A cohort's native Low/Temporal videos are first reconstructed through this module's
own render path and must match the existing videos frame for frame (decoded pixels exact, one environment)
before any new timing cell is rendered. See docs/phase3h_timing_dissociation.md.
"""

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from scripts import generate_ladder_dataset as renderer
from scripts.phase3c_core import atomic_write, digest, file_hash, frozen_write, read_json
from scripts.phase3d_core import audit_row
from scripts.phase3d_execution import permuted_row
from scripts.phase3e_order_materials import decoded_comparison

SCHEMA = "phase3h_timing_materials_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("low_boundary", "temporal_boundary")
# cell: (source condition, Event-1 onset frame, gap frames)
CELLS = {"low": ("low_boundary", 30, 0), "low_mid": ("low_boundary", 60, 0), "low_late": ("low_boundary", 75, 0),
         "temporal_early": ("temporal_boundary", 15, 45), "temporal": ("temporal_boundary", 30, 45)}
NATIVE = {"low_boundary": "low", "temporal_boundary": "temporal"}
NEW_CELLS = ("low_mid", "low_late", "temporal_early")
VIEWS = ("archived", "option_order_permuted")
TOKEN_FRAMES = 15  # one temporal token at the processor's default 2 fps sampling and temporal patch 2
MOTION_THRESHOLD, MOTION_PIXELS = 60, 20


def schedule(row, cell):
    """Event and boundary timing of `cell`, built from a native Low/Temporal source row."""
    condition, onset, gap = CELLS[cell]
    if row["condition"] != condition or row.get("feature_variant") != "full":
        raise ValueError("A timing cell must be built from its own full-feature source condition.")
    event, boundary = row["event_timing"], row["boundary_timing"]
    duration = event["first_event_end_frame"]-event["first_event_start_frame"]
    if (event["second_event_end_frame"]-event["second_event_start_frame"] != duration or
            boundary["gap_frames"] != CELLS[NATIVE[condition]][2] or
            event.get("unrelated_event_start_frame") is not None or
            boundary.get("visual_marker") != "none" or boundary.get("audio_marker") != "none"):
        raise ValueError("Source row does not use the fixed Level-5 Low/Temporal schedule.")
    first_end = onset+duration
    second_start, second_end = first_end+gap, first_end+gap+duration
    if onset < TOKEN_FRAMES or second_end >= row["total_frames"] or any(v % TOKEN_FRAMES for v in (onset, gap, duration)):
        raise ValueError("Timing cell must keep 1 s of lead-in, fit the video and stay token-aligned.")
    return ({**event, "first_event_start_frame": onset, "first_event_end_frame": first_end,
             "second_event_start_frame": second_start, "second_event_end_frame": second_end},
            {**boundary, "boundary_start_frame": first_end, "boundary_end_frame": first_end+gap, "gap_frames": gap})


def native_check(row):
    cell = NATIVE[row["condition"]]
    if schedule(row, cell) != (row["event_timing"], row["boundary_timing"]):
        raise ValueError("Source row is not on the native schedule of its condition.")
    return cell


def retimed_targets(row, event):
    """Target objects with the first mover on Event 1 and the second mover on Event 2."""
    first = row["first_object_id"]
    old = {first: (row["event_timing"]["first_event_start_frame"], row["event_timing"]["first_event_end_frame"]),
           3-first: (row["event_timing"]["second_event_start_frame"], row["event_timing"]["second_event_end_frame"])}
    new = {first: (event["first_event_start_frame"], event["first_event_end_frame"]),
           3-first: (event["second_event_start_frame"], event["second_event_end_frame"])}
    targets = copy.deepcopy(row["target_objects"])
    if sorted(o["id"] for o in targets) != [1, 2]:
        raise ValueError("Need both persistent target identities.")
    for obj in targets:
        if (obj["start_frame"], obj["end_frame"]) != old[obj["id"]]:
            raise ValueError("Target motion does not match the source event timing.")
        obj["start_frame"], obj["end_frame"] = new[obj["id"]]
    return targets


def render_spec(row, cell):
    """Renderer inputs for one cell; at the native cell this equals the Phase 3E reconstruction spec."""
    event, boundary = schedule(row, cell)
    objects = []
    for obj in retimed_targets(row, event):
        static = {k: copy.deepcopy(v) for k, v in obj.items() if not (k == "radius" and v is None)}
        objects.append({**static, "motion": renderer.make_motion(obj, obj["start_frame"], obj["end_frame"])})
    distractors = [{**{k: copy.deepcopy(v) for k, v in d.items() if not (k == "radius" and v is None)},
                    "path": {k: d[k] for k in ("from", "to", "direction")}} for d in row["distractors"]]
    timing = {**event, **boundary, "total_frames": row["total_frames"]}
    return {"objects": objects, "distractors": renderer.timed_distractors(distractors, timing), "timing": timing}


def render(path, row, cell):
    spec = render_spec(row, cell)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    renderer.render_video(path, row["condition"], row["fps"], row["duration_sec"],
                          spec["objects"], spec["distractors"], spec["timing"])
    return spec


def cell_row(row, cell, path):
    """Evaluation row for a re-timed video; prompt text, objects and answer are unchanged."""
    event, boundary = schedule(row, cell)
    result = {k: copy.deepcopy(v) for k, v in row.items() if not k.startswith(("archived_", "phase3"))}
    result.update(target_objects=retimed_targets(row, event), event_timing=event, boundary_timing=boundary,
                  eval_id=row["eval_id"]+f"__timing_{cell}", video_path=str(path), video_id=Path(path).name,
                  timing_cell=cell)
    audit_row(result)
    return result


def expected_motion(spec):
    """Frames whose content must change from the previous frame (any rendered object moves)."""
    def pixel(motion, frame):  # draw_shape truncates coordinates to int
        return tuple(int(v) for v in renderer.object_position(motion, frame))

    moving = set()
    for item in spec["objects"]+spec["distractors"]:
        for frame in range(1, spec["timing"]["total_frames"]):
            if pixel(item["motion"], frame) != pixel(item["motion"], frame-1):
                moving.add(frame)
    return moving


def observed_motion(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode {path}.")
    previous, frame_index, moving = None, 0, set()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if previous is not None:
                changed = np.abs(frame.astype(np.int16)-previous.astype(np.int16)).max(axis=2) > MOTION_THRESHOLD
                if int(changed.sum()) > MOTION_PIXELS:
                    moving.add(frame_index)
            previous, frame_index = frame, frame_index+1
    finally:
        capture.release()
    return moving, frame_index


def timing_audit(path, spec):
    """Decoded-frame motion must occur exactly in the scheduled frames; frame count is unchanged."""
    observed, frames = observed_motion(path)
    expected = expected_motion(spec)
    record = {"frames": frames, "expected_motion_frames": len(expected), "observed_motion_frames": len(observed),
              "unexpected_motion_frames": sorted(observed-expected), "missing_motion_frames": sorted(expected-observed),
              "event_timing": {k: spec["timing"][k] for k in ("first_event_start_frame", "first_event_end_frame",
                               "second_event_start_frame", "second_event_end_frame", "gap_frames")},
              "token_bins": {"event_1": [spec["timing"]["first_event_start_frame"]//TOKEN_FRAMES,
                                         spec["timing"]["first_event_end_frame"]//TOKEN_FRAMES],
                             "event_2": [spec["timing"]["second_event_start_frame"]//TOKEN_FRAMES,
                                         spec["timing"]["second_event_end_frame"]//TOKEN_FRAMES]}}
    record["passed"] = (frames == spec["timing"]["total_frames"] and not record["unexpected_motion_frames"]
                        and not record["missing_motion_frames"])
    return record


def prompts(row, cell, video_path, base, cohort, root):
    """Both A/B layouts of one cell; the permuted layout is a pure option swap. Paths are relative to `root`."""
    if cell == NATIVE[row["condition"]]:
        archived = {**{k: copy.deepcopy(v) for k, v in row.items() if not k.startswith(("archived_", "phase3"))},
                    "video_path": str(video_path), "timing_cell": cell}
        audit_row(archived)
    else:
        archived = cell_row(row, cell, video_path)
    result = []
    for view in VIEWS:
        prompt = archived if view == "archived" else {**permuted_row(archived), "timing_cell": cell}
        result.append({"cohort": cohort, "base_sample_id": base, "timing_cell": cell, "view": view,
                       "source_eval_id": row["eval_id"], "row": prompt, "video_sha256": file_hash(Path(root)/video_path)})
    return result


def prepare(sources, output_root, cohort, render_cache=None, cells=None):
    """sources: [{"base_sample_id", "pair_id", "rows": {condition: row}, "videos": {condition: path}}].

    Reconstructs both native videos, requires decoded-exact equality, then renders the new cells (or copies them
    from a checked matched-encoder cache) and writes a frozen plan with every prompt and audit.
    """
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new materials directory; preserve earlier attempts.")
    cache = read_json(Path(render_cache)/"cache_manifest.json") if render_cache else None
    if cache and cache["fingerprint"] != digest({k: v for k, v in cache.items() if k != "fingerprint"}):
        raise ValueError("Render cache manifest changed.")
    chosen = tuple(cells or CELLS)
    output.mkdir(parents=True)
    reconstructions, audits, rows = [], [], []
    for source in sources:
        base, pair = source["base_sample_id"], source["pair_id"]
        for condition in CONDITIONS:
            row, original = source["rows"][condition], Path(source["videos"][condition])
            native = native_check(row)
            expected = row.get("phase3c_video_sha256")
            if expected and file_hash(original) != expected:
                raise ValueError("Source video hash changed.")
            entry = cache["entries"].get(row["eval_id"]) if cache else None
            reconstruction = output/"reconstructions"/f"{row['eval_id']}.mp4"
            reconstruction.parent.mkdir(exist_ok=True)
            if entry:
                cached_copy(render_cache, entry, "reconstruction", reconstruction, row, native, file_hash(original))
            else:
                render(reconstruction, row, native)
            proof = {"base_sample_id": base, "pair_id": pair, "condition": condition,
                     "original_video_sha256": file_hash(original), **decoded_comparison(original, reconstruction)}
            reconstructions.append(proof)
            if not proof["decoded_pixels_exact"]:
                atomic_write(output/"reconstruction_audit.json", {"schema": SCHEMA, "audits": reconstructions,
                                                                  "status": "failed_native_reconstruction"})
                raise ValueError(f"Native reconstruction differs for {pair} {condition}; no timing cell rendered.")
            for cell in (c for c in chosen if CELLS[c][0] == condition):
                video = output/"videos"/f"{row['eval_id']}__timing_{cell}.mp4"
                if cell == native:
                    # The verified original itself; copied so that every prompt video lives inside the materials.
                    video.parent.mkdir(exist_ok=True)
                    shutil.copyfile(original, video)
                    spec = render_spec(row, cell)
                else:
                    if entry:
                        cached_copy(render_cache, entry, cell, video, row, cell, file_hash(original))
                        spec = render_spec(row, cell)
                    else:
                        spec = render(video, row, cell)
                audit = {"base_sample_id": base, "pair_id": pair, "timing_cell": cell,
                         "video_sha256": file_hash(video), "spec_sha256": digest(spec), **timing_audit(video, spec)}
                audits.append(audit)
                if not audit["passed"]:
                    atomic_write(output/"timing_audit.json", {"schema": SCHEMA, "audits": audits, "status": "failed"})
                    raise ValueError(f"Timing audit failed for {pair} {cell}.")
                rows += prompts(row, cell, Path("videos")/video.name, base, cohort, output)
            print("TIMING_MATERIALS", cohort, pair, condition, flush=True)
    plan = {"schema": SCHEMA, "cohort": cohort, "complete": True, "cells": list(chosen),
            "base_ids": [s["base_sample_id"] for s in sources], "pair_ids": [s["pair_id"] for s in sources],
            "rows": rows, "reconstruction_audits": reconstructions, "timing_audits": audits,
            "renderer_sha256": file_hash(renderer.__file__), "materials_code_sha256": file_hash(__file__),
            "opencv_version": cv2.__version__, "render_cache_fingerprint": cache["fingerprint"] if cache else None,
            "outcome_screening": False}
    plan["fingerprint"] = digest(plan)
    frozen_write(output/"plan.json", plan)
    return plan


def cached_copy(cache_root, entry, name, destination, row, cell, original_sha):
    if entry["original_video_sha256"] != original_sha or entry["spec_sha256"][name] != digest(render_spec(row, cell)):
        raise ValueError("Cached render does not correspond to this original and timing specification.")
    candidate = Path(cache_root)/entry["paths"][name]
    if (not candidate.resolve().is_relative_to(Path(cache_root).resolve()) or candidate.is_symlink() or
            file_hash(candidate) != entry["sha256"][name]):
        raise ValueError("Cached render path/checksum changed.")
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate, destination)


def make_render_cache(sources, output_root, cells=NEW_CELLS):
    """Matched-encoder cache for bases whose originals this environment reproduces byte for byte."""
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new render cache directory.")
    output.mkdir(parents=True)
    entries, skipped = {}, []
    for source in sources:
        for condition in CONDITIONS:
            row, original = source["rows"][condition], Path(source["videos"][condition])
            native = native_check(row)
            uid = row["eval_id"]
            if not original.is_file():
                skipped.append(uid)
                continue
            reconstruction = output/f"{uid}__reconstructed.mp4"
            render(reconstruction, row, native)
            if file_hash(reconstruction) != file_hash(original):
                reconstruction.unlink()
                skipped.append(uid)
                continue
            names = ["reconstruction"]+[c for c in cells if CELLS[c][0] == condition]
            paths = {"reconstruction": reconstruction.name}
            for cell in names[1:]:
                video = output/f"{uid}__timing_{cell}.mp4"
                render(video, row, cell)
                paths[cell] = video.name
            entries[uid] = {"original_video_sha256": file_hash(original), "paths": paths,
                            "sha256": {n: file_hash(output/p) for n, p in paths.items()},
                            "spec_sha256": {n: digest(render_spec(row, native if n == "reconstruction" else n))
                                            for n in paths}}
            print("TIMING_CACHE", uid, flush=True)
    manifest = {"schema": "phase3h_matched_encoder_cache_v1", "entries": entries, "skipped_not_byte_exact": skipped,
                "renderer_sha256": file_hash(renderer.__file__), "opencv_version": cv2.__version__,
                "purpose": "render new timing cells with the encoder that produced each base's originals"}
    manifest["fingerprint"] = digest(manifest)
    atomic_write(output/"cache_manifest.json", manifest)
    return manifest


def generate_fresh(output_root, first=1001, last=1050, batch=5):
    """Fresh, unscreened bases from the Phase 3B rescue-pool generator (default seed and settings)."""
    import subprocess
    script = Path(__file__).with_name("generate_phase3b_rescue_pool.py")
    for start in range(first, last+1, batch):
        subprocess.run([sys.executable, str(script), "--output_root", str(Path(output_root).resolve()),
                        "--start_base_id", str(start), "--count", str(batch), "--max_new_bases", str(last-30)],
                       check=True, cwd=PROJECT_ROOT)
        print("FRESH_BATCH", start, flush=True)
    rows = [json.loads(line) for line in (Path(output_root)/"L5_full/annotations.jsonl").read_text().splitlines() if line.strip()]
    bases = sorted({int(r["base_sample_id"]) for r in rows})
    if bases != list(range(first, last+1)):
        raise ValueError("Fresh pool does not contain exactly the requested base IDs.")
    return {"schema": SCHEMA+"_fresh_pool", "bases": len(bases), "rows": len(rows),
            "annotations_sha256": file_hash(Path(output_root)/"L5_full/annotations.jsonl")}


def rescue_sources(pairs_path, resolve):
    """Frozen rescue50 rows; `resolve(row)` returns the archived video path in this environment."""
    pairs = read_json(pairs_path)
    return [{"base_sample_id": int(p[CONDITIONS[0]]["base_sample_id"]), "pair_id": pair_id,
             "rows": {c: p[c] for c in CONDITIONS}, "videos": {c: resolve(p[c]) for c in CONDITIONS}}
            for pair_id, p in pairs.items()]


def fresh_sources(annotations_path, base_ids):
    """Fresh generator rows: the 'original' prompt variant of each base's Low and Temporal videos."""
    rows = [json.loads(line) for line in Path(annotations_path).read_text().splitlines() if line.strip()]
    result = []
    for base in base_ids:
        chosen = {r["condition"]: r for r in rows
                  if int(r["base_sample_id"]) == base and r.get("prompt_variant") == "original"}
        if set(chosen) != set(CONDITIONS):
            raise ValueError(f"Fresh base {base} lacks its original-variant Low/Temporal rows.")
        videos = {c: Path(chosen[c]["video_path"]) for c in CONDITIONS}
        videos = {c: v if v.is_absolute() else PROJECT_ROOT/v for c, v in videos.items()}
        result.append({"base_sample_id": base, "pair_id": f"phase3h_fresh_{base:04d}", "rows": chosen, "videos": videos})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("generate", "fresh", "rescue", "cache"), required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--annotations", help="fresh: generator L5_full/annotations.jsonl")
    parser.add_argument("--base_ids", default="1001-1050", help="fresh: inclusive range")
    parser.add_argument("--pairs", help="rescue/cache: frozen rescue50 pairs.json")
    parser.add_argument("--source_root", help="rescue: Phase 3D run root (project root and path map)")
    parser.add_argument("--originals", help="cache: directory holding <eval_id>__original.mp4")
    parser.add_argument("--render_cache")
    args = parser.parse_args()
    if args.action == "generate":
        first, last = (int(x) for x in args.base_ids.split("-"))
        result = generate_fresh(args.output_root, first, last)
    elif args.action == "fresh":
        first, last = (int(x) for x in args.base_ids.split("-"))
        result = prepare(fresh_sources(args.annotations, range(first, last+1)), args.output_root, "fresh")
    elif args.action == "cache":
        result = make_render_cache(rescue_sources(args.pairs, lambda r: Path(args.originals)/f"{r['eval_id']}__original.mp4"),
                                   args.output_root)
    else:
        from scripts.phase3b_paths import resolve_video_path
        plan = read_json(Path(args.source_root)/"plan.json")
        result = prepare(rescue_sources(args.pairs, lambda r: resolve_video_path(
            r["video_path"], plan["project_root"], plan["source_path_map"])), args.output_root, "rescue",
            args.render_cache, cells=NEW_CELLS)
    print(json.dumps({k: len(v) if isinstance(v, (list, dict)) else v for k, v in result.items()}, indent=2))
