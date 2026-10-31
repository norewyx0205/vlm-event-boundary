"""CPU-only audit of target/distractor co-motion and pre-encoding footprint overlap."""

import argparse
import json
import math
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts import phase3e_paired_design as design
from scripts.phase3c_core import atomic_write, digest, file_hash, read_json, read_jsonl

generator = design.fixture.generator
SCHEMA = "phase3e_target_distractor_interaction_audit_v1"
METRICS = ("co_motion_fraction", "mean_active_distractor_count",
           "minimum_center_distance_px", "footprint_overlap_fraction")


def position(obj, frame):
    # The renderer truncates the interpolated centre to integer pixel coordinates.
    return tuple(int(v) for v in generator.object_position(obj, frame))


@lru_cache(maxsize=32)
def shape_stamp(shape, radius):
    mask = np.full((2*radius+1, 2*radius+1, 3), 255, dtype=np.uint8)
    generator.draw_shape(mask, shape, "black", radius, radius, size=radius)
    result = mask[:, :, 0] == 0
    result.setflags(write=False)
    return result


def footprint_overlap(one, centre_one, two, centre_two):
    """Exact intersection of the generator's raster footprints, before video encoding."""
    r1, r2 = one.get("radius", 28), two.get("radius", 23)
    x1, y1 = centre_one
    x2, y2 = centre_two
    if any(x-radius < 0 or y-radius < 0 or x+radius >= generator.W or y+radius >= generator.H
           for (x, y), radius in ((centre_one, r1), (centre_two, r2))):
        # OpenCV polygon clipping can change boundary rasterization; reproduce it on the full canvas.
        masks = []
        for obj, (x, y), radius in ((one, centre_one, r1), (two, centre_two, r2)):
            mask = np.full((generator.H, generator.W, 3), 255, dtype=np.uint8)
            generator.draw_shape(mask, obj["shape"], "black", x, y, size=radius)
            masks.append(mask[:, :, 0] == 0)
        return int(np.count_nonzero(masks[0] & masks[1]))
    left, right = max(0, x1-r1, x2-r2), min(generator.W-1, x1+r1, x2+r2)
    top, bottom = max(0, y1-r1, y2-r2), min(generator.H-1, y1+r1, y2+r2)
    if right < left or bottom < top:
        return 0
    masks = []
    for obj, (x, y), radius in ((one, centre_one, r1), (two, centre_two, r2)):
        stamp = shape_stamp(obj["shape"], radius)
        masks.append(stamp[top-(y-radius):bottom-(y-radius)+1, left-(x-radius):right-(x-radius)+1])
    return int(np.count_nonzero(masks[0] & masks[1]))


def target_event(row, target):
    start, end = target["start_frame"], target["end_frame"]
    if not 0 <= start < end < row["total_frames"]:
        raise ValueError("Target event is outside the video or has no motion intervals.")
    frames = list(range(start, end))
    centres = {f: position(target, f) for f in frames}
    active_counts = {f: 0 for f in frames}
    overlap_any = {f: False for f in frames}
    details = []
    for dist in row["distractors"]:
        distances, overlaps, active = [], [], []
        for frame in frames:
            point = position(dist, frame)
            moving = point != position(dist, frame+1)
            pixels = footprint_overlap(target, centres[frame], dist, point)
            distances.append(math.dist(centres[frame], point))
            overlaps.append(pixels)
            active.append(moving)
            active_counts[frame] += int(moving)
            overlap_any[frame] |= pixels > 0
        details.append({"distractor_id": dist["id"], "label": dist["label"],
            "motion_kind": dist["motion_kind"], "start_frame": dist["start_frame"], "end_frame": dist["end_frame"],
            "same_shape": target["shape"] == dist["shape"], "same_color": target["color"] == dist["color"],
            "active_frames": [f for f, moving in zip(frames, active) if moving],
            "active_fraction": sum(active)/len(frames),
            "minimum_center_distance_px": min(distances), "mean_center_distance_px": float(np.mean(distances)),
            "minimum_distance_frame": frames[int(np.argmin(distances))],
            "footprint_overlap_frames": [f for f, pixels in zip(frames, overlaps) if pixels],
            "footprint_overlap_fraction": sum(p > 0 for p in overlaps)/len(frames),
            "maximum_overlap_pixels": max(overlaps),
            "co_motion_overlap_frames": [f for f, moving, pixels in zip(frames, active, overlaps) if moving and pixels]})
    return {"scene_seed": row["scene_seed"], "video_id": row["video_id"], "condition": row["condition"],
        "first_object_id": row["first_object_id"], "target_id": target["id"], "target_label": target["label"],
        "event_role": "first" if target["id"] == row["first_object_id"] else "second",
        "start_frame": start, "end_frame_exclusive": end, "motion_interval_count": len(frames),
        "active_distractor_ids": [d["distractor_id"] for d in details if d["active_frames"]],
        "co_motion_frames": [f for f in frames if active_counts[f]],
        "co_motion_fraction": sum(v > 0 for v in active_counts.values())/len(frames),
        "mean_active_distractor_count": float(np.mean(list(active_counts.values()))),
        "minimum_center_distance_px": min(d["minimum_center_distance_px"] for d in details),
        "footprint_overlap_fraction": sum(overlap_any.values())/len(frames),
        "footprint_overlap_frames": [f for f in frames if overlap_any[f]], "distractors": details}


def pair_record(left, right, comparison):
    if left["scene_seed"] != right["scene_seed"] or left["target_id"] != right["target_id"]:
        raise ValueError("Interaction pairs must keep the same named target and scene.")
    return {"scene_seed": left["scene_seed"], "target_id": left["target_id"], "comparison": comparison,
        "left_video_id": left["video_id"], "right_video_id": right["video_id"],
        "left_active_distractor_ids": left["active_distractor_ids"],
        "right_active_distractor_ids": right["active_distractor_ids"],
        "active_identity_set_changed": left["active_distractor_ids"] != right["active_distractor_ids"],
        "metrics": {m: {"left": left[m], "right": right[m], "right_minus_left": right[m]-left[m]}
                    for m in METRICS}}


def summarize_pairs(pairs):
    families = defaultdict(list)
    for pair in pairs:
        families[pair["scene_seed"]].append(pair)
    metrics = {}
    for metric in METRICS:
        levels = [{k: float(np.mean([p["metrics"][metric][k] for p in rows]))
                   for k in ("left", "right", "right_minus_left")} for rows in families.values()]
        metrics[metric] = {k: float(np.mean([r[k] for r in levels]))
                           for k in ("left", "right", "right_minus_left")}
        deltas = [r["right_minus_left"] for r in levels]
        metrics[metric].update(families_negative=sum(d < -1e-9 for d in deltas),
            families_positive=sum(d > 1e-9 for d in deltas), families_zero=sum(abs(d) <= 1e-9 for d in deltas),
            target_pairs_changed=sum(abs(p["metrics"][metric]["right_minus_left"]) > 1e-9 for p in pairs))
    return {"independent_scene_families": len(families), "target_pairs": len(pairs), "metrics": metrics,
        "active_identity_sets_changed": sum(p["active_identity_set_changed"] for p in pairs)}


def audit(plan):
    rows = plan["rows"]
    seeds = sorted({r["scene_seed"] for r in rows})
    legacy = plan.get("schema") in (design.LEGACY_SCHEMA, design.fixture.LEGACY_SCHEMA)
    design.validate_materials(rows, seeds, legacy_absolute=legacy)
    unique = {}
    for row in rows:
        unique.setdefault(row["video_id"], row)
    events = [target_event(row, target) for row in unique.values() for target in row["target_objects"]]
    lookup = {(e["scene_seed"], e["condition"], e["target_id"], e["event_role"]): e for e in events}
    order, boundary = [], []
    for seed in seeds:
        for target in (1, 2):
            for condition in design.fixture.CONDITIONS:
                order.append({**pair_record(lookup[seed, condition, target, "first"],
                    lookup[seed, condition, target, "second"], "same_target_first_to_second"), "condition": condition})
            for role in ("first", "second"):
                boundary.append({**pair_record(lookup[seed, "low_boundary", target, role],
                    lookup[seed, "temporal_boundary", target, role], "same_target_low_to_temporal"), "event_role": role})
    summary = {"order_first_to_second": {c: summarize_pairs([p for p in order if p["condition"] == c])
                                         for c in design.fixture.CONDITIONS},
               "boundary_low_to_temporal": {role: summarize_pairs([p for p in boundary if p["event_role"] == role])
                                             for role in ("first", "second")}}
    by_role = []
    for condition in design.fixture.CONDITIONS:
        for role in ("first", "second"):
            selected = [e for e in events if e["condition"] == condition and e["event_role"] == role]
            by_role.append({"condition": condition, "event_role": role, "target_events": len(selected),
                "independent_scene_families": len(seeds), "events_with_co_motion": sum(bool(e["co_motion_frames"]) for e in selected),
                "mean_co_motion_fraction": float(np.mean([e["co_motion_fraction"] for e in selected])),
                "events_with_footprint_overlap": sum(bool(e["footprint_overlap_frames"]) for e in selected),
                "mean_footprint_overlap_fraction": float(np.mean([e["footprint_overlap_fraction"] for e in selected])),
                "mean_minimum_center_distance_px": float(np.mean([e["minimum_center_distance_px"] for e in selected]))})
    co_motion = summary["boundary_low_to_temporal"]["second"]["metrics"]["co_motion_fraction"]
    systematic = max(co_motion["families_negative"], co_motion["families_positive"]) == len(seeds)
    return {"schema": SCHEMA, "status": "cpu_material_audit_complete", "plan_fingerprint": plan.get("fingerprint"),
        "rows_sha256": digest(rows), "independent_scene_families": len(seeds), "videos": len(unique),
        "target_events": len(events), "event_distractor_comparisons": sum(len(e["distractors"]) for e in events),
        "frame_convention": "start <= f < end; active means the rendered centre changes from f to f+1",
        "geometry": "integer-centre generator raster footprints, before MP4 compression; no near-distance cutoff",
        "draw_order": "distractors are drawn before targets; footprint overlap is not target occlusion",
        "model_inference_performed": False, "materials_modified": False, "co_motion_by_condition_and_role": by_role,
        "paired_summary": summary, "events": events, "order_pairs": order, "boundary_pairs": boundary,
        "systematic_second_event_co_motion_change": systematic,
        "distractor_timing_policy": "legacy_fixed_absolute" if legacy else "event_relative_near_targets",
        "event_relative_schedule_verified": not legacy,
        "interpretation": ("Absolute distractor schedules are fixed, but event-relative distractor motion is not. "
            "Order/boundary contrasts include dynamic-context changes; this audit does not measure their effect on the model."
            if legacy else "Distractors retain their identities, paths and event slots. Their motion windows follow the "
            "current boundary's target events; order/text variants within a condition use identical schedules. "
            "This audit measures stimulus matching, not model effects.")}


def report_text(result):
    lines = ["# Phase 3E-B Target-Distractor Interaction Audit", "", "## Material Passport", "",
        "- CPU metadata/raster-geometry audit, not model inference or a behavioural result.",
        f"- {result['independent_scene_families']} families; {result['videos']} planned videos; {result['target_events']} target events.",
        "- Original video specifications, labels, folds and scene seeds are unchanged.", "",
        "## Event-Relative Motion", "",
        "| Boundary | Target role | Events with co-motion | Mean co-motion fraction | Events with footprint overlap |",
        "| --- | --- | ---: | ---: | ---: |"]
    for row in result["co_motion_by_condition_and_role"]:
        lines.append(f"| {row['condition']} | {row['event_role']} | {row['events_with_co_motion']}/{row['target_events']} | "
            f"{row['mean_co_motion_fraction']:.3f} | {row['events_with_footprint_overlap']}/{row['target_events']} |")
    lines += ["", "## Paired Changes", "",
        "Each row averages within scene family first. Delta is right minus left.", "",
        "| Comparison | Metric | Left | Right | Delta | Families negative / zero / positive |",
        "| --- | --- | ---: | ---: | ---: | --- |"]
    for kind, groups in result["paired_summary"].items():
        for group, summary in groups.items():
            for metric, values in summary["metrics"].items():
                lines.append(f"| {kind}: {group} | {metric} | {values['left']:.3f} | {values['right']:.3f} | "
                    f"{values['right_minus_left']:+.3f} | {values['families_negative']} / {values['families_zero']} / {values['families_positive']} |")
    lines += ["", "## Interpretation", "", result["interpretation"], "",
        "Spatial proximity is continuous centre distance. Footprint overlap uses the actual generator shape primitive, "
        "not bounding-box overlap, codec pixels, attention weights or a measured interference effect.", "",
        "Full frame lists, distractor identities, raw paired endpoints and source hashes are in interaction_audit.json.", ""]
    return "\n".join(lines)


def save(root, plan):
    root = Path(root)
    result = audit(plan)
    result["code_sha256"] = {Path(p).name: file_hash(p) for p in
        (__file__, design.__file__, design.fixture.__file__, generator.__file__)}
    result["record_sha256"] = digest(result)
    atomic_write(root / "interaction_audit.json", result)
    (root / "interaction_audit.md").write_text(report_text(result), encoding="utf-8")
    return result


def historical_audit(paths):
    """Inspect saved motion windows only; never substitute today's generator for missing history."""
    sources = []
    for path in paths:
        path = Path(path).resolve()
        before = file_hash(path)
        rows = [r for r in read_jsonl(path) if r.get("difficulty_level") == 5 and
                r.get("condition") in design.fixture.CONDITIONS]
        videos = {}
        for row in rows:
            key = (row["base_sample_id"], row["condition"], row["video_id"])
            if key in videos and any(row[k] != videos[key][k] for k in ("target_objects", "distractors")):
                raise ValueError("Historical prompt views disagree on video annotations.")
            videos.setdefault(key, row)
        records = []
        for row in videos.values():
            moving = [d for d in row["distractors"] if d["motion_kind"] != "static"]
            missing = [d["id"] for d in moving if any(k not in d for k in ("start_frame", "end_frame"))]
            item = {"video_id": row["video_id"], "base_sample_id": row["base_sample_id"],
                "condition": row["condition"], "moving_distractors": len(moving),
                "missing_moving_window_ids": missing, "events": []}
            if not missing:
                for target in row["target_objects"]:
                    start, end = target["start_frame"], target["end_frame"]
                    frames = list(range(start, end))
                    active = {f: [d["id"] for d in moving if position(d, f) != position(d, f+1)] for f in frames}
                    item["events"].append({"target_id": target["id"],
                        "event_role": "first" if target["id"] == row["first_object_id"] else "second",
                        "co_motion_fraction": sum(bool(ids) for ids in active.values())/len(frames),
                        "active_distractor_ids": sorted({d for ids in active.values() for d in ids})})
            records.append(item)
        by_condition = {}
        for condition in design.fixture.CONDITIONS:
            selected = [v for v in records if v["condition"] == condition]
            by_condition[condition] = {"videos": len(selected),
                "videos_with_saved_moving_windows": sum(not v["missing_moving_window_ids"] for v in selected),
                "videos_missing_moving_windows": sum(bool(v["missing_moving_window_ids"]) for v in selected),
                "events_with_co_motion": {role: sum(e["co_motion_fraction"] > 0 for v in selected
                    for e in v["events"] if e["event_role"] == role) for role in ("first", "second")},
                "events_with_known_windows": {role: sum(e["event_role"] == role for v in selected
                    for e in v["events"]) for role in ("first", "second")}}
        after = file_hash(path)
        if before != after:
            raise ValueError("Historical annotations changed during the read-only audit.")
        sources.append({"path": str(path), "sha256": before, "source_bytes_unchanged": True,
            "prompt_rows": len(rows), "videos": len(videos), "by_condition": by_condition, "records": records})
    return {"schema": "phase3e_historical_annotation_timing_audit_v1", "sources": sources,
        "model_inference_performed": False, "historical_data_modified": False,
        "missing_times_reconstructed": False,
        "interpretation": "Missing saved distractor windows remain unknown. motion_timing labels and current generator "
            "code alone do not verify historical rendered motion or establish whether behavioural conclusions are affected."}


def save_history(root, paths):
    root = Path(root)
    result = historical_audit(paths)
    result["code_sha256"] = file_hash(__file__)
    result["record_sha256"] = digest(result)
    atomic_write(root / "historical_annotation_audit.json", result)
    lines = ["# Historical Level-5 Annotation Timing Audit", "", "## Material Passport", "",
        "Read-only CPU inspection of saved annotations. No historical videos were decoded or regenerated.", "",
        "| Source | Condition | Unique videos | Saved moving windows | Missing moving windows |",
        "| --- | --- | ---: | ---: | ---: |"]
    for source in result["sources"]:
        for condition, counts in source["by_condition"].items():
            lines.append(f"| {source['path']} | {condition} | {counts['videos']} | "
                f"{counts['videos_with_saved_moving_windows']} | {counts['videos_missing_moving_windows']} |")
    lines += ["", "## Interpretation", "", result["interpretation"], "",
        "Per-video known event overlap, missing distractor IDs and source SHA256 hashes are preserved in the JSON.", ""]
    (root / "historical_annotation_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--plan_path")
    inputs.add_argument("--historical_annotations", nargs="+")
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("Use a new output directory; do not overwrite frozen materials/audits.")
    root.mkdir(parents=True, exist_ok=True)
    if args.historical_annotations:
        result = save_history(root, args.historical_annotations)
        print(json.dumps({"sources": len(result["sources"]), "interpretation": result["interpretation"]}, indent=2))
        sys.exit(0)
    plan = read_json(args.plan_path)
    design.validate_plan(plan, allow_legacy=True)
    result = save(root, plan)
    print(json.dumps({k: v for k, v in result.items() if k not in ("events", "order_pairs", "boundary_pairs")}, indent=2))
