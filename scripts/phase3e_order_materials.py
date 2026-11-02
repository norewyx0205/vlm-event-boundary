"""Reconstruct frozen rescue50 stimuli before rendering order-reversal partners."""

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from scripts import generate_ladder_dataset as renderer
from scripts.phase3b_paths import resolve_video_path
from scripts.phase3c_core import atomic_write, digest, file_hash, frozen_write, read_json
from scripts.phase3d_core import audit_pairs, audit_row
from scripts.phase3d_execution import permuted_row

SCHEMA = "phase3e_rescue50_order_materials_v1"
CONDITIONS = ("low_boundary", "temporal_boundary")


def render_spec(row, reverse=False):
    if row["condition"] not in CONDITIONS or row["feature_variant"] != "full":
        raise ValueError("Only frozen full-feature Low/Temporal videos are supported.")
    targets = row["target_objects"]
    if len(targets) != 2 or {o["id"] for o in targets} != {1, 2}:
        raise ValueError("Need both persistent target identities.")
    by_id = {o["id"]: o for o in targets}
    objects = []
    for obj in targets:
        timing = by_id[3-obj["id"]] if reverse else obj
        if obj["end_frame"]-obj["start_frame"] != timing["end_frame"]-timing["start_frame"]:
            raise ValueError("Order reversal must retain each motion duration.")
        static = {k: copy.deepcopy(v) for k, v in obj.items() if not (k == "radius" and v is None)}
        objects.append({**static, "motion": renderer.make_motion(obj,
            timing["start_frame"], timing["end_frame"])})
    dist = [{**{k: copy.deepcopy(v) for k, v in d.items() if not (k == "radius" and v is None)},
             "path": {k: d[k] for k in ("from", "to", "direction")}}
            for d in row["distractors"]]
    timing = {**row["event_timing"], **row["boundary_timing"], "total_frames": row["total_frames"]}
    return {"objects": objects, "distractors": renderer.timed_distractors(dist, timing), "timing": timing}


def reversed_row(row, path):
    result = {k: copy.deepcopy(v) for k, v in row.items() if not k.startswith(("archived_", "phase3"))}
    original = {o["id"]: o for o in row["target_objects"]}
    for obj in result["target_objects"]:
        other = original[3-obj["id"]]
        obj.update(start_frame=other["start_frame"], end_frame=other["end_frame"])
    result.update(first_object_id=row["second_object_id"], second_object_id=row["first_object_id"],
        event_1=row.get("event_2"), event_2=row.get("event_1"),
        correct_option={"A": "B", "B": "A"}[row["correct_option"]],
        correct_relation={"before": "after", "after": "before"}[row["correct_relation"]],
        incorrect_relation=row["correct_relation"], eval_id=row["eval_id"]+"__order_reversed",
        video_path=str(path), video_id=Path(path).name)
    result["correct_sentence"] = result["option_"+result["correct_option"]]
    result["incorrect_sentence"] = result["option_"+row["correct_option"]]
    audit_row(result)
    return result


def decoded_comparison(left, right):
    a, b = cv2.VideoCapture(str(left)), cv2.VideoCapture(str(right))
    if not a.isOpened() or not b.isOpened():
        raise ValueError("Cannot decode an original/reconstructed video.")
    count, different, max_error, absolute_error, pixels = 0, 0, 0, 0, 0
    hashes = [hashlib.sha256(), hashlib.sha256()]
    try:
        if abs(a.get(cv2.CAP_PROP_FPS)-b.get(cv2.CAP_PROP_FPS)) > 1e-6:
            raise ValueError("Reconstructed video FPS differs.")
        while True:
            aa, x = a.read()
            bb, y = b.read()
            if aa != bb:
                raise ValueError("Reconstructed video frame count differs.")
            if not aa:
                break
            if x.shape != y.shape:
                raise ValueError("Reconstructed video dimensions differ.")
            hashes[0].update(x.tobytes())
            hashes[1].update(y.tobytes())
            delta = np.abs(x.astype(np.int16)-y.astype(np.int16))
            different += int(np.any(delta))
            max_error = max(max_error, int(delta.max()))
            absolute_error += int(delta.sum())
            pixels += delta.size
            count += 1
    finally:
        a.release()
        b.release()
    if not count:
        raise ValueError("Empty original/reconstruction.")
    return {"frames": count, "different_frames": different, "decoded_pixels_exact": different == 0,
        "max_channel_error": max_error, "mean_absolute_channel_error": absolute_error/pixels,
        "original_decoded_sha256": hashes[0].hexdigest(), "reconstruction_decoded_sha256": hashes[1].hexdigest(),
        "encoded_bytes_exact": file_hash(left) == file_hash(right)}


def render(path, row, reverse=False):
    spec = render_spec(row, reverse)
    renderer.render_video(path, row["condition"], row["fps"], row["duration_sec"],
        spec["objects"], spec["distractors"], spec["timing"])


def prepare(source_root, store_path, output_root, preflight_only=False, render_cache=None):
    from scripts.phase3e_rescue50_probes import load_store
    source, output = Path(source_root).resolve(), Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new materials directory; preserve earlier audit attempts.")
    manifest, preparation, _, _, _ = load_store(store_path)
    old = read_json(source/"plan.json")
    pairs = read_json(source/"pairs.json")
    if (file_hash(source/"pairs.json") != old["source_files"]["pairs.json"] or
            old["fingerprint"] != preparation["source_plan_fingerprint"] or
            audit_pairs(pairs) != preparation["case_metadata"]):
        raise ValueError("Original rescue50 source does not match the verified token store.")
    output.mkdir(parents=True)
    (output/"reconstructions").mkdir()
    (output/"videos").mkdir()
    cache = read_json(Path(render_cache)/"cache_manifest.json") if render_cache else None
    if cache and cache["fingerprint"] != digest({k: v for k, v in cache.items() if k != "fingerprint"}):
        raise ValueError("Render cache manifest changed.")
    ids = [next(c["pair_id"] for c in preparation["case_metadata"] if c["first_object_id"] == first)
           for first in (1, 2)] if preflight_only else preparation["case_ids"]
    audits, rows = [], []
    for pair_id in ids:
        for condition in CONDITIONS:
            original = pairs[pair_id][condition]
            original_path = resolve_video_path(original["video_path"], old["project_root"], old["source_path_map"])
            if file_hash(original_path) != original["phase3c_video_sha256"]:
                raise ValueError("Original video hash changed.")
            reconstruction = output/"reconstructions"/(original["eval_id"]+".mp4")
            cached = cache["entries"].get(original["eval_id"]) if cache else None
            if cached:
                if cached["original_video_sha256"] != file_hash(original_path) or cached["reversed_spec_sha256"] != digest(render_spec(original, True)):
                    raise ValueError("Cached video does not correspond to the same original and reversed motion specification.")
                for name in ("reconstruction", "reversed"):
                    candidate = Path(render_cache)/cached[name+"_path"]
                    if (not candidate.resolve().is_relative_to(Path(render_cache).resolve()) or candidate.is_symlink() or
                            file_hash(candidate) != cached[name+"_sha256"]):
                        raise ValueError("Cached render path/checksum changed.")
                shutil.copyfile(Path(render_cache)/cached["reconstruction_path"], reconstruction)
            else:
                render(reconstruction, original)
            proof = decoded_comparison(original_path, reconstruction)
            record = {"pair_id": pair_id, "condition": condition, "original_video_sha256": file_hash(original_path),
                "reconstruction_video_sha256": file_hash(reconstruction), **proof,
                "distractor_schedule": [d["motion"] for d in render_spec(original)["distractors"]]}
            audits.append(record)
            atomic_write(output/"reconstruction_audit.json", {"schema": SCHEMA, "audits": audits,
                "all_decoded_pixels_exact": all(r["decoded_pixels_exact"] for r in audits),
                "status": "running" if proof["decoded_pixels_exact"] else "failed_original_reconstruction"})
            if not proof["decoded_pixels_exact"]:
                raise ValueError("Original reconstruction does not match decoded frames; no GPU/reversal capture permitted.")
            reverse_path = output/"videos"/(original["eval_id"]+"__order_reversed.mp4")
            if cached:
                shutil.copyfile(Path(render_cache)/cached["reversed_path"], reverse_path)
            else:
                render(reverse_path, original, True)
            reverse = reversed_row(original, Path("videos")/reverse_path.name)
            for view in ("archived", "option_order_permuted"):
                row = reverse if view == "archived" else permuted_row(reverse)
                rows.append({"pair_id": pair_id, "condition": condition, "view": view,
                    "base_sample_id": original["base_sample_id"], "original_eval_id": next(t["eval_id"] for t in preparation["tasks"]
                        if t["pair_id"] == pair_id and t["condition"] == condition and t["view"] == view),
                    "row": row, "video_sha256": file_hash(reverse_path)})
            print("MATCHED_AND_REVERSED", pair_id, condition, flush=True)
    plan = {"schema": SCHEMA, "complete": not preflight_only, "preflight_only": preflight_only,
        "source_root": str(source), "source_store_sha256": file_hash(store_path),
        "source_preparation_fingerprint": preparation["fingerprint"], "source_plan_fingerprint": old["fingerprint"],
        "case_ids": ids, "outer_folds": preparation["outer_folds"], "settings": preparation["settings"],
        "source_runtime": preparation["source_runtime"], "rows": rows, "reconstruction_audits": audits,
        "renderer_sha256": file_hash(renderer.__file__), "materials_code_sha256": file_hash(__file__),
        "opencv_version": cv2.__version__, "new_cases": 0, "new_KO": 0}
    plan["render_cache_fingerprint"] = cache["fingerprint"] if cache else None
    plan["fingerprint"] = digest(plan)
    frozen_write(output/"plan.json", plan)
    return plan


def make_render_cache(pairs_path, source_root, output_root):
    pairs, source_root, output = read_json(pairs_path), Path(source_root), Path(output_root).resolve()
    prior = read_json(source_root/"reconstruction_audit.json")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new render cache directory.")
    output.mkdir(parents=True)
    entries = {}
    for proof in prior["audits"]:
        if not proof["decoded_pixels_exact"] or not proof["encoded_bytes_exact"]:
            continue
        row = pairs[proof["pair_id"]][proof["condition"]]
        uid = row["eval_id"]
        original = source_root/(uid+"__original.mp4")
        if file_hash(original) != row["phase3c_video_sha256"]:
            raise ValueError("Cached original bytes changed.")
        reconstruction, reverse = output/(uid+"__reconstructed.mp4"), output/(uid+"__order_reversed.mp4")
        render(reconstruction, row)
        if file_hash(reconstruction) != row["phase3c_video_sha256"]:
            raise ValueError("Local encoder no longer reproduces original bytes exactly.")
        render(reverse, row, True)
        entries[uid] = {"original_video_sha256": row["phase3c_video_sha256"],
            "reconstruction_path": reconstruction.name, "reconstruction_sha256": file_hash(reconstruction),
            "reversed_path": reverse.name, "reversed_sha256": file_hash(reverse),
            "reversed_spec_sha256": digest(render_spec(row, True))}
    manifest = {"schema": "phase3e_order_matched_encoder_cache_v1", "entries": entries,
        "source_pairs_sha256": file_hash(pairs_path), "renderer_sha256": file_hash(renderer.__file__),
        "opencv_version": cv2.__version__, "purpose": "retain the original encoder within every video pair"}
    manifest["fingerprint"] = digest(manifest)
    atomic_write(output/"cache_manifest.json", manifest)
    return manifest


def audit_local_backup(pairs_path, manifest_path, output_root, source_store=None):
    """Use checked original video bytes while the GPU VM is unavailable."""
    from scripts.recover_phase3e_rescue50 import CheckedBackup
    pairs, output = read_json(pairs_path), Path(output_root).resolve()
    cases = audit_pairs(pairs)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Keep previous reconstruction attempts; use a new output directory.")
    output.mkdir(parents=True)
    preparation = None
    if source_store:
        from scripts.phase3e_rescue50_probes import load_store
        _, preparation, _, _, _ = load_store(source_store)
        if cases != preparation["case_metadata"]:
            raise ValueError("Original backup cases differ from the frozen rescue50 store.")
    ids = preparation["case_ids"] if preparation else [
        next(c["pair_id"] for c in cases if c["first_object_id"] == first) for first in (1, 2)]
    audits, rows = [], []
    if preparation:
        (output/"videos").mkdir()
    with CheckedBackup(manifest_path) as backup:
        by_hash = {}
        for name, entry in backup.files.items():
            if name.endswith(".mp4"):
                by_hash.setdefault(entry["sha256"], []).append(name)
        for identity in ids:
            for condition in CONDITIONS:
                row = pairs[identity][condition]
                candidates = by_hash.get(row["phase3c_video_sha256"], [])
                if not candidates:
                    raise ValueError("Original video missing from checked backup.")
                original = output/(row["eval_id"]+"__original.mp4")
                original.write_bytes(backup.read(sorted(candidates)[0]))
                reconstructed = output/(row["eval_id"]+"__reconstructed.mp4")
                render(reconstructed, row)
                proof = decoded_comparison(original, reconstructed)
                audits.append({"pair_id": identity, "condition": condition,
                    "original_video_sha256": file_hash(original), **proof})
                print("LOCAL_RECONSTRUCTION", identity, condition, json.dumps(proof), flush=True)
                if preparation:
                    if not proof["decoded_pixels_exact"]:
                        atomic_write(output/"reconstruction_audit.json", {"audits": audits, "status": "failed"})
                        raise ValueError("Original frames differ; full order-reversal plan cannot be frozen.")
                    relative = Path("videos")/(row["eval_id"]+"__order_reversed.mp4")
                    render(output/relative, row, True)
                    reverse = reversed_row(row, relative)
                    for view in ("archived", "option_order_permuted"):
                        value = reverse if view == "archived" else permuted_row(reverse)
                        rows.append({"pair_id": identity, "condition": condition, "view": view,
                            "base_sample_id": row["base_sample_id"], "row": value,
                            "original_eval_id": next(t["eval_id"] for t in preparation["tasks"] if
                                t["pair_id"] == identity and t["condition"] == condition and t["view"] == view),
                            "video_sha256": file_hash(output/relative)})
        result = {"schema": SCHEMA, "stage": "local_original_reconstruction", "audits": audits,
            "all_decoded_pixels_exact": all(r["decoded_pixels_exact"] for r in audits),
            "backup_receipt": backup.receipt(), "renderer_sha256": file_hash(renderer.__file__),
            "opencv_version": cv2.__version__, "source_pairs_sha256": file_hash(pairs_path),
            "GPU_started": False}
        atomic_write(output/"reconstruction_audit.json", result)
        if preparation:
            plan = {"schema": SCHEMA, "complete": True, "preflight_only": False,
                "source_store_sha256": file_hash(source_store),
                "source_preparation_fingerprint": preparation["fingerprint"],
                "source_plan_fingerprint": preparation["source_plan_fingerprint"],
                "case_ids": ids, "outer_folds": preparation["outer_folds"], "settings": preparation["settings"],
                "source_runtime": preparation["source_runtime"], "rows": rows, "reconstruction_audits": audits,
                "renderer_sha256": file_hash(renderer.__file__), "materials_code_sha256": file_hash(__file__),
                "opencv_version": cv2.__version__, "new_cases": 0, "new_KO": 0,
                "backup_manifest_sha256": backup.manifest_sha256}
            plan["fingerprint"] = digest(plan)
            frozen_write(output/"plan.json", plan)
            result["plan_fingerprint"] = plan["fingerprint"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root")
    parser.add_argument("--source_store")
    parser.add_argument("--pairs_path")
    parser.add_argument("--backup_manifest")
    parser.add_argument("--render_cache")
    parser.add_argument("--make_render_cache_from")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--preflight_only", action="store_true")
    args = parser.parse_args()
    if args.make_render_cache_from:
        result = make_render_cache(args.pairs_path, args.make_render_cache_from, args.output_root)
        print(json.dumps({"entries": len(result["entries"]), "fingerprint": result["fingerprint"]}))
    elif args.backup_manifest:
        result = audit_local_backup(args.pairs_path, args.backup_manifest, args.output_root, args.source_store)
        print(json.dumps({"all_decoded_pixels_exact": result["all_decoded_pixels_exact"]}))
    else:
        if not args.source_root or not args.source_store:
            parser.error("Need source_root and source_store for material preparation.")
        result = prepare(args.source_root, args.source_store, args.output_root, args.preflight_only, args.render_cache)
        print(json.dumps({k: result[k] for k in ("schema", "complete", "case_ids", "fingerprint")}, indent=2))
