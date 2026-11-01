"""Independently replay recovered tokens using original union groups, not pooling."""

import argparse
import json
import sys
from pathlib import Path, PurePosixPath

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.phase3c_core import atomic_write, file_hash, read_json
from scripts.phase3d_core import statement
from scripts.recover_phase3e_rescue50 import CheckedBackup, LAYERS, OUTPUT_SCHEMA, load_plan


def reference_position(index, row, option):
    """Split original option sections and intersect the reference identity group."""
    runs = []
    for p in sorted(index["group_positions"]["options_all"]):
        if not runs or p != runs[-1][-1]+1:
            runs.append([])
        runs[-1].append(p)
    if len(runs) != 2:
        raise ValueError("Cannot independently resolve the two contextual option spans.")
    reference = statement(row, row["option_"+option])["reference_id"]
    candidates = set(runs[0 if option == "A" else 1]) & set(
        index["group_positions"][f"option_target_{reference}_mentions"])
    if not candidates:
        raise ValueError("Original reference-object token support is absent.")
    return max(candidates)


def load_partial(path):
    path = Path(path).resolve()
    manifest = read_json(path)
    if (manifest.get("schema") != OUTPUT_SCHEMA or manifest.get("complete") is not False or
            manifest.get("ready_for_formal_probe_training") is not False or
            manifest.get("recovered_prompt_cells") != 100 or manifest.get("missing_prompt_cells") != 100 or
            manifest.get("recovered_statement_rows") != 200 or manifest.get("shape") != [200, 4, 4096] or
            manifest.get("missing_by_view") != {"option_order_permuted": 100}):
        raise ValueError("Require the complete archived-only recovery, not a formal mixed-view training store.")
    resources = {"report.md", "examples.json", "decisions.json", "recovered_prompts.json", "readout_maps.json",
                 "layer_replay.json", "backup_receipts.json", "missing_capture_tasks.json", "recovered_features.npz"}
    if set(manifest["files"]) != resources:
        raise ValueError("Missing or unbound recovered resources.")
    for name, sha in manifest["files"].items():
        target = path.parent/name
        if Path(name).name != name or target.is_symlink() or file_hash(target) != sha:
            raise ValueError("Recovered resource path/hash changed.")
    with np.load(path.parent/"recovered_features.npz", allow_pickle=False) as payload:
        features = payload["features"]
    if features.shape != (200, 4, 4096) or features.dtype != np.float32 or not np.isfinite(features).all():
        raise ValueError("Invalid recovered numeric array.")
    return manifest, features


def verify(plan_path, recovery_manifest, output_path):
    plan_path, recovery_manifest = Path(plan_path).resolve(), Path(recovery_manifest).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise ValueError("Use a new verification receipt path.")
    if output_path in (plan_path, recovery_manifest) or output_path.is_relative_to(recovery_manifest.parent):
        raise ValueError("Keep independent verification outside the immutable recovery directory.")
    plan = load_plan(plan_path)
    manifest, features = load_partial(recovery_manifest)
    if manifest["preparation_sha256"] != file_hash(plan_path) or manifest["preparation_fingerprint"] != plan["fingerprint"]:
        raise ValueError("Recovery belongs to a different preparation.")
    root = recovery_manifest.parent
    examples = read_json(root/"examples.json")
    expected = [e for e in plan["statement_examples"] if e["view"] == "archived"]
    missing = read_json(root/"missing_capture_tasks.json")
    if (examples != expected or
            {t["eval_id"] for t in missing} != {t["eval_id"] for t in plan["tasks"] if t["view"] != "archived"}):
        raise ValueError("Recovered or missing prompt identities changed.")
    restored = read_json(root/"recovered_prompts.json")
    lookup = {r["eval_id"]: r for r in restored}
    if len(lookup) != 100 or set(lookup) != {e["eval_id"] for e in examples}:
        raise ValueError("Missing/duplicated original prompt provenance.")
    receipts = read_json(root/"backup_receipts.json")
    source = receipts["phase3b"]
    pairs_path = next(Path(p) for p in plan["source_files_sha256"] if Path(p).name == "pairs.json")
    pairs = read_json(pairs_path)
    replayed, current_eval, raw, index, index_path = 0, None, None, None, None
    with CheckedBackup(source["manifest_path"], [p["path"] for p in source["verified_parts"].values()]) as backup:
        if backup.manifest_sha256 != source["manifest_sha256"]:
            raise ValueError("Original backup manifest differs from recovery provenance.")
        for i, example in enumerate(examples):
            provenance = lookup[example["eval_id"]]
            if current_eval != example["eval_id"]:
                current_eval = example["eval_id"]
                index_path = provenance["index_member"]
                index = backup.json(index_path)
                if index["eval_id"] != current_eval or backup.files[index_path]["sha256"] != provenance["index_sha256"]:
                    raise ValueError("Original raw index identity/hash changed.")
                parent = str(PurePosixPath(index_path).parent)
                raw = [backup.tensor(parent+f"/layer_{layer:02d}.pt") for layer in LAYERS]
            row = pairs[example["pair_id"]][example["condition"]]
            position = reference_position(index, row, example["option"])
            if provenance["mapping"]["option_readout_audit"][example["option"]]["reference_last_position"] != position:
                raise ValueError("Independent reference position differs from the exported mapping.")
            for j, tensor in enumerate(raw):
                original = tensor["vectors"][tensor["positions"].index(position)].float().numpy()
                if not np.array_equal(original, features[i, j]):
                    raise ValueError("Recovered vector differs from independently selected original raw token.")
                replayed += 1
        backup.receipt()
    result = {"status": "verified", "scope": "independent original-group token selection and exact vector replay",
        "recovery_manifest_sha256": file_hash(recovery_manifest), "preparation_sha256": file_hash(plan_path),
        "statement_rows": len(examples), "vectors_replayed_exactly": replayed,
        "numeric_components_replayed_exactly": replayed*4096, "new_model_forwards": 0,
        "formal_probe_training_ready": False, "missing_AB_permutation_prompts": len(missing),
        "verifier_sha256": file_hash(Path(__file__)), "recovery_code_sha256": manifest["code_sha256"]}
    atomic_write(output_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan_path", "recovery_manifest", "output_path"):
        parser.add_argument("--"+name, required=True)
    print(json.dumps(verify(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
