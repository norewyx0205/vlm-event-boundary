"""Fit resumable CPU linear probes from a complete, gate-verified real feature store."""

import argparse
import fcntl
import os
import platform
import re
import tempfile
import time
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import sklearn
import scipy
from scipy.stats import spearmanr

try:
    from .phase3c_core import atomic_write, digest, file_hash, frozen_write, read_json
    from .phase3d_probes import (SETTINGS, bootstrap_mean, case_evidence, categorical_controls,
                                 crossfit, prediction_metrics, validate_rows)
except ImportError:
    from phase3c_core import atomic_write, digest, file_hash, frozen_write, read_json
    from phase3d_probes import (SETTINGS, bootstrap_mean, case_evidence, categorical_controls,
                                crossfit, prediction_metrics, validate_rows)

DATASET_SCHEMA = "phase3d_feature_dataset_v1"
GATES = ("processor_input_parity", "standard_first_token_parity", "full_logit_parity",
         "capture_noop", "source_state_replay", "identity_patch", "same_block_timing",
         "new_option_order_input_audit", "one_to_one_full_visual_support")
DEFAULT_READOUTS = [f"{kind}:L{layer}_output" for kind in ("source_pair", "option_pair") for layer in range(36)]
DEFAULT_READOUTS += ["source_pair:L16_input"] + [f"decision:L{layer}_output" for layer in (16, 20, 24, 28, 32, 35)]
DEFAULT_READOUTS += [f"option_{option}_{kind}:L20_output" for option in ("A", "B") for kind in ("pair", "relation")]


def resource(root, spec):
    path = (root/spec["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file() or file_hash(path) != spec["sha256"]:
        raise ValueError("Feature-store resource path/hash is invalid.")
    return path


def load_dataset(manifest_path):
    path = Path(manifest_path).resolve()
    root, manifest = path.parent, read_json(path)
    if (manifest.get("schema") != DATASET_SCHEMA or manifest.get("artifact_type") != "real" or
            manifest.get("complete") is not True):
        raise ValueError("Formal probes require complete real captures, not mock or partial data.")
    gates = read_json(resource(root, manifest["gate_summary"]))
    if (gates.get("schema") != "phase3d_capture_gate_summary_v1" or gates.get("complete") is not True or
            any(gates.get(name) is not True for name in GATES)):
        raise ValueError("Missing/failed Phase 3D capture gates; do not train a formal probe.")
    cases = read_json(resource(root, manifest["case_metadata"]))
    folds = read_json(resource(root, manifest["outer_folds"]))
    rows = read_json(resource(root, manifest["row_metadata"]))
    if len(cases) != 50 or gates.get("base_count") != 50:
        raise ValueError("Use all 50 frozen bases; no probe-specific exclusion or resampling.")
    labels = validate_rows(rows, cases, folds, require_complete=True)
    if gates.get("case_labels_sha256") != digest(sorted(labels.items())):
        raise ValueError("Capture gates bind a different relation-label cohort.")
    primary = [row for row in rows if row["view"] == "archived" and row["intervention"] == "baseline"]
    controls = [row for row in rows if row["view"] == "option_order_permuted"]
    case_lookup = {case["base_sample_id"]: case for case in cases}
    if any(row.get("correct_option") not in ("A", "B") or
           (row["view"] == "archived" and row["correct_option"] != case_lookup[row["base_sample_id"]]["correct_option"])
           for row in rows):
        raise ValueError("Feature rows changed the actual correct-answer identity.")
    if any(not re.fullmatch(r"[0-9a-f]{64}", row.get("input_fingerprint", "")) for row in rows):
        raise ValueError("Each actual prompt input needs a SHA256 fingerprint.")
    for control in controls:
        original = next(row for row in primary if row["base_sample_id"] == control["base_sample_id"] and
                        row["condition"] == control["condition"])
        if (not control.get("input_fingerprint") or control["input_fingerprint"] == original.get("input_fingerprint") or
                control.get("correct_option") == original.get("correct_option")):
            raise ValueError("Pure option-order control reused the old prompt or answer label.")
    if any(type(row.get("answer_margin")) not in (int, float) or not np.isfinite(row["answer_margin"]) for row in rows):
        raise ValueError("Every view/patch needs its actually measured answer margin.")
    names = manifest["feature_names"]
    pattern = re.compile(r"(?:source_pair|option_pair|decision|query|option_[AB]_(?:pair|relation)):L(?:[0-9]|[12][0-9]|3[0-5])_output$")
    if len(set(names)) != len(names) or any(not pattern.fullmatch(name) and name != "source_pair:L16_input" for name in names):
        raise ValueError("Invalid or duplicate layer/site feature name.")
    with np.load(resource(root, manifest["features"]), allow_pickle=False) as payload:
        values = payload["features"]
    if (values.ndim != 3 or values.shape[:2] != (len(rows), len(names)) or values.shape[2] <= 0 or
            not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all()):
        raise ValueError("Expected finite [row, named feature, hidden dimension] arrays.")
    return manifest, rows, cases, folds, names, values


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def frozen_numeric_scores(path, values):
    """Apply an exported numeric probe without fitting/scaling on new data."""
    with np.load(path, allow_pickle=False) as payload:
        coefficient, intercept = payload["coefficient"], payload["intercept"]
        mean, scale = payload["scaler_mean"], payload["scaler_scale"]
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 2 or coefficient.shape != (1, values.shape[1]) or
            mean.shape != scale.shape or mean.shape != (values.shape[1],) or intercept.shape != (1,) or
            any(not np.isfinite(value).all() for value in (values, coefficient, intercept, mean, scale)) or
            (scale <= 0).any()):
        raise ValueError("Frozen numeric readout has incompatible/nonfinite features or scaling.")
    return ((values-mean)/scale) @ coefficient[0] + intercept[0]


def saved_result(root, fingerprint):
    path = root/"result.json"
    if not path.exists():
        return None
    result = read_json(path)
    if result.get("complete") is not True or result.get("fingerprint") != fingerprint:
        raise ValueError("Probe checkpoint/configuration mismatch; use a new output root.")
    for name, sha in result["output_sha256"].items():
        candidate = (root/name).resolve()
        if not candidate.is_relative_to(root.resolve()) or file_hash(candidate) != sha:
            raise ValueError("Completed probe checkpoint bytes changed.")
    return result


def write_result(root, result, fingerprint, settings, readout_name="unspecified", case_metadata=None):
    root.mkdir(parents=True, exist_ok=True)
    files, records = [], []
    for record in result["models"]:
        pipe = record["model"]
        classifier, scaler = pipe.named_steps["probe"], pipe.named_steps["scale"]
        names = (pipe.named_steps["vocabulary"].get_feature_names_out() if "vocabulary" in pipe.named_steps else
                 np.asarray([str(i) for i in range(classifier.coef_.shape[1])]))
        name = f"{record['model_id']}.npz"
        save_npz(root/name, coefficient=classifier.coef_, intercept=classifier.intercept_,
                 scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, feature_columns=np.asarray(names, dtype=str))
        files.append(name)
        records.append({key: value for key, value in record.items() if key != "model"})
    atomic_write(root/"models.json", records)
    atomic_write(root/"held_out_predictions.json", result["predictions"])
    files.extend(["models.json", "held_out_predictions.json"])
    summary = {"complete": True, "fingerprint": fingerprint, "target": result["target"],
               "label_permutation": result["label_permutation"],
               "readout_name": readout_name, "is_primary_readout": readout_name == settings["primary_feature"],
               "held_out_metrics": prediction_metrics(result["predictions"])}
    if result["target"] == "relation" and not result["label_permutation"]:
        separate_option = re.fullmatch(r"option_([AB])_(?:pair|relation):L\d+_output", readout_name)
        if separate_option and case_metadata is not None:
            try:
                from .phase3d_probes import statement_truth_predictions
            except ImportError:
                from phase3d_probes import statement_truth_predictions
            truth = statement_truth_predictions(result["predictions"], case_metadata, separate_option[1])
            atomic_write(root/"statement_truth_predictions.json", truth)
            files.append("statement_truth_predictions.json")
            summary["statement_truth_diagnostic"] = {
                "held_out_metrics": prediction_metrics(truth),
                "interpretation": "Same frozen canonical-relation probe re-expressed using actual option semantics; not independent evidence or an A/B-label fit."}
        evidence = case_evidence(result["predictions"])
        atomic_write(root/"case_evidence.json", evidence)
        files.append("case_evidence.json")
        primary = [row for row in evidence if row["view"] == "archived"]
        summary["boundary_evidence"] = bootstrap_mean([row["delta_S_boundary"] for row in primary],
                                                       settings["seed"], settings["bootstrap_replicates"])
        transfer = [row for row in primary if "delta_S_temporal_to_low_L16" in row]
        if transfer:
            x = np.asarray([row["delta_S_temporal_to_low_L16"] for row in transfer])
            y = np.asarray([row["delta_M_temporal_to_low_L16"] for row in transfer])
            rho = float(spearmanr(x, y).statistic) if np.ptp(x) > 0 and np.ptp(y) > 0 else None
            downstream = re.fullmatch(r"(?:option_pair|decision|query|option_[AB]_(?:pair|relation)):L(\d+)_output", readout_name)
            summary["patch_associated_evidence"] = {
                **bootstrap_mean(x, settings["seed"], settings["bootstrap_replicates"]),
                "unpatched_downstream_text_readout": bool(downstream and int(downstream[1]) > 16),
                "case_level_evidence_margin_spearman": rho}
    summary["output_sha256"] = {name: file_hash(root/name) for name in files}
    atomic_write(root/"result.json", summary)
    return summary


def _run(args):
    manifest_path = Path(args.manifest_path).resolve()
    manifest, rows, cases, folds, names, features = load_dataset(manifest_path)
    settings = dict(SETTINGS)
    selected = args.feature_names or DEFAULT_READOUTS
    if len(set(selected)) != len(selected) or not set(selected) <= set(names) or settings["primary_feature"] not in selected:
        raise ValueError("Need the fixed L20 primary readout and every requested named feature.")
    root = Path(args.output_dir).resolve()
    if root == manifest_path.parent or root in manifest_path.parents or manifest_path.parent in root.parents:
        raise ValueError("Use an isolated probe root, outside the immutable feature store.")
    config = {"schema": "phase3d_linear_probes_v1", "manifest_sha256": file_hash(manifest_path),
              "feature_names": selected, "settings": settings,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "sklearn": sklearn.__version__, "scipy": scipy.__version__},
              "protocol_sha256": file_hash(Path(__file__).resolve().parents[1]/"docs/phase3d_protocol.md"),
              "code_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in
                              ("run_phase3d_probes.py", "phase3d_probes.py", "phase3d_core.py")}}
    config["fingerprint"] = digest(config)
    frozen_write(root/"run_config.json", config)
    jobs = [(name, features[:, names.index(name), :], "relation", False, None) for name in selected]
    for name in ("source_pair:L16_output", "option_pair:L20_output"):
        if name not in names:
            raise ValueError("Missing fixed condition-only control feature.")
        jobs.append(("condition__"+name, features[:, names.index(name), :], "condition", False, None))
    for kind in ("static_attributes", "option_lexical"):
        jobs.append((kind, categorical_controls(rows, cases, kind), "relation", True, None))
    original_labels = np.asarray([case["y"] for case in cases])
    for permutation in range(settings["label_permutations"]):
        shuffled = np.random.default_rng(settings["seed"]+1000+permutation).permutation(original_labels)
        labels = {case["base_sample_id"]: int(label) for case, label in zip(cases, shuffled)}
        jobs.append((f"label_permutation_{permutation:03d}", features[:, names.index(settings["primary_feature"]), :],
                     "relation", False, labels))
    started, outputs = time.perf_counter(), {}
    for index, (name, values, target, categorical, permutation) in enumerate(jobs, 1):
        destination = root/name.replace(":", "__")
        fingerprint = digest([config["fingerprint"], name, permutation])
        summary = saved_result(destination, fingerprint)
        reused = summary is not None
        if summary is None:
            print(f"Phase 3D probes {index}/{len(jobs)} starting: {name}; checkpoint={destination}", flush=True)
            def progress(message):
                print(f"  {name}: {message}; elapsed {(time.perf_counter()-started)/60:.1f} min", flush=True)
            result = crossfit(rows, values, cases, folds, settings, target, categorical, permutation, progress)
            summary = write_result(destination, result, fingerprint, settings, readout_name=name, case_metadata=cases)
        outputs[name] = {"result_path": str(destination.relative_to(root)/"result.json"),
                         "sha256": file_hash(destination/"result.json")}
        print(f"Phase 3D probes {index}/{len(jobs)}: {name}; elapsed "
              f"{(time.perf_counter()-started)/60:.1f} min; "
              f"status={'reused' if reused else 'newly_fit'}; checkpoint={destination}", flush=True)
    aggregate = {"schema": config["schema"], "artifact_type": "real", "complete": True,
                 "base_count": 50, "fingerprint": config["fingerprint"], "results": outputs,
                 "primary_feature": settings["primary_feature"],
                 "interpretation": "Held-out linear evidence, not proof of mediation or unique routing. "
                                   "Layer profiles are descriptive; bases, not condition rows, are independent."}
    atomic_write(root/"aggregate_summary.json", aggregate)
    return aggregate


@contextmanager
def probe_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root/".probe.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another probe process already owns this output root.") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run(args):
    manifest = Path(args.manifest_path).resolve()
    root = Path(args.output_dir).resolve()
    if root == manifest.parent or root in manifest.parents or manifest.parent in root.parents:
        raise ValueError("Use an isolated probe root, outside the immutable feature store.")
    with probe_lock(root):
        return verify_outputs(root, manifest) if getattr(args, "verify_only", False) else _run(args)


def verify_outputs(root, manifest_path):
    """Check every expected job against the actual dataset and held-out folds."""
    root, manifest_path = Path(root).resolve(), Path(manifest_path).resolve()
    _, rows, cases, folds, names, values = load_dataset(manifest_path)
    config, summary = read_json(root/"run_config.json"), read_json(root/"aggregate_summary.json")
    if (config.get("fingerprint") != digest({k: v for k, v in config.items() if k != "fingerprint"}) or
            config["manifest_sha256"] != file_hash(manifest_path) or config["settings"] != SETTINGS or
            config["protocol_sha256"] != file_hash(Path(__file__).resolve().parents[1]/"docs/phase3d_protocol.md") or
            any(file_hash(Path(__file__).with_name(name)) != sha for name, sha in config["code_sha256"].items()) or
            summary.get("complete") is not True or summary.get("fingerprint") != config["fingerprint"]):
        raise ValueError("Probe summary/configuration/dataset binding changed.")
    selected = config["feature_names"]
    if (len(set(selected)) != len(selected) or not set(selected) <= set(names) or SETTINGS["primary_feature"] not in selected):
        raise ValueError("Invalid frozen relation readout list.")
    jobs = [(name, "relation", None) for name in selected]
    jobs += [("condition__"+name, "condition", None) for name in ("source_pair:L16_output", "option_pair:L20_output")]
    jobs += [(name, "relation", None) for name in ("static_attributes", "option_lexical")]
    for permutation in range(SETTINGS["label_permutations"]):
        shuffled = np.random.default_rng(SETTINGS["seed"]+1000+permutation).permutation([c["y"] for c in cases])
        labels = {case["base_sample_id"]: int(label) for case, label in zip(cases, shuffled)}
        jobs.append((f"label_permutation_{permutation:03d}", "relation", labels))
    if set(summary["results"]) != {name for name, _, _ in jobs}:
        raise ValueError("Missing or foreign relation/control/null result.")
    key = lambda row: (row["base_sample_id"], row["condition"], row["view"], row["intervention"])
    expected_rows = {key(row): (index, row) for index, row in enumerate(rows)}
    base_fold = {base: f"outer_{fold['outer_fold']}" for fold in folds for base in fold["test_base_ids"]}
    expected_models = {f"outer_{fold['outer_fold']}": fold for fold in folds}
    for count, (name, target, permutation) in enumerate(jobs, 1):
        directory = root/name.replace(":", "__")
        ref = summary["results"][name]
        if (ref["result_path"] != str(directory.relative_to(root)/"result.json") or
                file_hash(directory/"result.json") != ref["sha256"]):
            raise ValueError("Misbound or altered per-readout result resource.")
        result = saved_result(directory, digest([config["fingerprint"], name, permutation]))
        if result is None or result["readout_name"] != name or result["target"] != target or result["label_permutation"] != (permutation is not None):
            raise ValueError("Readout/target/permutation identity differs from its requested job.")
        models = read_json(directory/"models.json")
        if len(models) != len(folds) or {m["model_id"] for m in models} != set(expected_models):
            raise ValueError("Missing/duplicate outer-fold model.")
        for model in models:
            fold = expected_models[model["model_id"]]
            if (model["train_base_ids"] != fold["train_base_ids"] or model["test_base_ids"] != fold["test_base_ids"] or
                    model["chosen_C"] not in SETTINGS["C_grid"]):
                raise ValueError("Exported model does not bind the held-out fold/C grid.")
        predictions = read_json(directory/"held_out_predictions.json")
        if len(predictions) != len(rows) or {key(p) for p in predictions} != set(expected_rows):
            raise ValueError("Incomplete/duplicated held-out observation grid.")
        for prediction in predictions:
            _, original = expected_rows[key(prediction)]
            if any(prediction.get(field) != value for field, value in original.items()):
                raise ValueError("Held-out prediction refers to a different input/condition/intervention.")
            label = (int(original["condition"] == "temporal_boundary") if target == "condition" else
                     permutation[original["base_sample_id"]] if permutation is not None else original["y"])
            if (prediction["probe_label"] != label or prediction["probe_target"] != target or
                    prediction["model_id"] != base_fold[original["base_sample_id"]] or
                    prediction["probe_prediction"] != int(prediction["probe_logit"] > 0)):
                raise ValueError("Prediction uses a foreign label, target or non-held-out model.")
        # Numeric exports must reproduce all saved held-out logits without refitting.
        numeric = None if name in ("static_attributes", "option_lexical") else (
            SETTINGS["primary_feature"] if permutation is not None else name.removeprefix("condition__"))
        if numeric is not None:
            feature = values[:, names.index(numeric), :]
            for model in models:
                subset = [p for p in predictions if p["model_id"] == model["model_id"]]
                indices = [expected_rows[key(p)][0] for p in subset]
                logits = frozen_numeric_scores(directory/(model["model_id"]+".npz"), feature[indices])
                if not np.allclose(logits, [p["probe_logit"] for p in subset], atol=1e-10, rtol=1e-10):
                    raise ValueError("Exported numeric probe does not reproduce held-out logits.")
        if count % 10 == 0 or count == len(jobs):
            print(f"Phase 3D CPU verification: {count}/{len(jobs)} readouts/controls", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_names", nargs="+")
    parser.add_argument("--verify_only", action="store_true", help="Verify complete probes and numeric held-out replay without refitting.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
