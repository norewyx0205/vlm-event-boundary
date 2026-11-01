"""CPU-only condition-generalization follow-up; preserve the mixed-reference run."""

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy
import sklearn
from sklearn.preprocessing import StandardScaler

from scripts.phase3c_core import atomic_write, digest, file_hash, frozen_write, read_json
from scripts.phase3e_condition_probes import (CONDITIONS, READOUTS, REGIMES, SCHEMA,
    crossfit_condition, evaluation, selections, training_audit)
from scripts.phase3e_option_probes import SETTINGS, metrics, paired_analysis
from scripts.phase3e_option_store import load_store
from scripts.run_phase3d_probes import probe_lock
from scripts.run_phase3e_option_probes import verify_contrast_projection, verify_job, write_job

PROTOCOL = "phase3e_condition_generalization.md"
DEPENDENCIES = ("phase3e_condition_probes.py", "run_phase3e_condition_probes.py",
    "phase3e_option_probes.py", "run_phase3e_option_probes.py", "phase3e_option_store.py",
    "phase3d_probes.py", "run_phase3d_probes.py", "phase3e_paired_design.py",
    "phase3e_option_features.py", "phase3e_preflight.py", "phase3c_core.py", "phase3d_core.py")


def disjoint_output(root, *sources):
    for source in sources:
        if root == source or root in source.parents or source in root.parents:
            raise ValueError("Keep follow-up outputs separate from immutable source/reference directories.")


def reference_snapshot(root, manifest_path, plan, examples, features, names):
    config = read_json(root / "config.json")
    repo = Path(__file__).resolve().parents[1]
    runtime = {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}
    if (config["schema"] != "phase3e_option_probe_run_v1" or
            config["manifest_sha256"] != file_hash(manifest_path) or config["settings"] != SETTINGS or
            config["runtime"] != runtime or not set(READOUTS) <= set(config["readouts"]) or
            set(config["code_sha256"]) != set(DEPENDENCIES)-{
                "phase3e_condition_probes.py", "run_phase3e_condition_probes.py"} or
            config["protocol_sha256"] != file_hash(repo / "docs/phase3e_b_paired_option_protocol.md")):
        raise ValueError("Mixed-reference source, settings, runtime or protocol does not match.")
    for name, sha in config["code_sha256"].items():
        if Path(name).name != name or file_hash(repo / "scripts" / name) != sha:
            raise ValueError("Mixed-reference code differs from the frozen original.")
    snapshot = {"config_sha256": file_hash(root / "config.json"), "jobs": {}}
    for name in READOUTS:
        job = root / name.replace(":", "__")
        result = read_json(job / "result.json")
        expected = {f"outer_{fold['outer_fold']}.npz" for fold in plan["folds"]} | {
            "models.json", "predictions.json", "paired_analysis.json"}
        if not result.get("complete") or set(result["files"]) != expected:
            raise ValueError("Mixed-reference job is incomplete or has unexpected resources.")
        values = features[:, names.index(name), :]
        predictions = verify_job(job, examples, values, plan["folds"], False)
        if (result["held_out_metrics"] != metrics(predictions) or
                read_json(job / "paired_analysis.json") != paired_analysis(predictions, plan["rows"])):
            raise ValueError("Mixed-reference statistics do not replay.")
        snapshot["jobs"][name] = {"result_sha256": file_hash(job / "result.json"), "files": result["files"]}
    return snapshot


def seal_job(job, examples, rows, folds, selection, regime, name, reused):
    predictions = read_json(job / "predictions.json")
    atomic_write(job / "training_audit.json", training_audit(examples, folds, selection, regime))
    atomic_write(job / "evaluation.json", evaluation(predictions, rows))
    result = {"schema": SCHEMA, "complete": True, "regime": regime, "readout": name,
        "reused_mixed_reference": reused, "held_out_metrics": metrics(predictions),
        "files": {p.name: file_hash(p) for p in sorted(job.iterdir()) if p.is_file() and p.name != "result.json"}}
    atomic_write(job / "result.json", result)


def verify_saved(job, examples, values, plan, selection, regime, name, reference):
    result = read_json(job / "result.json")
    expected = {f"outer_{fold['outer_fold']}.npz" for fold in plan["folds"]} | {
        "models.json", "predictions.json", "paired_analysis.json", "training_audit.json", "evaluation.json"}
    if (result.get("schema") != SCHEMA or result.get("complete") is not True or
            result.get("regime") != regime or result.get("readout") != name or
            result.get("reused_mixed_reference") != (regime == "mixed_full") or
            set(result["files"]) != expected or
            {p.name for p in job.iterdir()} != expected | {"result.json"}):
        raise ValueError("Completed follow-up job contract changed.")
    if any((job / relative).is_symlink() for relative in expected):
        raise ValueError("Probe resources must not be symlinks.")
    predictions = verify_job(job, examples, values, plan["folds"], False)
    audit = training_audit(examples, plan["folds"], selection, regime)
    if read_json(job / "training_audit.json") != audit or result["held_out_metrics"] != metrics(predictions):
        raise ValueError("Source-only training audit or held-out metrics changed.")
    selected = set(selection["selected_example_ids"][regime])
    records = read_json(job / "models.json")
    for record in records:
        candidates = record["inner_scores"]
        if ({row["C"] for row in candidates} != set(SETTINGS["C_grid"]) or len(candidates) != len(SETTINGS["C_grid"]) or
                any(not 0 <= row["inner_balanced_accuracy"] <= 1 for row in candidates) or
                record["chosen_C"] != max(candidates, key=lambda row: (row["inner_balanced_accuracy"], -row["C"]))["C"]):
            raise ValueError("Saved hyperparameter selection changed.")
        indices = [i for i, row in enumerate(examples) if row["scene_seed"] in record["train_scene_seeds"]
                   and row["example_id"] in selected]
        scaler = StandardScaler().fit(np.asarray(values[indices], dtype=np.float64))
        with np.load(job / (record["model_id"]+".npz"), allow_pickle=False) as parameters:
            if (not np.allclose(parameters["scaler_mean"], scaler.mean_, rtol=1e-12, atol=1e-12) or
                    not np.allclose(parameters["scaler_scale"], scaler.scale_, rtol=1e-12, atol=1e-12)):
                raise ValueError("Saved scaler was not fitted on the selected source training rows.")
    if regime == "mixed_full" and any(file_hash(job / relative) != sha
                                     for relative, sha in reference["jobs"][name]["files"].items()):
        raise ValueError("Reused full-mixed numeric resources differ from the original.")
    paired = paired_analysis(predictions, plan["rows"])
    measured = evaluation(predictions, plan["rows"])
    if read_json(job / "paired_analysis.json") != paired or read_json(job / "evaluation.json") != measured:
        raise ValueError("Paired evidence or cross-condition evaluation does not replay.")
    projection = verify_contrast_projection(job, examples, values, predictions, paired)
    return {**measured, "training_audit": audit, "reused_mixed_reference": regime == "mixed_full",
            "direct_D_projection_replay": projection}


def report_text(summary):
    lines = ["# Phase 3E-B: Cross-Condition Truth Probing", "", "## Material Passport", "",
        "- Existing verified captures; no new GPU inference, KO or patch.",
        "- L16/L17/L18 block outputs; L20 downstream reference; reference-last option token.",
        "- Same 40 scene families, grouped outer/inner folds and statement-truth labels.",
        "- Low-only, Temporal-only and budget-matched mixed are new fits. Full mixed is verified reuse.",
        "- Each frozen source-trained model evaluates both conditions of unseen test families.", "",
        "## Training And Testing", "",
        "| Readout | Training | Rows per outer fit | Test Low accuracy | Test Temporal accuracy | T - L (pp) | Low AUC | Temporal AUC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, regimes in summary["results"].items():
        for regime, result in regimes.items():
            overall = {row["condition"]: row for row in result["metrics"] if row["group_by"] is None}
            low, temporal = (overall[condition] for condition in CONDITIONS)
            counts = {row["train"]["statement_rows"] for row in result["training_audit"]["outer_models"]}
            count_text = ",".join(map(str, sorted(counts)))
            lines.append(f"| {name} | {regime} | {count_text} | {low['accuracy']:.2%} | {temporal['accuracy']:.2%} | "
                f"{100*(temporal['accuracy']-low['accuracy']):+.2f} | {low['mean_within_fold_roc_auc']:.3f} | "
                f"{temporal['mean_within_fold_roc_auc']:.3f} |")
    lines += ["", "AUC is averaged within held-out folds; logits from distinct fitted models are not pooled.",
        "Source-only scaling and inner regularization selection never use the other condition.", "",
        "## Paired Evidence", "", "Each row uses one frozen readout rule for both test conditions.", "",
        "| Readout | Training | Low evidence | Temporal evidence | T - L | Diagnostic 95% interval of difference |",
        "| --- | --- | ---: | ---: | ---: | --- |"]
    for name, regimes in summary["results"].items():
        for regime, result in regimes.items():
            boundary = result["boundary_summary"]
            difference = boundary["evidence_temporal_minus_low"]
            interval = difference["diagnostic_95_interval"]
            lines.append(f"| {name} | {regime} | {boundary['evidence_low']['mean']:.3f} | "
                f"{boundary['evidence_temporal']['mean']:.3f} | {difference['mean']:+.3f} | "
                f"[{interval[0]:.3f}, {interval[1]:.3f}] |")
    lines += ["", "Scores are comparable between conditions within a row, not across independently trained regimes or sites.",
        "Intervals resample scene families conditional on saved probes; no full-refit uncertainty or new significance tests.",
        "Family accuracy levels/differences, both order-contrast levels, subject/option breakdowns and source-row audits are in JSON.",
        "", "## Budget Control", "",
        "The matched mixed source uses complementary subject/layout halves in the two conditions, chosen without features or outcomes.",
        "It retains both true event orders and both statements for every selected prompt. Within every family and condition,",
        "truth, subject, layout and correct A/B position are balanced. Each source has 32 independent training families and",
        "one total classifier sample-weight unit per family. Full mixed has more views, not more independent families.", "",
        "## Interpretation", "",
        "In-domain cells measure source-condition linear accessibility; off-diagonal cells measure transfer of that readout rule.",
        "A cross-condition accuracy loss can reflect threshold or distribution changes; AUC provides a ranking-based comparison.",
        "These are representation diagnostics, not new causal interventions or a direct test of a unique circuit.", ""]
    return "\n".join(lines)


def run(manifest_path, mixed_reference_root, output_root, verify_only=False):
    manifest_path, reference_root, root = map(lambda path: Path(path).resolve(),
        (manifest_path, mixed_reference_root, output_root))
    disjoint_output(root, manifest_path.parent, reference_root)
    manifest, plan, examples, features, _ = load_store(manifest_path)
    reference = reference_snapshot(reference_root, manifest_path, plan, examples, features, manifest["feature_names"])
    selection = selections(examples, plan["rows"], SETTINGS["seed"])
    repo = Path(__file__).resolve().parents[1]
    config = {"schema": SCHEMA, "manifest_sha256": file_hash(manifest_path), "settings": SETTINGS,
        "readouts": list(READOUTS), "regimes": list(REGIMES), "selection_sha256": digest(selection),
        "mixed_reference": reference, "code_sha256": {name: file_hash(repo / "scripts" / name) for name in DEPENDENCIES},
        "protocol_sha256": file_hash(repo / "docs" / PROTOCOL), "runtime": {"python": platform.python_version(),
            "numpy": np.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__}}
    with probe_lock(root):
        if verify_only:
            if read_json(root / "config.json") != config or read_json(root / "selection_manifest.json") != selection:
                raise ValueError("Frozen follow-up configuration or training selections changed.")
        else:
            frozen_write(root / "config.json", config)
            frozen_write(root / "selection_manifest.json", selection)
        results = {}
        for name in READOUTS:
            values = features[:, manifest["feature_names"].index(name), :]
            results[name] = {}
            for regime in REGIMES:
                job = root / name.replace(":", "__") / regime
                if not (job / "result.json").exists():
                    if verify_only:
                        raise ValueError("Missing completed follow-up job.")
                    if job.exists() and any(job.iterdir()):
                        raise ValueError("Partial follow-up job; inspect it and use a new output root.")
                    if regime == "mixed_full":
                        job.mkdir(parents=True, exist_ok=True)
                        source = reference_root / name.replace(":", "__")
                        for relative in reference["jobs"][name]["files"]:
                            shutil.copy2(source / relative, job / relative)
                    else:
                        fitted = crossfit_condition(examples, values, plan["folds"], selection, regime)
                        write_job(job, fitted, examples, plan["rows"])
                    seal_job(job, examples, plan["rows"], plan["folds"], selection, regime, name, regime == "mixed_full")
                results[name][regime] = verify_saved(job, examples, values, plan, selection, regime, name, reference)
                print("VERIFIED_CONDITION_PROBE", name, regime, flush=True)
        summary = {"schema": SCHEMA, "complete": True, "scene_families": len(plan["scene_seeds"]),
            "statement_rows_per_job": len(examples), "new_saved_outer_models": 5*len(READOUTS)*3,
            "reused_outer_models": 5*len(READOUTS), "results": results,
            "new_GPU_inference": False, "new_KO": False, "original_mixed_results_modified": False}
        if verify_only:
            if read_json(root / "summary.json") != summary or (root / "report.md").read_text() != report_text(summary):
                raise ValueError("Aggregate condition-generalization report changed.")
        else:
            atomic_write(root / "summary.json", summary)
            (root / "report.md").write_text(report_text(summary), encoding="utf-8")
        if reference_snapshot(reference_root, manifest_path, plan, examples, features, manifest["feature_names"]) != reference:
            raise ValueError("Original mixed-reference artifacts changed during the follow-up.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--mixed_reference_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args()
    result = run(args.manifest_path, args.mixed_reference_root, args.output_root, args.verify_only)
    print(json.dumps({key: result[key] for key in ("complete", "scene_families", "statement_rows_per_job",
        "new_saved_outer_models", "reused_outer_models", "new_GPU_inference")}, indent=2))
