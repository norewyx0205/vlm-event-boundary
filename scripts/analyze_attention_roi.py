import argparse
import csv
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path


ROI_ORDER = (
    "target_1",
    "target_2",
    "distractors",
    "visual_marker",
    "boundary_flash",
    "background",
)
LAYER_STAGES = {
    "early": range(0, 12),
    "middle": range(12, 24),
    "late": range(24, 36),
}
EPSILON = 1e-12


def read_attention_results(path):
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".zip":
        rows = []
        sources = []
        with zipfile.ZipFile(path) as archive:
            for member in sorted(archive.namelist()):
                if not member.endswith(".json") or member.endswith("_config.json"):
                    continue
                try:
                    payload = json.loads(archive.read(member).decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
                    continue
                if not isinstance(payload, list) or not payload:
                    continue
                if not all(
                    isinstance(row, dict) and row.get("layer_roi_profiles")
                    for row in payload
                ):
                    continue
                source = f"{path}::{member}"
                rows.extend((source, row) for row in payload)
                sources.append(source)
        return rows, sources
    files = [path] if path.is_file() else sorted(path.rglob("*.json"))
    rows = []
    sources = []
    for input_file in files:
        try:
            payload = json.loads(input_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, list) or not payload:
            continue
        if not all(isinstance(row, dict) and row.get("layer_roi_profiles") for row in payload):
            continue
        rows.extend((input_file, row) for row in payload)
        sources.append(input_file)
    return rows, sources


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def value(metrics, explicit_name, legacy_name, default=0.0):
    item = metrics.get(explicit_name)
    if item is None:
        item = metrics.get(legacy_name, default)
    return float(item) if item is not None else default


def object_label(target):
    label = str(
        target.get("reference_label")
        or target.get("label")
        or " ".join(
            part
            for part in (target.get("color"), target.get("shape"))
            if part
        )
        or f"target {target.get('id')}"
    ).strip()
    if label.lower().startswith("the "):
        label = label[4:]
    return label


def prompt_subject_id(result, targets):
    sentences = [
        result.get("option_A"),
        result.get("option_B"),
        result.get("correct_sentence"),
    ]
    for sentence in sentences:
        sentence = str(sentence or "").strip().lower()
        for target in targets:
            labels = {
                object_label(target).lower(),
                str(target.get("label") or "").strip().lower(),
            }
            for label in labels:
                if label and (sentence.startswith(label) or sentence.startswith(f"the {label}")):
                    return int(target.get("id"))
    return None


def target_roles(result):
    targets = sorted(result.get("target_objects") or [], key=lambda item: int(item["id"]))
    subject_id = prompt_subject_id(result, targets)
    starts = [int(target.get("start_frame") or 0) for target in targets]
    first_start = min(starts) if starts else None
    last_start = max(starts) if starts else None
    roles = {}
    for target in targets:
        target_id = int(target["id"])
        start = int(target.get("start_frame") or 0)
        if first_start == last_start:
            mover_role = "co-mover"
        else:
            mover_role = "first mover" if start == first_start else "second mover"
        subject_role = (
            "subject"
            if subject_id == target_id
            else "non-subject"
            if subject_id is not None
            else "subject unknown"
        )
        semantic_label = object_label(target)
        roles[f"target_{target_id}"] = {
            "target_id": target_id,
            "semantic_label": semantic_label,
            "mover_role": mover_role,
            "subject_role": subject_role,
            "direction": target.get("direction"),
            "radius": target.get("radius"),
            "size_label": target.get("reference_label") or target.get("size_label"),
            "display_label": (
                f"T{target_id} | {semantic_label} | {mover_role} | {subject_role}"
            ),
        }
    return roles


def layer_stage(layer):
    for name, indices in LAYER_STAGES.items():
        if layer in indices:
            return name
    return "outside_preregistered_qwen3_stages"


def base_metadata(source_path, result):
    return {
        "source_attention_file": str(source_path),
        "eval_id": result.get("eval_id"),
        "video_id": result.get("video_id"),
        "base_sample_id": result.get("base_sample_id"),
        "dataset_version": result.get("dataset_version"),
        "feature_variant": result.get("feature_variant"),
        "size_scene_variant": result.get("size_scene_variant"),
        "condition": result.get("condition"),
        "prompt_variant": result.get("prompt_variant"),
        "correct_option": result.get("correct_option"),
        "prediction": result.get("prediction"),
        "is_correct": result.get("is_correct"),
        "attention_semantics": result.get("attention_semantics"),
        "head_reduction": result.get("head_reduction"),
        "roi_assignment_method": result.get("roi_assignment_method"),
        "roi_padding": result.get("roi_padding"),
    }


def flatten_roi_metrics(source_path, result):
    output = []
    roles = target_roles(result)
    metadata = base_metadata(source_path, result)
    for profile in result.get("layer_roi_profiles") or []:
        layer = int(profile.get("layer", 0))
        visual_fraction = float(profile.get("visual_attention_fraction") or 0.0)
        for roi in ROI_ORDER:
            metrics = (profile.get("spatial_roi") or {}).get(roi)
            if not metrics:
                continue
            role = roles.get(roi, {})
            output.append({
                **metadata,
                "layer": layer,
                "layer_stage": layer_stage(layer),
                "roi": roi,
                "roi_display_label": role.get("display_label", roi.replace("_", " ").title()),
                "semantic_label": role.get("semantic_label"),
                "mover_role": role.get("mover_role"),
                "subject_role": role.get("subject_role"),
                "direction": role.get("direction"),
                "radius": role.get("radius"),
                "size_label": role.get("size_label"),
                "visual_attention_fraction": visual_fraction,
                "all_token_attention_share": value(
                    metrics, "all_token_attention_share", "attention_mass"
                ),
                "visual_normalized_attention_mass": value(
                    metrics,
                    "visual_normalized_attention_mass",
                    "normalized_visual_attention",
                ),
                "effective_token_area_share": value(
                    metrics, "effective_token_area_share", "token_fraction"
                ),
                "effective_token_count": value(
                    metrics, "effective_token_count", "token_count"
                ),
                "mean_attention_per_effective_token": value(
                    metrics,
                    "mean_attention_per_effective_token",
                    "mean_attention_per_token",
                ),
                "area_normalized_enrichment": value(
                    metrics, "area_normalized_enrichment", "enrichment"
                ),
                "metric_schema": (
                    "explicit_mass_area_v2"
                    if "all_token_attention_share" in metrics
                    else "legacy_v1_aliases"
                ),
            })
    return output


def log_ratio(first, second):
    return math.log((float(first) + EPSILON) / (float(second) + EPSILON))


def target_contrasts(source_path, result):
    roles = target_roles(result)
    metadata = base_metadata(source_path, result)
    output = []
    for profile in result.get("layer_roi_profiles") or []:
        spatial = profile.get("spatial_roi") or {}
        target_1 = spatial.get("target_1")
        target_2 = spatial.get("target_2")
        if not target_1 or not target_2:
            continue
        layer = int(profile.get("layer", 0))
        mass_1 = value(
            target_1,
            "visual_normalized_attention_mass",
            "normalized_visual_attention",
        )
        mass_2 = value(
            target_2,
            "visual_normalized_attention_mass",
            "normalized_visual_attention",
        )
        area_1 = value(target_1, "effective_token_area_share", "token_fraction")
        area_2 = value(target_2, "effective_token_area_share", "token_fraction")
        enrichment_1 = value(
            target_1, "area_normalized_enrichment", "enrichment"
        )
        enrichment_2 = value(
            target_2, "area_normalized_enrichment", "enrichment"
        )
        delta_mass = log_ratio(mass_1, mass_2)
        log_area_ratio = log_ratio(area_1, area_2)
        delta_enrichment = log_ratio(enrichment_1, enrichment_2)
        output.append({
            **metadata,
            "layer": layer,
            "layer_stage": layer_stage(layer),
            "target_1_role": (roles.get("target_1") or {}).get("display_label"),
            "target_2_role": (roles.get("target_2") or {}).get("display_label"),
            "target_1_mover_role": (roles.get("target_1") or {}).get("mover_role"),
            "target_2_mover_role": (roles.get("target_2") or {}).get("mover_role"),
            "target_1_subject_role": (roles.get("target_1") or {}).get("subject_role"),
            "target_2_subject_role": (roles.get("target_2") or {}).get("subject_role"),
            "target_1_visual_mass": mass_1,
            "target_2_visual_mass": mass_2,
            "target_1_area_share": area_1,
            "target_2_area_share": area_2,
            "target_1_enrichment": enrichment_1,
            "target_2_enrichment": enrichment_2,
            "delta_log_visual_mass_t1_over_t2": delta_mass,
            "log_area_ratio_t1_over_t2": log_area_ratio,
            "delta_log_enrichment_t1_over_t2": delta_enrichment,
            "decomposition_residual": (
                delta_enrichment - (delta_mass - log_area_ratio)
            ),
            "target_1_has_more_total_visual_mass": mass_1 > mass_2,
            "target_1_has_higher_enrichment": enrichment_1 > enrichment_2,
        })
    return output


def stage_contrasts(layer_rows):
    grouped = defaultdict(list)
    for row in layer_rows:
        key = (
            row["source_attention_file"],
            row["eval_id"],
            row["layer_stage"],
        )
        grouped[key].append(row)
    output = []
    metrics = (
        "delta_log_visual_mass_t1_over_t2",
        "log_area_ratio_t1_over_t2",
        "delta_log_enrichment_t1_over_t2",
        "decomposition_residual",
    )
    for (_source, _eval_id, stage), rows in sorted(grouped.items()):
        if stage == "outside_preregistered_qwen3_stages":
            continue
        summary = {
            key: value
            for key, value in rows[0].items()
            if key not in metrics and key not in {"layer", "layer_stage"}
        }
        summary["layer_stage"] = stage
        summary["layer_count"] = len(rows)
        for metric in metrics:
            summary[f"mean_{metric}"] = sum(row[metric] for row in rows) / len(rows)
        output.append(summary)
    return output


def archive_audit(source_rows):
    grouped = defaultdict(list)
    for source_path, result in source_rows:
        grouped[str(source_path)].append(result)
    output = []
    for source_path, rows in sorted(grouped.items()):
        values = lambda key: sorted({str(row.get(key)) for row in rows})
        semantics = values("attention_semantics")
        assignments = values("roi_assignment_method")
        paddings = values("roi_padding")
        frame_groups_present = all(bool(row.get("source_frame_groups")) for row in rows)
        metadata_complete = (
            semantics != ["None"]
            and assignments != ["None"]
            and paddings != ["None"]
            and frame_groups_present
        )
        output.append({
            "source_attention_file": source_path,
            "rows": len(rows),
            "attention_semantics": " | ".join(semantics),
            "roi_assignment_method": " | ".join(assignments),
            "roi_padding": " | ".join(paddings),
            "head_reduction": " | ".join(values("head_reduction")),
            "has_source_frame_groups": frame_groups_present,
            "has_fractional_metric_aliases": all(
                "all_token_attention_share"
                in (((row.get("layer_roi_profiles") or [{}])[0].get("spatial_roi") or {}).get("background") or {})
                for row in rows
            ),
            "layer_counts": " | ".join(
                sorted({str(len(row.get("layer_roi_profiles") or [])) for row in rows})
            ),
            "measurement_metadata_complete": metadata_complete,
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, nargs="+")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    source_rows = []
    sources = []
    for input_path in args.input_path:
        rows, input_sources = read_attention_results(input_path)
        source_rows.extend(rows)
        sources.extend(input_sources)
    if not source_rows:
        raise ValueError("No attention JSON arrays with layer_roi_profiles were found.")

    roi_rows = []
    contrast_rows = []
    for source_path, result in source_rows:
        roi_rows.extend(flatten_roi_metrics(source_path, result))
        contrast_rows.extend(target_contrasts(source_path, result))
    stage_rows = stage_contrasts(contrast_rows)
    audit_rows = archive_audit(source_rows)
    audit_signatures = {
        (
            row["attention_semantics"],
            row["roi_assignment_method"],
            row["roi_padding"],
            row["head_reduction"],
            row["has_source_frame_groups"],
        )
        for row in audit_rows
    }
    cross_archive_comparison_ready = (
        all(row["measurement_metadata_complete"] for row in audit_rows)
        and len(audit_signatures) == 1
    )

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "layer_roi_metrics.csv", roi_rows)
    write_csv(output_dir / "layer_target_contrasts.csv", contrast_rows)
    write_csv(output_dir / "stage_target_contrasts.csv", stage_rows)
    write_csv(output_dir / "attention_archive_audit.csv", audit_rows)
    summary = {
        "source_files": sorted({str(path) for path in sources}),
        "attention_rows": len(source_rows),
        "layer_roi_rows": len(roi_rows),
        "layer_target_contrast_rows": len(contrast_rows),
        "stage_target_contrast_rows": len(stage_rows),
        "layer_stage_definition": {
            name: [min(indices), max(indices)]
            for name, indices in LAYER_STAGES.items()
        },
        "contrast_definition": {
            "delta_mass": "log(T1 visual-normalised mass / T2 visual-normalised mass)",
            "delta_enrichment": "log(T1 area-normalised enrichment / T2 area-normalised enrichment)",
            "identity": "delta_enrichment = delta_mass - log(T1 area share / T2 area share)",
        },
        "cross_archive_comparison_ready": cross_archive_comparison_ready,
        "cross_archive_warning": (
            None
            if cross_archive_comparison_ready
            else "Attention semantics or ROI measurement metadata differ or are missing; do not interpret cross-archive heatmap differences as model effects."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(
        f"Analyzed {len(source_rows)} attention rows from {len(set(sources))} files; "
        f"wrote Phase 0 tables to {output_dir}"
    )


if __name__ == "__main__":
    main()
