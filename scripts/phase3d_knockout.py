"""Equal-weight KO50 planning; provenance candidates are not validated reuse."""

try:
    from .phase3c_core import digest
    from .phase3c_knockout25 import WINDOWS, tasks
    from .phase3d_core import audit_pairs
except ImportError:
    from phase3c_core import digest
    from phase3c_knockout25 import WINDOWS, tasks
    from phase3d_core import audit_pairs


def intent(pairs, source_ko25_summary=None):
    cases = audit_pairs(pairs)
    result = {"schema": "phase3d_knockout50_intent_v1", "case_count": 50,
              "case_ids": [case["pair_id"] for case in cases],
              "query_groups": ["query_all", "options_all"],
              "key_groups": ["target_1", "target_2", "both_targets"],
              "whole_event2_enabled": True, "windows": [list(window) for window in WINDOWS],
              "primary_reporting": "All 50 independent bases equally; no pilot/new stratification.",
              "reuse_status": "candidate_only_requires_exact_live_input_runtime_mask_and_output_gates",
              "whole_event_control_limit": "No disjoint same-event 512-key matched background.",
              "source_ko25_summary_sha256": None}
    if source_ko25_summary is not None:
        if (source_ko25_summary.get("complete") is not True or
                source_ko25_summary.get("artifact_type") != "real" or
                source_ko25_summary.get("case_count") != 25 or
                source_ko25_summary.get("paired_rows") != 2925):
            raise ValueError("Reuse planning requires a complete real KO25 summary.")
        result["source_ko25_summary_sha256"] = digest(source_ko25_summary)
    result["paired_task_count"] = len(tasks(result, "count_only"))
    result["condition_forward_count"] = 2*result["paired_task_count"]
    result["validated_reused_tasks"] = 0
    result["fingerprint"] = digest(result)
    return result


def planned_tasks(config):
    if config["fingerprint"] != digest({key: value for key, value in config.items() if key != "fingerprint"}):
        raise ValueError("Frozen KO50 intent changed.")
    return tasks(config, config["fingerprint"])
