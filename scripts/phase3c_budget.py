"""Fixed-token-budget supports for the exploratory Phase 3C dose follow-up."""

import copy
import hashlib
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from . import phase3c_support as source
    from .phase3c_core import CONDITIONS, digest, event_layout, file_hash, frozen_write, read_json, support_record
except ImportError:
    import phase3c_support as source
    from phase3c_core import CONDITIONS, digest, event_layout, file_hash, frozen_write, read_json, support_record


SCHEMA = "phase3c_budget_followup_v1"
BUDGETS = (64, 128, 256)
LAYERS = (12, 16)
SEEDS = (42, 43, 44)
FAMILIES = ("context_only", "targets_plus_context")


def support_name(family, budget, seed):
    return f"{family}_b{budget}_s{seed}"


def allocate_budget(budget, capacities):
    total = sum(capacities.values())
    if type(budget) is not int or not 0 < budget <= total or any(value <= 0 for value in capacities.values()):
        raise ValueError("Invalid fixed token budget or temporal-bin capacities.")
    quotas = {key: budget * count // total for key, count in capacities.items()}
    order = sorted(capacities, key=lambda key: (-(budget * capacities[key] % total), key))
    for key in order[:budget - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def audit_supports(pair_id, pair, mapping, budgets=BUDGETS, seeds=SEEDS):
    # Reuse the original strict observed-donor correspondence audit, not shape matching.
    original = source.audit_supports(pair_id, pair, mapping)
    old = mapping["support_audit"]["supports"]
    whole = list(zip(old["whole_event2"]["low_positions"], old["whole_event2"]["temporal_positions"]))
    targets = set(zip(old["both_targets_event2"]["low_positions"], old["both_targets_event2"]["temporal_positions"]))
    layout = [event_layout(pair[condition], mapping["processor_records"][condition]["video_metadata"],
        mapping["processor_records"][condition]["visual_positions"],
        mapping["processor_records"][condition]["prompt_token_count"]) for condition in CONDITIONS]
    bins = defaultdict(list)
    for left, right in whole:
        key = (layout[0]["lookup"][left][0], layout[1]["lookup"][right][0])
        bins[key].append((left, right))
    capacities = {key: len(items) for key, items in bins.items()}
    supports, specs, counts = {}, {}, []
    budgets, seeds = tuple(budgets), tuple(seeds)
    if not budgets or len(set(budgets)) != len(budgets) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Budgets/seeds must be nonempty and unique.")
    for seed in seeds:
        ranked = {key: sorted([item for item in items if item not in targets],
            key=lambda item: digest([SCHEMA, seed, pair_id, key, layout[0]["lookup"][item[0]][1:]]))
            for key, items in bins.items()}
        previous = {family: set() for family in FAMILIES}
        for budget in sorted(budgets):
            quotas = allocate_budget(budget, capacities)
            chosen = {family: [] for family in FAMILIES}
            for key, items in sorted(bins.items()):
                target = [item for item in items if item in targets]
                pool, quota = ranked[key], quotas[key]
                if len(target) > quota or len(pool) < quota:
                    raise ValueError("Ineligible fixed budget: target inclusion/context capacity conflicts with frozen bin quota.")
                chosen["context_only"].extend(pool[:quota])
                chosen["targets_plus_context"].extend(target + pool[:quota - len(target)])
                counts.append({"seed": seed, "budget": budget, "low_temporal_index": key[0],
                    "temporal_temporal_index": key[1], "quota_each_family": quota,
                    "target_count": len(target), "available_context_count": len(pool)})
            for family, positions in chosen.items():
                current = set(positions)
                if len(current) != budget or not previous[family] <= current:
                    raise ValueError("Supports must meet exact budgets and be nested across doses.")
                if family == "context_only" and current & targets:
                    raise ValueError("Context-only support contains target-union positions.")
                if family == "targets_plus_context" and not targets <= current:
                    raise ValueError("Mixed support does not include the complete target union.")
                previous[family] = current
                name = support_name(family, budget, seed)
                low, temporal = ([item[side] for item in positions] for side in (0, 1))
                supports[name] = support_record(low, temporal, low, temporal, name)
                specs[name] = {"family": family, "budget": budget, "sampling_seed": seed}
    return {"pair_id": pair_id, "supports": supports, "support_specs": specs,
        "token_budget_by_bin": counts, "target_union_count": original["target_union_count"],
        "whole_count": len(whole), "nested_within_seed": True,
        "alignment": original["alignment"], "quota_rule": "whole-grid-proportional-largest-remainder",
        "content_limit": "Non-target-position residuals may encode targets; equal budgets do not match information content."}


def task_grid(config, stage):
    if stage not in ("replay", "preflight", "patch"):
        raise ValueError("Unknown budget-followup stage.")
    ids = config["preflight_case_ids"] if stage == "preflight" else config["case_ids"]
    tasks = []
    for pair_id in ids:
        for condition in CONDITIONS:
            if stage == "replay":
                tasks.append({"pair_id": pair_id, "condition": condition, "kind": "capture_replay"})
                continue
            for support in sorted(config["support_audit"][pair_id]["supports"]):
                for layer in LAYERS:
                    tasks.append({"pair_id": pair_id, "condition": condition,
                        "kind": "identity" if stage == "preflight" else "support_patch",
                        "support": support, "layer": layer, "location": "block_output"})
    for task in tasks:
        task["task_id"] = digest(task)
    return tasks


def code_hashes():
    return {name: file_hash(Path(__file__).parent / name)
            for name in ("phase3c_budget.py", "run_phase3c_budget.py")}


def prepare(args, freeze=False):
    try:
        from .run_phase3c_support import saved_rows, verify_analysis
    except ImportError:
        from run_phase3c_support import saved_rows, verify_analysis
    root, prior = Path(args.output_root).resolve(), Path(args.source_followup).resolve()
    for protected in (Path(args.plan_dir).resolve(), Path(args.source_run).resolve()):
        if root == protected or protected in root.parents or root in protected.parents:
            raise ValueError("The original plan/execution must remain read-only.")
    if root == prior or prior in root.parents or root in prior.parents:
        raise ValueError("Keep the completed support follow-up read-only; use a separate output root.")
    old, frozen, pairs, mappings, baselines = source.prepare(args.plan_dir, args.source_run,
        args.source_followup, args.source_backup)
    saved_rows(prior, "preflight", old, pairs, baselines, complete=True)
    saved_rows(prior, "patch", old, pairs, baselines, complete=True)
    verify_analysis(type("Source", (), {"output_root": str(prior)})(), old)
    audits = {key: audit_supports(key, pairs[key], mapping) for key, mapping in mappings.items()}
    if len(pairs) != 12 or len({pair[CONDITIONS[0]]["base_sample_id"] for pair in pairs.values()}) != 12:
        raise ValueError("Reuse exactly the 12 independent frozen pilot cases.")
    config = {**old, "schema": SCHEMA, "output_root": str(root), "layers": list(LAYERS),
        "budgets": list(BUDGETS), "sampling_seeds": list(SEEDS), "support_families": list(FAMILIES),
        "source_followup_root": str(prior),
        "source_followup_config_sha256": file_hash(prior / "support_config.json"),
        "source_followup_analysis_sha256": file_hash(prior / "analysis/aggregate_summary.json"),
        "followup_code_sha256": code_hashes(), "support_audit": audits,
        "primary_patch_count": 864, "technical_identity_count": 144, "capture_replay_count": 24,
        "expected_replay_signatures": replay_signatures(pairs, mappings, baselines),
        "replay_policy": "exact full-vocabulary logits and all whole-event L12/L16 vectors in every condition",
        "runtime_policy": "all fields exact except recorded GPU memory-capacity-only differences; exact replay required",
        "seed_aggregation": "equal-weight within-case means before case-level bootstrap; seeds are not independent cases",
        "interpretation": "Exploratory equal-budget support/dose comparison on reused cases, not causal mediation."}
    config.pop("fingerprint")
    config["fingerprint"] = digest(config)
    path = root / "budget_config.json"
    if freeze:
        frozen_write(path, config)
    elif not path.is_file() or read_json(path) != config:
        raise ValueError("Budget configuration missing/changed; preserve old artifacts and explicitly freeze a new root.")
    return config, frozen, pairs, mappings, baselines


def tensor_signature(value):
    import torch
    value = value.detach().cpu().contiguous()
    return {"dtype": str(value.dtype), "shape": list(value.shape),
            "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}


def replay_signatures(pairs, mappings, baselines):
    import torch
    try:
        from .phase3c_interventions import selected_state
    except ImportError:
        from phase3c_interventions import selected_state
    result = {}
    for pair_id, pair in pairs.items():
        for condition in CONDITIONS:
            row = baselines[pair[condition]["eval_id"]]
            index = read_json(row["capture_index_path"])
            saved = torch.load(index["vectors_path"], map_location="cpu", weights_only=True)
            side = "low_positions" if condition == CONDITIONS[0] else "temporal_positions"
            positions = mappings[pair_id]["support_audit"]["supports"]["whole_event2"][side]
            result[pair[condition]["eval_id"]] = {"logits": tensor_signature(saved["logits"]),
                "positions": positions, "sites": {str(layer): tensor_signature(
                    selected_state(saved["states"], layer, "block_output", positions)["vectors"])
                    for layer in LAYERS}}
            del saved
    return result


def validate_config(config):
    if (config.get("schema") != SCHEMA or config.get("artifact_type") != "real" or
            config["fingerprint"] != digest({key: value for key, value in config.items() if key != "fingerprint"}) or
            config["followup_code_sha256"] != code_hashes()):
        raise ValueError("Invalid/changed budget binding.")


def runtime_compatibility(expected, observed):
    normalized = copy.deepcopy(observed)
    changes = []
    before = expected.get("gpu_hardware", {}).get("devices", [])
    after = normalized.get("gpu_hardware", {}).get("devices", [])
    if len(before) != len(after):
        raise ValueError("Runtime device count changed.")
    for index, (source_device, live_device) in enumerate(zip(before, after)):
        a, b = source_device.get("total_memory_bytes"), live_device.get("total_memory_bytes")
        if a != b:
            if type(a) is not int or type(b) is not int or min(a, b) <= 0:
                raise ValueError("Invalid memory-capacity metadata.")
            changes.append({"device": index, "source_total_memory_bytes": a, "observed_total_memory_bytes": b})
            live_device["total_memory_bytes"] = a
    if normalized != expected:
        raise ValueError("Compute/runtime/placement changed; only capacity metadata differences may enter exact replay.")
    return {"memory_capacity_changes": changes, "exact_replay_required": True}


def collapse_seeds(rows, seeds=SEEDS):
    keys = ("pair_id", "base_sample_id", "analysis_stratum", "first_object_id", "direction", "layer", "family", "budget")
    groups = defaultdict(dict)
    metrics = ("margin_delta", "source_aligned_margin_delta", "recovery", "strict_incorrect_to_correct",
               "strict_sign_crossing", "categorical_flip", "replaced_token_count")
    for row in rows:
        group = groups[tuple(row[key] for key in keys)]
        if row["sampling_seed"] in group:
            raise ValueError("Duplicate sampling seed in a case/support/dose cell.")
        group[row["sampling_seed"]] = row
    result = []
    for cell, group in sorted(groups.items(), key=lambda item: str(item[0])):
        if set(group) != set(seeds):
            raise ValueError("Incomplete sampling seeds; analysis is blocked.")
        item = dict(zip(keys, cell))
        item.update(seed_count=len(seeds), location="block_output")
        for name in metrics:
            values = [group[seed][name] for seed in seeds]
            if any(value is None for value in values):
                if any(value is not None for value in values):
                    raise ValueError("Inconsistent missing recovery denominator.")
                item[name] = None
            else:
                item[name] = statistics.mean(values)
        values = [group[seed]["margin_delta"] for seed in seeds]
        item.update(seed_margin_min=min(values), seed_margin_max=max(values),
                    seed_margin_sd=statistics.stdev(values) if len(values) > 1 else 0)
        result.append(item)
    return result
