"""Strict baseline eligibility and capture-equivalent, patch-free GPU auditing."""

import hashlib
import json
import math
import platform
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path


CONDITIONS = ("low_boundary", "temporal_boundary")
MEASUREMENT_FILES = (
    "run_phase3b_patching.py", "phase3b_core.py", "probe_attention_roi.py",
    "run_eval.py", "activation_patching_core.py", "phase3b_paths.py",
)
BEHAVIORS = {
    "temporal_rescue": (False, True), "stable_both_correct": (True, True),
    "both_wrong": (False, False), "temporal_degradation": (True, False),
}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def measurement_hashes():
    return {name: sha256(Path(__file__).parent / name) for name in MEASUREMENT_FILES}


def pair_failures(pair, baselines):
    failures, correct = [], []
    expected = pair["low_boundary"].get("phase3b_prompt_pair_behavior")
    if expected not in BEHAVIORS:
        failures.append("unknown archived prompt-pair behavior")
    if pair["temporal_boundary"].get("phase3b_prompt_pair_behavior") != expected:
        failures.append("conditions disagree on archived behavior")
    for condition in CONDITIONS:
        row, saved = pair[condition], baselines.get(pair[condition]["eval_id"])
        if saved is None:
            failures.append(f"{condition}: missing baseline")
            continue
        decision = saved.get("capture_decision", saved.get("decision", {}))
        margin = decision.get("margin")
        if margin is None or not math.isfinite(margin) or margin == 0:
            failures.append(f"{condition}: non-finite/missing/zero A/B margin")
        prediction = decision.get("prediction")
        if prediction not in ("A", "B"):
            failures.append(f"{condition}: first token is not A/B")
        if prediction != row.get("archived_prediction"):
            failures.append(f"{condition}: archived prediction mismatch")
        if decision.get("correct_option") != row["correct_option"]:
            failures.append(f"{condition}: correct-option mismatch")
        sign_correct = margin is not None and math.isfinite(margin) and margin > 0
        if (prediction == row["correct_option"]) != sign_correct:
            failures.append(f"{condition}: prediction and margin sign disagree")
        correct.append(sign_correct)
        parity = saved.get("standard_parity") or {}
        if not parity.get("first_token_match") or not parity.get("logits_allclose"):
            failures.append(f"{condition}: standard-generation parity failed/missing")
        if not (saved.get("archived_input_parity") or {}).get("matches"):
            failures.append(f"{condition}: archived processor metadata failed/missing")
    if expected in BEHAVIORS and tuple(correct) != BEHAVIORS[expected]:
        failures.append("strict margin-sign behavioral category changed")
    return failures


def runtime_signature():
    import torch
    from run_phase3b_patching import gpu_hardware_metadata
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "packages": {name: version(name) for name in (
            "transformers", "qwen-vl-utils", "numpy", "accelerate", "av", "torchcodec",
        )},
        "gpu_hardware": gpu_hardware_metadata(),
    }


class BaselineEngine:
    def __init__(self, settings, expected_placement=None):
        from activation_patching_core import locate_decoder_layers
        from run_eval import configure_reproducibility
        from run_phase3b_patching import load_phase3b_model, validate_preflight_hardware
        validate_preflight_hardware(settings)
        configure_reproducibility(settings.seed, deterministic=True)
        self.model, self.processor = load_phase3b_model(settings)
        self.placement = {name: str(device) for name, device in self.model.hf_device_map.items()}
        if expected_placement is not None and self.placement != expected_placement:
            raise RuntimeError("Baseline model placement differs from the VM preflight.")
        self.layers, _ = locate_decoder_layers(self.model)
        if len(self.layers) != 36:
            raise RuntimeError("Baseline gate requires the fixed 36-layer decoder.")

    def audit(self, prepared):
        import torch
        from activation_patching_core import (
            decision_from_logits, hidden_from_layer_output, validate_archived_input_metadata,
        )
        from phase3b_core import GROUPS
        from probe_attention_roi import model_forward, standard_first_token
        metadata = validate_archived_input_metadata(prepared)
        groups = prepared["groups"]
        if any(not groups[name] for name in GROUPS):
            raise RuntimeError("A planned token group is empty in the baseline gate.")
        union = sorted({position for name in GROUPS for position in groups[name]})
        snapshots = {}

        def collect(layer):
            def hook(_module, _inputs, output):
                hidden = hidden_from_layer_output(output)
                positions = torch.as_tensor(union, device=hidden.device, dtype=torch.long)
                snapshots[layer] = hidden[0].index_select(0, positions).detach().cpu()
                return output
            return hook

        with ExitStack() as stack:
            for layer, module in enumerate(self.layers):
                stack.callback(module.register_forward_hook(collect(layer)).remove)
            with torch.inference_mode():
                output = model_forward(self.model, {
                    **dict(prepared["inputs"]), "use_cache": False, "return_dict": True,
                })
        if len(snapshots) != 36:
            raise RuntimeError("Not all capture-equivalent hooks executed.")
        logits = output.logits[0, -1, :].float().detach().cpu()
        del output, snapshots
        decision = decision_from_logits(logits, self.processor, prepared["row"]["correct_option"])
        standard_id, standard_logits = standard_first_token(self.model, prepared["inputs"])
        standard_logits = standard_logits.float().detach().cpu()
        finite = torch.isfinite(logits) & torch.isfinite(standard_logits)
        finite_match = torch.equal(torch.isfinite(logits), torch.isfinite(standard_logits))
        difference = (logits[finite] - standard_logits[finite]).abs()
        return {
            "eval_id": prepared["row"]["eval_id"],
            "capture_decision": decision,
            "standard_parity": {
                "first_token_match": standard_id == decision["predicted_first_token_id"],
                "logits_allclose": finite_match and bool(finite.any()) and bool(torch.allclose(
                    logits[finite], standard_logits[finite], rtol=0.001, atol=0.25,
                )),
                "max_abs_diff": float(difference.max()) if difference.numel() else None,
            },
            "archived_input_parity": metadata,
            "input_metadata": prepared["input_metadata"],
            "captured_layer_count": 36,
            "video_sha256": sha256(prepared["video_path"]),
            "input_tensor_sha256": {
                name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                for name, value in prepared["inputs"].items() if torch.is_tensor(value)
            },
        }
