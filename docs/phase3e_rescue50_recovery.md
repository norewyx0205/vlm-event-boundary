# Rescue50 Option-State Recovery

## Scope

Recover the primary archived prompts on CPU before scheduling new inference.
The original 50 Low/Temporal pairs and their videos, text and baseline logits
remain unchanged. No KO is repeated. The frozen protocol is
`phase3e_rescue50_option_protocol.md`.

## Recovery

`recover_phase3e_rescue50.py` reads existing local full backups without
extracting entire archives. It checks the SHA256 of each opened archive part
and every consumed member. Absent unused parts are explicitly recorded; this
is not a claim that a whole partially restored backup has been re-verified.

Phase 3D supplies sealed processor audits, per-option identity spans,
statement semantics and raw baseline checks. Phase 3B supplies raw option
token vectors at each block output. Reuse requires the original video hash,
sampled metadata, prompt token IDs, attention mask, text/visual positions and
baseline evidence to match. The historical processor-record schemas have
different auxiliary fields; compare explicit behavioral input fields exactly
and retain Phase 3D's separate sealed tensor-hash binding.

Full baseline logits match exactly. Single-token anchors (decision position
and relation words) also match exactly. Multi-token FP32 means can differ
between CPU and GPU reduction kernels. Their acceptance bound is derived
componentwise from group length, FP32 epsilon and absolute input values,
including subtraction roundoff. It is not an empirical tolerance selected
to accept a discrepancy. Save the actual differences in the replay ledger.

Export the original reference-object last-token vector directly, not an
object difference or reconstructed mean. The result includes all four fixed
readouts (L16/L17/L18/L20), exact statement IDs, source provenance and a
missing-task manifest. A partial result is marked `complete: false` and
`ready_for_formal_probe_training: false`.

## Independent Verification

`verify_phase3e_rescue50_recovery.py` selects reference tokens independently:
split the original options_all positions into A/B sections, intersect the
appropriate object-identity mention group, and take its last token. Replay
the recovered vectors exactly from the original per-layer tensors. No
pooling, fitting or model inference is used by this verifier.

## Current Coverage

The 1 November recovery restored all 100 original prompts (200 statements),
with four readouts per statement. All 100 original video files were checked.
It replayed 400 layer payloads, 1,200 exact single-token anchors and 2,400
historical feature vectors. Maximum multi-token pooling difference was
0.000003815, within the derived FP32 envelope.

The 100 pure A/B-permuted prompts still require their contextual vectors.
Their Phase 3D captures persist pooled groups and differences, not individual
reference-object tokens. Swapping old vectors does not reproduce the new
context. Keep these cells in the acquisition manifest; do not replace or
exclude bases and do not train the requested balanced-view experiment on the
archived-only partial set.

Next acquisition scope: 100 normal-forward A/B control prompts on the same
100 existing videos, plus technical parity checks. No new videos, cases,
event-order reversals, KO or patches are required. Start with a small real
model preflight, retain original model/input settings and source logits,
then export the complete 400-statement store before fitting Low-only,
Temporal-only and mixed probes.

## Commands

```bash
env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 scripts/recover_phase3e_rescue50.py \
  --plan_path /path/to/rescue50_preparation/plan.json \
  --phase3b_manifest /path/to/phase3b_backup/backup_manifest.json \
  --phase3d_manifest /path/to/phase3d_backup/backup_manifest.json \
  --phase3d_parts /path/to/restored/part_001.zip \
  --output_root /path/to/new/raw_recovery

python3 scripts/verify_phase3e_rescue50_recovery.py \
  --plan_path /path/to/rescue50_preparation/plan.json \
  --recovery_manifest /path/to/new/raw_recovery/recovery_manifest.json \
  --output_path /path/to/separate/independent_verification.json
```
