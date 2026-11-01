# Phase 3E Follow-Up: Original Rescue50 Option Probing

## Decision

On 1 November 2026 the primary probe comparison was moved back to the
original frozen 50 rescue bases, matching Phase 3C/3D patching, KO50 and
singleton KO. The completed 40-new-family paired-video experiment remains
unchanged as supplementary evidence, not a substitute for these 50 bases.

## Fixed Materials And Measurement

- Keep all 50 original Low/Temporal pairs and archived prompt variants.
- Keep video bytes, target identities/paths, event timings, distractors,
  model revision, precision, processor sampling and historical outer folds.
  Do not regenerate an old video with the new 40-family generator.
- Read normal-forward option representations: the last contextual token of
  the reference-object mention, at L16/L17/L18, with L20 as downstream reference.
  Do not revert to within-video T1-minus-T2 features.
- Train statement truth, not the model's answer correctness or A/B letter.
- Compare the same statement in its paired Temporal and Low videos with one
  frozen held-out probe. These original pairs do not reverse event order.
- KO keeps the original options_all queries to Event-2 visual keys route.
  The probe reads a subset of the same option side, not every KO query token.
  No new KO or KO-conditioned probing is required to align cohorts.

## Training And Testing

Preserve Low-only, Temporal-only and full-mixed training. Each fitted probe
tests both conditions of held-out bases. Scaler fitting, inner regularization
selection and all label-based choices use training bases/source conditions
only. All views of a base stay in the same outer/inner fold; each base has one
total training weight.

Include the already-defined pure A/B permutation as a training control,
without changing object roles or the sentence contents. This gives 200 prompt
cells and 400 statement rows: 100 original cells plus 100 permuted cells.
Original prompts still have 46 correct-A and four correct-B bases; the full
original-plus-permuted set has 50 correct-A and 50 correct-B prompts within
each boundary condition. A permuted prompt is not required to retain rescue
behavior and is not an original KO measurement.

Each outer fit uses 40 training bases: 160 statement rows for a single
condition, or 320 for full mixed training. Keep a 160-row matched mixed
control by selecting complementary original/permuted views across the two
conditions, with a deterministic per-base assignment fixed before fitting.
Retain both statements of each selected prompt. Save exact selected IDs and
report their answer-position counts within each training fold.

For direct comparison with existing KO, the primary held-out reporting uses
only the 100 original prompts (200 statements). Report permuted-view results
separately and additionally show all-view results. Joining uses pair_id,
condition, archived eval_id, video hash and original input/baseline evidence,
not just a matching numeric base ID.

Report Low/Temporal accuracy, within-fitted-model AUC, signed statement-truth
evidence levels and paired differences. Bootstrap 50 bases, keeping every
condition/view together. The cohort was selected for rescue behavior: these
results characterize the intervention cohort rather than population accuracy.
Option permutation balances answer positions but does not create same-text
event-order reversals; keep lexical controls in the analysis.

## Implementation And Gates

`scripts/prepare_phase3e_rescue50.py` validates exact source pairs/mappings,
canonical labels, historical fold assignments, the 50 joint KO reference
pairs, and 200 singleton KO pairs. It freezes capture and statement manifests
in a separate new output root. Its audit scope is metadata and checksum-bound
analysis tables, not a replacement for raw-checkpoint replay.

The old 289-feature store does not contain the new last-reference-token
readouts. Object-mention means/differences cannot reconstruct them. However,
Phase 3B's original capture code saved individual vectors for options_all at
every decoder block. First restore the full backup and verify whether each
original prompt's vectors, contextual positions, input and logit provenance
match the frozen KO50 source. Recover supported readouts on CPU. Check pure
A/B-permuted views separately; do not pretend that swapping activation rows
reproduces a new contextual forward. Capture only genuinely missing states or
views. The GPU budget is unknown until this inventory is checked.

Raw-state reuse verification, missing-state capture, feature export and probe
fitting for this rescue50 readout are pending; the 40-family capture/probe
runner must not silently accept this different design.

Before collecting the full set, replay original processor tensors and
baseline logits on a small preflight, check contextual token positions,
read-only hook parity and available persistent storage. Preserve failures
for diagnosis instead of dropping/replacing a case. Restore required sources
from the verified backups if necessary. Do not launch GPU work or regenerate
materials as part of CPU preparation.

```bash
python3 scripts/prepare_phase3e_rescue50.py \
  --source_root /path/to/phase3d/evidence \
  --pairs_path /path/to/original/pairs.json \
  --mappings_path /path/to/original/mappings.json \
  --singleton_root /path/to/singleton/evidence \
  --output_root /path/to/new/rescue50_preparation
```
