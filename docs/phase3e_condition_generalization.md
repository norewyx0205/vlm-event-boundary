# Phase 3E-B Follow-Up: Condition Generalization

## Material Passport

- Date: 1 November 2026.
- Status: CPU implementation and validation; scientific results require a completed run.
- Approved scope: condition-specific statement-truth training and cross-condition testing.
- Source: the existing complete, verified event-relative v2 paired-video feature store.
- Preserve all previous captures, mixed-probe models, reports and backups unchanged.
- No new videos, GPU forwards, KO, patches or layer search.

## Question And Fixed Readouts

Does a truth readout learned in Low generalize to Temporal, and vice versa?
Distinguish condition-specific linear accessibility from transfer of a shared
readout rule. Keep statement truth as the label, not boundary or A/B choice.
Use the last contextual reference-object token at L16/L17/L18 block outputs;
L20 is a fixed downstream reference. This is a follow-up motivated by the
already observed mixed-probe results, not an independent preregistration.

| Training regime | Rows per outer fit | Test Low | Test Temporal |
| --- | ---: | --- | --- |
| Low only | 512 | Within condition | Cross condition |
| Temporal only | 512 | Cross condition | Within condition |
| Matched-budget mixed | 512 | Mixed reference | Mixed reference |
| Full mixed, original model reuse | 1,024 | Original reference | Original reference |

All regimes have the same 32 training and eight held-out scene families per
outer fold. Every source-trained model evaluates all 256 statement rows of
those eight test families, including both boundary conditions. All order,
subject and A/B variants of a family stay together. Reuse the frozen five
outer folds and four inner folds, regularization grid, model class and solver.

## Source-Only Fitting

For Low-only fitting, both inner training and inner validation use Low rows
only; Temporal rows cannot affect the scaler, C selection or coefficients.
Apply the resulting scaler and classifier unchanged to both test conditions.
Reverse those roles for Temporal-only. Matched mixed uses only its declared
subset for both inner training and validation. No test-condition adaptation or
calibration is performed. One scene contributes one total classifier sample
weight, so row-budget changes do not change total regularization weight.

## Matched Mixed Selection

Each family has eight prompts per boundary and two statements per prompt.
Select four prompts per boundary, keeping both event-order partners and both
statements. Select complementary halves of the subject-by-layout combinations
in Low and Temporal. A deterministic, seed-42 family phase counterbalances
which half belongs to which condition. Selection never uses activations,
model decisions, probe results or label correctness to choose a subset.

Within each family and selected condition, truth labels, first mover,
subject, layout and correct A/B letter are balanced. Identical option text at
the same option position occurs with both truth labels. The half-budget set
cannot retain every joint subject/layout combination within each condition;
its subset and family phases are saved explicitly. Both test conditions
always retain the complete factorial, irrespective of training regime.

## Reuse And Verification

Verify the old mixed run's feature-store hash, settings, code, protocol,
runtime, fold membership, numeric model replay and paired statistics before
copying its four readouts. The new result directory contains byte-identical
copies of those numeric model resources plus provenance; original files are
never changed. Save 60 newly fitted outer models and reuse 20 original ones.

For every job, save source-row selection audits, inner-fold row hashes,
scaler parameters, coefficients, predictions, paired statistics and hashes.
Independent verification replays scores/probabilities and direct D
projections, reconstructs source-only scaler parameters, and recomputes all
statistics without refitting classifiers. Reject changed provenance, partial
jobs and overlapping output/source directories. Failed jobs are inspected,
not silently retried or overwritten.

## Outcomes

- Primary: held-out accuracy/balanced accuracy in the four source/test cells.
- Secondary: mean within-outer-fold ROC-AUC; do not pool independently fitted
  probe logits into a single cross-fold AUC.
- Paired: raw Low and Temporal signed truth evidence and their difference,
  raw true-video/false-video scores, and paired family accuracy changes.
- Retain subject and option-position breakdowns. Show original levels with
  every reported difference. Resample 40 families with the existing 2,000
  bootstrap replicates, conditional on saved probes; no new significance tests.

Compare logits only within a fixed fitted readout, not between separately
trained regimes/layers. Cross-condition accuracy loss can arise from
threshold/distribution changes; it is not alone evidence of distinct semantic
codes. AUC helps distinguish ranking from threshold sensitivity. No new causal
claim about temporal gaps or a specific attention route follows from probing.

## Execution

```bash
env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 scripts/run_phase3e_condition_probes.py \
  --manifest_path /path/to/feature_store/manifest.json \
  --mixed_reference_root /path/to/original_mixed_probes/v1 \
  --output_root /path/to/new_condition_probes/v1
```

Repeat with the same arguments and `--verify_only` for independent frozen
model/statistic replay. This CLI makes no network or VM lifecycle requests.
