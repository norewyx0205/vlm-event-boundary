# Phase 3D: Temporal-Relation Evidence and L16 Transfer

Protocol date: 2026-10-19 (Europe/Amsterdam).
This is a pre-specified follow-up informed by prior results, not formal
preregistration or a new independently sampled rescue population.

Status: approved scientific design; CPU metadata preparation, compact capture
primitives, linear-probe fitting and KO50 task planning are implemented and
unit-tested. The live GPU collection/export runner, KO50 reuse/execution and
real-model technical gates remain pending. No scientific probe results or
GPU-readiness claim follow from unit tests. No automatic VM launch.

Reference: Song, Paperno and Gatt, *Who Is Left of Whom? Tracing Spatial Evidence
and Role Binding in Relative-Position Reasoning*, Sections 5-6:
https://arxiv.org/pdf/2609.35486 . Adapt its distinction between readable
object-relation information and intervention-tested source-to-query transfer;
do not assume spatial axes or layer ranges transfer unchanged to this task.

## 1. Cohort, Isolation and Shared Gates

Use all 50 frozen, VM-baseline-verified Phase 3B/3C rescue bases, once per base.
Primary reporting treats every base equally, without old/new/pilot strata.
Keep historical provenance for auditing and safe reuse, not case selection.
Do not select on visual-patch success or replace a case after a new outcome.
This is a selective rescue cohort, not a random population sample. Late answer
decodability is additionally conditioned on rescue-based behavioral selection.

The archived prompt remains the primary view. Derive a second view by swapping
the exact A/B option strings and answer label, keeping object identities,
sentence contents, video and canonical relation unchanged. Existing
`prompt_variant=swapped` is not a guarantee of this pure permutation: archived
mirroring can reverse sentence subject/reference. Record these as distinct
transformations. Do not require the derived view to remain a behavioral rescue
and never count it as a new independent base.

Before launch, freeze all case/view IDs, canonical labels, positions, supports,
pooling, folds, fitting settings and source/code hashes. Live GPU checks must
verify processor bytes, complete logits, greedy first token, source-state
replay, disabled hooks and same-state identity replacements. Missing or failed
technical gates stop work; no silent case dropping or tolerance relaxation.
New prompt views require fresh tokenization and baselines; never attach an
archived prompt's input hashes, outputs or capture to a changed option order.

Use isolated `/data/yuxuanstorage/vlm_phase3d/` configuration, checkpoints,
captures, probes, analysis and backups. Phase 3B/3C sources remain read-only.
Storage estimation, reserve checks, resumable per-task checkpoints, elapsed-time
progress, verified full backup, email and explicitly authorized idle-checked
SURF Pause are required before unattended GPU execution. Only verified source
results can be reused; never rerun completed work merely to relabel provenance.

## 2. Experiment A: Knockout50

Extend the exact existing KO25 routes to all 50 frozen bases:

- `query_all` and `options_all` queries to T1, T2 and both-target Event-2 ROI
  keys, each with the existing causal-edge-budget-matched background control.
- Separately, `options_all` queries to the complete matched Event-2 grid.
- All heads; nine separate windows: 0-3, 4-7, ..., 32-35 (zero-based).
- Apply each intervention independently to low and temporal conditions.

This gives 5,850 paired tasks / 11,700 condition forwards. At most 2,925 prior
paired KO25 tasks are reuse candidates; accept them only after exact case,
input, runtime, mask, edge-budget and output provenance/parity checks. Do not
rewrite old task IDs, manifests or analyses. New computation and reused rows
have equal weight in the new complete 50-base primary summaries.

Report `delta_M_temporal`, `delta_M_low`, and
`compression = delta_M_low - delta_M_temporal`, plus intact/KO margins, strict
sign crossings, categorical and non-A/B diagnostics. Preserve matched-background
contrasts. Whole-event KO has no equal-sized, same-event disjoint background
control; larger effects do not isolate target specificity, content from routing,
or softmax competition. Unmodified condition/prompt text is not evidence that
all Event-2 information was removed: other layers and token routes remain.

## 3. Experiment B: Relation Probing

Canonical label: `y=1` for **T1-before-T2**, `y=0` for **T1-after-T2**. Derive it
from target event timing and verify first/second-object metadata. It must be
identical across low/temporal and option-order views. Check sentence truth and
archived answer labels separately: the word in `correct_relation` is relative
to the sentence subject, which need not be T1. Never use A/B prediction,
correctness, condition or patch success as the relation label.

Use regularized linear logistic probes, not nonlinear probes. Within each
layer/site/fold, fit one common relation readout on unmodified low and temporal
training bases, with equal condition weight; evaluate both with the same
frozen coefficients and intercept. The pure option-permuted view is a held-out
robustness evaluation, not an extra training example in the primary analysis.

Freeze five relation-stratified outer folds, seed 42, at **base** level. Every
condition, prompt view, token group and patched version of a base stays in its
outer test fold. Fit scalers/centering and any feature transforms using training
bases only. Select L2 strength inside four grouped inner training folds only:
`C = 0.001, 0.01, 0.1, 1.0`; ties favor the smallest C. Use liblinear/L2/dual,
class-weight balanced, tolerance 1e-7 and max_iter 10000. Convergence warnings
stop fitting. Each paired training condition gets sample weight 0.5. Record
NumPy/scikit-learn/Python versions in the immutable run configuration. Do not
tune by outer-test accuracy, boundary gain or patch effect. No PCA or
alternative pooling selected by test performance.

Primary downstream feature: for each literal identity, pool its mention tokens
separately within A and B, give the two options equal weight, then take the
canonical T1-minus-T2 vector. Retain separate option/object token vectors so the
pure permutation and lexical controls can be audited. Do not treat a probe that
identifies the visible word 'before' versus 'after' as relational understanding.
Keep decision-position and source-visual readouts separate; do not combine
positions/layers into one high-capacity classifier. The analogous source
feature is mean(literal T1/Event2 ROI) minus mean(literal T2/Event2 ROI).
Require both groups to be nonempty and entirely within the matched Event-2
support; insufficient support stops the run, not case dropping or silent ROI
cropping. Preserve source counts/positions as geometry-shortcut diagnostics.

Descriptive layerwise readouts cover decoder block outputs 0-35. Record L16
visual block input and block output explicitly. L16 input can be matched to
L15 output only after verifying the architecture's intervening operations.
The primary downstream site is **L20 option-object representations**, motivated
by earlier whole-option interventions. The pre-specified propagation window is
L17-L24; other layer profiles are descriptive, not independent winning-layer
tests. Previously sparse visual stores are not full-layer text captures.

For a frozen probe logit `z(h)=w.h+b` predicting `y=1`, define true-relation
evidence `S_y(h)=(2*y-1)*z(h)`. The boundary-associated evidence gain is
`delta_S_boundary = S_y(h_temporal) - S_y(h_low)`. Do not label `h_T-h_L` itself
as temporal-order information. Report held-out balanced accuracy, class counts,
log loss, continuous evidence and paired case-level gains. Positive correct
relation evidence in an incorrect low run is a valid result: representation
and effective use need not coincide.

Required controls use the same base-level train/test separation:

- Base-label permutation with all paired views permuted together and the
  fitting procedure repeated; never shuffle individual token/condition rows.
  Run 20 fixed-seed permutations at the primary L20 readout as null diagnostics,
  not a high-resolution confirmatory permutation test.
- A separate condition-only probe; successful condition decoding is not
  evidence of relation decoding.
- Static object-feature and text-only/lexical/answer-position baselines. Fit
  any vocabulary/attribute encoding on training folds only; retain unseen
  attributes. Report shortcuts rather than silently residualizing them away.
- The explicit pure A/B permutation: retain canonical y and compare relation
  readout under the answer-label swap. Do not screen it for correct behavior.

## 4. Experiment C: Patch-Conditioned Probing

Intervene only at **L16 visual block output**, using the already defined
512-position, complete one-to-one observed-donor Event-2 support. Primary
direction: temporal to low; reverse is the fixed comparison. Keep recipient
inputs, positional encoding and masks unchanged. No new patch location grid,
dummy tokens, interpolation, donor duplication or direction steering.

Apply the already fitted outer-fold probe to held-out patched cases. Never fit,
calibrate, select pooling or tune regularization on patched representations or
their behavioral outcomes. Record actual donor/recipient supports and capture
location semantics, not just tensor dimensions.

L16 block-output visual replacement cannot alter option outputs already
computed inside that block. Primary propagation is measured at unpatched **L20
option positions**, with L17-L24 as the propagation profile. Same-block text
states must remain unchanged as a timing check. A readout from a directly
transplanted visual vector is not evidence of downstream transfer.

For the held-out true label y, compute
`delta_S_patch = S_y(h_low_patched) - S_y(h_low)` and the corresponding change
in first-answer-token correct-option A/B margin. Retain signed per-case values,
including adverse patches and cases that do not flip. Compute the association
across bases at the fixed L20 readout, not across pooled layer/token/condition
rows. Keep model answer margin and probe evidence clearly named and separate.
Reverse-direction changes are recipient-relative, never silently sign-flipped.

## 5. Statistics, Claims and Remaining Implementation Gates

There are **50 independent bases**, not 100 independent conditions or more
independent observations after adding views/tokens. Use paired base-level
bootstrap and base-level effect correlations. Cross-fitted predictions come
from different training folds: use raw signed logits from one shared readout
within each fold for paired comparisons, save fold/probe IDs, and do not imply
that a unit is globally calibrated across independently fitted probes. A
bootstrap of saved out-of-fold scores is conditional on those fitted probes;
it does not include complete model-refitting uncertainty. Keep layer profiles
descriptive and L20/L16 primary sites fixed.

The supported claim is that decodable temporal-relation changes are consistent
with the causal intervention's downstream and behavioral effects. Probe success
alone is not causal computation; paired score/margin changes do not prove
mediation or a unique routing mechanism. L20 necessity would require separately
approved mediation/path-specific intervention, outside this phase.

Before GPU launch, finish and test: exact per-option token spans and source
pooling against the live tokenizer; baseline and patch-conditioned compact
capture/identity/replay/export gates; KO50 verified reuse/merge; independent
analysis; storage/backup/lifecycle orchestration. See the implementation status
and feature-store contract in `phase3d_implementation.md`. The CPU preparation
below freezes only the audited metadata/fold inputs, not a launch-ready GPU
protocol:

```bash
python scripts/prepare_phase3d.py \
  --pairs_path analysis/phase3c/depth50_ko25_results_20261014/prefix/run/pairs.json \
  --mappings_path analysis/phase3c/depth50_ko25_results_20261014/prefix/run/mappings.json \
  --source_capture_refs analysis/phase3c/depth50_ko25_results_20261014/prefix/run/source_capture_refs.json \
  --capture_root analysis/phase3c/depth50_ko25_results_20261014/prefix/run/captures \
  --capture_root analysis/phase3c/depth50_ko25_results_20261014/run/captures \
  --output_dir analysis/phase3d/preparation_v3
```
