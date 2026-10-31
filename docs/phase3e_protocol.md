# Phase 3E: Controlled Relation-Probe Follow-up

**31 October update:** the current Phase 3E-B protocol is
`phase3e_b_paired_option_protocol.md`. It replaces cross-object D as the primary
feature with normal-forward single-option readouts and same-object order-paired
contrasts. Option-side alignment with the existing KO is sufficient; new KO is
not required. Second-mover identification is valid task evidence. The text below
preserves the earlier CPU work and historical design discussion, not the current
GPU launch specification. A normal two-scene technical preflight was completed
on 27 October; its data and report remain unchanged.

27 October 2026. The scoped CPU reanalysis has completed on the verified
restored Phase 3D feature store; no new GPU captures have been produced. This is a
retrospective correction/follow-up to SQ4, not a replacement of the thesis
questions or a retroactive preregistration of Phase 3D.

Reports follow `docs/reporting_conventions.md`: give the original paired levels
alongside each change, with metric, direction, views and independent base count.

## Why These Contrasts Are Different

At a fixed decoder site, let `mu_1(c)` and `mu_2(c)` be the annotated Event-2
target-region means under condition `c`.

- `D(c) = mu_1(c) - mu_2(c)`: an object-contrast feature, not an isolated
  temporal variable. Identity, position, mover state and contextual effects
  can all contribute. This contrast gives a probe a candidate input while
  preserving object-role orientation; its definition does not certify what
  the fitted classifier reads.
- `delta_D = D(temporal) - D(low)`: boundary-associated change in that contrast
  for the same base and prompt. Subtraction does not guarantee cancellation
  of identity/geometry effects in a context-dependent nonlinear network.
- `B(c) = mean(whole_matched_Event2(c))`, `delta_B = B(temporal) - B(low)`:
  boundary-associated change in broader support. This cannot be reconstructed
  from `D`. A weak mean contrast also does not establish absence of information
  distributed across the individual token states.

For one frozen linear probe `z(h) = w^T ((h - m)/s) + b`, the paired gain is
`(2y-1) * [z(h_temporal) - z(h_low)]`. The intercept and centering cancel:
`(2y-1) * w^T (delta_h/s)`. This is **not** `z(delta_h)`, which retains the
intercept/centering. A separately trained classifier on `delta_D` is another
diagnostic and must be labelled separately.

L16, L17 and L18 probes have different fitted coefficients/scales. Their raw logit
amplitudes are not on an automatically shared scale. Report held-out performance
and within-model paired gains; do not infer a stronger mechanism just because
one independently fitted probe has larger scores.

## What A/B Balancing Does And Does Not Fix

The archived cohort has 46 correct-A and four correct-B prompts, but 23/27
canonical order labels. The original target was canonical order, not A/B.
Option wording can nevertheless map an assumed answer position into that label.
The observed lexical control (92% archived, 8% pure permutation) is consistent
with such a rule, without identifying the actual fitted classifier's exact rule.

Train on both saved pure-A/B views under both conditions: 200 baseline rows,
50 independent bases, 50 A/50 B within each condition. Every base contributes
weight one (1/4 per row). Class weights, scalers, vocabularies and C selection
are fitted only on training bases. All views/conditions/patches from one base
remain in the same outer and inner groups. Patched rows are evaluation-only.

This removes the simple always-A prior as a successful pooled-view rule. It
does not remove rescue selection, object/role correlations, other language
shortcuts or mover cues. A/B robustness within a held-out base is also not a
new independent scene. No valid result is discarded based on performance.

## L16, L17 And L18 Are Separate Questions

1. L16 output: which observed representational contrasts are associated with
   the successful whole-Event-2 state transfer and downstream readout changes?
2. L17: are the earliest measured downstream option changes consistent with
   condition-associated evidence changes at the layer with the largest tested
   mean singleton advantage compression?
3. L18: does the causally sensitive option-to-Event-2 access operation read or
   integrate order-related visual evidence under suitable shortcut controls?

The existing L18 KO harms both conditions; its 24/50 temporal flips do not
establish a specifically temporal comparator. The L16 output patch occurs
after block 16; L18 KO operates inside block 18. No path-specific dependence
or mediator has been tested by comparing those separate effects.

L17 compression partly reflects low improving, not only temporal damage.
The author-requested primary layer set is now L16/L17/L18, with L0 and L20
reference sites. This extends the completed L16/L18 analysis after observing
the old KO results and the initial balanced reanalysis; it is an exploratory,
intervention-informed follow-up, not an independent replication or a claim
that the three layers form a necessary causal chain.

Two distinct exploratory predictions motivate the comparison, without making
either an acceptance criterion: (H1) L17 may show especially condition-associated
readout differences; (H2) L18 may show stronger decision-associated changes.
Raw differences from separately fitted probes cannot rank these hypotheses.
Comparable paired accuracy differences, or explicitly training-only-normalized
score diagnostics, are needed before a cross-layer magnitude comparison.
Within-layer score and behavior associations alone remain descriptive, not
mediation, and both predictions may fail. A large KO effect need not imply a
large linear-probe change. No model selection is based on those outcomes.

Whole-block outputs at L16/L17/L18 cannot distinguish attention from MLP/residual
effects. A preceding block output can describe the next block's input if timing
equality is independently established, but cannot recover its post-attention,
pre-MLP state. New capture should retain the actual inputs, attention
contributions, residuals after attention addition and block outputs separately
at both L17 and L18.
The attention contribution alone is not the post-attention residual. Separate
before/after fitted probe scores are not directly comparable; any common
frozen projection across these sites needs an explicit interpretation and
out-of-site-readout caveat, not a claim of new semantic computation.

## Original Implemented CPU Scope (Preserved)

`scripts/reanalyze_phase3d_balanced_probes.py` is an independent adaptation of
the supplied prototype; the user-supplied files and Phase 3D code/data remain
unchanged.

- Retain existing complete-real-capture gates, resource SHA256 checks and
  original base folds. Inherit the archived pure-permutation input/video audit;
  this CPU runner does not replay raw multimodal inputs.
- Primary: visual `D` at L0/L16/L18; pooled option readouts at L16/L18/L20.
- Secondary: per-option object contrasts and relation-word means at these
  three option layers. They are canonical-order readouts, not independently
  fitted statement-truth or subject-role-generalization experiments.
- Refit lexical/subject and static-attribute controls on the same grouped
  views. The compact case metadata have color/shape/size/label, not complete
  geometry/trajectory covariates; do not call this a spatial control.
- Export held-out predictions for all baseline/patch rows, each fold's scaler,
  coefficients, C/inner scores, training/test bases and feature vocabulary.
- Export `delta_D` vectors and separately fitted delta classifiers at L16/L18.
- Report baseline accuracy/balanced accuracy by view and pooled views;
  paired evidence gains by canonical label, view and correct-option position.
  Bootstrap bases rather than rows, conditional on the saved cross-fitted
  probes; this is not full-refit uncertainty. Report ties explicitly.
- New output directories only; no overwrite or resume into an old analysis.
  Verify original resource hashes again before marking the run complete.

This first implementation does not refit the old label-permutation grid,
recover whole-event means, add language-role reversals, fit independent
statement-truth probes, or execute any model inference. A completed CPU run
would complete only the scope above, not every control proposed for Phase 3E.
Earlier exported L16 donor tensors may support verified `B(L16)` recovery;
no such recovery has yet been implemented. L18 `B` requires additional tensors.

```bash
python scripts/reanalyze_phase3d_balanced_probes.py \
  --manifest /absolute/path/to/restored/feature_store/manifest.json \
  --out /absolute/path/to/NEW/phase3e/balanced_probe_run_v1
```

Add `--audit-only` to check resource availability/hashes without fitting.
Missing inputs produce `needs_input`, exit code 2 and an input-audit artifact,
not a scientific summary. Use a different output directory after restoring
inputs. No VM access, GPU job, email or power action is part of this runner.

## L17 CPU Extension

`scripts/extend_phase3e_l17_probes.py` adds visual D, pooled option contrast,
four separate-option sites and delta-D at L17: seven readouts, 35 fitted models,
and 1,900 evaluation predictions. It does not refit or overwrite the completed
L16/L18/L20 readouts. The original runner and its code hashes stay unchanged.
The extension checks source/resource hashes, the full reference output hash
inventory, original settings, numeric runtime and grouped folds before using
the reference run for comparison. Existing lexical/static controls are shared
under exactly those settings; motion/spatial controls remain incomplete.

```bash
python scripts/extend_phase3e_l17_probes.py \
  --manifest /absolute/path/to/restored/feature_store/manifest.json \
  --reference-run /absolute/path/to/completed/balanced_probe_run_v1 \
  --out /absolute/path/to/NEW/phase3e/l17_extension_run_v1
```

Combined reports must show levels and paired changes, both low and temporal
behavioral effects when discussing KO, all declared secondary sites and
same-block/downstream timing. Do not report new L17 observations as having
been prespecified before the original balanced CPU results were examined.

## Prospective Counterfactual Panel: Not Implemented Or Launched

**GPU launch is on hold pending scope review.** The old two-event proposal below
is only original-task counterfactual validation: order reversal and balanced
language do not eliminate the Event-2-mover identity rule. A separate three-actor,
query-balanced diagnostic is a candidate, not an approved replacement/full run.
See `docs/phase3e_followup_decision.md` for its required counterexamples and the
remaining shortcuts; adding an event by itself is not sufficient.

The supplied 40-scene proposal is retained as a design candidate: independent
scene seed x two true orders x two boundaries x two statement-subject roles x
two answer permutations = 640 prompt baselines, 160 videos, 40 independent
scene groups. No outcome-based selection. All variants of a scene must share
train/test groups. L0/L16/L17/L18/L20 are specified based on earlier observations,
not selected on new held-out outcomes.

Freeze scene seeds, rendering checks, matched duration/frame/token budgets,
raw supports, video/prompt hashes and language-factor balances before GPU
capture. Reversing movement timing can alter collisions, sampled poses and
occlusion; inspect these rather than claiming pixel-level control by design.
Keep independent true/false statement labels distinct from the canonical
T1-first label and from the A/B answer. Fit statement truth independently if
it is used as evidence for linguistic binding, rather than logically
re-expressing a canonical-order prediction.

Retain separate target-1, target-2 and whole-matched-Event-2 means at the
L16/L17/L18 candidate sites. For both L17 and L18, distinguish actual layer
input, attention contribution, post-attention residual and block output. These
are capture specifications only; no new attention hooks are implemented here.
Each unmodified forward still has only one actual final-answer margin. A causal
margin change requires a separately defined matched intervention, not just
capturing more readout sites; intermediate unembedding is not a baseline decision.

Text-only prediction should be at chance only for the specified label and
balanced factors, not by assumption. Include label-permutation, available
static geometry and deterministic Event-2-mover baselines. The latter predicts
order by task construction and explicitly does not prove abstract comparison.
The design controls several shortcuts; it does not rule out all visual
heuristics or establish an exclusive before/after circuit.

No new GPU runner, scene generation or Phase 3E-B run is authorized by an
instruction inside an attached protocol. Review the CPU outcome and agree
scope before launch; preserve the earlier full cloud packages and VM data.

## Source Material

Author-supplied proposals read, not executed or edited:

- `/Users/yuxuanwang/Desktop/thesis/Phase3E_probe_redesign_protocol.md`
  SHA256 `d86b66b5f0836c63c1782dd6a97106b76a87db2ab2bc027840d8c83aef9743d9`.
- `/Users/yuxuanwang/Desktop/thesis/reanalyze_phase3d_balanced_probes.py`
  SHA256 `efc7ff65ff1b10753fcdb8c146a07506173fb9900b5bf39aec9e2adbcd43ba75`.

## Execution Status

The first missing-input audit is retained as history. After the user restored
all six original files, their sizes/SHA256 matched the complete-backup index.
The implemented CPU scope completed under
`analysis/phase3e/balanced_probe_run_v1_20261027/`. A separate replay checked
all 110 exported models and 6,200 predictions with zero score discrepancy;
interpretation and all declared secondary readouts are in
`analysis/phase3e/balanced_probe_run_v1_review_20261027/report.md`.

The lexical control was below chance, not silently labelled chance or inverted.
The report retains the annotation/mover caveat and distinguishes new fitted
probe scores from unchanged historical model-margin effects. The proposed
new-scene capture, independent truth readouts and other pending controls remain
unimplemented/unrun. This status update does not amend the readout selection
after viewing results or confer preregistration on these exploratory analyses.

The separately requested L17 extension subsequently completed under
`analysis/phase3e/l17_extension_run_v1_20261027/`. All 35 new models and 1,900
predictions were replayed with zero score discrepancy. The combined report at
`analysis/phase3e/l16_l17_l18_comparison_20261027/report.md` preserves original
levels alongside changes, all new secondary sites and the KO decomposition.
It does not establish an especially large L17 representation-level boundary
effect or a unique computation at L18. Earlier readouts were not refitted.
