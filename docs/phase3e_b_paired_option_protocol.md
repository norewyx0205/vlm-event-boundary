# Phase 3E-B: Paired-Video Option Probing

## Material Passport

- Date: 31 October 2026; revised after Albert's meeting feedback.
- Stage: materials, compact-capture/export and CPU-fitting implementation;
  new-cohort GPU execution and scientific probe results remain pending.
- Current scope: normal-forward option-side probing on matched order-reversal videos.
- Replaces: the old cross-object D feature as the main input and the earlier proposal
  to make KO-conditioned probing a required new experiment.
- Existing Phase 3B-3E-A results and the 27 October preflight remain frozen.

## Research Question

With visual identities, object paths and option wording controlled, how do true
event order and temporal boundary change option representations at L16-L18?
Use the existing option-to-Event-2 knockout findings as causal motivation for
this endpoint, rather than expanding the present intervention grid.

Identifying the second mover is valid evidence in the two-event task. Excluding
that strategy or adding a third event is not an acceptance criterion.

## Paired Materials

Use 40 fresh scene families, fixed seeds 2026103101 through 2026103140, without
behavioural screening. T1 and T2 are persistent object IDs, not first/second mover.
Keep each object's shape, colour, initial position, direction and entire path
fixed. Reverse only which target is scheduled in the first versus second event.
Keep absolute distractor schedules fixed. This is **not** an appearance swap,
an annotation-only renaming, a spatial mirror or a query-role reversal.

| Object identity | Video A example | Video B example |
| --- | --- | --- |
| Black triangle, same path | First target event | Second target event |
| Blue circle, same path | Second target event | First target event |

Cross both orders with low/temporal boundaries, both sentence subjects and both
literal A/B permutations: 160 videos, 640 baseline prompt cells. Each family
contributes 16 cells and is one independent statistical unit. Keep the complete
before/after option strings identical within an order-reversal pair: statement
truth and the correct letter must reverse while the wording remains fixed.
Within each boundary, each family has four correct-A and four correct-B cells.

The generator reuses the audited 18-second, 15-FPS, 512-by-512 level-5 fixture:
two targets, two static and two moving distractors, two-second target events,
zero/three-second gap. The same fixed identities also undergo both event orders,
so appearance and fixed path alone cannot determine the relation label.

## Primary Readout And Label

Train an L2-regularized linear logistic probe on each **single video's single
option representation**, not a T1-minus-T2 contrast and not an average of the
contradictory A/B statements. The primary site is the last contextual token of
the reference-object mention in each option. It has already seen the subject,
relation word and full reference identity. Resolve it from actual prompt spans,
not standalone token IDs. Keep reference-span means and option-final states as
declared secondary readouts. Earlier subject mentions are descriptive order
readouts, not equivalent full-statement truth sites under causal attention.

Primary label: is this exact named-object before/after statement true in this
video? Compute it from the actual subject/reference IDs and target event times,
independently of the correct A/B annotation, then verify agreement. The 640
prompts provide 1,280 statement rows, with 640 true / 640 false labels. The
question is not to predict whether a statement happens to be placed at A.

Primary layers: L16/L17/L18 block outputs; L20 is the downstream reference.
Retain L0 visual means as a descriptive source reference. At L17/L18 also keep
block residual input, projected attention contribution, actual post-attention
residual and block output as separate sites. A block input is not a normalized
Q/K/V representation. Different fitted site-specific probe logits are not one
shared cross-layer scale; compare held-out performance across sites.

## Same-Object Paired Contrasts

For each named object occurrence o in an identical option and a fixed site,
define D_order(o,c) = h(o,V_left,c) - h(o,V_right,c), where the right video has
the opposite order and the same boundary c. The object's identity and textual
role are the same on both sides; use the actual matching reference token, not
the two different objects within one video. Source ROI contrasts of the same
visual object are secondary, not the main option-side measure.

Store both explicit left/right directions and the truth label in the left video.
Do not orient every feature as first-minus-second and then pretend its label
varies. The two directions belong to one pair, not two independent cases.

Primary analysis projects the contrast through the already frozen normal
single-video probe. If s(h)=w^T((h-m)/sigma)+b, then:

```
s(h_left) - s(h_right) = w^T(D_order / sigma)
order_consistent_contrast = (2*y_left - 1) * (s(h_left) - s(h_right))
```

This separates a change of probe evidence from fitting a new contrast classifier.
As a secondary analysis, pairwise training may use both D and -D with opposite
left-statement truth labels, under the same scene-family folds. Report that as
pairwise discrimination, not single-video decoding accuracy.

At fixed order, define D_boundary(o,order)=h(o,temporal)-h(o,low). The statement's
truth label stays fixed. Report both original scores and their paired signed
gain. Difference-in-differences asks whether the order contrast changes with
boundary. Average mirrored directions within a family before uncertainty
estimation; do not count duplicate views as additional independent evidence.

## Option-Side Alignment With Existing KO

The important existing route is options_all queries attending whole_event2
visual keys. KO masks those edges; this new probe observes normal option states.
The primary reference-token readout is a recorded subset of options_all, not a
claim that all query tokens are one pooled probe feature. Capture full option
span indices and the exact primary/secondary readout indices for auditing.

Matching the option-side endpoint is sufficient for the current research scope.
**No new knockout, patch or KO-conditioned probe evaluation is scheduled.**
The previous L16 visual patch and L17/L18 KO results remain independent causal
results. KO-conditioned probing is optional later, not a prerequisite to this
experiment and not something the earlier probes already performed.

## Training And Evaluation

Keep all order reversals, conditions, subjects, layouts, A/B statements and both
contrast directions from a family in one outer/inner fold. The supplied CPU plan
fixes five outer folds and four inner folds within each outer training set.
Fit scaling, lexical vocabularies, regularization selection and classifiers only
on training families. Every family has equal total training/reporting weight.

Report held-out accuracy/balanced accuracy, statement-truth evidence, paired
order contrast and temporal-minus-low gain. Show both raw levels and differences.
Measure the model's own correct-option logit margin separately. Retain all
wrong answers, zero-margin ties and non-A/B outputs. Report accuracy and margin
by boundary, subject and answer placement, alongside the pooled values.

Run lexical-only and static-identity controls with identical family folds. The
CPU contingency audit verifies that identical statement words occur with both
truth labels within each family/boundary. It is a label-design audit, not a
trained lexical control. A second-mover rule is a legitimate task reference.

## Compute, Storage And Gates

The full design contains 640 normal baseline cells; each is checked against an
unhooked replay and independent first-token generation (1,920 forwards total).
The first 32 cells are the two-scene technical gate and are verified/reused,
not repeated as additional scientific observations. There are zero new KO/patch tasks.

The old preflight retained approximately 1.22 GiB for two families; copying that
raw layout to 40 families projects to 24.4 GiB, or 60.8 GiB for two copies plus
a 12-GiB reserve. Before launch, measure a compact collector saving all planned
object-occurrence/option readouts, token mappings and first-token logits, rather
than retaining full selected visual-vector blocks for every prompt. Freeze its
resource budget and backup method after that measurement. No old evidence is
deleted to make the estimate fit.

CPU acceptance: exact factorial coverage, unchanged named identities/paths,
opposite actual schedules, identical paired wording, independently checked
truth/A-B balance, complete same-object correspondences, and nested scene folds.
Render two preselected families as CPU examples, without model screening.
Before formal inference, extend and test the compact reference-last-token
collector, then check real-model readout alignment/no-op/timing parity and space.
Reuse the existing email, verified full-backup and quiescent-Pause lifecycle.

## Current Delivery

- `phase3e_paired_design.py` and `prepare_phase3e_paired_design.py`: fixed
  materials, labels, contrasts and scene folds.
- `phase3e_option_features.py`: contextual reference-last, reference-mean and
  option-final readouts; compact normal-forward hooks. Raw residual timing
  proofs retain only reference-last and option-final positions, not entire
  Event-2 vector grids. Source means remain available as descriptive artifacts.
- `run_phase3e_paired_capture.py`: explicit prepare/preflight/capture/verify/
  export/package/finish stages, input hashes, exact hook parity, generation
  parity, resumable checkpoints, measured disk budget and verified backup.
- `phase3e_option_store.py`: complete, real-capture-only training contract.
- `phase3e_option_probes.py` and `run_phase3e_option_probes.py`: nested
  scene-family cross-fitting, lexical/static controls, numeric frozen-model
  replay, direct D projection checks, paired raw levels/differences and reports.
- `export_phase3e_option_smoke.py`: read-only CPU export of the old two-scene
  backup at L17/L18. This is partial technical data, rejected for formal fitting.

Default CPU fitting covers reference-last at L16/L17/L18/L20 outputs and the
already declared input/attention/residual sites, with lexical/static controls.
L0 option fitting is not included. `--include_secondary` also fits reference
means and option-final states. Direct-D classifier training remains optional
and is not part of the current primary runner.

The new 40-family captures have not been run. The old 27 October captures have
passed a 32-prompt/64-statement CPU export smoke test: 64 directed order pairs
and 32 boundary pairs preserve actual contextual reference tokens/positions.
This supplies no new scientific probe results.

See [execution notes](phase3e_b_execution.md) for stage commands and gates.
