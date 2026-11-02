# Phase 3F: Layout-Symmetric Patch/KO, Equal-Budget KO Controls and Patch x KO

## Material Passport

- Decision date: 2 November 2026, after the direct-D results and the A/B audit.
- Cohort: the exact frozen 50 rescue bases of Phase 3D `run_v2`; no screening,
  replacement or pilot/new weighting. All 50 bases count equally.
- Source: Phase 3D `run_v2` is read-only. Its archived baselines, KO50 joint
  rows, archived L16 patches and both-view processor audits/captures are reused
  after hash verification. No historical checkpoint is rewritten.
- Model, runtime, eager attention, processor settings and GPU budget are
  inherited from the source runtime.
- Status at freezing: not started. Results never edit this protocol.

## Why

1. The cohort has 46 correct-A and 4 correct-B archived prompts. Patch and KO
   were measured only in that layout, while the model prefers B. A margin gain
   in the archived layout can partly be a shift towards A.
2. The Event-2 KO route lacked an equal-edge-budget control.
3. The middle-layer integration account (L16 visual state, read at L17/L18)
   rested on adjacent-layer correlations; no joint intervention tested it.

## Layout Views

`archived` is the frozen prompt. `option_order_permuted` swaps the two literal
option strings and the correct letter; video tensors, visual positions and the
text before the video are unchanged. For every base and condition, the two
views have opposite correct letters, so averaging their margins cancels any
constant letter preference:

```text
S   = (M_correct-A view + M_correct-B view) / 2      layout-invariant part
b   = (M_correct-B view - M_correct-A view) / 2      B preference
dS  = mean over the two views of an intervention's margin change
```

Gate: L16 block-output states at whole-Event-2 positions must be bitwise
identical between views (causal attention; options follow the video). Only
then may the archived donor be reused for permuted-view patches.

## New Interventional Forwards (50 bases)

| Block | Views | Conditions | Windows | Forwards |
| --- | --- | --- | --- | ---: |
| L16 whole-Event-2 patch, both directions | permuted | recipient Low and Temporal | L16 output | 100 |
| KO options -> whole Event 2 | permuted | both | 16, 17, 18, 19, 16-19 | 500 |
| KO options -> whole Event 1 (control) | both | both | 18, 16-19 | 400 |
| KO options -> 512 random non-event visual keys (control) | both | both | 18, 16-19 | 400 |
| KO-only reference, Event 2 | both | Low | 17-18, 17-23 | 200 |
| Temporal->Low L16 patch + KO options -> Event 2 | both | Low recipient | 17, 18, 17-18, 17-23 | 400 |
| Total | | | | 2,000 |

Technical gates add one per base x condition x view (200): exact replay of the
saved full-vocabulary baseline logits, a disabled-mask no-op and the Event-2
L16 state check. Preflight runs the full grid on the two source preflight bases
(one of each mover order) and is not the experiment.

Archived references that are not recomputed: Event-2 KO singletons L16-L19
(Phase 3D singleton run), the 16-19 joint row and the archived L16 patches
(Phase 3D `run_v2`). Their decisions are frozen into this run with hashes.

## Routes And Budgets

- Queries: `options_all` of the actual view's prompt.
- Event 2: the source one-to-one whole-Event-2 support (512 positions).
- Event 1: every visual position whose sampled-frame group lies more than half
  inside the first target-event interval, by the same inclusive dominance rule
  used for Event 2. Required: exactly as many positions as Event 2, disjoint
  from it, identical frames in Low and Temporal.
- Random: the same count drawn once per base and condition from visual
  positions outside both target events, ranked by a fixed hash of
  (seed 42, base, condition, temporal index, cell). Not outcome-driven.
- All three routes must have identical causal edge budgets per view.
- KO: pre-softmax negative infinity, all 32 heads, renormalised; same
  `AttentionKnockout` as Phase 3C/3D.

## Patch x KO

Patch: Temporal->Low whole-Event-2 replacement at L16 block output, the same
mapping and observed-donor replacement checks as Phase 3D. KO is installed in
the same forward at the listed later windows; the L16 timing signature must
equal the unpatched baseline (KO starts at L17).

```text
R0 = M(patch) - M(base)                rescue without KO
Rw = M(patch + KO_w) - M(KO_w)         rescue that survives blocking window w
retained_w = mean(Rw) / mean(R0)       descriptive, not a mediation estimate
```

L17-23 covers all blocks after the patch up to the last depth at which the
visual-position patch had a measurable effect (near zero from L24). Blocking
the option->Event-2 route does not block Event-2 information that reaches
other positions first; a surviving rescue is therefore not evidence against
middle-layer use, and a removed rescue is evidence for this route only.

## Outputs

Per layout and layout-symmetric margin changes, compression, strict sign
crossings and actual answer flips, with diagnostic 95% case-bootstrap
intervals over 50 bases. Report both endpoint values with every difference.
No result selects layers, bases or routes post hoc. Not a circuit claim.
