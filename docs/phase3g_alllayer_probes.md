# Phase 3G: All-Layer Reference-Token Capture And Layerwise 2x2 Probes

## Material Passport

- Decision date: 2 November 2026, after the direct-D audit showed (i) no
  A/B-slot shortcut, (ii) saturated pairwise accuracy and (iii) only four
  captured layers, so neither the onset nor the depth of any Low/Temporal
  difference could be located.
- Cohort and prompts: exactly the 400 prompt cells of `rescue50_direct_D_v3`
  (50 frozen rescue bases x Low/Temporal x archived/pure A/B permutation x
  original/order-reversed video). No new videos, text, cases or screening.
- Source: the completed v3 order-capture root and the Phase 3D `run_v2`
  baselines are read-only. Normal forward only; no patch or KO.
- Status at freezing: not started.

## Capture

One uncached forward per prompt cell. Read the block output of all 36 text
decoder blocks (L0-L35, zero-based) at the two frozen reference-last option
tokens (statement A and statement B), float32, unmodified.

Gates per cell, all exact (`torch.equal`):

1. Inputs: the processor record equals the historical audit (original) or
   the v3 checkpoint record (reversed).
2. Full-vocabulary logits equal the frozen baseline (original: Phase 3D
   payload; reversed: v3 capture payload). This is also the hook no-op proof.
3. The newly captured L16, L17, L18 and L20 vectors equal the frozen v3/rescue50
   vectors at the same token positions.

Export builds an `(800, 36, 4096)` store in the frozen v3 endpoint order and
requires its four historical layers to equal the v3 store bit for bit.
Preflight: the two source preflight bases (16 cells). Then full, export,
verify, verified backup, e-mail and Pause.

## Layerwise Analysis (CPU)

For every layer L0-L35, unchanged folds (5 outer by base, 4 inner), L2
logistic regression, source-only scaling and regularisation selection:

- Direct D: `D = h(original) - h(reversed)`, `(D, y)` and `(-D, 1-y)`,
  no intercept; sources Low-only, Temporal-only, mixed-matched.
- Single-video h: original-video statements only, as the 3E h-probe;
  sources Low-only, Temporal-only.

Each frozen probe is tested on held-out Low and Temporal. Reported per
layer: pairwise accuracy and truth-signed evidence (D), evidence divided by
||D||, single-video AUC of D-probe endpoint projections (within fold, across
bases), h-probe accuracy and AUC. Two contrasts with base-clustered
bootstrap intervals: test-condition gap (Temporal - Low, same probe) and
training-source gap (Temporal-trained - Low-trained, same test). Absolute
scores of probes fitted at different layers or sources are not ranked.
