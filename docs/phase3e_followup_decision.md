# Phase 3E Follow-up Decision: Hold GPU Launch

**Superseded for experiment scope on 31 October:** see
`phase3e_b_paired_option_protocol.md`. The current design uses fixed-object/path
order reversals and normal option-side probing. It does not require excluding
the second-mover strategy or adding KO-conditioned probing. This file retains
the earlier decision history; it does not override the revised protocol.

27 October 2026. Design/meeting memo only. The implemented balanced CPU scope
and L17 extension are complete. Neither the old 640-prompt plan nor a larger
diagnostic is authorized for launch by this memo. Thesis RQs are unchanged.

## Current Evidence And Claim Boundary

The tested pooled temporal option-readout accuracies are L16 53%, L17 60%,
L18 95%, L20 100%, from two views of the same 50 rescue bases. This is probe
accuracy, not the model answering all those inputs correctly. It localizes a
difference in linear readability within the tested sites, not a new computation.

The previous L16 visual intervention still affects unpatched option readouts:
L17 mean signed evidence 0.98513 -> 1.26426 (delta +0.27913; 41/50 positive),
L18 2.25536 -> 2.67660 (+0.42123; 44/50), and L20 3.08244 -> 3.68377
(+0.60133; 49/50). These are different fitted probes, not a common calibrated
axis. Effects cannot be ranked by those raw score changes.

Balancing probe-training views does not rebalance the archived patch/KO
cohort (46 correct-A / four correct-B) or create an independent intervention
replication. The prior 92% lexical result is consistent with the always-A
heuristic, not proof that the fitted classifier learned exactly that rule.
The revised 30% control remains below chance and is neither inverted nor
presented as evidence that all language shortcuts are gone.

L17 has the largest tested mean singleton gap compression; L18 has the largest
singleton temporal damage. L17 compression partly comes from low improving,
and L18 also damages low. Neither observation identifies a specialized boundary
operation or establishes that the transplanted states use the KO-tested route.

The apparent timing gap is not a contradiction: the intervention replaces 512
matched visual-token vectors, whereas the option probe reads a restricted,
pooled object-mention contrast with a linear classifier. Distributed content,
pooling loss, nonlinear accessibility or ordinary residual transformations can
make these effects differ without a new semantic operation first appearing at
L18. Those alternatives belong beside the integration hypothesis, not after
an assumption that the probe must reveal every causally effective feature.

## Three Bounded Questions

1. Which boundary-associated visual contrasts are available at L16/L17/L18?
   Capture separate target means and whole matched Event-2 support, not just
   D=T1-minus-T2; retain D, delta-D, B and delta-B with actual token supports.
2. How do order-label readouts differ across L17/L18 attention operations?
   Distinguish actual block input, attention contribution, post-attention
   residual and post-MLP block output. Observational site differences are not
   equivalent to the effect of blocking a specified attention edge set.
3. Which results survive order changes, query-role and answer-position balance,
   and explicit tests of event-specific mover rules? This is stronger evidence
   for query-dependent relation evaluation, not a promise of abstract computation.

L0 and L20 remain early and downstream references. No full-depth or head search
is added. All new choices are informed by previous results and remain exploratory.

## Prospective Levels: Not The Completed CPU Workstream

The labels below refer to possible GPU follow-ups, not to completed Phase 3E-A.

### Level A / Candidate 3E-B1: Within The Original Two-Event Task

The earlier 40-scene, 640-prompt proposal changes true target order, boundary,
statement subject and answer arrangement. It can test transfer beyond selected
rescues and control several static/language correlations. It **cannot** remove
the deterministic fact that knowing the second mover identifies the true order
when the two targets each move exactly once. Reversing order simply reverses
the mover label; the rule remains valid.

Its scope is original-task counterfactual validation, not a clean exclusion of
the Event-2-mover explanation. The 640 figure counts baseline prompt forwards,
not technical replays, KO/patch controls or a new task's complete compute cost.
Do not launch it merely because its old baseline count was already specified.

### Level B / Candidate 3E-B2: Separate Query-Dependent Diagnostic

Adding a third event alone does not guarantee control of a shortcut. A useful
design requires explicit paired counterexamples and a balance audit. For example:

| Visual sequence | Event-2 mover | Fixed queried pair | True relation |
| --- | --- | --- | --- |
| X then Y then Z | Y | X versus Z | X before Z |
| Z then Y then X | Y | X versus Z | X after Z |

Here the Event-2 mover and queried pair are the same, while the label reverses.
Use three distinct actors with one scheduled event each, counterbalance event
permutations and queried pairs, and balance canonical relation, subject role
and correct A/B position. Hold all scene-family variants together in CV.

Acceptance requires auditing the conditional label counts for each candidate
shortcut input, including Event-2 mover + queried pair + subject/answer layout.
Here layout means displayed option contents/order, not the ground-truth correct
answer supplied as a predictor. Check canonical-order and actual A/B labels
separately; balance marginal totals is not sufficient.
The identity-only baseline must not predict the ground-truth label by design;
check the actual generated table rather than assuming this from event count.
Render and inspect the opposing examples before any full model collection.

This excludes the **Event-2 mover-identity-only** rule if implemented correctly,
not all event shortcuts. Event-1/final-mover rules, static poses, event durations,
occlusion and motion history can remain informative. An Event-2 hidden state
may legitimately contain preceding-event information; it is not equivalent to
the identity of the object currently moving. Do not require the entire Event-2
representation to be label-uninformative, or call this diagnostic a proof of
abstract before/after computation.

A local combinatorial check of all six orders, three queried pairs, two subject
roles, two displayed layouts and two boundary labels gives 144 symbolic cells.
In that table, lookup using Event-2 mover plus displayed query/layout information
has a 50% label ceiling. First- or last-mover identity plus that text information
instead permits 83.3%: a concrete warning that one removed shortcut is not all
shortcuts. These are label-contingency bounds, not trained-model accuracies,
rendered videos or a proposed 144-forward run. The audit is preserved under
`analysis/phase3e/meeting_20261029/design_audit.json`.

This changes the task distribution. Keep its data, labels, metrics and reports
separate from the original two-target rescue results. A three-actor/query-pair
schema needs independent validation: the current Phase 3D label/token helpers
assume exactly two literal target identities and cannot simply accept a third
target unchanged. No schema, generator or GPU runner has been implemented here.

## Probe Scores And Actual Decisions

One unmodified baseline forward yields one final first-answer-token margin.
Collecting pre-/post-attention states does **not** yield separate actual model
decisions "before attention" and "after attention". Intermediate unembedding
would be another diagnostic, not the measured baseline decision.

Report baseline margin by factorial cell and its association with readouts.
To measure a causal decision change attributable to an operation, a separately
specified matched KO/state intervention must be compared with its baseline
and no-op control, using actual full-vocabulary logits and correct token/label
alignment. Such forwards are additional scope, not implicitly included here.

Across sites, compare held-out performance and clearly defined paired score
changes. A common frozen projection across sites is an out-of-site diagnostic;
separately fitted raw probe logits are not a calibrated before/after scale.
Retain continuous scores, categorical flips/ties, and both low and temporal
effects. Do not equate decodability, feature use and causal mediation.

## Decision For Albert

First show the completed corrected analysis and the distinction between L17
compression and L18 temporal damage. Ask whether the bounded thesis account
is sufficient, or whether one limited, technically audited follow-up would
materially strengthen SQ4. If more work is warranted, agree which level and
whether causal intervention forwards are needed before fixing sample size,
rendering, token alignment, capture hooks, storage budget and lifecycle.

Until that decision, no GPU run is queued. Existing experimental artifacts and
the completed CPU output inventories remain unchanged. Full cloud preservation
is user-confirmed; no new VM or independent cloud-checksum observation is made.

## Subsequent Authorization: Technical Preflight Only

Later on 27 October, the user explicitly authorized implementation and a small
technical preflight, with completion email and VM Pause. The separate frozen
scope is documented in `phase3e_technical_preflight.md`: two fresh scene seeds,
eight two-event videos and 32 factorial prompt cells, with read-only capture
and numerical timing/parity gates. The full 40-scene experiment and three-actor
diagnostic remain unlaunched. This updates the earlier launch hold only for
that limited technical scope, not the scientific interpretation above.
