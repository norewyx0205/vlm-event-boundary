# Phase 3E: Direct Paired-Difference Probe Training

## Material Passport

- Decision: direct same-object paired-video differences are the primary probe
  training input, following Albert's meeting feedback.
- Status: revised design; the original rescue50 cohort still lacks captured
  order-reversal partners. No new GPU experiment has been launched.
- Cohort: retain the exact original 50 scene bases and historical five outer
  folds. Do not substitute the completed 40-new-family experiment.
- Earlier single-video statement-truth fits remain unchanged as controls.
  Their training input was h, not D; they do not complete this protocol.
- Historical capture protocols remain immutable because their hashes bind
  completed runs. This document changes the next analysis, not old provenance.

## Controlled Video Pair

For each original base and each boundary condition, pair the original video
with a new version in which the same two objects exchange their event slots.
Keep each object's identity, appearance, initial location, path and motion
duration fixed. Reverse actual motion scheduling, not IDs or appearances.
Keep question, both literal option strings, option arrangement, subject and
reference roles fixed across this pair. Compute truth from event times.

Example, for the unchanged statement "The black triangle moves before the
blue circle":

| | Original video | Order-reversed video |
| --- | --- | --- |
| Black triangle, same path | First | Second |
| Blue circle, same path | Second | First |
| Statement truth | True | False |
| Readout | Last contextual token of blue circle | Corresponding contextual token of blue circle |

Original Low/Temporal videos alone are not this pair: their true event order
is identical. Subtracting their representations must not be relabelled as
an order-reversal feature.

## Primary Feature And Label

At each fixed layer l in {16, 17, 18, 20}, for statement s and boundary c:

```
D_order(i,c,s,l) = h(original_order,i,c,s,l) - h(reversed_order,i,c,s,l)
y_left = truth of the unchanged statement s in the left/original-order video
```

The hidden state is the last reference-object contextual option token on each
side, not a visual ROI vector and not a difference between two different
objects. Both sides refer to the same named object in the same sentence role.
This controls identity and wording in the comparison; it does not assert that
all appearance-dependent neural interactions algebraically disappear.

Train directly on these difference vectors. Include (D_order, y_left) and
(-D_order, 1-y_left). Store explicit left/right video and statement IDs.
Do not orient every input as true-minus-false or first-minus-second with an
unchanging label. Both directions and both statements of a pair belong to
one base and are not independent observations.

Use a linear L2 logistic classifier with zero intercept and source-training
scaling that does not centre D. This preserves antisymmetry: the score of -D
is the negative of the score of D. Fit scaling, regularization and weights
only on training bases. A symmetric source has a zero feature mean by design.
Each base contributes one total classifier-weight unit.

## Boundary Comparison And Folds

Retain four sources: Low-only, Temporal-only, full mixed, and equal-budget
mixed. Every frozen source-trained difference classifier evaluates Low and
Temporal differences on held-out bases. Low-only fitting must not use
Temporal features for scaling or regularization selection, and conversely.

Keep the historical 40-train/10-test base assignment in each outer fold and
four grouped inner folds on the outer training bases. Every original/reversed
video, Low/Temporal condition, A/B view, statement and subtraction direction
from one base stays in the same fold. A reversed partner is not a new base.

Keep both original and pure-A/B-permuted layouts. Match exactly the same layout
across each video pair; never pair option A with a different statement at B.
Original-layout reporting is primary for comparison with old KO. Permuted
and combined-layout results are reported separately.

With 50 bases, two orders, two boundaries and two A/B layouts, the complete
target store has 400 prompt cells and 800 single-statement endpoint vectors.
It supplies 400 unique unordered statement pairs and 800 directed D examples.
Half the endpoint prompts already have verified rescue50 captures; the other
200 prompt cells require new reversed-order capture if all inputs/sites pass
reuse checks. These need 100 new videos, not 50 new independent scenes.

Per outer fit, Low-only/Temporal-only/matched-mixed sources each use 320
directed D rows from 40 bases; full mixed uses 640. The extra directions are
augmentation, not a doubled independent sample size. The matched source uses
complementary A/B layouts across conditions and keeps both statements and
both difference directions of every selected pair.

## Outcomes

- Primary: held-out pairwise statement-truth discrimination in each boundary,
  with accuracy, balanced accuracy and within-model AUC.
- Boundary effect: Low and Temporal levels and their difference under the
  same frozen pairwise classifier. Cluster summaries/uncertainty by 50 bases.
- Signed paired evidence: (2*y_left-1)*score(D_order). Compare Low/Temporal
  order contrasts with the same fitted direction, not independently scaled
  logits from different layers or training sources.
- Save both endpoint h vectors and IDs. When showing a projected difference,
  also show its two linear endpoint projections. Those projections are
  algebraic components of a pairwise score, not calibrated single-video truth
  probabilities; a pairwise classifier does not evaluate a lone video.
- Keep model answer logits/margins and the earlier single-video probe results
  separate. No new KO, patch or KO-conditioned probing is required.

D_boundary = h(Temporal)-h(Low) at fixed event order remains a separate
boundary analysis. It is not substituted for D_order when training the primary
order-reversal classifier. Comparing D_order across boundaries tests whether
the temporal gap changes order-sensitive differences.

## Next Gates

1. Inventory exact original annotations, frames, renderer, fonts and encoding.
   Check that reconstructed originals match the original stimuli before using
   that renderer to make reversal partners. Do not silently replace old videos.
2. Resolve actual moving-distractor schedules from original assets where
   annotations omit start/end frames. Keep schedules identical within each
   original/reversed pair in a fixed boundary condition. Audit interactions;
   a newly changing distractor must not stand in for the order manipulation.
3. Freeze reversed video specs, identical statement correspondence, opposite
   truth labels, A/B controls and unchanged historical base folds. Do not
   screen out reversed videos because the model answers them incorrectly.
4. Generate and audit videos; tokenize and capture missing endpoints only
   after a small real-model preflight. Reuse old vectors only with exact source,
   input, model and contextual-site correspondence.
5. Fit the direct-D probes and replay saved models independently. Report
   pairwise discrimination, not the earlier single-video decoding percentages.

The existing 40-family order-reversal store can test implementation separately,
but it cannot be merged into the rescue50 cohort or presented as its completed
direct-difference experiment. The current rescue50 store alone must fail the
order-pair completeness gate rather than produce fabricated differences.
