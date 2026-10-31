# Reporting Levels And Changes

Author-requested convention, 27 October 2026. Apply to experimental reports,
tables, figures, meeting summaries and email drafts in this project.

- Whenever reporting a change, include the corresponding original levels:
  `before mean -> after mean (delta = after - before)`. For condition contrasts,
  name the order explicitly, for example `low -> temporal`.
- Name the metric and its units. In particular, distinguish signed probe
  evidence from model correct-answer logit margin and prediction accuracy.
- Use the same paired cases, prompt views, frozen readout and weighting for
  both endpoints and the change. State independent base count and view pooling;
  do not silently average only the cases that flipped or improved.
- For accuracy changes, report original correct/total counts and rates, with
  the absolute difference in percentage points where appropriate.
- State whether an uncertainty interval describes a level or a paired change.
  Compute statistics from full-precision paired values before formatting;
  rounded endpoint subtraction can differ slightly from the rounded delta.
- Do not compare raw score magnitudes from separately fitted probes as if
  their units were calibrated. A new readout does not change old model effects.
