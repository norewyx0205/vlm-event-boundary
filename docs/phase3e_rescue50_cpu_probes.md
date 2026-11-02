# Rescue50 Condition-Generalization Probes

## Fixed Source

This CPU follow-up consumes only the complete
`phase3e_rescue50_option_store_v1` store. It preserves the original 50 rescue
bases, videos, archived prompts, model captures and historical outer folds.
The earlier 40-family paired-video experiment remains a separate supplement.
The source protocol and GPU capture dependencies are not edited.

The store joins 200 recovered original-prompt statement vectors with 200
new pure-A/B-permuted statement vectors. Each vector is the last contextual
reference-object token at L16/L17/L18 or L20 block output. No within-video
object subtraction is used for training. Labels indicate statement truth.

## Four Training Sources

- Low-only: both statements and both A/B views in Low.
- Temporal-only: the corresponding Temporal rows.
- Full mixed: all rows from both conditions.
- Budget-matched mixed: one whole prompt view in each condition per base,
  choosing complementary archived/permuted views and retaining both statements.

Matched assignments are fixed with seed 42 within original correct-letter
strata before fitting. Exact assignments and per-fold answer-letter counts
are saved. This balances the complete matched source across conditions, but
individual folds need not have exactly equal answer-letter counts.

Every outer fit holds out 10 of the original 50 bases and trains on 40.
Single-condition and matched-mixed fits use 160 statement rows; full mixed
uses 320. Each training base contributes one total classifier-weight unit.
All conditions/views/statements of a base stay in the same outer and inner
fold. Four inner folds use the historical canonical-relation stratifier on
outer training bases, with seed 42 plus outer-fold index.

The existing L2 logistic probe, four-value C grid, source-only scaler and
source-only inner regularization selection are retained. Every frozen model
evaluates both conditions of its held-out bases. A lexical-only classifier
uses the same sources/folds but no visual features, event times or labels as
inputs. Four readouts times four regimes times five outer folds produce 80
numeric models; lexical controls add 20. All are new fits for this cohort
and last-reference-token readout, not reused 40-family coefficients.

## Evaluation

Primary reporting uses archived prompts to match the old KO/patch setup.
Pure A/B permutation and combined-view metrics are separate. Report Low and
Temporal levels plus their difference for accuracy, signed truth evidence
and model margins. Report accuracy separately from model answer accuracy.
AUC is averaged within outer models; do not pool unrelated logit scales.

For each same-text Low/Temporal pair, apply one frozen held-out model.
Replay the score difference directly as
`(h_temporal - h_low) @ (coefficient / scaler_scale)`, with no centering or
intercept applied to the difference. This contrast is not the training input.
Aggregate within base before 2,000-replicate diagnostic bootstrap. Intervals
are conditional on saved probes, not a nested full-refit sampling estimate.

Answer-position balancing does not undo the original rescue selection or
balance every lexical association. Keep the lexical-control results and
view-specific breakdowns. This cohort characterizes the same 50 intervention
bases, not the general frequency of a temporal-gap benefit.

## Saved Validation

The runner checks store/resource hashes, complete cells, canonical statement
truth, original rescue behavior and historical outer folds. It saves numeric
models, source selections, inner choices, predictions and base-level paired
tables. Verification independently applies exported coefficients, reconstructs
source-only scalers/vocabularies, checks direct-D projection and regenerates
the statistics without training any classifier. Completed jobs are replayed;
partial jobs are rejected rather than silently overwritten.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/run_phase3e_rescue50_probes.py \
  --manifest_path /path/to/rescue50/feature_store/manifest.json \
  --output_root /path/to/new/rescue50_probes

python3 scripts/run_phase3e_rescue50_probes.py \
  --manifest_path /path/to/rescue50/feature_store/manifest.json \
  --output_root /path/to/new/rescue50_probes --verify_only
```

No GPU, video regeneration, KO or patch is performed by either command.
