# Phase 3D Implementation Status

Updated 2026-10-19 (Europe/Amsterdam).

## Implemented and Unit-Tested

- `phase3d_core.py` / `prepare_phase3d.py`: canonical timing-derived labels,
  pure A/B permutation descriptors, base folds and metadata-only capture audit.
- `phase3d_features.py`: actual contextual per-option/identity spans, with
  exact agreement against the audited section-scoped token groups. Compact
  means at all 36 block outputs and the actual L16 input; raw vectors retained
  only for complete L16 donor support. Explicit one-to-one donor reordering.
  Identity/no-op, L15-to-L16-input equivalence and unaltered upstream/L16 text
  readouts are tested. Patch propagation first becomes eligible at L17.
- `phase3d_probes.py`: grouped nested linear logistic probes. Only archived,
  unmodified low/temporal training rows enter fitting, scaling and inner-fold
  C selection. Permuted-option and patched rows are strictly held out by base.
  Static-attribute, lexical/role, condition-only and paired-label null controls
  remain distinct from the main relation evidence.
- `run_phase3d_probes.py`: checked real feature-store loading, per-readout
  resumable checkpoints, numeric model/scaler exports without pickle,
  per-case signed boundary/patch evidence and margin association. Exported
  numeric probes can be applied without any refitting. Every outer fold
  reports progress. Completed checkpoints are hash-verified before reuse.
- `phase3d_knockout.py`: 50-base equal-weight intent, 5,850 paired tasks and
  11,700 condition forwards. A prior KO25 summary is only a provenance
  candidate, never automatically accepted as verified reuse.

Source means are float32 pooled vectors, not raw full-prompt all-layer tensors.
Numeric probe fitting and exported scoring convert these vectors to float64,
so exported scalers do not silently change float32 standardization rounding.
Keep the two options equally weighted before taking the literal T1-minus-T2
contrast. Readouts at a transplanted visual position are not downstream transfer
evidence; output metadata labels unpatched downstream text readouts separately.

## Still Required Before GPU Launch

The full CPU-processor audit for all newly permuted prompts, real-model paired
capture/export executor, source replay/parity checks, KO50 verified reuse/
execution/independent completeness checks, and storage/backup/email/Pause
orchestration are **not** yet implemented as a Phase 3D launch command.
The current collector is an API used by a future gated executor; it does not
itself load a model or start a job. Do not treat the existing Phase 3C runner as
a Phase 3D runner or manually mark gates passed to train from partial data.

No actual new model activations or scientific probe results have been produced.
Toy/synthetic unit tests validate code, not the scientific hypothesis.

## Feature-Store Contract

The future executor must create an isolated, complete store before the CPU
training CLI accepts it. Manifest paths are relative to and contained within
the store; every referenced file needs an exact SHA256. Never replace files in
a completed source store or attach old processor hashes to a changed prompt.

`manifest.json` fields:

```json
{
  "schema": "phase3d_feature_dataset_v1",
  "artifact_type": "real",
  "complete": true,
  "case_metadata": {"path": "case_metadata.json", "sha256": "..."},
  "outer_folds": {"path": "outer_folds.json", "sha256": "..."},
  "row_metadata": {"path": "rows.json", "sha256": "..."},
  "gate_summary": {"path": "capture_gates.json", "sha256": "..."},
  "features": {"path": "features.npz", "sha256": "..."},
  "feature_names": ["option_pair:L20_output"]
}
```

This example shows the shape of the manifest, not a ready or sufficient store.
The default trainer needs source-pair and option-pair block outputs 0-35,
source-pair L16 input and decision L35 output. The collector also exposes query
and other decision-layer outputs for explicitly named descriptive readouts.
`features.npz` has a floating-point, finite `features` array of shape
`[row, feature_name, hidden_dimension]`, loaded with `allow_pickle=False`.

Use the canonical case metadata and five outer folds from the latest CPU
preparation. Row metadata has `base_sample_id`, canonical `y`, actual recipient
`condition`, `view`, `intervention`, actual `correct_option`, measured
`answer_margin` and actual prompt `input_fingerprint`, plus provenance as needed.
For each of 50 bases, exactly six rows are required:

1. Archived baseline low and temporal (two rows).
2. Pure A/B-permuted baseline low and temporal (two rows).
3. Archived temporal-to-low L16 patch and low-to-temporal L16 patch (two rows).

There are 300 condition/view/intervention rows but only 50 independent bases.
The permuted views must have genuinely different prompt fingerprints and
swapped correct-answer identities; no behavior-based reselection is allowed.

`capture_gates.json` must have schema `phase3d_capture_gate_summary_v1`,
`complete=true`, `base_count=50` and `case_labels_sha256` equal to
`digest(sorted(canonical_base_label_map.items()))`. The following are required
true: `processor_input_parity`, `standard_first_token_parity`,
`full_logit_parity`, `capture_noop`, `source_state_replay`, `identity_patch`,
`same_block_timing`, `new_option_order_input_audit`,
`one_to_one_full_visual_support`. These must be generated by independently
checked real stage ledgers, not user-filled booleans. The formal exporter and
stage-ledger verifier are still pending.

## CPU Training After Real Capture Completes

The CPU trainer needs NumPy, SciPy and scikit-learn. It records all three
versions, plus Python, in its frozen run configuration. Do not downgrade or
replace the pinned GPU environment to match a local CPU test environment;
use a separate CPU environment when needed. Changing versions requires a new
probe output root rather than mixing checkpoints from different runtimes.

```bash
python scripts/run_phase3d_probes.py \
  --manifest_path /data/yuxuanstorage/vlm_phase3d/run_v1/feature_store/manifest.json \
  --output_dir /data/yuxuanstorage/vlm_phase3d/run_v1/probes
```

The defaults fit 74 relation readouts (36 source, 36 option, source L16 input,
decision L35), two condition-only controls, static/lexical controls and 20
fixed-seed null fits at the primary option L20 readout. No GPU is used. Primary
L20 is required even if a smaller explicitly named readout list is requested;
other layer profiles remain descriptive, never an automatic best-layer search.

Only when every requested readout/control is complete does the CLI produce a
complete aggregate summary. Each readout contains held-out predictions, case
evidence, inner-fold scores, base membership and exported coefficients/scalers.
Bootstrap intervals are diagnostic and conditional on the saved cross-fitted
probes. Correlations use per-base paired effects, not layer/token rows, and do
not establish mediation. Twenty label permutations are null diagnostics, not
a high-resolution confirmatory test.

```bash
python3 -m unittest tests.test_phase3d tests.test_phase3d_features tests.test_phase3d_probes -v
```
