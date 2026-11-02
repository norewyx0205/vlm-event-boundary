# Rescue50 Direct-Difference Execution

## Scope

This implements `phase3e_paired_difference_training.md`, not the earlier
single-video truth-probe training. Keep the original 50 rescue cases,
historical outer folds, Low/Temporal conditions and literal A/B views.
Add order-reversed versions without replacing, screening or expanding bases.

- `phase3e_order_materials.py`: exact original reconstruction, matched-encoder
  cache, unchanged identity/path/distractor geometry, real motion retiming.
- `run_phase3e_order_capture.py`: eight-original/eight-reversed prompt gate,
  200 missing reversed-prompt captures, verified endpoint-store export.
- `phase3e_order_probes.py`, `run_phase3e_order_probes.py`: direct D/-D fitting
  with opposite truth labels, no centering/intercept and frozen-model replay.
- `run_phase3e_order_unattended.py`: tmux, failure/success mail, complete public
  backup, checksum validation and exclusive idle-workspace Pause.

## Matched Encoding

The original cohort contains two encoder histories. The six early cases
(12 Low/Temporal videos) are reproduced byte-for-byte by local OpenCV 4.13 /
Lavf61.7.100. The subsequent 44 cases use the VM OpenCV 5.0 / Lavf62.12.101
encoding. Applying one encoder to both groups produces small decoded-pixel
differences, so no tolerance was relaxed and neither group was replaced.

The local cache contains rechecked byte-identical original reconstructions and
their reversed renders. VM preparation checks each cache video's source hash,
reversed motion-spec hash and file hash. It independently compares original
and reconstructed decoded frames on the VM. Other cases render in their
original VM environment. All 100 original videos passed exact decoded-frame
comparison before the complete 100 reversed-video/200-prompt plan was frozen.

No claim is made that decoded frame hashes are portable across decoder
implementations. Exact comparison always decodes both sides in one environment.
Reversed versions use the matching encoder within each original/reversed pair.
Target timing changes, while paths, duration, object roles in text and
boundary-specific distractor schedules remain fixed.

## GPU Gates

Two original cases, covering both first-mover identities, supply eight prompts
(two boundaries times two A/B layouts). Before collecting each corresponding
reversed prompt, replay its original input through the frozen model. Require
exact historical first-token logits and exact recovered reference-token states
at all four block outputs. Source reuse is not established just by filenames.

Reversed prompts must retain full prompt token IDs, contextual reference-token
positions and sampling metadata, but change actual visual input tensors.
Each receives an unhooked forward, a read-only-hook capture and a standard
first-answer-token generation parity check. Wrong reversed answers are kept.

Only a verified eight-prompt preflight permits full capture. Its checkpoints
are reused; 192 reversed prompts then remain. The initial full store contains
800 endpoint vectors: 400 original and 400 reversed statement vectors. Direct
training constructs 800 directed D rows from 400 unique unordered statement
pairs. Mirrored directions are augmentation, not independent bases.

## Run Commands

```bash
python scripts/run_phase3e_order_capture.py --action prepare \
  --output_root /data/yuxuanstorage/vlm_phase3e/rescue50_direct_D_v3 \
  --materials_root /data/yuxuanstorage/vlm_phase3e/rescue50_order_materials_v2 \
  --source_root /data/yuxuanstorage/vlm_phase3d/run_v2 \
  --source_store /data/yuxuanstorage/vlm_phase3e/rescue50_reference_last_v1/feature_store/manifest.json

python scripts/run_phase3e_order_unattended.py --action launch --stage full \
  --output_root /data/yuxuanstorage/vlm_phase3e/rescue50_direct_D_v3 \
  --email_notify --pause_after --confirm_exclusive_workspace
```

The full stage runs preflight, full capture, endpoint export and independent
raw verification in that order. Any failing child stops later children.
Both successful and failed attempts attempt a full public backup and email.
Pause follows an idle check; it never interrupts another active worker. A
backup failure is disclosed, and the idle VM may still be paused under the
user's explicit success-or-failure shutdown instruction. Pause does not delete
persistent files. The final portal state must be checked separately.

The completion email describes capture/export, not finished CPU probe fitting.
After verified endpoints are retrieved, fit on the local CPU:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/run_phase3e_order_probes.py \
  --manifest_path /path/to/feature_store/manifest.json \
  --output_root /path/to/new/direct_D_probes

python3 scripts/run_phase3e_order_probes.py \
  --manifest_path /path/to/feature_store/manifest.json \
  --output_root /path/to/new/direct_D_probes --verify_only
```

The four sources are Low-only, Temporal-only, full mixed and equal-budget
mixed. Each frozen fitted model tests both boundaries of held-out bases.
Reports use pairwise accuracy, endpoint projection levels and paired evidence,
not a re-labelled version of old single-video decoding accuracy.

## Verification Record

On 2 November 2026, 124 local Phase 3E regression tests passed. All 19 dedicated
direct-D/material/capture/lifecycle tests also passed in the VM Python 3.12
environment after the budget and input-audit fixes. The additional local test
verifies the direct-D training-unit metadata and result replay. These are engineering tests; actual model
preflight and capture
status are recorded in the run's lifecycle JSON and raw checkpoints.
Historical Phase 3B-3E artifacts and single-video controls remain unchanged.

The first v1 launch stopped at the runtime gate: its new executor had
incorrectly used a 14-GiB weight-placement budget instead of the historical
10.0 GiB. Preparation now reads the budget from the frozen source runtime,
with a regression test enforcing preservation. No runtime or replay tolerance
was relaxed. The original v1 log, config and verified backup are retained;
the corrected placement attempt used an isolated v2 capture root.

V2 passed runtime validation and the first exact original logits/token replay,
then rejected the first reversed input because it compared all video metadata
as if they were sampling settings. CPU comparison isolated a change in
`ambiguous_visual_cells` (911 original, 909 reversed), a motion-dependent ROI
assignment diagnostic. The v3 input audit records both counts and requires
exact equality of every other video-metadata field. It also enforces unchanged
text readouts/groups, masks, visual positions and nonpixel input tensors;
visual pixels must change while their shape/dtype remain fixed. V1 and v2
outputs and verified backups remain unchanged.

V3 subsequently passed the fixed two-case/eight-prompt CPU input audit and
all eight real-model preflight prompts. Exact historical original logits and
four-layer reference states replayed for all eight; reversed captures passed
read-only-hook no-op and standard-generation parity. The lifecycle then
started full capture, reusing the eight preflight endpoints. The public proof
and deployment record are under
`analysis/phase3e/direct_difference_20261102/deployment_v3/`.
V3 then completed all 200 reversed prompt captures, export and raw verification
successfully. Its lifecycle took 3,581.985 seconds, including finishing work.
A fresh post-run verification independently replayed all 200 checkpoints.
The complete VM package `phase3c_backup_20261102_120340_428783` was downloaded
and checksum-verified locally; it is approximately 470 MiB.

Local CPU fitting and independent saved-model replay completed 80 outer models
in `analysis/phase3e/direct_difference_20261102/probes_v2/`. The inherited
`single_video` training-unit description was corrected to
`direct_same_object_paired_order_difference`; all model arrays and statistics
exactly matched the retained metadata-only predecessor `probes_v1/`.
The actual features were D in both runs. No capture code or input was changed
by this metadata correction.

Main interpretation and the separate model-answer results are documented in
`analysis/phase3e/direct_difference_20261102/analysis_notes.md`. The complete
capture package, final CPU models, exact CPU source snapshot and figure are
staged in `baidu_upload_staging_v1/`; a cloud upload is not yet confirmed.
