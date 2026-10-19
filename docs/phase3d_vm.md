# Phase 3D VM Runbook

No live GPU preflight has been performed for this implementation. CPU/toy
tests are not scientific model validation. These commands do not Resume a VM.
Use the persistent volume; retain frozen Phase 3B/3C evidence and off-VM backups.

## Prepare Without GPU

Use the complete `scale50_v1` prefix as the L16 source, not the sparse extra-depth
capture directory. The optional `depth50_v1` source supplies the old KO25.
Preparation checks frozen videos and capture indexes but does not load weights.
Adapt paths only if the preserved source directories have actually moved.

```bash
python scripts/run_phase3d_vm.py --action prepare \
  --source_run /data/yuxuanstorage/vlm_phase3c/scale50_v1 \
  --ko_source /data/yuxuanstorage/vlm_phase3c/depth50_v1 \
  --output_root /data/yuxuanstorage/vlm_phase3d/run_v1

python scripts/run_phase3d_vm.py --action plan --stage preflight \
  --output_root /data/yuxuanstorage/vlm_phase3d/run_v1
```

The frozen plan binds source/code/protocol hashes, all 50 canonical cases,
five grouped folds, the two first-mover preflight representatives and KO reuse
candidates. Changed code/settings need a new output root, not overwritten
checkpoints. Old metadata-only `preparation_v3` is not adopted or modified.

Inference remains Transformers 5.9.0, Torch 2.11.0, qwen-vl-utils 0.0.14, the
frozen Qwen revision, fp16/eager and the prior two-A10 placement. The same
source/live runtime and exact baseline replay are required; GPU capacity
metadata differences alone do not authorize a numerically different replay.
The GPU environment also needs NumPy/SciPy/scikit-learn for export verification.
Fitting can use `--probe_python /absolute/path/to/cpu/python`. Dependencies are
checked before preflight/full; no packages are automatically upgraded. Preserve
the pinned inference dependencies when preparing a CPU environment. Interpreter
paths preserve virtual-environment symlinks; resolving them to the system Python
would bypass the environment's installed packages.

## Two-Case Preflight

```bash
python scripts/run_phase3d_vm.py --action launch --stage preflight \
  --output_root /data/yuxuanstorage/vlm_phase3d/run_v1 \
  --session_name phase3d_preflight --gpus 0,1 \
  --email_notify --pause_after --confirm_exclusive_workspace
```

The pipeline first audits both original/permuted prompts using only the
processor. Changed prompts get new input hashes, contextual mention/relation
spans and measured baselines, never inherited predictions. It then loads the
model, checks original source replay and no-op/identity/disabled-mask parity,
and tests both L16 patch directions. Original frozen rescue behavior must
reproduce; derived option-permuted views are never screened/reselected.

Review `preflight_summary.json` and the `audit`, `capture`, `patch` task records:
one-to-one full 512-position donor/recipient mapping, actual L16 input/output,
36-block hook coverage, exact unchanged upstream and same-block text, and raw
first-answer logits. The feature verifier reconstructs these checks rather than
trusting an overall `passed` boolean. New downstream readout begins at L17.
A successful preflight is not a full result. With the flags above it packages
and requests Pause after preflight; Resume manually and inspect before full.

## Full, After Review

```bash
python scripts/run_phase3d_vm.py --action launch --stage full \
  --output_root /data/yuxuanstorage/vlm_phase3d/run_v1 \
  --session_name phase3d_full --gpus 0,1 \
  --email_notify --pause_after --confirm_exclusive_workspace
```

Full refuses to start without the exact two-case preflight. Sequence:

1. Processor audit of 200 original/permuted condition views.
2. Compact baseline capture of those 200 views; all original sources pass
   exact replay before the remaining intervention grid starts.
3. 100 bidirectional L16 whole-Event-2 replacements and downstream capture.
4. Independent verification/export of all 300 rows, 289 features per row.
5. KO50: 11,700 condition tasks, paired into 5,850 outcomes; both text-query
   groups, target/background routes and the separate whole-event route.
6. CPU cross-fitted probes, controls and immutable numeric exports.
7. Independent complete-grid and probe-artifact verification, then full backup.

Primary pooled-option L20 remains fixed. Secondary readouts preserve A/B
object-pair and relation-word representations separately; decision probes cover
L16/L20/L24/L28/L32/L35. They are not pooled into a higher-capacity classifier
or promoted based on test accuracy. Default: 83 relation fits + 24 control/null
jobs. Statement-truth diagnostics are explicitly re-expressions of canonical
relation probes, not new A/B-label training or independent evidence.

KO reuse never compares plan-specific task IDs. It uses the actual case,
query/key/control/window semantic key, original mask/edge budgets, unchanged
implementation, full live baseline/input/runtime replay, sealed source A/B
logits/decisions and provenance. Original summary-only inherited rows are
recomputed. The preserved KO25 currently has 4,122 condition candidates
(2,061 pairs), not verified reuse; 864 inherited paired summaries lack the
complete mask chain. Exact live gates may reject candidates. New and verified
reused rows have equal weight across all 50 bases, with no pilot/new strata.

## Progress And Resume

```bash
tmux attach -t phase3d_full
tail -n 60 /data/yuxuanstorage/vlm_phase3d/run_v1_lifecycle/job.log
python -m json.tool /data/yuxuanstorage/vlm_phase3d/run_v1/progress_status.json
python -m json.tool /data/yuxuanstorage/vlm_phase3d/run_v1_lifecycle/job_status.json
```

Every task prints its start, completed/total, resumed count, elapsed time and
checkpoint path. KO is checkpointed per condition, so failure in its matched
partner does not repeat the completed forward. CPU fitting prints each outer
fold/readout. Inspect failures before adding `--retry_failed` to the same
launch command. This retries only missing failed tasks, never completed
checkpoints; failed error records remain preserved. Do not launch duplicate
jobs or manually Pause an active experiment. Closing the browser/PC is safe
after `Detached job started`; tmux continues on the VM.

The launcher checks conservative space for compact evidence, feature export,
probe products, selected videos, a full backup copy and a 12-GiB reserve.
Workers stop before exhausting that reserve. No cleanup is automatic.

## Backup, Email And Pause

SMTP and SURF private configs use the existing owner-only locations, outside
the repository/run/backups. Do not put passwords/tokens in CLI arguments or Git.
`--email_notify` and `--pause_after --confirm_exclusive_workspace` are opt-in.
The VM must be exclusive to this experiment. No stage pauses between pending
full-run GPU/CPU work. Success or failure leads to:

`preserve status/logs -> full multipart package -> checksums -> idle check ->
technical email -> optional SURF Pause request`.

An incomplete/reports-only backup, another research/GPU process or a failed
precheck blocks Pause. Email failure is recorded but does not prevent a safe
backup/Pause. An accepted Pause request is not confirmation that billing has
stopped; check the SURF portal. Completion is bound to this attempt's nonce,
not a stale successful analysis in the directory.

The transport reuses the `phase3c_backup_*` folder prefix; the manifest has
`schema=phase3d_backup_v1`. It contains all available new compact captures,
features, probes, task/mapping/provenance records, logs and selected videos,
not a duplicate of all historical Phase 3B/3C activations. Keep the historical
backups separately. A verified VM package is not an off-VM backup.

```bash
python scripts/backup_phase3d.py --run_root /data/yuxuanstorage/vlm_phase3d/run_v1
python scripts/backup_phase3d.py --verify /path/to/downloaded/phase3c_backup_TIMESTAMP
```

Transfer every part, `backup_manifest.json` and `SHA256SUMS`, then verify on the
destination. No source evidence or off-VM copy is deleted automatically.
