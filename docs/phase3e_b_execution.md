# Phase 3E-B Execution Notes

## Material Passport

- Date: 31 October 2026.
- Scope: implementation and local CPU verification of the paired-option protocol.
- No new GPU inference, KO, patch or scientific probe fitting has been performed.
- Formal inference still requires real-model compact-collector preflight.

## Data Flow

1. Generate the corrected v2 40-family materials without behavioural screening.
2. Review the CPU target/distractor interaction audit; audit actual processor
   spans and run the first two families (32 prompts).
3. Check exact capture/no-hook logits, independent greedy parity, L17/L18
   residual arithmetic, paired reference-token alignment and video tensor budgets.
4. Measure compact output bytes and require room for all captures, the feature
   store, a full backup and a 12-GiB untouched reserve.
5. Capture the remaining cells; verify and reuse the 32 preflight checkpoints.
6. Export all 1,280 statement rows into the immutable feature store.
7. Package/verify captures and materials; send completion mail and request Pause.
8. Fit and verify grouped CPU probes from the portable feature store locally.

Use `run_phase3e_paired_unattended.py` for detached execution. It launches tmux,
uses the persistent model caches and explicit GPU visibility, and runs each
stage in a separate child process. Full launch requires a verified preflight;
it does not start full capture automatically after the technical gate.

Both successful and failed experiments produce a checksum-verified snapshot of
all available public files, then a completion email. A failed run's snapshot is
not a completed scientific result. Automatic Pause requires exclusive-workspace
confirmation, a verified full snapshot and quiescence. A failed backup or a
remaining worker blocks Pause and sends an attention message. SMTP failure
alone does not prevent Pause after the other gates pass.

## CPU Interaction Audit

Both material preparation commands now emit `interaction_audit.json` and
`interaction_audit.md`. To audit an existing plan without changing it or
rendering video:

```bash
python scripts/audit_phase3e_interactions.py \
  --plan_path /absolute/path/to/plan.json \
  --output_root /absolute/path/to/new_interaction_audit
```

The audit deduplicates prompt views, records actual co-motion frames and
distractor identities, and measures shape-footprint overlap and centre distance.
Paired changes retain both raw levels and average within scene family first.
Capture validates the audit's plan/row hashes and binds its file hash in the
frozen configuration. Use a newly prepared root; do not retrofit or overwrite
an old frozen capture configuration.

The original v1 audit found second-event co-motion in 80/80 Low events versus
0/80 Temporal events (100% to 0%, -100 percentage points). The corrected v2
generator schedules distractors per boundary condition using the original
Level-5 `near_targets` helper. The same fixed 40 families now give 80/80 in both
conditions (100% to 100%, 0 percentage points). Identity, path, duration and
event-slot matching are enforced; within a boundary, reversing target order
does not change the distractor schedule.

Use a new root such as `paired_option_event_relative_v2`, never overwrite v1
videos/configurations. A v1 plan cannot launch formal capture, even if its own
hash is valid. Re-render all 160 videos, audit decoded frames and bytes, and run
the 32-prompt compact preflight on those corrected videos before full capture.
The primary probe and 640-prompt design are unchanged.

For a read-only historical annotation check:

```bash
python scripts/audit_phase3e_interactions.py \
  --historical_annotations /path/to/L5_full/annotations.jsonl \
  --output_root /absolute/path/to/new_historical_audit
```

Missing saved moving-distractor windows are reported as unknown, not filled in
from current generator settings. Available Part-1 annotations record
`near_targets` but not those windows; verifying historical rendered timing
would require the corresponding videos.

## VM Stages

Use a new run directory, outside all historical results. `SOURCE_PLAN` is an
existing checked plan carrying the frozen Qwen revision/runtime/sampling settings.

```bash
ROOT=/data/yuxuanstorage/vlm_phase3e/paired_option_event_relative_v2
STORAGE=/data/yuxuanstorage
SOURCE_PLAN=/absolute/path/to/frozen_runtime/plan.json
BACKUPS=/data/yuxuanstorage/backups

python scripts/run_phase3e_paired_capture.py --stage prepare \
  --output_root "$ROOT" --storage_root "$STORAGE"

python scripts/run_phase3e_paired_unattended.py --action plan --stage preflight \
  --output_root "$ROOT" --storage_root "$STORAGE" --source_plan "$SOURCE_PLAN" \
  --backup_dir "$BACKUPS"

python scripts/run_phase3e_paired_unattended.py --action launch --stage preflight \
  --output_root "$ROOT" --storage_root "$STORAGE" --source_plan "$SOURCE_PLAN" \
  --backup_dir "$BACKUPS" --gpus 0,1 --session_name phase3e-paired-preflight \
  --email_notify --pause_after --confirm_exclusive_workspace
```

Review the new two-family technical result and storage projection before a
separate full launch. If Pause was enabled, resume and confirm VM use first:

```bash
python scripts/run_phase3e_paired_unattended.py --action launch --stage full \
  --output_root "$ROOT" --storage_root "$STORAGE" --source_plan "$SOURCE_PLAN" \
  --backup_dir "$BACKUPS" --gpus 0,1 --session_name phase3e-paired-full \
  --email_notify --pause_after --confirm_exclusive_workspace
```

Status and logs are in the sibling `<run>_lifecycle/job_status.json` and
`job.log`. Code/protocol snapshots are included in the backup. Existing tmux
sessions, another workspace job, active GPU compute, or changed capture code
block a launch. No failed experiment is automatically retried.

## Individual Stages

These lower-level commands are for supervised operation/diagnostics. They do
not provide failure notifications or automatic sequencing by themselves:

```bash
python scripts/run_phase3e_paired_capture.py --stage preflight \
  --output_root "$ROOT" --storage_root "$STORAGE" --source_plan "$SOURCE_PLAN"

python scripts/run_phase3e_paired_capture.py --stage capture \
  --output_root "$ROOT" --storage_root "$STORAGE" --source_plan "$SOURCE_PLAN"

python scripts/run_phase3e_paired_capture.py --stage verify --output_root "$ROOT"
python scripts/run_phase3e_paired_capture.py --stage export --output_root "$ROOT"

python scripts/run_phase3e_paired_capture.py --stage package \
  --output_root "$ROOT" --storage_root "$STORAGE" --backup_dir "$BACKUPS"
```

Inspect the verified package path returned by `package`. The `finish` command
requires that exact package, not its parent backup directory:

```bash
python scripts/run_phase3e_paired_capture.py --stage finish \
  --output_root "$ROOT" --storage_root "$STORAGE" \
  --backup_dir /data/yuxuanstorage/backups/ACTUAL_VERIFIED_PACKAGE
```

`finish` requests Pause, but final portal state/billing stop must still be checked.
It uses the same GPU/CPU research-worker idle guard as the detached lifecycle,
both before notification and immediately before Pause. Its CLI takes the shared
workspace lock before the run lock. A worker detected after email blocks Pause
and is recorded in the completion receipt.
Do not call `finish` after the detached lifecycle; that job already handles
notification and Pause. Lifecycle packages use schema
`phase3e_paired_option_job_backup_v1`; individual-stage packages use
`phase3e_paired_option_backup_v1`. Both use the existing supported
`phase3c_backup_...` directory prefix, with their actual scope in the manifest.
A failed gate preserves existing outputs, stops the stage and does not initiate
another GPU experiment or delete files. An incomplete tensor without its
committed checkpoint requires inspection; it is not silently overwritten.

## CPU Fitting

After recovering and checksum-verifying the full package outside the VM, the
`feature_store` is sufficient for these CPU commands. Keep probe output separate:

```bash
python scripts/run_phase3e_option_probes.py \
  --manifest_path /absolute/path/to/feature_store/manifest.json \
  --output_root /absolute/path/to/new_probe_results

python scripts/run_phase3e_option_probes.py --verify_only \
  --manifest_path /absolute/path/to/feature_store/manifest.json \
  --output_root /absolute/path/to/new_probe_results
```

Default fitting produces 11 reference-last readouts plus two controls, each with
five outer models (65 frozen models). The site set includes the three primary
block outputs, L20 output and the declared L16/L17/L18 input and L17/L18 attention
sites. Add `--include_secondary` in both commands for reference-mean/option-final
probes (175 outer models including controls). All variants of a scene remain
together throughout inner selection and outer testing.

`report.md` includes original low/temporal evidence and their difference, and
true/false-video scores and their difference. JSON artifacts retain every
statement, pair, family aggregate, uncertainty interval, model margin, tie and
non-A/B output. Direct contrasts are checked as `w dot (D / scale)` without
applying a scaler offset or classifier intercept to D.

## Local Verification

The local regression suite passes 112 tests, including event-relative interaction
audits, feature/probe tests and lifecycle tests. Lifecycle tests use real temporary
ZIP packages and
checksum verification; model execution, email and SURF calls are mocked. They
cover success, partial failure, stale backup rejection, failed packaging,
quiescence, CPU workers with no GPU compute, workers starting during notification,
SMTP failure, tmux launch and exclusive-workspace confirmation. Interaction tests
cover all fixed families, end-exclusive motion intervals, exact shape overlap
including canvas clipping, unchanged input metadata and audit tamper detection.
Timing tests reject fixed-Low windows in Temporal, swapped distractor slots and
off-by-one motion windows. Historical checks distinguish missing windows from
observed zero co-motion. Legacy validation is read-only and explicit.
No VM/GPU job was started by these local tests.

The old two-scene backup can be exported without starting the VM:

```bash
python scripts/export_phase3e_option_smoke.py \
  --backup_dir /absolute/path/to/phase3c_backup_20261027_180633_971812 \
  --output_root /absolute/path/to/new_smoke_export
```

That export has 64 statements, 24 L17/L18 feature readouts and a 4,096-d hidden
dimension. It cannot supply the new L16/L20 reference-last readouts or the 40
new scene families, so it is deliberately not a formal training feature store.

The portable formal store's `source.json` preserves per-prompt input hashes,
logical identity spans and actual primary/secondary token positions. Its vectors
are indexed by option/readout/site, not by the old cross-object feature name.
