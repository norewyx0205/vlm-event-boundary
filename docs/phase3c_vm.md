# Phase 3C VM Runbook

This is an execution/lifecycle guide, not a change to the
[frozen scientific protocol](phase3c_protocol.md). Phase 3B is read-only source
evidence. The new runner does not rescreen, refreeze, modify case IDs or rerun
Part 1/Phase 1/Phase 3B. Real CPU eligibility, cohort freeze and GPU gates still
need to be completed on the VM before the primary pilot can start.

## Notebook and Readiness

Open `notebooks/phase3c_vm.ipynb` in the VM's JupyterLab with the pinned `phase3b`
kernel. It is a separate workflow; leave the historical Phase 3B and Colab
notebooks unchanged. Its default `PHASE3C_ACTION = "status"` only reads existing
records. Run all does not start a processor, GPU job, backup, email or Pause.
Reported summaries/progress remain explicitly unverified.

Choose one action per execution of the notebook:

- `prepare`: archive audit, resumable processor-only mapping, then freeze.
  Requires `PHASE3C_CONFIRM_CPU_PREPARATION = True`. Individual `audit`, `mapping`
  and `freeze` actions are also supported. A processor failure/incomplete quota
  blocks freeze; no subsequent GPU action is invoked automatically.
- `verify`: read-only reconstruction of the existing freeze and checksum
  validation of available completed gates. This may scan large activation files,
  but does not load a processor/model or request GPU work.
- `baseline` / `preflight`: explicit GPU gates with
  `PHASE3C_CONFIRM_GPU_RUN = True`; inspect each result before the next stage.
- `dry_run_full`: validate prerequisites and print the background launch command
  without API, SMTP, tmux, GPU or file writes.
- `full`: requires both GPU confirmation and
  `PHASE3C_PREFLIGHT_REVIEWED = True`. The confirmation does not bypass artifact
  validation. Background execution defaults to tmux, email/Pause to disabled.
- `analyze`: verify completed primary evidence and rerun CPU analysis; no GPU
  inference. Immutable analysis settings still apply.
- `backup`: explicit full stopped-run package with
  `PHASE3C_CONFIRM_BACKUP = True`. The end status cell normally only displays
  the lifecycle's existing package, never duplicates a running job's backup.

Once the selection is frozen, `prepare` validates/reuses it without rerunning
the processor or changing progress timestamps. Individual preparation actions
are blocked after freeze; use `status`/`verify`, preserving frozen evidence.
No notebook output substitutes for real artifacts and there are no mock model
predictions. The notebook forwards the same scientific parameters and guards
as the CLI, without changing the frozen protocol.

Readiness can also be viewed from a terminal at any time:

```bash
python scripts/inspect_phase3c.py \
  --plan_dir /data/yuxuanstorage/vlm_phase3c/pilot_v3

python scripts/inspect_phase3c.py \
  --plan_dir /data/yuxuanstorage/vlm_phase3c/pilot_v3 --verify --json
```

The inspector reports exact missing paths, recorded progress/elapsed time/ETA
and a suggested next action. Missing/incomplete gates do not become verified
because a summary exists. An incomplete checkpoint is distinguished from a
recorded failure or corrupted provenance. Inspection never repairs a partial
freeze or writes files. A verified artifact snapshot does not authorize launch
or confirm the latest attempt/mail/Pause succeeded; runtime guards still apply.

## Persistent Layout

Use `/data/yuxuanstorage`, not `/mnt/scratch` or the VM home disk:

```text
/data/yuxuanstorage/vlm_phase3b/a10_runs/run_v4_vm_verified/  frozen source
/data/yuxuanstorage/vlm_phase3c/pilot_v3/                   preparation
  selection/                                             frozen 12-case cohort
  processor_audit/                                       CPU eligibility audit
  execution_v1/                                          new scientific evidence
    preparation_snapshot/                                checksummed preparation copy
    phase3c_vm_config.json                                independent VM binding
    execution_config.json                                scientific runtime binding
    vm_last_status.json                                  current attempt, not old success
    captures/                                            all 36 + 3 timing sites
    baseline/, preflight/                                 gates
    patch/, routing/, knockout/                           primary task checkpoints
    analysis/, logs/                                     verified report and logs
  execution_v1_lifecycle/                                 launch/status/email/Pause journal
/data/yuxuanstorage/backups/phase3c_backup_<timestamp>/      backup parts + checksums
```

The VM binding fixes selected video hashes, preparation snapshot, execution and
orchestration code, GPU allocation, weight budget, paths and analysis settings.
Do not change these mid-run. Existing scientific baseline/preflight checkpoints
can be adopted only if their execution request matches exactly. Freeze failures
or changed provenance require investigation, not deletion or relaxed checks.

## Preparation and Gates

Activate the existing pinned `phase3b` Python environment. Keep Qwen model,
revision, FP16, eager attention and package versions specified in the protocol;
this wrapper does not install or update dependencies. Both A10 GPUs hold one
model in model-parallel mode, not two data-parallel replicas.

From the repository, complete the protocol's archive audit, processor-only
mapping audit and 12-case freeze first. The cohort is eight rescues plus four
stable controls, balanced by first mover. Then set these shell variables:

```bash
cd /data/yuxuanstorage/vlm-event-boundary
PLAN=/data/yuxuanstorage/vlm_phase3c/pilot_v3
RUN=$PLAN/execution_v1
```

Read-only baseline planning checks local artifacts without loading weights,
writing files, contacting SMTP/SURF or starting tmux:

```bash
python scripts/run_phase3c_vm.py --stage plan --target_stage baseline \
  --plan_dir "$PLAN" --output_root "$RUN" --gpus 0,1
```

Run the two gates explicitly, inspecting each result before the next:

```bash
python scripts/run_phase3c_vm.py --stage baseline \
  --plan_dir "$PLAN" --output_root "$RUN" --gpus 0,1

python scripts/run_phase3c_vm.py --stage preflight \
  --plan_dir "$PLAN" --output_root "$RUN" --gpus 0,1
```

Baseline validates all 24 low/temporal conditions and captures the required
positions at all 39 semantic sites. Preflight verifies the fixed 152-forward
technical grid on the two frozen representative cases. Read their summaries:

```bash
python -m json.tool "$RUN/baseline/summary.json"
python -m json.tool "$RUN/preflight/summary.json"
```

Both must pass with complete coverage. Review mapping, parity, identity no-op,
DeepStack timing and knockout-mask diagnostics. Technical smoke outcomes are
not primary effect estimates. `full` never runs these gates implicitly.

## Detached Primary Run

Install `tmux` if absent. Existing private SMTP and SURF configuration may be
reused; their validated files remain outside the repository, preparation,
execution, backup and lifecycle directories. Never put passwords/tokens in
shell arguments, Git, notebooks, logs or emails.

By default email and automatic Pause are disabled. Validate the complete plan
before invoking an unattended job:

```bash
python scripts/launch_phase3c_vm.py --dry_run --stage full \
  --plan_dir "$PLAN" --output_root "$RUN" --gpus 0,1 \
  --email_notify --pause_policy finished --confirm_exclusive_workspace
```

The dry run makes no API, SMTP, tmux or GPU calls and writes nothing. It requires
the baseline and preflight artifacts to be ready and prints public commands,
not credentials. Before real launch, check persistent free space (`df -h`),
confirm no colleagues/other jobs need the VM, and check the existing private
configuration with the established `phase3b_email.py check` and
`surf_workspace.py check` commands. Do not send a token/password to anyone.

Start one detached job:

```bash
python scripts/launch_phase3c_vm.py --stage full \
  --plan_dir "$PLAN" --output_root "$RUN" --gpus 0,1 \
  --session_name phase3c_full \
  --email_notify --pause_policy finished --confirm_exclusive_workspace
```

`full` runs `patch -> routing -> knockout -> analyze`, stopping on the first
failed stage. The fixed grid is 696 bidirectional observed-donor visual patches,
24 intact routing diagnostics and 2,592 all-head knockout forwards. Scientific
CLIs validate and skip completed atomic tasks; a complete stage is verified
without loading model weights again. Partial failures remain checkpointed.
Use `--retry_failed` only after diagnosing the recorded failure; the flag must
be supplied explicitly when restarting the launcher.

The launcher checks duplicate sessions, path/provenance constraints and enabled
SMTP/SURF authentication before launch. Locks protect the execution root from
duplicate orchestrators and backups. The running lifecycle repeats readiness
checks; a printed `Detached job started` alone is not evidence of successful
GPU execution. Confirm progress in the persistent logs/status.

```bash
tail -f "${RUN}_lifecycle/job.log"
python -m json.tool "${RUN}_lifecycle/job_status.json"
python -m json.tool "$RUN/vm_last_status.json"
tmux attach -t phase3c_full
```

Ctrl+C in `tail -f` stops only the display. In tmux, Ctrl+B then D detaches; do
not press Ctrl+C in the experimental command unless deliberately cancelling.
Closing your browser, JupyterLab Terminal tab or local computer does not stop
the detached VM job. Do not shut down that session or manually Pause the VM
while work is active. The tmux session may disappear after completion; logs and
status remain on persistent storage.

## Completion, Notification and Pause

Success requires a new attempt ID, exit code zero, matching immutable bindings,
verified fixed-grid checkpoints and current complete analysis with input/output
checksums. An old `aggregate_summary.json` cannot make a failed restart succeed.
Baseline/preflight completion emails explicitly do not claim a completed pilot.

The lifecycle order is:

```text
experiment stops -> validate attempt -> snapshot public status/logs
-> package all available evidence -> verify all parts and file checksums
-> confirm no other GPU/research jobs -> send completion email
-> optionally request SURF Pause
```

`--pause_policy success` pauses only after success; `finished` also permits
Pause after an experimental failure, but never without a verified full backup.
`off` never requests Pause. All Pause policies require explicit whole-workspace
authorization except `off`. A low-space, missing-video, changed-snapshot,
unverified/reports-only backup or active-job error blocks Pause and leaves an
attention status/email. The VM may then still be charging.

SMTP failure after the experiment cannot block checkpoint backup or an otherwise
eligible Pause. Provider error details are sanitized. Pause is attempted once;
an ambiguous timeout is not automatically retried. A successful request/email
does not prove the workspace is paused or billing has stopped: check SURF.
Private SMTP/SURF configuration is never bundled or attached to emails.

These are technical completion notifications, not scientific interpretations.
There is no fixed-hours shutdown timer and no automatic deletion of evidence.

## Local Backup

Automatic packages include all available Phase 3C activations (also failed or
partial attempts), selected original videos, preparation snapshot, scientific
checkpoints, public logs, status and analysis. They do not duplicate the entire
frozen Phase 3B source archive; retain that previously verified backup separately.
`complete=true` in a backup manifest means packaging completed, not that the
experiment succeeded. The manifest records the available activation file count
and the stopped attempt's status.

Independent packaging/verification is also available:

```bash
python scripts/backup_phase3c.py --run_root "$RUN" \
  --backup_dir /data/yuxuanstorage/backups
```

Download **all** ZIP parts, `backup_manifest.json` and `SHA256SUMS` using the
timestamped path from the email/status. On your Mac, substitute that exact path:

```bash
mkdir -p /Users/yuxuanwang/Desktop/thesis/Phase3C_VMbackup
scp -i /Users/yuxuanwang/.ssh/id_ed25519 -o IdentitiesOnly=yes -r \
  ywang12@145.38.205.113:/data/yuxuanstorage/backups/phase3c_backup_TIMESTAMP \
  /Users/yuxuanwang/Desktop/thesis/Phase3C_VMbackup/

python3 /Users/yuxuanwang/Desktop/thesis/Gatt/videnGenCode/scripts/backup_phase3c.py \
  --verify /Users/yuxuanwang/Desktop/thesis/Phase3C_VMbackup/phase3c_backup_TIMESTAMP
```

Verification is CPU-only and does not extract ZIPs or require the VM. Only a
successful verification on your computer establishes your local copy. The VM
cannot confirm that automatically. Reports-only packages omit activations and
videos and cannot authorize Pause or replace the full scientific backup.
