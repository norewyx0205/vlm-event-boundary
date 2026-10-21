# Phase 3D: Singleton Knockout Follow-Up

## Frozen Scope

This supplement decomposes the existing **options_all -> whole_event2**
four-block attention knockout, using the **same 50 independent rescue bases**,
archived prompts, low/temporal conditions, and 512-position matched visual
support. Every base has equal weight. There are no pilot/new strata, new
stimuli, new probes, new patch sites, head searches or counterfactual panels.

The only new interventions are all-head knockouts in block **16**, **17**,
**18**, or **19**, separately. Indices are zero-based: L16 is the 17th block.
There are 4 x 50 x 2 = **400 interventional condition forwards**, yielding
200 paired outcomes. The two-case preflight's 16 KOs belong to this same grid
and are resumed, not repeated. Baseline replay and disabled-hook checks are
additional technical forwards, not included in 400.

The existing L16-19 joint knockout contributes 50 read-only reference pairs.
It is matched by route, window, condition and pair identity, then checked for
input, baseline, edge budget, mask semantics, runtime and original reuse
provenance. It is not rerun or selected by effect. No old artifact is changed.

L16 block-output visual replacement occurs after block 16 attention/MLP;
singleton L16 KO acts inside block 16 attention. Adjacent peaks do not establish
the same computational operation or a unique before/after circuit.

## Gates and Outputs

The source must be a complete, intact Phase 3D KO50 run. Preparation is CPU-only
and freezes code/protocol hashes, case IDs, mappings, source checkpoints, raw
baseline logits, and joint-reference provenance in a separate output root.
Do not move/delete the source or modify pinned code after freezing. Keep the
separate full Phase 3D backup; this supplement does not duplicate its activation
stores. A partially interrupted preparation requires a new empty output root;
completed preparation and execution can be resumed normally.

Preflight uses the source's fixed two representatives, one per first-mover
order. Live processor tensors and full-vocabulary first-token baseline logits
must replay exactly; disabled hooks must be exact no-ops. Runtime compatibility
is inherited from the original dual-A10 engine. No tolerance relaxation,
silent case dropping or rescue re-screening is allowed. All 100 condition gates
must finish before new full-grid tasks run. Resume verifies sealed task/payload
hashes; failures require explicit `--retry_failed`. Progress includes task count,
elapsed time, resumed count and checkpoint path. No automatic experiment
expansion is triggered.

Analysis reports paired temporal margin change, low margin change and
compression = low change - temporal change. Margin is the correct-option logit
minus the incorrect-option logit at the first answer token. Actual predictions,
categorical flips, strict sign crossings, strict wrong/correct transitions,
zero ties, non-A/B outputs and actual all-head causal-edge budgets are retained.
Diagnostic 95% bootstrap intervals resample **base cases**, not condition rows.
All 50 cases are required in each cell. The joint intervention cuts four times
the layer-edge budget of a singleton; it is not a budget-matched or additive
comparison. Weak singleton effects with a strong joint effect could reflect
accumulation, redundancy, nonlinear interaction or compensation. Leave-one-layer-
out interventions remain a separately decided conditional follow-up.

## VM Commands

Do not launch this until the code has been pushed/pulled and VM use confirmed.
Use the existing pinned inference environment, without upgrading dependencies.

```bash
cd /data/yuxuanstorage/vlm-event-boundary
python scripts/run_phase3d_single_layer_ko.py --action prepare \
  --source_run /data/yuxuanstorage/vlm_phase3d/run_v2 \
  --output_root /data/yuxuanstorage/vlm_phase3d_single_layer_ko/run_v1

python scripts/run_phase3d_single_layer_ko.py --action plan \
  --output_root /data/yuxuanstorage/vlm_phase3d_single_layer_ko/run_v1

python scripts/run_phase3d_single_layer_ko.py --action run --stage preflight \
  --output_root /data/yuxuanstorage/vlm_phase3d_single_layer_ko/run_v1 --gpus 0,1
```

Inspect `preflight_summary.json` and logs. Only then launch full, with existing
private SMTP/SURF configuration kept outside the repository and artifact roots:

```bash
python scripts/run_phase3d_single_layer_ko.py --action launch --stage full \
  --output_root /data/yuxuanstorage/vlm_phase3d_single_layer_ko/run_v1 \
  --gpus 0,1 --email_notify --pause_after --confirm_exclusive_workspace

tail -f /data/yuxuanstorage/vlm_phase3d_single_layer_ko/run_v1_lifecycle/job.log
```

`Ctrl+C` stops `tail`, not the detached job. Do not manually pause an active run.
The lifecycle sends a technical success/failure email, creates and verifies a
full backup, checks workspace quiescence, then requests Pause if opted in.
Precheck/backup failure blocks automatic Pause. Portal status is authoritative
for billing; an email or accepted request is not proof of paused state. Storage
checks reserve 12 GiB plus conservative output/backup space, deleting nothing.
Backup folder names retain the transport prefix `phase3c_backup`; the manifest
schema `phase3d_single_layer_ko_v1_backup` identifies this follow-up.
Its new `.pt` files contain raw logits, not new activation captures. The generic
backup field `activation_tensors_included` indicates full, not reports-only,
transport coverage; `new_activation_capture=false` and
`historical_activation_stores_included=false` state the actual scope.

```bash
python scripts/run_phase3d_single_layer_ko.py --action run --stage verify \
  --output_root /data/yuxuanstorage/vlm_phase3d_single_layer_ko/run_v1
```

Verification and report generation are CPU-only. Outputs live under `analysis/`:
`case_decomposition.{json,csv}`, `aggregate_cells.{json,csv}`,
`single_layer_vs_joint.png`, `report.md`, and `aggregate_summary.json`.
After any run, download every backup part, manifest and SHA256SUMS, verify
off-VM, and retain the frozen Phase 3D backup separately. The standalone
`--action backup` also supports a failed/partial run; it is not scientific
completion. VM backup verification does not establish off-VM preservation.
