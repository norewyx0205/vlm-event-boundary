# Phase 3C Fixed-Budget Dose Follow-Up

Exploratory post-pilot amendment approved on 2026-10-11. This reuses the same
8 rescue and 4 stable independent bases; it is not a new confirmatory cohort.
The completed Phase 3B, Phase 3C pilot and support-decomposition artifacts are
read-only dependencies. No old patch, knockout or activation capture is rerun.

## Fixed Grid

- Decoder block outputs L12 and L16; both temporal-to-low and low-to-temporal.
- Exact total budgets 64, 128 and 256 observed-donor visual-token positions.
- Context-only versus the complete cross-condition target-ROI union plus context.
- Three fixed sampling seeds: 42, 43, 44. Supports are nested across budgets
  within each seed. A budget-independent hashed spatial-cell ranking defines
  the context samples, not observed divergence or intervention outcomes.
- Both compositions use exactly the same quota in each paired temporal bin.
  Quotas are proportional to the frozen whole-event bin counts, with a fixed
  largest-remainder tie rule. Reject target-inclusion/capacity conflicts; no
  relaxed quotas, target trimming, dummy vectors, interpolation or duplication.

The two compositions share the ranked context prefix. At a fixed budget, the
mixed intervention substitutes all target-union positions for an equal number
of context positions; it does not just add more tokens. Context residuals may
already represent targets. Equal counts do not match attention/content/geometry.

## Technical Gates

CPU preparation first verifies the completed source pilot and the completed
support follow-up, hashes their evidence and freezes all mappings/config/code.
CPU preparation also reads the source tensors to freeze exact full-vocabulary
logit signatures and full Event-2 donor-vector signatures at L12/L16.

Before any new primary patch, run 24 in-memory capture replays (every condition)
and 144 same-state identity patches (both frozen representatives, all supports,
seeds, budgets, layers and conditions). Require exact source input, logit and
donor-vector bytes; identity patches require zero logit difference. No new `.pt`
files are saved. A failed gate blocks primary patching and is not quietly dropped.

The resumed VM reported a memory-capacity-only change on the second A10.
All compute/runtime/model/precision/device-map fields must still match exactly.
Only visible-memory-capacity metadata differences may enter the exact replay
gate; they are recorded in a separate frozen observed-runtime file. Any other
difference, any replay mismatch or subsequent runtime change blocks execution.
Original source runtime records are never edited or relabelled.

## Analysis

There are 864 primary patched forwards. Keep per-seed records, then average
the three seeds equally within each case/composition/budget/layer/direction.
Bootstrap independent bases, not seeds, layers or forwards. Strict-rescue
fractions describe seed runs within cases, not additional rescued independent
cases. Report correct-option margin changes, donor-aligned changes, Recovery,
seed variation and paired mixed-minus-context contrasts. Prior whole-event and
target-union outcomes remain separate read-only references with different budgets.

All intervals remain exploratory pilot diagnostics. A dose response or a
composition difference does not uniquely establish information content, routing,
additive mediation or a population-level effect. L12/L16 are fixed based on the
preceding pilot, not independently discovered in this run.

## VM Execution

New persistent root: `/data/yuxuanstorage/vlm_phase3c/budget_followup_v1`.
No source activations/videos are copied. Keep at least 12 GiB free; no automatic
deletion. Each replay/identity/patch has a unique checkpoint. Verified completed
tasks resume without recomputing; failed tasks need explicit diagnosis/retry.

```bash
python scripts/run_phase3c_budget.py --action prepare \
  --plan_dir /data/yuxuanstorage/vlm_phase3c/pilot_v3 \
  --source_run /data/yuxuanstorage/vlm_phase3c/pilot_v3/execution_v1 \
  --source_followup /data/yuxuanstorage/vlm_phase3c/support_followup_v1 \
  --source_backup /data/yuxuanstorage/backups/phase3c_backup_20261010_145907_284325 \
  --output_root /data/yuxuanstorage/vlm_phase3c/budget_followup_v1 --gpus 0,1
```

Keep arguments unchanged; use `--action launch --stage full` with
`--email_notify --pause_after --confirm_exclusive_workspace` only with the user's
authorization. The detached `phase3c_budget` tmux job executes replay -> identity
preflight -> primary patches -> CPU analysis. Each stage prints progress/ETA and
persists checkpoints. The full sequence fails closed at any technical gate.

After the stopped attempt, save and verify a complete package of **new** evidence,
send the technical completion email, then request SURF Pause after an idle check.
The new package does not duplicate source tensors/videos. Retain the locally
verified complete source backup and prior support backup. Download all new parts,
manifest and SHA256SUMS and verify with `scripts/backup_phase3b.py --verify`.
The email does not confirm local backup, portal Paused state or billing stop.
