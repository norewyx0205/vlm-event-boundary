# Phase 3C Support-Decomposition Follow-Up

This is an **exploratory amendment after inspecting the completed pilot**, not
an independent confirmatory replication or preregistered analysis. The user
selected support decomposition and token-budget controls on 2026-10-10.

## Frozen Scope

Reuse all 12 frozen Phase 3C bases (8 rescue, 4 stable) and both directions at
block outputs L0, 4, 8, 12, 16 and 20. Do not rank cases/layers using new effects.
Whole-Event-2 and both-target-union effects are verified, read-only references.
Do not rerun them, the old knockout grid, Phase 3B or activation capture.

Add two observed-donor, event-relative, one-to-one interventions:

- **Non-target complement:** all matched Event-2 cells outside the union of
  either condition's padded target ROI. This support and the target union
  exactly partition the original whole-event mapping in both directions.
- **Token-budget-matched non-target context:** sample from that complement
  using a fixed hash of seed, base, paired temporal bin and spatial cell.
  Match the target union's token count separately within every paired bin.
  Use the same donor/recipient correspondence in both directions. Reject
  insufficient pools; no interpolation, duplicate vectors or null filling.

Non-target context can include distractors. These are **position supports**,
not pure target-free information: contextual residuals may encode objects after
attention. Budget matching does not match all spatial/content/attention factors.

L0 retains the known pre-DeepStack timing limitation. No timing-control rerun:
the original separately captured pre/post states and controls remain available.
L4+ are unaffected by subsequent DeepStack addition.

## Controls and Outcomes

First run 48 exact same-state identity patches on the original two frozen
representatives (both mover orders), covering both supports, all six sites and
both conditions. Any failure blocks the primary follow-up.

Then run 288 new patches: 12 bases x 2 directions x 2 supports x 6 layers.
Compare raw correct-option margin change, donor-aligned change, Recovery and
strict sign crossing. Keep rescue/stable strata and directions separate.
Report paired target-minus-budget-context contrasts, whole-minus-target and
whole-minus-target-minus-complement non-additivity. The last quantity is not
additive causal mediation; nonlinear interactions remain possible.

Case-bootstrap intervals remain exploratory diagnostics. Reusing the same
cases does not increase the number of independent experimental units.

## Execution and Storage

Use `scripts/run_phase3c_support.py` with a new sibling output root. Explicit
`--action prepare` independently verifies the original completed pilot and
freezes the new support/config hashes before any new intervention. New captures
are unnecessary; original capture hashes and input tensor bytes are rechecked.
Runtime, GPU placement and precision must exactly match the donor execution.
Never modify or create files in the source plan, execution or Phase 3B root.

Example VM layout:

```text
/data/yuxuanstorage/vlm_phase3c/pilot_v3/execution_v1/     frozen source
/data/yuxuanstorage/vlm_phase3c/support_followup_v1/      new config/checkpoints
/data/yuxuanstorage/vlm_phase3c/support_followup_v1_lifecycle/
```

Use the existing pinned environment; no dependency update is needed:

```bash
python scripts/run_phase3c_support.py --action prepare \
  --plan_dir /data/yuxuanstorage/vlm_phase3c/pilot_v3 \
  --source_run /data/yuxuanstorage/vlm_phase3c/pilot_v3/execution_v1 \
  --source_backup /data/yuxuanstorage/backups/phase3c_backup_20261010_145907_284325 \
  --output_root /data/yuxuanstorage/vlm_phase3c/support_followup_v1
```

Keep those four arguments unchanged when running `--action run --stage preflight`.
Inspect all 48 identity results, then use `--action launch --stage full` with
`--email_notify --pause_after --confirm_exclusive_workspace` if the user has
authorized whole-VM Pause. The tmux session is `phase3c_support`. Long stages
print measured progress/ETA and checkpoint each task. Resume verifies completed
keys before loading weights; failures require explicit diagnosis/retry.

The lifecycle verifies a new successful attempt, saves all new evidence, backs
it up with checksums, sends the completion email and only then requests Pause
after checking the workspace is idle. Failure also backs up stopped evidence
before an eligible Pause. At least 12 GiB free is required; no automatic deletion.

The small follow-up backup includes **all newly produced artifacts**, but does
not duplicate the source activations/videos. Its manifest records this external
dependency and the exact source backup manifest hash. Retain the separate full
Phase 3C backup and verify both packages locally. Local source backup verification
was completed before this follow-up's GPU launch; source files remain untouched.

After downloading every part, manifest and SHA256SUMS of the follow-up package,
the shared checksummed-package verifier also handles this new schema:

```bash
python scripts/backup_phase3b.py --verify /path/to/downloaded/followup_package
```

Keep using `backup_phase3c.py --verify` for the separate full source package.
