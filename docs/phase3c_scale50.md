# Phase 3C Whole-Event-2 Expansion to 50 Rescues

Approved 2026-10-13. Expand the existing frozen 50-rescue Phase 3B cohort,
not 50 newly selected confirmation cases. Retain its original/swapped prompts
and mover distribution. No case may be dropped/replaced based on patch effect,
and a failed live baseline/mapping gate stops the experiment before patching.

Fixed scope: decoder block-output L16 primary, L12 comparison; whole matched
Event-2 visual support; observed-donor one-to-one replacement; both directions.
Total 200 task rows. The previous 8 pilot and 8 replication rescues provide
64 prior reference rows and 32 condition captures (16 pairs). These references are
replayed on the current VM, not silently mixed into the new results. The 34
remaining rescues yield 136 new effects and 68 new condition captures.

All 100 condition baselines pass archived-input/first-token checks, full-logit
standard-generation parity, disabled-hook exact parity, finite strict rescue
margins and explicit hook/mapping audits before any patch. Prior captures must
also match current full logits, input tensor bytes and both planned state slices
exactly. Representative T1-first/T2-first identity patches precede the primary
stage. After freeze, failures are preserved rather than skipped or relaxed.

New captures store only whole-event positions at the two intervention sites.
This is an explicitly approved storage amendment: previous 36-layer/DeepStack
evidence remains read-only, but the newly expanded context support is not saved
at untested layers. No dummy, interpolation, duplication or new KO/ROI/budget
grid. Actual mapped supports/coverage are retained for each direction.

Pin software/model/revision/FP16/eager/placement as before. Log capacity-only
GPU metadata changes separately; they cannot waive exact replay. Checkpoints
bind code, manifest, mapping, runtime compatibility and source evidence hashes.

Analysis separates original pilot 8, independent replication 8, newly tested
34 and the descriptive full 50. Base is the independent unit, not layers or
directions. Report correct-option margins, actual categorical predictions,
strict sign crossings, non-A/B outputs and zero-margin ties separately. A
positive A/B margin alone is not a rescue if greedy output is not the correct
option. Bootstrap intervals are diagnostics for the selective cohort, not
population inference; the 50 include settings-development cases.

Isolated root: `/data/yuxuanstorage/vlm_phase3c/scale50_v1`. Source and previous
backups are never overwritten/deleted. A conservative capture/backup storage
gate reserves 12 GiB. Complete backup includes new compact captures, selected
videos and source-reference hashes; it does not duplicate previous large
activation stores, whose separately verified backups must be retained.

The detached job runs baseline -> patch -> CPU analysis/plots -> complete
backup verification -> email -> idle/exclusive-VM SURF Pause request, on success
or experimental failure. Backup or idle failures block Pause. A completion
email is technical status, not confirmation of local backup or billing stop.
The user's meeting email to Albert remains unsent/on hold.

```bash
python scripts/run_phase3c_scale.py --action prepare \
  --source_run /data/yuxuanstorage/vlm_phase3b/a10_runs/run_v4_vm_verified \
  --pilot_plan /data/yuxuanstorage/vlm_phase3c/pilot_v3 \
  --pilot_run /data/yuxuanstorage/vlm_phase3c/pilot_v3/execution_v1 \
  --replication_run /data/yuxuanstorage/vlm_phase3c/replication_v1 \
  --output_root /data/yuxuanstorage/vlm_phase3c/scale50_v1
```

Use the same paths with `--action launch --email_notify --pause_after
--confirm_exclusive_workspace` after tests and CPU preparation. The launch
requires a free detached session and exclusive workspace authorization.
