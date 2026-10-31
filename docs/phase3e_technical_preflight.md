# Phase 3E Two-Scene Technical Preflight

Authorized on 27 October 2026: implement and run a small technical preflight;
send a completion email and pause the exclusive-use VM afterwards. This does
not authorize the proposed 40-scene experiment.

Version note, 31 October: the following frozen-scope description records the
original v1 technical run. New preparations use v2 event-relative distractor
timing: untimed scene geometry, then `timed_distractors` for the current boundary.
The original v1 videos/captures are preserved and readable as legacy technical
data. The corrected 40-family paired-option protocol requires a new preflight
on corrected videos; the old checkpoints cannot be reused for it.

## Frozen Scope

Two fresh deterministic scene seeds, 2026102701 and 2026102702, each have both
target motion orders, low/temporal boundary conditions, both sentence subjects
and both literal A/B permutations: eight videos and 32 prompt cells. The
independent unit is the scene seed. Seeds are fixed before model outputs.

Existing level-5 geometry/rendering helpers supply two targets, two static and
two moving target-like distractors. Each video is 18 seconds, 512 x 512, 15 FPS;
target events last two seconds, with either zero or three seconds between them.
Identities, paths and distractor absolute schedules are fixed within a seed.
Only target event timing/order varies visually. The fixed distractor schedule
is an explicit counterfactual control, not a replay of the archived dataset.

This two-event task does not eliminate the Event-2-mover shortcut. The run tests
data generation and capture correctness, not abstract relation computation.

## Gates And Saved States

Before model loading, audit all decoded frames, semantic labels, complete
factorial balance, actual contextual token groups, video-tensor identity across
text variants and equal visual/Event-2 support counts. Do not infer counts from
the old 512-token grid or silently crop mismatched supports.

Save separate T1, T2 and whole Event-2 means, per-option object mentions and
relation words, subject/reference mentions, options, query and decision groups.
Fixed block outputs: L0, L16, L17, L18, L20. Also capture L16/L17/L18 block inputs,
L17/L18 projected attention contributions and actual post-attention residuals.
Keep raw vectors at the selected union of positions for the timing checks.
The L0 block-output hook is before the parent model's subsequent DeepStack
visual-feature addition; the L16-L20 sites have no such post-hook injection.
After verification, the CPU exporter preserves D_low, D_temporal, delta-D,
B_low, B_temporal and delta-B at each saved site, matching scene, event order,
subject and displayed option layout. These are vector contrasts, not assumed
pure temporal directions; two seeds do not support probe training.

Each of the 32 prompts gets an unhooked forward, read-only captured forward and
independent greedy one-token generation. Hooks must leave full-vocabulary logits
exactly unchanged. Generation must match the chosen token and the existing
frozen numerical parity tolerances. Check exact residual addition and adjacent
block input/output equality. A wrong behavioural answer does not fail a
technical gate or remove a scene.

No probes are fitted; no patch/KO is executed. The 96 forwards include technical
replays and are not 96 independent observations. Execution uses the previous
pinned model revision, fp16, eager attention and two-A10 model parallelism.
Check at least 14 GiB free, preserving 12 GiB as reserve. Compact captures and
full logits are checkpointed per prompt, sealed and independently reloaded.

## Execution

```sh
python scripts/run_phase3e_preflight.py --stage prepare --output_root NEW_ROOT
python scripts/run_phase3e_preflight.py --stage capture --output_root NEW_ROOT \
  --source_plan /data/yuxuanstorage/vlm_phase3d/run_v2/plan.json
python scripts/run_phase3e_preflight.py --stage verify --output_root NEW_ROOT
```

Use an isolated code snapshot and new output root, not frozen Phase 3B-3D
directories. Preserve source hashes, runtime, plan, processor records, all
captures, verification results and videos in a checksummed full backup. Download
and verify this small package before finishing. Notify using the existing VM
private SMTP configuration; after compute has exited and quiescence is checked,
request Pause and verify the final portal/API state. Never copy credentials
into outputs or the code snapshot.
