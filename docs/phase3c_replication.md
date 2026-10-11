# Phase 3C Independent-Case Replication and ROI Diagnostics

Approved 2026-10-12 after the fixed-budget pilot. This is a pre-specified
out-of-pilot-stimulus replication within the same behavioral pool, not formal
preregistration, external-dataset replication or population inference.

## Independent Cohort

Use the completed, baseline-verified Phase 3B archive. Exclude **base IDs**, not
just prompt IDs, from every previous Phase 3C run (the same original 12 bases).
Use one prompt pair per base. Select ascending unused base IDs within quotas:
4 rescue T1-first, 4 rescue T2-first, 2 stable T1-first and 2 stable T2-first.
No prior/new patch effect, residual divergence or diagnostic energy is used to
rank the replication cases.

Order: archive/exclusion audit -> CPU processor/mapping and budget eligibility
-> live GPU baseline eligibility -> freeze cohort -> new all-layer capture
-> exact replay/identity gates -> interventions -> analysis.
CPU-ineligible cases and live behavior/zero-margin exclusions remain in ledgers.
Technical parity/input/hook failures stop the run; they are not case-selection
exclusions. After freeze, do not drop or replace failed cases.

## Fixed Intervention Grid

Decoder block outputs L12/L16, both directions, all 12 new bases:

- Original single-ROI references T1/Event2 and T2/Event2, with actual directional
  coverage recorded; these retain the original observed-donor correspondence.
- Complete target union, complete matched Event-2 grid and non-target complement.
- Fixed budgets 64/256, context-only versus target-union-plus-context, seeds
  42/43/44; exact per-bin quotas and nested observed-donor supports.

Total: 240 anchor patches + 576 budget patches = 816 new forwards.
Three seeds are averaged within case, never counted as independent cases.
Primary contrast: rescue temporal-to-low L16/budget256 mixed minus context.
Secondary: context budget256 minus64, and whole-event minus target union at L16.
Single-ROI summaries also use dynamic first-mover/second-mover Event-2 roles:
the first mover is inactive during Event2, while the second mover performs its
own event. Never derive event identity from the literal T1/T2 annotation ID.
Retain each reference's actual directional coverage and replaced position count.
L12 and reverse-direction results are fixed comparison analyses. Categorical
rescues remain separate from continuous margin effects.

All 36 decoder block states and 3 post-DeepStack sites are captured for the
needed position union. Verify exact full logits and input bytes against live
selection baselines. Replays cover all 24 conditions; identity patches cover
all 17 supports x 2 layers x 2 conditions x 2 frozen representatives = 136.
No dummy, interpolation or duplicate-donor filling. Only source-observed vectors.
Runtime/model/placement fields are pinned; any visible-memory-capacity-only
difference is recorded separately and new source states have an exact replay gate.

## Why Small-ROI Patching May Be Weak

The pilot establishes a support-size/composition effect, not a unique mechanism.
Potential explanations still include distributed/contextual representations,
different donor-recipient delta magnitudes, ROI/cell localization limitations,
event-position/timestamp information and which states downstream tokens read.
ROI is a geometric support label, not an information-isolation boundary. The
pinned Qwen3-VL vision blocks already mix features through non-causal attention
within packed visual chunks; decoder residuals at L12/L16 have undergone further
contextual mixing. Thus outside-ROI positions can potentially encode target
information before transplantation. This architecture-based account is a
plausible hypothesis, not evidence that a particular temporal-order signal is
present or read at those positions:
https://raw.githubusercontent.com/huggingface/transformers/v5.9.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
DeepStack reinjection concerns layers0-2, not L12/L16. Incomplete ROI coverage
cannot explain fully-covered temporal-to-low reference patches on its own.
Mapping coverage is conditional on the declared ROI support: Phase 3B's groups
use `MIN_ROI_OVERLAP=0.10` and exclusive maximum-ROI assignment. Therefore even
100% mapped coverage need not cover every target-touching cell. CPU geometry
diagnostics report omitted positive-overlap cells and retained padded-target
overlap mass. Preserve the original definition for replication; do not silently
rename the complement as pure object-free background or change the ROI threshold.

`analyze_phase3c_footprint.py` reads the existing pilot's all-layer captures on
CPU and measures aligned donor-recipient squared-difference energy in the target
union versus its complement. The union and complement exactly partition the
observed whole-event grid. It records token share, energy share and per-token
energy across all36 layers, with cases as units. This is **descriptive transplant
magnitude**, not causal importance, attention or a direct measure of task
information. It is not used to select the new cohort. A large outside-ROI
energy share would make a support/magnitude account plausible but would not
identify the stored content or prove mediation.

## Attention Knockout Decision

**Disabled in this replication.** Do not repeat the completed target-only grid.
The prior KO tested options/query queries attending Event-2 ROI keys, whereas
the stronger patch includes many non-ROI positions. Weak ROI KO does not test
all broad visual-to-language reading routes and does not establish their absence.
Conversely, successful residual transplantation does not prove a routing change.

If the independent support pattern replicates, a later separately approved pilot
can test broader options-to-Event2 routes with matched key/query/visible-edge
budgets and fixed early/middle windows. Keep low/temporal interventions paired;
report delta temporal margin, delta low margin and advantage compression.
Whole-Event2 KO lacks a disjoint same-event full-size control, so it must not be
presented as an exactly matched ROI-vs-background comparison. Broader 128/256-key
controls are feasible, but do not isolate content from softmax competition.
The interpretation remains functional route contribution, not proof that routing
is the unique/primary mechanism. Zhang et al.'s image/LLaVA findings motivate
questions, not Qwen3-VL video-specific layer ranges:
https://arxiv.org/abs/2411.18620

## Execution and Storage

Persistent isolated root: `/data/yuxuanstorage/vlm_phase3c/replication_v1`.
Retain prior evidence untouched. New capture/backup needs roughly two copies
of the new all-layer tensors; require at least 24 GiB free before launch and
reserve 12 GiB against conservative maximum capture estimates. No auto deletion.
Screening persists baseline JSON only, not tensors for rejected/reserve cases.
Only the frozen24 conditions receive persistent new captures.

```bash
python scripts/run_phase3c_replication.py --action prepare \
  --source_run /data/yuxuanstorage/vlm_phase3b/a10_runs/run_v4_vm_verified \
  --exclude_plan /data/yuxuanstorage/vlm_phase3c/pilot_v3 \
  --reference_run /data/yuxuanstorage/vlm_phase3c/budget_followup_v1 \
  --output_root /data/yuxuanstorage/vlm_phase3c/replication_v1 --gpus 0,1
```

Use the same arguments with `--action launch`, adding
`--email_notify --pause_after --confirm_exclusive_workspace` with authorization.
The detached `phase3c_replication` job runs screen -> capture -> interventions;
all technical gates block primary forwards on failure. Checkpoints resume without
recomputing verified tasks; failures require diagnosis, not silent retry/drop.
The job verifies complete new-evidence and selected-video backup before email
and an idle-checked SURF Pause request. Success/failure packages are evidence of
available artifacts, not proof of experiment completion. Download every ZIP part,
manifest and SHA256SUMS and verify using `backup_phase3b.py --verify` locally.
Retain separate prior backups; email does not confirm local backup or billing stop.
