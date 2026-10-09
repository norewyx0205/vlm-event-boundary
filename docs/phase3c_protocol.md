# Phase 3C: Visual-State Sufficiency and Attention-Routing Pilot

Protocol date: 2026-10-09.

Status: the methodological requirements below are fixed for the follow-up
pilot. CPU archive auditing, processor-only mapping/control auditing and cohort
freeze tools are implemented. The implementation defaults below are explicit;
the actual case IDs and configuration are frozen only after processor
eligibility passes. The GPU baseline/technical-preflight entry point and reusable
residual/mask interventions, fixed primary grid and CPU aggregate analysis are
implemented but not GPU-validated. No new intervention results are claimed by
the implementation tests.
This is a pre-specified follow-up protocol, not a retrospective
preregistration of Phase 3B.

## Questions and Scope

1. Visual-state transplantation: does replacing a larger, explicitly aligned
   support at the appropriate intervention location transfer the observed
   temporal-boundary advantage?
2. Information routing: do specified attention edges make a functional
   contribution to that advantage?

A weak residual-state patch cannot answer the second question. Likewise,
attention knockout does not separately identify routing patterns and the
representational content read through them.

Use the existing low/temporal stimuli, a fixed prompt per base and the verified
Phase 3B evidence as inputs. Do not regenerate the 50-case experiment, modify
its artifacts or add audio/visual conditions. Attention-pattern transplant,
path patching, Q/K/V-specific interventions and head-level searches are outside
this initial pilot.

## 1. Pilot Cohort and Freeze

- Select 8 independent rescue bases and 4 independent stable-both-correct
  control bases. No base may occur twice or in both strata.
- Target 4 Target-1-first and 4 Target-2-first rescues, and 2 of each mover order
  among stable controls. Record an explicit amendment if eligibility prevents
  this balance; do not silently change the quotas.
- Rescue eligibility requires low incorrect with negative correct-option
  margin and temporal correct with positive margin. Stable controls require
  positive margins and correct predictions in both conditions. Exclude
  non-finite/zero-margin baselines and non-A/B first-token predictions.
- Verify behavior for the selected prompt pair, not just the base category.
  Preserve original/swapped labels and correct-option identity; do not force an
  A/B quota or add a second prompt as another independent case.
- Fix a deterministic selection rule using baseline behavior, mover order and
  processor-only eligibility. Neither old patch effect magnitude nor any new
  intervention result may rank cases.
- Freeze the final case manifest and selection summary after processor-only
  eligibility checks and before new patched/knockout outcomes are produced.
  Save candidate exclusions and the selection/configuration hashes.
- Revalidate baselines on the execution environment. A failure stops the pilot
  pending a logged eligibility amendment; it must not relax the gate or silently
  remove the case. No replacement may depend on a new intervention effect.

## 2. Intervention Audit

For every case, condition, direction and planned support, record:

- Processor input IDs, sampled source-frame groups, grid dimensions, visual
  positions, timestamps, event timing and relative event progress.
- Literal target identities and dynamic first/second-mover roles. Never infer
  Event 1 from Target 1.
- The donor and recipient position lists, mapped pairs, support sizes,
  unmatched positions and coverage on both sides.
- The correspondence rule, one-to-one status and event-progress error.
  Equal tensor shapes do not establish token correspondence.
- The actual intervention location, capture location and DeepStack injection
  layers, including whether the captured state is before or after addition.

Use actual donor/recipient labels for each patch direction. Do not reuse a
low-as-source mapping label as though it denotes the donor in temporal-to-low.

The case-220 example has 15 versus 16 ROI visual-token positions, not different
numbers of temporal bins. Temporal-to-low replaces 15/15 low recipient positions;
low-to-temporal replaces 15/16 temporal recipient positions. Completeness must
therefore be evaluated separately by direction.

Primary mapping requires an explicit, deterministic one-to-one correspondence
over the declared support. Report residual event-progress differences even when
every position is mapped. Event-relative alignment is not identical absolute
position alignment. Resolve the numerical eligibility thresholds before freeze.

## 3. Visual-State Transplantation

Compare the following in both temporal-to-low and low-to-temporal directions:

1. The existing single-ROI intervention as a reference.
2. An expanded, explicitly matched both-target support.
3. The entire matched Event-2 visual grid, including background/context cells.

Primary replacements use observed donor activations only. Require full recipient
coverage for a support labelled complete. Do not describe a larger intersection
with remaining unmatched positions as complete replacement.

Nearest-neighbour assignment, interpolation, duplicated donor vectors and
null/zero filling are separate robustness interventions, not equivalent to
one-to-one donor transplantation. Keep their methods and analyses separate from
the primary results. Do not insert dummy tokens into the input sequence: keep
the processor inputs, sequence length, recipient M-RoPE and causal mask intact.

Capture any newly required positions in a separate Phase 3C activation store.
Phase 3B saved the planned position union, not every visual activation. Reuse
old tensors only when support, capture semantics and provenance match exactly.

### DeepStack Timing Control

For the pinned 8B checkpoint, vision-layer indices `[8, 16, 24]` yield three
DeepStack feature sets. Transformers 5.9.0 adds these after decoder blocks
0, 1 and 2. These vision indices are not decoder injection indices. This direct
post-hook reinjection affects layer 0 in the old four-layer grid, not layers
4, 8, 12 or 16.

- `pre_deepstack`: capture donor block output before addition; replace recipient
  at that same location, before its DeepStack addition.
- `post_deepstack`: capture donor after addition and before the next block;
  replace recipient at that corresponding post-addition location.

Do not put a donor pre-addition vector into the post-addition location. Compare
location-matched states and record both capture semantics. Test early injection
layers separately; fix the remaining early/middle patch grid before GPU execution.
Report recipient position encoding and unmodified paths as interpretation limits.

## 4. Attention Knockout

The first pilot uses all-head, layer/window-level knockout. Fix numerical
windows before results; do not import the paper's nine-layer window by default
or choose the best window after inspecting this pilot without labelling that
choice exploratory.

Primary route families prohibit `options_all` or `query_all` positions from
attending to visual Target 1, Target 2 or both-target key positions. Keep options
and the generic query separate: target descriptions and temporal relations occur
in the options in the current prompt. Fix the visual event scope before execution.

Attention matrices are indexed `[query_position, key_position]`. Thus blocking
text-query rows and visual-key columns tests visual-to-text information transfer.
Earlier video queries cannot attend to later question keys under the existing
causal mask. Video-ROI targets and textual target mentions are distinct groups.

Apply the same route definition and layer/window to low and temporal inputs,
using each condition's independently audited groups. Modify the actual
pre-softmax attention mask, not an exported attention array. Preserve the
original causal mask, sequence and positional encoding. Log the convention that
remaining attention is renormalized after the blocked edges are removed.

### Matched Controls and Edge Budgets

Use distractor/background key controls matched as closely as possible to the
target route. Record per condition and per head:

- Query and key counts, event window and temporal location.
- The number of distinct selected edges that were causally visible before
  knockout; already-masked edges do not count as an intervention.
- The desired and achieved control budget and any matching discrepancy.
- Baseline attention mass on the blocked edges, as a diagnostic rather than a
  substitute for matched edge counts or an outcome-dependent selection rule.

Pre-specify control construction and numeric matching tolerances. Insufficient
matching must be explicit and fail the primary eligibility rule when outside
tolerance. Do not silently substitute a larger or temporally different control.

Verify disabled-knockout parity, correct edge orientation, empty-group rejection,
effective masking and finite logits. Do not leave a query with all its permitted
keys blocked. Capture/index/hook controls must also pass for each new patch
location, including same-state identity replacement.

## 5. Outcomes and Analysis

Margin remains the first-answer-token logit of the correct option minus that of
the incorrect option. For each matched case and knockout setting, retain:

```text
delta_M_temporal = M_temporal_KO - M_temporal
delta_M_low      = M_low_KO - M_low
advantage_base   = M_temporal - M_low
advantage_KO     = M_temporal_KO - M_low_KO
compression     = advantage_base - advantage_KO
                = delta_M_low - delta_M_temporal
```

Report all components, not compression alone. Positive compression can result
from temporal impairment, low improvement or unequal same-direction changes.
Only a decomposition dominated by temporal impairment supports a selective
loss of temporal benefit; compression alone does not establish this pattern.

Retain both conditions' raw baseline/intervened logits and margins. For patching,
also retain direction, margin change and the denominator of any reported
Recovery. Keep categorical flips, strict sign crossings and zero-margin ties
separate as secondary outcomes. Do not classify a greedy tie as strict rescue.

Use base case as the independent unit. Analyze rescue and stable-control strata
separately; layers, heads and repeated settings are not independent samples.
Case-bootstrap intervals are exploratory pilot diagnostics, not population
inference or simultaneous evidence after searching a grid. A non-significant
effect is not evidence of equivalence; any practical-smallness threshold must
be set before results and evaluated with its uncertainty.

## 6. Interpretation Boundary

The strongest proposed conclusion, if supported by the matched decomposition
and controls, is:

> The tested attention routes make a functional contribution to the
> temporal-boundary advantage, whereas the current visual-state transplantation
> does not fully transfer that advantage.

Do not conclude from weak patch/strong knockout alone that routing is the only
or primary mechanism. Knockout changes access to content and attention
competition. Broader state replacement is not an isolated sufficiency proof;
other recipient pathways remain intact. Results from image/LLaVA information-flow
studies motivate the hypothesis but do not fix Qwen3-VL video layer ranges.

## 7. Artifact Isolation and Execution Gates

Use a new `phase3c/<run_id>/` artifact root and Phase 3C schema, with separate
configuration, frozen selection, audit, captures, patch results, knockout
results, analysis and report. Phase 3B is read-only evidence. Do not overwrite
its manifests, fingerprints, checkpoints or reports, including failed runs.

The final configuration must contain the cohort IDs, model/revision, processor
and package versions, hardware/model placement, precision, seed, ROI/event
rules, mapping/control tolerances, intervention grids, hook semantics and code
hashes. The current reference is Qwen3-VL-8B revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, Transformers 5.9.0, FP16,
eager attention and seed 42; verify the actual runtime rather than assuming it.

Execution order:

1. Processor-only coverage audit and deterministic cohort selection.
2. Freeze case IDs, numerical configuration and hashes.
3. Execution-environment baseline/parity gate and technical hook/mask preflight.
4. Matched visual capture, scope replacement and DeepStack timing controls.
5. Matched low/temporal knockout with audited edge-budget controls.
6. CPU merge, decomposed analysis, interpretation audit and report.
7. Checksummed VM backup, local download and local verification.

Checkpoint each completed intervention with a unique case/condition/method/
support/location key. Resume only compatible completed keys, preserve failures
and never duplicate rows or mark an incomplete attempt successful. Print elapsed
time, completed/remaining work and checkpoint paths; base ETA on measured work.
Persistent VM artifacts belong under `/data/yuxuanstorage`. Detached launch,
email and optional SURF Pause require explicit configuration and must not be
activated by this protocol document. Automatic packaging does not confirm a
local backup.

## CPU Preparation Implementation

`scripts/prepare_phase3c.py --stage audit` validates the explicitly real source
configuration, manifest/mapping hashes, per-shard capture fingerprints, archived
video-hash agreement and strict paired baseline/parity evidence. It reads no
patch outcomes, loads no tensor/model packages and writes only a new artifact
root. Its balanced case preview is not a frozen selection.

`scripts/audit_phase3c_mappings.py` uses only the pinned processor and model
configuration, with GPUs hidden and no model weights loaded. It rechecks video
bytes, actual token IDs/visual positions and sampled-frame indices, then creates
expanded donor mappings and knockout control budgets. It checkpoints each pair,
prints elapsed time/progress and stops once the fixed mover quotas can be met
from an audited prefix. Resume reuses compatible records and checks their video
hashes. Failed audits require an explicit `--retry_failed` to retry.

`scripts/prepare_phase3c.py --stage freeze` independently reconstructs the
support/control audits, rejects unaudited earlier-ranked candidates, and freezes
24 condition rows from 12 independent bases. It will not overwrite a different
selection. Even a successful freeze remains `gpu_ready=false` until a separate
execution-environment baseline and technical preflight pass.

Initial implementation defaults, fixed before any new intervention outcomes:

- Same pinned model/input settings as the verified Phase 3B VM run.
- Strict phase dominance `>0.5` and full primary recipient coverage `1.0` in
  both directions; maximum paired event-progress difference `0.10`.
- Equal Event-2 bin counts and spatial grid dimensions, with chronological,
  one-to-one relative-event bin alignment and identical spatial cells.
- Expanded both-target support is the cross-condition ROI union in those
  observed grid cells. It can include background on one side; record this
  semantic difference. It uses real donor vectors, not a fill/interpolation.
- Reference residual-post patch layers `0,4,8,12,16,20`; separate location-matched
  pre/post DeepStack controls at decoder layers `0,1,2`.
- Knockout at non-overlapping four-layer windows `0-3,4-7,...,32-35`, all 32 heads,
  Event-2 visual keys, and separate options/query rows.
- Required background controls match key counts per temporal bin and visible
  causal edges exactly. Background requires all target/distractor overlap weights
  below the inherited `0.10` threshold; tied object cells are not background.
  Distractor controls are optional diagnostics only when the same budget is
  available, with unavailable routes explicitly reported.
- No robustness fill methods or practical-equivalence threshold are activated.

These are a small-pilot configuration, not evidence that its layer windows or
alignment tolerance are optimal. Any amendment must use a new fingerprint/root
and remain independent of intervention outcomes.

Example CPU-only preparation on the VM, from the repository root:

```bash
python scripts/prepare_phase3c.py --stage audit \
  --source_run_root /data/yuxuanstorage/vlm_phase3b/a10_runs/run_v4_vm_verified \
  --output_dir /data/yuxuanstorage/vlm_phase3c/pilot_v3

python scripts/audit_phase3c_mappings.py \
  --plan_dir /data/yuxuanstorage/vlm_phase3c/pilot_v3 \
  --project_root /data/yuxuanstorage/vlm-event-boundary

python scripts/prepare_phase3c.py --stage freeze \
  --output_dir /data/yuxuanstorage/vlm_phase3c/pilot_v3
```

The source videos must be present; source archive hashes do not replace the
processor audit. An optional `--path_map` JSON file remaps relocated video roots
without changing the original manifests. `--max_pairs` limits newly processed
pairs, leaving the remaining audit resumable. These commands do not launch
patching, knockout, notification or automatic Pause.

## Baseline and Technical Preflight Implementation

After the real CPU audit and freeze, explicitly invoke
`scripts/run_phase3c_preflight.py`. It supports only `baseline` and `preflight`;
there is no `full` stage. These commands do load the pinned FP16 model on CUDA.
The default uses one model split across both A10 GPUs, with the same 10-GiB
per-GPU weight-placement budget as the verified Phase 3B execution. No
quantization or CPU/disk offload is permitted. Single-GPU mode must be explicit.
The execution root must be below `--storage_root` (default
`/data/yuxuanstorage`); Hugging Face and Torch caches are also redirected to that
volume, avoiding another model download onto the ephemeral root disk.

The `baseline` stage audits all 24 low/temporal rows, rechecks the exact frozen
processor inputs/video bytes and records hashes of every input tensor. It
compares capture-hook logits against an unmodified forward exactly, and against
standard greedy first-token generation using the inherited `rtol=0.001`,
`atol=0.25` and exact first-token match. The strict rescue/stable margin signs and
archived predictions must still hold on the execution hardware.

Each successful capture saves the planned position union at all 36 block-output
sites plus three distinct post-DeepStack sites. Group indices and mean-norm
trajectories reference that union; overlapping groups do not duplicate vectors.
Capture files are checksummed and each attempt has a separate directory,
including failed attempts. A failed baseline cannot silently change the cohort.

The frozen selection includes two technical preflight IDs: the first eligible
rescue under each mover order, chosen before new intervention effects. The fixed
technical grid has 152 forwards over those two pairs and both conditions:

- 44 same-state identity replacements: block outputs at layers
  `0,1,2,4,8,12,16,20`, plus post-DeepStack at `0,1,2`.
- 40 cross-condition transplant smoke tests: each ROI, both-target and whole
  Event-2 support at layer 12, and whole support at both matching pre/post
  DeepStack locations at layers `0,1,2`.
- 4 disabled-knockout no-ops over all 36 layers.
- 64 active-knockout smoke tests: separate query/options and T1/T2/both routes,
  each paired with its exact background edge-budget control in window `0-3`;
  options/both-target and background also checked at `12-15` and `32-35`.

Post-DeepStack capture/replacement occurs at the actual addition function's
return, not by putting a pre-addition donor vector at the next block input.
The residual controller checks all 36 block calls and injections `0,1,2`.
Knockout modifies the actual self-attention additive mask before softmax, verifies
actual causal visibility against the audited budget, and measures the selected
edges' probabilities after softmax. Blocked probabilities must be exactly zero;
the remaining attention row sums must be within `0.005` of one. Identity and
disabled-mask logits must be exactly unchanged. These are technical gates, not
tests that require a desired causal effect.

Example explicit GPU stages, only after completing CPU eligibility and freeze:

```bash
python -u scripts/run_phase3c_preflight.py --stage baseline \
  --plan_dir /data/yuxuanstorage/vlm_phase3c/pilot_v3 \
  --output_dir /data/yuxuanstorage/vlm_phase3c/pilot_v3/execution_v1 \
  --gpus 0,1 --execution_mode model_parallel --gpu_weight_budget_gib 10

python -u scripts/run_phase3c_preflight.py --stage preflight \
  --plan_dir /data/yuxuanstorage/vlm_phase3c/pilot_v3 \
  --output_dir /data/yuxuanstorage/vlm_phase3c/pilot_v3/execution_v1 \
  --gpus 0,1 --execution_mode model_parallel --gpu_weight_budget_gib 10
```

Each stage writes its frozen task manifest, `rows.jsonl`, `summary.json`,
`progress_status.json` and preserved `errors.json` beneath the execution root.
Resume skips compatible passed tasks. A stored failure stops further work until
an explicit `--retry_failed`; retries never discard failure history. `--max_tasks`
limits new work, checkpoints it and exits nonzero when the gate is incomplete.
Configuration/code/input/capture hashes and actual hardware/model placement bind
the stages; mere presence of a summary file is not sufficient to pass.

`preflight/boundary_smoke_diagnostics.json` retains both margin changes and
advantage compression for matched completed knockout smoke tests. Every smoke
result is marked `is_primary_effect_estimate=false`; do not choose primary
windows/cases from it or treat the technical grid as the main pilot.

These commands do not send email or request SURF Pause. They have not been
launched on the VM as part of implementation. Preserve earlier preparation
roots: the updated protocol/preparation hashes require a fresh plan root, hence
`pilot_v3` above. The Phase 3B source remains read-only. A passed technical stage
does not mark the primary Phase 3C experiment complete.

## Fixed Primary Grid and Analysis Implementation

`scripts/run_phase3c.py` has three explicitly invoked stages: `patch`, `routing`
and `knockout`. All require the complete 24-row GPU baseline and 152-task
technical-preflight gates, independently rechecked before loading weights.
`knockout` additionally requires the intact `routing` diagnostics. Patch and
routing may be run independently; neither inspects effects to choose cases,
layers or windows. No stage automatically launches another stage.

- `patch`: 696 forwards. For each of 12 cases and both recipient conditions,
  replace each single-ROI reference, expanded both-target support and whole
  Event-2 grid at layers `0,4,8,12,16,20`. Whole-grid timing controls also cover
  pre/post-DeepStack at layers `0,1,2`. The layer-0 pre-addition whole-grid task
  serves both comparisons and is computed once, not duplicated. This is 29
  unique interventions per condition. Only observed, location-matched donor
  vectors are used; reference coverage and expanded full coverage remain
  distinct in results.
- `routing`: 24 intact forwards, one per case/condition. Observe all 36 layers
  for each of the 12 query/key/control routes without changing the mask or
  logits. Save selected-edge attention mass averaged over query rows and heads
  per layer. Exact logits parity with the GPU baseline is mandatory. This is a
  descriptive diagnostic, never a primary intervention effect estimate.
- `knockout`: 2,592 forwards: 12 cases x 2 conditions x 2 text-query groups x
  3 target-key groups x 9 four-layer windows x 2 target/background settings.
  Every background control retains its exact frozen per-bin key and visible
  causal-edge budget. Optional distractor availability is recorded by the CPU
  audit but is not added to this primary grid. No missing control shrinks the
  required grid silently.

Each completed primary task is atomically saved under
`<stage>/task_checkpoints/<task_id>.json`. Resume validates its fingerprint,
specification, input tensor hashes and actual frozen donor/recipient positions.
These files are authoritative even after interruption before `rows.jsonl`
consolidation. A completed stage rebuilds and verifies consolidated rows and
summary checksums; CPU analysis requires exact agreement with the per-task
files. Stored failures block additional work unless `--retry_failed` is explicit;
the error history is retained. `--max_tasks` limits new tasks, not the cohort,
and an incomplete stage exits nonzero. A fully completed requested stage is
verified and reused without loading model weights again.

Example commands, only after the baseline and technical preflight above pass:

```bash
python -u scripts/run_phase3c.py --stage patch \
  --plan_dir /data/yuxuanstorage/vlm_phase3c/pilot_v3 \
  --output_dir /data/yuxuanstorage/vlm_phase3c/pilot_v3/execution_v1 \
  --gpus 0,1 --execution_mode model_parallel --gpu_weight_budget_gib 10

python -u scripts/run_phase3c.py --stage routing \
  --plan_dir /data/yuxuanstorage/vlm_phase3c/pilot_v3 \
  --output_dir /data/yuxuanstorage/vlm_phase3c/pilot_v3/execution_v1 \
  --gpus 0,1 --execution_mode model_parallel --gpu_weight_budget_gib 10

python -u scripts/run_phase3c.py --stage knockout \
  --plan_dir /data/yuxuanstorage/vlm_phase3c/pilot_v3 \
  --output_dir /data/yuxuanstorage/vlm_phase3c/pilot_v3/execution_v1 \
  --gpus 0,1 --execution_mode model_parallel --gpu_weight_budget_gib 10

python scripts/analyze_phase3c.py \
  --plan_dir /data/yuxuanstorage/vlm_phase3c/pilot_v3 \
  --execution_dir /data/yuxuanstorage/vlm_phase3c/pilot_v3/execution_v1
```

The analyzer loads no model weights or activation tensors. It validates all
source/freeze/execution gates, capture bytes and primary task manifests, then
writes case-level and aggregate JSON/CSV tables, a report and static figures.
Only a complete verified grid can yield `analysis/aggregate_summary.json` with
`complete=true`. There is no partial-result success mode. Do not treat an older
analysis file as completion of a newer failed attempt; use its input hashes and
the current stage gates.

Results separate rescue and stable-control strata. Bootstrap intervals use
2,000 deterministic base-case resamples per setting, seed 42; repeated layers,
directions and prompt conditions are not independent replicates. Target-minus-
background contrasts are computed within each base before aggregation. Tables
retain temporal and low margin changes, boundary compression, donor-aligned
patch effects, Recovery denominators, coverage, categorical flips, strict sign
crossings, zero-margin ties and non-A/B first tokens. No effect threshold or
equivalence claim is added.

Figures include scope-effect curves, location-matched DeepStack timing curves,
KO decomposition heatmaps and descriptive all-layer activation-norm curves for
the two representative rescue IDs already frozen for technical preflight. These
representatives are not reselected using intervention results. The analysis
configuration and output checksums are separate from the GPU execution binding;
amended analysis settings require a separate analysis directory.

These primary entry points still do not send email, detach tasks or request
SURF Pause. Use a persistent terminal session for explicit execution and back up
the separate Phase 3C artifacts afterward. No VM/GPU tasks have been launched
as part of implementation. Completing code/tests does not mean the real CPU
mapping audit, freeze or GPU stages have completed.

## Sources

- [Zhang et al., Cross-modal Information Flow in Multimodal Large Language Models,
  CVPR 2025](https://arxiv.org/html/2411.18620v2).
- [Transformers 5.9.0 Qwen3-VL implementation](https://raw.githubusercontent.com/huggingface/transformers/v5.9.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py).
- [Pinned Qwen3-VL-8B configuration](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/config.json).
