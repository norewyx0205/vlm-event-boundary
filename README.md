# VLM Event Boundary Ladder Experiment

This project evaluates video-text models on forced-choice event-order matching. Each video contains two target events performed by 2D geometric objects. The model receives two `before/after` statements and must choose the statement that matches the video.

Example prompt options:

```text
A: The orange circle moves before the blue square.
B: The orange circle moves after the blue square.
```

Each video is evaluated twice with mirrored prompts:

- `original`: correct sentence in option A
- `swapped`: correct sentence in option B

This counterbalances answer position and supports response-bias analysis.

## Project Structure

```text
vlm-event-boundary/
  data/
    ladder_v2/
      level_1_simple/
        videos/
        annotations.jsonl
      level_2_randomized/
        videos/
        annotations.jsonl
      level_3_non_target_static_distractors/
        videos/
        annotations.jsonl
      level_4_target_like_static_distractors/
        videos/
        annotations.jsonl
      level_5_target_like_moving_distractors/
        videos/
        annotations.jsonl
      level_6_hard_temporal_interference/
        videos/
        annotations.jsonl
    README.md
  scripts/
    common.py
    experiment_artifacts.py
    generate_ladder_dataset.py
    run_eval.py
    analyze_results.py
    activation_patching_core.py
    select_activation_patching_cases.py
    select_activation_patching_candidates.py
    run_activation_patching.py
    analyze_activation_patching.py
    make_mirrored_annotations.py
    check_ladder_dataset.py
  results/
  notebooks/
    colab_eval.ipynb
  README.md
```

Legacy root scripts are kept for backwards compatibility, but new experiments should use the `scripts/` pipeline. In particular, `scripts/run_eval.py` is the canonical evaluation implementation; the root `run_eval.py` is only a thin wrapper for older commands.

## Difficulty Ladder

| Level | Name | Description |
| --- | --- | --- |
| 1 | `level_1_simple` | Two target objects, no distractors, short fixed/simple videos. Sanity check. |
| 2 | `level_2_randomized` | Randomized target positions, motion directions, and which object moves first. No distractors. |
| 3 | `level_3_non_target_static_distractors` | Static distractors with colors/shapes distinct from the targets. |
| 4 | `level_4_target_like_static_distractors` | Static distractors share target colors/shapes. Tests target binding. |
| 5 | `level_5_target_like_moving_distractors` | Moving distractors share target colors/shapes and move near target events. |
| 6 | `level_6_hard_temporal_interference` | Target-like moving distractors near the boundary plus a later unrelated event. |

All levels include four boundary conditions:

- `low_boundary`
- `temporal_boundary`
- `visual_boundary`
- `audio_boundary`

## Generate Ladder Data

Default generation:

```bash
python scripts/generate_ladder_dataset.py \
  --dataset_version ladder_v2 \
  --samples_per_level 30 \
  --seed 42
```

Useful generation arguments:

```text
--samples_per_level
--level_count                Number of levels to generate, default 6
--levels                     Comma-separated specific levels to generate, e.g. 6 or 4,5,6
--fps
--level_durations            Comma-separated durations for generated levels, default 10,12,14,16,18,20
--event_duration_sec
--temporal_gap_sec
--visual_marker_sec
--audio_beep_duration_sec
--static_distractors
--moving_distractors
--disable_unrelated_later_motion
--seed
--output_root
```

Each `annotations.jsonl` is evaluation-level: one row per prompt, not one row per unique video. Every video has two rows with unique `eval_id`, for example:

```text
level_2_sample_001_low_boundary_original
level_2_sample_001_low_boundary_swapped
```

For a fixed `base_sample_id`, the two target objects keep the same color and shape across all difficulty levels and all four boundary conditions. Across levels, only the difficulty manipulation changes: motion path, target order, distractors, and temporal complexity.

After generation, verify the dataset controls:

```bash
python scripts/check_ladder_dataset.py --root data/ladder_v2
```

## Level 5 Feature Ablation

Generate the structurally paired Level 5 pilot:

```bash
python scripts/generate_l5_feature_ablation.py \
  --dataset_version l5_feature_ablation_v1 \
  --samples_per_variant 30 \
  --size_stress_samples_per_cell 10 \
  --output_root data/l5_feature_ablation_v1 \
  --seed 42
```

The main variants are `L5_full`, `L5_shape_only`, `L5_color_only`, and
`L5_size_only`. They share motion paths, event order, distractor timing, and
boundary timing; only the visual feature encoding changes. The main experiment
contains `30 x 4 x 2 x 4 = 960` prompt evaluations.

`L5_size_only` renders every object as a black circle. Prompts refer to the
targets as `the smallest circle` and `the largest circle`; every distractor
radius lies strictly between the two target radii. The annotation-level
experimental labels remain `small` and `large`, while
`target_*_reference_label` records the unambiguous prompt wording.

To add only the new size datasets without regenerating the existing three main
variants:

```bash
python scripts/generate_l5_feature_ablation.py \
  --dataset_version l5_feature_ablation_v1 \
  --variants size_only \
  --samples_per_variant 30 \
  --size_stress_samples_per_cell 10 \
  --output_root data/l5_feature_ablation_v1 \
  --seed 42
```

The separate `size_stress_pilot/` uses a 2x2 design:

| Scene | Absolute target size | Distractor count |
| --- | --- | ---: |
| `large_few` | radii 28 / 50 | 1 |
| `large_many` | radii 28 / 50 | 4 |
| `small_few` | radii 14 / 28 | 1 |
| `small_many` | radii 14 / 28 | 4 |

With 10 base samples per cell, four boundaries, and mirrored prompts, this pilot
contains `4 x 10 x 4 x 2 = 320` prompt evaluations.

The separate `size_clear_contrast_pilot/` repeats the same 2x2 design with
larger target-distractor size margins. This is intended to test whether the
previous size-only pattern survives when the smallest/largest contrast is
visually clear enough for the model's coarse visual-token resolution.

| Scene | Absolute target size | Distractor count |
| --- | --- | ---: |
| `clear_large_few` | radii 28 / 72 | 1 |
| `clear_large_many` | radii 28 / 72 | 4 |
| `clear_small_few` | radii 14 / 48 | 1 |
| `clear_small_many` | radii 14 / 48 | 4 |

Generate only the clear-contrast pilot:

```bash
python scripts/generate_l5_feature_ablation.py \
  --dataset_version l5_feature_ablation_v1 \
  --size_clear_contrast_only \
  --size_clear_contrast_samples_per_cell 10 \
  --output_root data/l5_feature_ablation_v1 \
  --seed 42
```

Validate the pilot:

```bash
python scripts/check_l5_feature_ablation.py \
  --root data/l5_feature_ablation_v1
```

Evaluate all variants:

```bash
python scripts/run_eval.py \
  --annotation_root data/l5_feature_ablation_v1 \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --dataset_name_prefix l5_feature_ablation_v1_main_ \
  --output_dir results
```

`--annotation_root` loads the model once and evaluates every immediate child `annotations.jsonl`. This is substantially faster than launching one process per level or variant.

Evaluate the independent size/crowding pilot:

```bash
python scripts/run_eval.py \
  --annotation_root data/l5_feature_ablation_v1/size_stress_pilot \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --dataset_name_prefix l5_feature_ablation_v1_size_stress_ \
  --output_dir results
```

Analyze the latest run for each variant:

```bash
python scripts/analyze_results.py \
  --input results \
  --dataset_name_prefix l5_feature_ablation_v1_main_ \
  --latest_per_dataset \
  --output_dir analysis/l5_feature_ablation_v1 \
  --plots
```

Analyze the 2x2 pilot:

```bash
python scripts/analyze_results.py \
  --input results \
  --dataset_name_prefix l5_feature_ablation_v1_size_stress_ \
  --latest_per_dataset \
  --output_dir analysis/l5_feature_ablation_v1_size_stress \
  --plots
```

Evaluate the clear-contrast pilot:

```bash
python scripts/run_eval.py \
  --annotation_root data/l5_feature_ablation_v1/size_clear_contrast_pilot \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --dataset_name_prefix l5_feature_ablation_v1_size_clear_contrast_ \
  --output_dir results
```

Analyze the clear-contrast pilot:

```bash
python scripts/analyze_results.py \
  --input results \
  --dataset_name_prefix l5_feature_ablation_v1_size_clear_contrast_ \
  --latest_per_dataset \
  --output_dir analysis/l5_feature_ablation_v1_size_clear_contrast \
  --plots
```

The size analysis reports prompt accuracy, strict both-correct pair accuracy,
boundary-condition effects, response-position sensitivity, and factorial
estimates for target size, distractor count, and their interaction.
Its matched boundary plots are `accuracy_by_size_scene_condition.png` and
`strict_pair_by_size_scene_condition.png`. The main feature-ablation plots use
the same `Full / Shape only / Color only / Size only` ordering for prompt
accuracy and strict pair accuracy.

## Diagnostic And Mechanism Probes

You can create diagnostic forced-choice prompts from an existing annotation file
without regenerating videos. These prompts separate object identity, motion
binding, and event-order tracking more cleanly than the final before/after task.
The `size_extreme_identity` diagnostic uses a video-grounded spatial relation
prompt, for example whether the largest circle starts left/right or above/below
the smallest circle. This avoids the earlier semantic shortcut where statements
such as "the smallest circle is smaller than every distractor" could be answered
from wording alone.

```bash
python scripts/make_diagnostic_annotations.py \
  --annotation_root data/l5_feature_ablation_v1/size_clear_contrast_pilot \
  --output_path data/diagnostics/l5_size_clear_contrast_diagnostics/annotations.jsonl

python scripts/run_eval.py \
  --annotation_path data/diagnostics/l5_size_clear_contrast_diagnostics/annotations.jsonl \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --dataset_name l5_size_clear_contrast_diagnostics \
  --output_dir results

python scripts/analyze_results.py \
  --input results \
  --dataset_name_prefix l5_size_clear_contrast_diagnostics \
  --latest_per_dataset \
  --output_dir analysis/l5_size_clear_contrast_diagnostics \
  --plots
```

`analyze_results.py` automatically writes diagnostic tables and plots when
`diagnostic_type` is present in raw results, including
`accuracy_by_diagnostic_type_condition.csv` and
`strict_pair_by_diagnostic_type_condition.png`.

For causal perturbation, create codec controls, masked-video variants, and
fixed-duration temporal interventions, then evaluate them with the same runner.
`original` points to the source video, while `reencode_control` passes unchanged
frames through exactly the same OpenCV encode and ffmpeg mux path as every
intervention. `--max_base_samples` samples complete base stimuli, so both
mirrored prompts and all selected boundary conditions remain paired.

```bash
python scripts/make_roi_perturbation_dataset.py \
  --annotation_path data/l5_feature_ablation_v1/size_clear_contrast_pilot/L5_size_only_clear_small_many/annotations.jsonl \
  --output_root data/perturbations/l5_clear_small_many \
  --max_base_samples 4 \
  --perturbations original,reencode_control,mask_target_1,mask_target_2,mask_distractors,mask_background_control,remove_visual_marker,gap_removed,gap_shortened,gap_shifted \
  --mask_mode dynamic \
  --mask_scope all_frames \
  --mask_padding 6 \
  --sham_clearance 4

python scripts/visualize_roi_perturbations.py \
  --annotation_path data/perturbations/l5_clear_small_many/annotations.jsonl \
  --condition visual_boundary \
  --output_path analysis/l5_clear_small_many_roi_qa.png

python scripts/run_eval.py \
  --annotation_path data/perturbations/l5_clear_small_many/annotations.jsonl \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --dataset_name l5_clear_small_many_perturbation \
  --output_dir results

python scripts/analyze_results.py \
  --input results \
  --dataset_name_prefix l5_clear_small_many_perturbation \
  --latest_per_dataset \
  --output_dir analysis/l5_clear_small_many_perturbation \
  --plots
```

The dynamic mask follows each object's annotated path instead of erasing its
whole trajectory. `all_frames` tests dependence on persistent object identity;
use `--mask_scope motion_window` as a narrower motion-evidence ablation.
On the current 512-pixel stimuli, a local padding sweep found that `0` pixels
left a codec halo, `3` removed the visible edge, and `6` provided a conservative
clean mask without approaching neighbouring objects; therefore `6` is the
recommended default and should still be checked in the generated QA sheet.
Visual-marker frames are protected from object masks, and
`remove_visual_marker` reconstructs the underlying scene only where the marker
appears. By default, `mask_background_control` matches the distractor mask's
per-frame union area and follows its centroid trajectory through pixels that do
not contain annotated objects. The stats record exact area-match rate and both
trajectory lengths. Candidate offsets are re-ranked with dense temporal sampling,
and generation fails if the realised path-length error exceeds
`--sham_max_path_relative_error` (default `0.10`); target-matched sham references
are also available. The default `--sham_clearance 4`, combined with the 6-pixel
mask padding, keeps sham assignments at least 10 pixels outside annotated object
masks. Source audio is preserved by default. The
dataset folder records `perturbation_stats.jsonl`, separating mask-area
assignments from the union of pixels that actually changed. For the re-encode
control it also records decoded-video MAE, MSE, PSNR, and changed-pixel rate.

`gap_removed`, `gap_shortened`, and `gap_shifted` apply only to temporal-boundary
videos with no motion inside the gap. They preserve total frame count: the gap
is removed, shortened to `--gap_shortened_sec` (default 1 second), or moved
before the first target event. Updated event and boundary timings are written to
the derived annotations, together with explicit remapped `start_frame` and
`end_frame` values for every moving distractor. Analysis automatically uses
`reencode_control` as the
primary perturbation baseline when present and writes separate codec-control
prompt and strict-pair tables for `original` versus `reencode_control`, plus
`codec_prediction_consistency.csv` for exact A/B/UNKNOWN agreement.
It also writes `accuracy_by_temporal_intervention.csv`,
`strict_pair_by_temporal_intervention.csv`, and a combined prompt/strict plot
for the fixed-duration gap ablation. `model_input_by_perturbation_condition.csv`
reports realised sampled frames, video grids, and visual-token counts so the
fixed-input-budget assumption can be checked after evaluation.

For small-sample attention inspection, first build a behavioral case manifest.
This avoids selecting whichever rows happen to occur first in the annotation
file and carries the archived main-evaluation prediction into the probe.

```bash
python scripts/select_attention_cases.py \
  --annotation_path data/l5_feature_ablation_v1/size_clear_contrast_pilot/L5_size_only_clear_small_many/annotations.jsonl \
  --main_results results/.../raw_results.jsonl \
  --perturbation_results results/.../raw_results.jsonl \
  --output_path analysis/attention/l5_clear_small_many_cases.jsonl \
  --max_video_pairs 4
```

The selector prioritizes matched behavioral cases such as distractor-mask
repairs, target-mask failures, perturbation negative controls, and contrasting
temporal/visual pair outcomes. A temporal/visual contrast is an atomic two-pair
bundle: selection includes both conditions for the same base stimulus or neither.
Then run the ROI probe. Eager attention is
required. The probe prefills every prompt token except the final token and uses
the final prompt token as a one-token query whose logits predict the first A/B
answer token.

```bash
python scripts/probe_attention_roi.py \
  --annotation_path analysis/attention/l5_clear_small_many_cases.jsonl \
  --output_path analysis/attention/l5_clear_small_many_attention.json \
  --visualization_dir analysis/attention/l5_clear_small_many_figures \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --model_revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --expected_transformers_version 5.9.0 \
  --seed 42 \
  --deterministic \
  --attn_implementation eager \
  --max_samples 8 \
  --roi_padding 8 \
  --roi_assignment overlap \
  --roi_padding_sensitivity 0,4,8,12 \
  --parity_atol 0.25 \
  --no-require_standard_logits_match \
  --minimum_standard_top10_overlap 0.8 \
  --minimum_standard_logits_cosine_similarity 0.999 \
  --visualization_layer -1 \
  --head_reduction mean \
  --empty_cache_each_sample
```

The split-cache first token must exactly match standard greedy `model.generate`;
rows carrying `archived_prediction` must also match the main evaluation. Top-10
overlap and full-vocabulary logit cosine similarity provide distribution-level
hard checks. Elementwise FP16 `allclose` is retained as a diagnostic because
prefix splitting can change CUDA accumulation order without changing the answer
or high-probability token ranking. Maximum and mean logit differences, top-10
overlap, cosine similarity, and top-1 margins are archived for audit. The probe
maps model-visible video tokens through
`video_grid_thw`, accounts for Qwen3-VL spatial merging, and uses the processor's
sampled source-frame indices.
When one temporal patch combines multiple sampled frames, ROI overlap and phase
membership are computed for each constituent frame and averaged. Patches that
cross an event boundary retain fractional phase weights and are displayed as
`Mixed phase`, rather than being assigned from a rounded mean frame alone.
For every inspected evaluation row it writes:

- a decision-position attention overlay on representative source frames
- a temporal attention profile with event and boundary phases
- a decoder-layer by ROI visual-normalised attention-mass heatmap
- a decoder-layer by area-normalised ROI enrichment heatmap explicitly labelled
  as not showing total attention
- a layerwise Target 1/Target 2 contrast plot for `log(T1/T2)` mass and enrichment
- an ROI-padding sensitivity plot
- JSON metadata with the sampled grid, decision query, first answer token parity, spatial ROI mass,
  temporal phase mass, and per-layer ROI profiles

Target columns are role-aware, for example `T1 - smallest / second mover /
non-subject`, rather than displaying only the internal annotation ID. ROI
enrichment is visual-normalised attention mass divided by effective token-area
share. It measures attention density relative to ROI size, not total attention.
Merged cells are assigned fractionally by rasterized overlap with each ROI;
legacy center-point assignment remains available through `--roi_assignment
center`. Attention remains a qualitative association: interpret it together
with matched perturbation effects, not as standalone causal evidence.

The probe retains the legacy metric keys and also writes explicit names:

- `all_token_attention_share`
- `visual_normalized_attention_mass`
- `effective_token_area_share`
- `mean_attention_per_effective_token`
- `area_normalized_enrichment`

For legacy archives that do not contain explicit
`all_token_attention_share`, the analyzer reconstructs it as
`visual_attention_fraction * normalized_visual_attention`. It does not relabel
the legacy raw `attention_mass` sum as a global share, because that sum is not
normalised when `head_reduction=max`.

Existing attention JSON files can be analysed without loading Qwen or using a
GPU:

```bash
python scripts/analyze_attention_roi.py \
  --input_path analysis/attention/l5_clear_small_many_attention.json \
  --output_dir analysis/attention/l5_clear_small_many_metrics
```

When discovering results inside a directory or ZIP archive, the analyzer
excludes `*_incompatible_*.json` checkpoints and `*_errors.json` manifests by
default. Pass an exact JSON path only when intentionally auditing a quarantined
technical pilot.

The analysis writes `layer_roi_metrics.csv`, `layer_target_contrasts.csv`,
`stage_target_contrasts.csv`, `paired_stage_target_contrasts.csv`,
`attention_archive_audit.csv`, and `summary.json`.
Contrasts are defined as:

```text
delta_mass       = log(T1 visual-normalised mass / T2 visual-normalised mass)
delta_enrichment = log(T1 area-normalised enrichment / T2 area-normalised enrichment)
delta_enrichment = delta_mass - log(T1 effective area / T2 effective area)
```

The fixed layer-stage partition used for subsequent Qwen3-VL analyses is early
`0-11`, middle `12-23`, and late `24-35`. This partition was specified before
expanding the attention analysis to Phase 1 stimuli and remains fixed for all
subsequent analyses. Stage rows remain grouped by evaluation/base stimulus; layers are
not treated as independent experimental samples. Multiple attention JSON files
or complete output ZIP archives may be passed to `--input_path` to audit
attention semantics, ROI assignment, padding, frame-group support, and metric
schema before comparing runs. For example:

```bash
python scripts/analyze_attention_roi.py \
  --input_path 0803_output.zip 0805_output.zip \
  --output_dir analysis/attention/archive_audit
```

### Matched feature attention calibration

Phase 1 compares the same four base stimuli across `full`, `color_only`,
`shape_only`, and `size_only`. Each selected base retains all four boundary
conditions and both mirrored prompts, producing `4 x 4 x 4 x 2 = 128` attention
rows. The selector validates that target trajectories, distractor trajectories,
event timing, and boundary timing match across feature variants. It also balances
which target moves first and maximises coverage of `both_correct`,
`position_sensitive`, and `both_wrong` archived behavioral pairs.

```bash
python scripts/select_feature_attention_cases.py \
  --annotation_root data/l5_feature_ablation_v1 \
  --main_results \
    results/.../l5_feature_ablation_v1_main_L5_full/.../raw_results.jsonl \
    results/.../l5_feature_ablation_v1_main_L5_color_only/.../raw_results.jsonl \
    results/.../l5_feature_ablation_v1_main_L5_shape_only/.../raw_results.jsonl \
    results/.../l5_feature_ablation_v1_main_L5_size_only/.../raw_results.jsonl \
  --output_path analysis/attention/l5_feature_calibration_cases.jsonl \
  --base_samples 4
```

The adjacent `*_summary.json` is the selection audit. It records all candidate
and selected base IDs, mover balance, pair-outcome coverage, the number of
cross-feature structural checks, and mirrored-prompt video checks. Each selected
base is explicitly recorded as 16 video pairs and 32 evaluation rows. Original
and swapped rows must reference identical video, target, distractor, event, and
boundary metadata. Run the 128-row probe without per-case plots; aggregate
figures are generated by the analysis script.

```bash
python scripts/probe_attention_roi.py \
  --annotation_path analysis/attention/l5_feature_calibration_cases.jsonl \
  --output_path analysis/attention/l5_feature_calibration_attention.json \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --model_revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --expected_transformers_version 5.9.0 \
  --seed 42 --deterministic \
  --attn_implementation eager \
  --max_samples 128 \
  --roi_padding 8 \
  --roi_assignment overlap \
  --roi_padding_sensitivity 0,4,8,12 \
  --parity_atol 0.25 \
  --no-require_standard_logits_match \
  --minimum_standard_top10_overlap 0.8 \
  --minimum_standard_logits_cosine_similarity 0.999 \
  --resume \
  --continue_on_error \
  --log_every 8 \
  --no-model_loading_progress \
  --no-verbose_failures \
  --head_reduction mean \
  --empty_cache_each_sample \
  --no-plots

python scripts/analyze_attention_roi.py \
  --input_path analysis/attention/l5_feature_calibration_attention.json \
  --output_dir analysis/attention/l5_feature_calibration_metrics
```

The probe summary records model-load time, per-row inference time, and total
wall time. Feature calibration analysis first averages original/swapped rows
within each video pair and then clusters summaries and deterministic bootstrap
intervals by `base_sample_id`. It writes feature-by-stage, feature-by-boundary,
feature-by-first-mover, feature-by-prompt, and feature-by-behavioral-outcome CSVs,
plus three aggregate mass-versus-enrichment PNGs. With only four base samples,
these intervals are calibration diagnostics rather than confirmatory inference.
The probe preserves the selector's archived pair-outcome and first-mover labels
in every attention result so the behavioral-outcome table remains auditable.
The explicit `0.25` FP16 tolerance is reported as an elementwise diagnostic;
first-token identity, archived A/B prediction, top-10 overlap, and logit cosine
similarity provide the hard parity checks. `--resume` reuses validated eval IDs
from the existing JSON, while each newly completed row is written atomically.
With `--continue_on_error`, a genuinely invalid row is isolated in
`*_errors.json` and the remaining expensive probes continue. Paired feature
summaries automatically exclude incomplete original/swapped pairs.
Console output is intentionally compact: `--log_every 8` reports periodic
progress, model-weight bars are hidden unless `--model_loading_progress` is
passed, and isolated failures print a one-line summary. Their complete
tracebacks are still preserved in `*_errors.json`; use `--verbose_failures`
only for interactive debugging. An incompatible resume checkpoint is retained
as `*_incompatible_<timestamp>.json`, with the detailed reason recorded in the
probe summary, before a clean run starts.

Across the behavioral experiments, `analyze_results.py` produces feature-level
accuracy, strict mirrored-pair accuracy, the accuracy-strict gap `d`,
position-sensitive pair rates, paired boundary/feature differences, and
swap-consistency diagnostics. Report plots include:

- prompt accuracy versus strict both-correct accuracy
- mirrored-pair outcome proportions
- feature-by-boundary prompt and strict accuracy
- visual-boundary effects by feature condition
- correct-option A/B response-position sensitivity

## Baseline And Synthetic References

The legacy generator is kept for two reference settings outside the ladder:

- `baseline_boundary_videos`: very simple sanity-check cases.
- `synthetic_boundary_videos`: harder pre-ladder synthetic cases with distractors and later unrelated motion.

Generate both:

```bash
python generate_2d_boundary_videos.py --dataset all
```

Generate only one:

```bash
python generate_2d_boundary_videos.py --dataset baseline
python generate_2d_boundary_videos.py --dataset hard
```

Evaluate them with the same Qwen runner:

```bash
python scripts/run_eval.py \
  --annotation_path baseline_boundary_videos/annotations.jsonl \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --dataset_name baseline_qwen3_sanity_check \
  --output_dir results

python scripts/run_eval.py \
  --annotation_path synthetic_boundary_videos/annotations.jsonl \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --dataset_name synthetic_qwen3_reference \
  --output_dir results
```

## Run Qwen Evaluation

Run one level:

```bash
python scripts/run_eval.py \
  --annotation_path data/ladder_v2/level_1_simple/annotations.jsonl \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --dataset_name ladder_v2_level_1_simple \
  --output_dir results
```

Run Qwen3-VL:

```bash
python scripts/run_eval.py \
  --annotation_path data/ladder_v2/level_5_target_like_moving_distractors/annotations.jsonl \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --dataset_name ladder_v2_level_5_target_like_moving_distractors \
  --output_dir results
```

Run the complete ladder with one model load:

```bash
python scripts/run_eval.py \
  --annotation_root data/ladder_v2 \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --dataset_name_prefix ladder_v2_ \
  --output_dir results \
  --seed 42 \
  --deterministic \
  --attn_implementation eager
```

The runner keeps CUDA caching enabled by default. `--empty_cache_each_sample` is available only for unusually tight GPU-memory situations because it generally reduces throughput.

### Reproducible evaluation

The evaluator already uses greedy decoding (`do_sample=False`, one beam). For repeatable
Qwen3-VL runs on the same GPU/runtime, also use:

```bash
PYTHONHASHSEED=42 python scripts/run_eval.py \
  --annotation_root data/ladder_v2 \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --model_revision <commit-hash> \
  --dataset_name_prefix ladder_v2_ \
  --output_dir results \
  --seed 42 \
  --deterministic \
  --attn_implementation eager
```

After the first run, copy `environment.model_commit_hash` from its `config.json` into
`--model_revision`. Each config also records the annotation SHA-256, package versions,
CUDA/cuDNN versions, GPU name, seed, and attention backend. Exact equality is expected
only when the model commit, annotations, package/runtime versions, hardware, and command
are unchanged. Different GPU types or CUDA stacks can still produce small floating-point
differences near a decision boundary.

`eager` attention is the conservative reproducibility setting. If throughput matters
more, use `--attn_implementation sdpa`; keep that choice fixed across compared runs.
If strict deterministic mode reports an unsupported operation, add
`--deterministic_warn_only` and record that relaxation.

### Model-visible video inputs

Source-video properties such as 512 x 512 resolution and 15 fps do not by
themselves determine what the model receives. Qwen3-VL video preprocessing can
sample by `fps` or by `num_frames`; these are mutually exclusive controls. The
evaluator therefore exposes both options, but rejects commands that set both:

```bash
python scripts/run_eval.py \
  --annotation_path data/ladder_v2/level_5_target_like_moving_distractors/annotations.jsonl \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --dataset_name ladder_v2_level_5_target_like_moving_distractors \
  --output_dir results \
  --video_num_frames 32
```

If neither `--video_fps` nor `--video_num_frames` is supplied, the run uses the
processor/qwen-vl-utils default sampling behavior. Do not report this as "all
15 fps source frames were passed to the model" unless the archived input
metadata verifies it.

Each raw result row now includes `input_metadata` with:

- `video_kwargs`, excluding verbose `video_metadata`
- stringified `video_metadata`
- decoded `video_inputs` shapes and frame counts from their first dimension
- `pixel_values_videos` shape after processor preprocessing
- `video_grid_thw`
- visual-token counts derived from `video_grid_thw`
- video-token count from `mm_token_type_ids`, when available
- `input_ids`, `attention_mask`, and `mm_token_type_ids` shapes

`config.json` also records the requested temporal sampler, pixel budget,
model-load settings, decoding settings, and output parser. After evaluating a
run with the new logger, produce a table-ready summary with:

```bash
python scripts/analyze_results.py \
  --input results/Qwen_Qwen3-VL-8B-Instruct/ladder_v2_level_5_target_like_moving_distractors/<timestamp>/raw_results.jsonl \
  --output_dir analysis/ladder_v2_level5_inputs \
  --plots
```

The analyzer writes `model_input_by_boundary.csv`, grouped by boundary
condition. Use this file for the thesis table reporting source duration,
sampled-frame count, `video_grid_thw`, and visual-token counts. Older raw
results created before this metadata was added cannot support that table
without rerunning evaluation or separately probing the processor.

Quick smoke test:

```bash
python scripts/run_eval.py \
  --annotation_path data/ladder_v2/level_1_simple/annotations.jsonl \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --dataset_name smoke_ladder_v2_level_1_simple \
  --output_dir results \
  --max_samples 4
```

Results are saved to:

```text
results/<safe_model_name>/<dataset_name>/<timestamp>/
  raw_results.jsonl
  summary.json
  config.json
```

## Analyze Results

Analyze a single run:

```bash
python scripts/analyze_results.py \
  --input results/Qwen_Qwen3-VL-8B-Instruct/ladder_v2_level_5_target_like_moving_distractors/<timestamp>/raw_results.jsonl \
  --output_dir analysis/ladder_v2_qwen3_level5 \
  --plots
```

Analyze a directory containing multiple run folders:

```bash
python scripts/analyze_results.py \
  --input results \
  --dataset_name_prefix ladder_v2_level_ \
  --output_dir analysis/ladder_v2_qwen3_all \
  --plots
```

The analyzer saves:

- `accuracy_by_difficulty.csv`
- `accuracy_by_difficulty_condition.csv`
- `strict_pair_overall.csv`
- `strict_pair_by_difficulty.csv`
- `strict_pair_by_condition.csv`
- `strict_pair_by_difficulty_condition.csv`
- `accuracy_by_correct_option.csv`
- `accuracy_by_prompt_variant.csv`
- `prediction_distribution.csv`
- `swap_consistency_summary.csv`
- `swap_consistency_details.csv`
- `swap_consistency_by_level_condition.csv`
- `paired_boundary_summary.csv`
- `paired_boundary_details.csv`
- `summary.json`
- optional `accuracy_by_difficulty_condition.png`
- optional `strict_pair_by_difficulty_condition.png`
- optional `accuracy_vs_strict_pair_by_difficulty.png`
- optional `accuracy_vs_strict_pair_by_boundary.png`

Prompt-level accuracy and strict both-correct pair accuracy are treated as
co-primary descriptive metrics. The strict metric counts a video as correct
only when both its original and swapped prompt rows are answered correctly,
which makes it substantially less sensitive to A/B response-position bias.

For analyses containing only one difficulty level, the difficulty-condition
plots automatically switch to boundary-condition bar charts instead of
collapsing all points onto one x coordinate. CSV outputs that do not apply to
the selected experiment scope are omitted rather than written as empty files.

`paired_boundary_summary.csv` compares each non-low boundary condition against `low_boundary` within the same `difficulty_level` and `base_sample_id`, reporting:

- `temporal_boundary_minus_low_boundary`
- `visual_boundary_minus_low_boundary`
- `audio_boundary_minus_low_boundary`

This is especially useful for Level 5, where aggregate accuracy can hide whether a boundary condition consistently helps or hurts the same stimuli.

Swap consistency categories:

- `both_correct`
- `both_wrong`
- `original_correct_swapped_wrong`
- `original_wrong_swapped_correct`

## Dependencies

Generation:

```bash
pip install opencv-python numpy imageio-ffmpeg
```

Qwen evaluation:

```bash
pip install torch "transformers==5.9.0" accelerate "qwen-vl-utils==0.0.14" "decord==0.6.0"
```

The attention probe checks the Transformers version before loading model
weights. In Colab, dependency verification uses a fresh Python subprocess, so a
reinstall can take effect without discarding `/content` checkpoints or archived
main-evaluation results from the current runtime.

For smaller GPUs, install `bitsandbytes` and add `--load_in_4bit --video_fps 1 --video_max_pixels 150000`.

Analysis uses the same `opencv-python` and `numpy` dependencies as generation.

## Colab

Use `notebooks/colab_eval.ipynb` for Colab. It contains cells for:

- cloning/pulling the repo
- installing dependencies
- generating baseline and synthetic reference datasets
- generating the ladder dataset
- running baseline, synthetic, and ladder evaluations
- running Qwen3-VL on each level
- analyzing saved results

### Artifact-first notebook modes

The notebook is an experiment orchestrator rather than an unconditional
`Run all` script. Its configuration cell resolves one mode for every independent
experiment:

- `skip`: do not run or consume the experiment.
- `reuse`: validate and use existing real artifacts without recomputation.
- `analyze`: reuse raw outputs and rebuild only CPU analysis and plots.
- `run`: execute the real generation/evaluation/probe stage.

The default `part1_reuse` profile never regenerates completed datasets or loads
Qwen for completed Part 1, ROI, or Phase 1 experiments. The older standalone
Phase 0 case study is skipped by default because its metric decomposition is
already incorporated into Phase 1; set `attention_phase0=reuse` when its own
archive has also been restored. Available
profiles are `part1_reuse`, `analysis_only`, `full_reproduction`, and `smoke`.
Override only the active experiment in the central configuration, for example:

```python
PIPELINE_PROFILE = "part1_reuse"
EXPERIMENT_MODE_OVERRIDES = {
    "attention_phase1": "analyze",
}
```

For a genuine full rerun, select `full_reproduction`. `run` remains explicit so
a missing artifact can never silently trigger an expensive model evaluation.

Completed runs should be restored from the timestamped ZIP produced by the final
notebook cell. Formal runs should list exact paths in `ARTIFACT_ARCHIVES`;
automatic latest-ZIP discovery remains available but is disabled by default.
New-schema manifests must contain `artifact_type`, schema version, source commit,
model name/revision, creation time, and configuration fingerprint, and configured
model provenance must match before safe extraction. Legacy archives without the
new schema remain usable for backward compatibility but are explicitly reported
as `legacy_unverified`, with every unavailable check recorded as a warning. They
must not be described as fully validated artifacts. Missing dependencies fail
before model loading with a path-specific message. Archives marked
`artifact_type=mock` are rejected from the research pipeline; mock data is
reserved for unit tests and smoke fixtures.

New archives record `artifact_schema_version=2`, `artifact_type=real`, the
pipeline profile, all experiment modes, restored archive provenance, and a
stable configuration fingerprint.
Notebook cell outputs are intentionally not versioned as evidence: raw results,
summary JSON/CSV files, figures, configurations, and provenance remain in the
timestamped artifact archive.

`activation_patching_phase3` is an independent experiment mode and defaults to
`skip`. Enabling it does not rerun Part 1 or Phase 1. The Phase 3 cell consumes
the archived `L5_full` behavioural evaluation, checkpoints each expensive GPU
stage, and can later rebuild its analysis without loading Qwen.

## Phase 3: Temporal-Boundary Activation Patching

The first Phase 3 experiment is a narrow matched-pair causal pilot. It asks where
the hidden states of low-boundary and temporal-boundary videos diverge, then tests
whether position-aligned high-divergence locations causally change the
first-answer-token decision. A small medium/low-divergence comparison set prevents
the exploratory divergence-effect analysis from being restricted to the top of
the observed range.
It does not run RSA/CKA or an exhaustive layer-by-group patch sweep.

### Design

- Primary temporal-rescue bases: `5,11,14,15,17,19` from `L5_full`.
- Stable both-correct controls: bases `1,2`.
- Both `original` and `swapped` prompts are retained, but behaviour is verified
  separately for each prompt pair. Original temporal-rescue pairs are the primary
  analysis; swapped pairs are mirrored controls unless they independently satisfy
  the rescue criterion. Base-level selection and prompt-pair behaviour are stored
  in separate fields.
- Hidden states are captured at decoder-layer residual-post for all 36 layers.
- Coarse groups cover all video tokens, target/distractor ROIs, event phases,
  object mentions, before/after terms, option spans, and the decision position.
  Option spans are located inside the rendered contextual `A:` / `B:` block,
  rather than by standalone sentence tokenization. Expected target, relation,
  and option text groups are fail-fast: a tokenizer/chat-template mismatch cannot
  silently produce an empty analysis group.
- Pairwise metrics are cosine distance and relative L2 change after explicit
  mean pooling. Token-wise diagnostics are computed only when the exact input
  sequence positions are identical; equal tensor shape alone is insufficient.
  Every row records whether identical positions or an event-relative map supplied
  the correspondence. This pilot does not implement event-relative mapping.
- `phase_gap` is explicitly excluded from pairwise divergence and patch selection
  in this first pilot: low-boundary has no semantically matched inter-event gap.
  A later analysis must define either an absolute-time control window or an
  event-relative mapping before testing this phase.
- Candidate selection ranks position-aligned locations within each matched pair
  by the mean of cosine- and relative-L2 descending percentile scores. The fixed
  six-location budget contains four high-divergence primary candidates, one
  medium-divergence comparison, and one low-divergence comparison, with at most
  one location per token group.
- Because captures are taken at decoder-layer residual-post, a non-decision token
  patched after the terminal decoder layer cannot influence an already-computed
  decision position. Such terminal-layer locations are structurally excluded;
  only `decision_position` remains eligible at the final layer.
- Patching is bidirectional: temporal-to-low recovery and low-to-temporal
  disruption. The primary causal analysis uses only `positionwise_replace` at
  identical sequence positions. `pooled_mean_delta`, if explicitly requested in
  a separate run, is exported as an exploratory group-level mean-shift
  intervention and is never pooled with standard activation-patching results.
- The primary behavioural quantity is the correct-minus-incorrect A/B first-token
  logit margin. Categorical answer flips remain secondary.

With the default 8 bases and 2 mirrored prompts, this yields 16 matched pairs,
7,488 auditable layer/group rows (including the explicitly excluded `phase_gap`
rows), 96 selected locations, and 192 bidirectional patch runs. Thus the
methodological cleanup does not increase the GPU patch budget. The first pair
is a fail-fast preflight. Before scaling to the remaining pairs, the patch stage
also checks a repeated no-patch forward and same-state identity patches.

Position-wise replacement is used only when the source and target groups contain
the same sequence positions. Each divergence and patch row records this alignment
assumption, whether event-relative mapping was used, and which intervention family
the result belongs to.
The low and temporal videos are still different inputs and Event 2 occurs at a
different absolute time, so same-position visual patches are a documented first
pass rather than a claim of perfect event-semantic alignment. The selected-case
manifest retains all event annotations for a later event-relative extension.
Accordingly, analysis figures and tables label `phase_event_2` divergence as
**non-position-aligned, descriptive only**; it is not presented as causal evidence.

The contextual text-span and terminal-layer eligibility cleanup advances the
Phase 3 runtime schema to `temporal_boundary_activation_patching_v3_contextual_text_spans`.
Older pilot checkpoints must remain archived and must not be resumed into this
final rerun.

### Standalone commands

Select the auditable case manifest from an archived/current `L5_full` run:

```bash
python scripts/select_activation_patching_cases.py \
  --annotation_path data/l5_feature_ablation_v1/L5_full/annotations.jsonl \
  --main_results results/<model>/l5_feature_ablation_v1_main_L5_full/<run>/raw_results.jsonl \
  --output_path analysis/activation_patching_phase3/selected_cases.jsonl \
  --expected_model_name Qwen/Qwen3-VL-8B-Instruct \
  --expected_model_revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
```

Map divergence, select candidates, and run the bidirectional intervention:

```bash
python scripts/run_activation_patching.py divergence \
  --manifest_path analysis/activation_patching_phase3/selected_cases.jsonl \
  --output_path analysis/activation_patching_phase3/pairwise_divergence.jsonl \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --model_revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --expected_transformers_version 5.9.0 --deterministic --resume

python scripts/select_activation_patching_candidates.py \
  --divergence_path analysis/activation_patching_phase3/pairwise_divergence.jsonl \
  --output_path analysis/activation_patching_phase3/patch_candidates.jsonl \
  --top_k_per_pair 6 --medium_k_per_pair 1 --low_k_per_pair 1 \
  --max_per_token_group 1 --required_patch_method positionwise_replace

python scripts/run_activation_patching.py patch \
  --manifest_path analysis/activation_patching_phase3/selected_cases.jsonl \
  --candidate_path analysis/activation_patching_phase3/patch_candidates.jsonl \
  --output_path analysis/activation_patching_phase3/patching_results.jsonl \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --model_revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --expected_transformers_version 5.9.0 --deterministic --resume

python scripts/analyze_activation_patching.py \
  --divergence_path analysis/activation_patching_phase3/pairwise_divergence.jsonl \
  --candidate_path analysis/activation_patching_phase3/patch_candidates.jsonl \
  --patching_path analysis/activation_patching_phase3/patching_results.jsonl \
  --output_dir analysis/activation_patching_phase3/analysis --plots
```

Every GPU checkpoint has a configuration fingerprint. A changed manifest,
candidate table, model revision, sampling configuration, ROI definition, or
Transformers runtime is rejected before model loading rather than mixed with an
older partial run. Isolated failures are written to an error manifest; with the
default `--require_complete`, the process exits after checkpointing so rerunning
the same command retries only unresolved work.

The patch stage also runs no-patch and same-state identity-patch controls on the
first matched pair. Analysis writes separate position-wise and pooled-mean-delta
CSVs, and stratifies standard patching by primary original rescue, independently
rescued swapped prompts, mirrored controls, and stable both-correct controls.
The high/medium/low candidate labels are retained in the divergence-effect plot.
Divergence is descriptive; only position-aligned patch effects enter the primary
causal-mechanistic summaries.

## Phase 3B: Scaled Rescue and Video-Token Patching

Phase 3B is a separate schema and does not replace the Phase 3A pilot. It screens
new `L5_full` videos using only `low_boundary` and `temporal_boundary` with both
mirrored prompts. The original pre-specified screen allowed 300 new base samples
and targeted 60 mapping-eligible rescues. After exhausting that cap with fewer
than 50 rescues, the current notebook explicitly amends the cap to 500 new bases
and stops at 50 **mapping-eligible independent rescue bases**. The amendment is
recorded in `rescue_pool/budget_amendments.json`; it is not presented as part of
the original pre-specified budget. Batches contain five bases by default, and
completed batches are reused after interruption.
The processor-only audit occurs before the 50-case primary cohort is frozen.
If fewer than 50 eligible cases are found at the cap, the cohort is not silently
filled with non-rescues. Original/A and swapped/B rescues are counted separately;
a 25A+25B extension is created only if enough B-rescues arise naturally.

The frozen analysis manifest includes 50 primary rescues, up to 5 opposite
prompt pairs, and up to 10 stable-both-correct controls by default. Opposite prompts that
independently rescue are labelled separately from mirrored controls; each pair
retains its own behavioral classification. A separate two-case preflight chooses
one `target_1`-first and one `target_2`-first primary case.
The selection summary also freezes one representative pair per first-mover
stratum (median base ID, chosen before activation patching); full-run plots use
these IDs rather than selecting cases from patch effects or observed margins.

Six visual groups represent literal target/distractor identity by Event 1/2.
The four first-/second-mover roles are derived per case from `first_object_id`,
never inferred from target ID. The text groups distinguish the generic query,
A/B option bodies, target mentions inside those options, before/after terms,
and the final answer-predicting prompt position. Visual tokens are assigned
exclusively to one ROI and mapped one-to-one by event-relative temporal progress
and merged spatial cell. Cases failing mapping coverage or alignment are
ineligible; there is no pooled-mean fallback. Distractor patches are labelled
aggregate-distractor interventions because individual distractor identity is
not aligned across the videos.

For every selected condition, all 36 residual-post decoder layers are captured
as per-token vectors, token indices, counts, and pooled group means. Patching
uses layers `0,4,...,32` plus layer 35 for `decision_position` only. Video
groups use explicit `event_relative_replace`; query, options, and decision use
`positionwise_replace`. Both temporal-to-low and low-to-temporal directions are
run. The primary outcome is the correct-minus-incorrect first-token A/B logit
margin, with categorical flips secondary. Same-state identity patches are
technical no-op controls; a separate preflight global temporal-relocation
control tests cross-position effects. Its shift is derived separately for each
matched pair from the low-to-temporal Event 2 onset displacement (normally 45
source frames), not a fixed one-second offset. This control moves the entire
low-boundary video in absolute time and focuses its patch summary on Event 2
visual groups; it does not isolate an Event 2-only video edit. The original
low video and its codec-matched re-encode are checked for prediction parity,
margin change, and decoded-frame PSNR before interpreting relocation effects.
These are decoder residual-stream
interventions at visual-token positions, **not** vision-encoder patches.
CPU analysis writes all-layer divergence and fixed-grid causal heatmaps, plus
representative Target-1-first and Target-2-first activation-norm trajectories.

### Colab workflow

In `notebooks/colab_eval.ipynb`, set only
`EXPERIMENT_MODE_OVERRIDES["activation_patching_phase3b"] = "run"`; keep old
experiments at `skip`/`reuse`. The frozen `0922_phase3_output.zip` archive is
loaded from the repository root. Include that file when committing/pushing to
GitHub so a fresh Colab clone can restore it; otherwise set
`ARTIFACT_ARCHIVES` to an existing Colab/Drive path. Advance
`PHASE3B_STAGE` explicitly:

1. `screen`: batched generation/evaluation, processor mapping audit, and frozen
   selection. Re-running with the amended budget resumes completed batches;
   it does not regenerate or re-evaluate the existing 300 new bases. The cell prints the current
   batch, elapsed time, evaluated-new-base count, and last audited eligible
   rescue count. A heartbeat updates
   `/content/drive/MyDrive/vlm_phase3b/analysis/screening_status.json` every
   60 seconds while a child step is running; the separate
   `screening_progress.json` records the latest *completed* mapping audit.
   Before reusing a Drive pool, screening checks the saved generation settings
   and generator-code hashes. A changed generator or missing config requires a
   new pool directory rather than silently mixing stimuli.
   After the 50-case screen, a CPU-only repeat of `screen` can repair a truncated
   cached mapping manifest without repeating model inference. The repaired
   selection is frozen under `selection_v2_controls`; the original `selection`
   directory remains untouched. The notebook checks that primary and preflight
   rescue manifests match the original selection before proceeding.
2. `preflight`: two first-mover-balanced cases, technical controls, patching,
   analysis, and a temporal-relocation control. If screening exhausted its
   budget below 50 eligible rescues, this stage freezes a separate
   `technical_preflight_selection` from the saved screening and mapping audit.
   Its checkpoints and analyses use `technical_*` directories; it does not
   create or substitute for the formal primary cohort. Running this stage
   does not re-run screening.
   The formal `selection_v2_controls` preflight uses separate
   `preflight_v2_controls` checkpoints, `preflight_v2_controls_analysis`, and
   `relocation_control_v2_controls` outputs so earlier preflight artifacts are
   never reused under a changed selection fingerprint. Capture/patch child logs
   are streamed to the notebook and retained under the checkpoint `logs/` folder.
3. `full`: five-case independent shards for the frozen primary and secondary
   analysis cohorts, followed by CPU analysis. Requires 50 formally frozen
   independent primary rescues and a preflight from that formal selection;
   technical preflight outputs cannot satisfy this gate. If screening ends
   below 50, review and amend the screening budget before attempting `full`.
   Capture and patch stages report pair/shard timing and the current checkpoint
   path; patching logs its completed-location count every three token groups.

The notebook mounts Google Drive for every active Phase 3B mode. Its default
`/content/drive/MyDrive/vlm_phase3b` root persists the generated video pool,
batch evaluation results, screening/mapping audits, and GPU activation
checkpoints across Colab sessions. The ordinary timestamped download ZIP
contains lightweight behavioral results, selection/mapping audits, analyses,
figures, and provenance, but deliberately excludes large `.pt` activation
tensors and `.mp4` videos. The ZIP alone cannot resume video patching; keep
the Drive video pool and activation directory as the underlying records.
Reuse the same
repository commit, model revision, manifests, and mapping file when resuming a
shard; the fingerprint also includes key source-file hashes, so uncommitted
code changes cannot silently reuse old captures. A mismatch is rejected.
The capture stage may restart an incompatible setup-only shard containing just
`run_config.json` and, optionally, `capture_errors.json`; it first moves the
entire failed shard into `incompatible_failed_shards/` and records that path in
the new configuration. Any activation, result, or unrecognized file retains
strict fingerprint protection and requires a separate output directory.
Archived processor parity compares tensor shape/dtype, sampled-frame tensor
dimensions, video grid, and token counts, not the CPU/CUDA device label. Device differences are
reported separately in `index.json` under
`archived_input_parity.tensor_device_differences`. Metadata mismatches fail
before activation capture, while prediction and generation parity remain
mandatory after inference.
At the amended 500-new-base screening cap, the new pool has at most 2,000 prompt
evaluations. The fixed grid has 218 bidirectional patches per matched pair:
10,900 for 50 primary rescues, plus up to 3,270 for the default 15 secondary
cases. The preflight should be inspected before committing to that GPU cost.

### Standalone entry points

```bash
python scripts/run_phase3b_screening.py \
  --existing_annotation_path data/l5_feature_ablation_v1/L5_full/annotations.jsonl \
  --existing_result_path results/<model>/l5_feature_ablation_v1_main_L5_full/<run>/raw_results.jsonl \
  --model_name Qwen/Qwen3-VL-8B-Instruct

python scripts/select_phase3b_cases.py \
  --annotation_paths <existing-annotations> <batch-annotations...> \
  --result_paths <existing-results> <batch-results...> \
  --mapping_path analysis/phase3b/mapping_audit/video_mapping_manifest.jsonl \
  --output_dir analysis/phase3b/selection

python scripts/run_phase3b_patching.py --stage capture \
  --manifest_path analysis/phase3b/selection/preflight_case_manifest.jsonl \
  --mapping_path analysis/phase3b/selection/selected_video_mappings.jsonl \
  --output_dir <persistent-checkpoint-root>/preflight --shard_index 0

python scripts/run_phase3b_patching.py --stage patch \
  --manifest_path analysis/phase3b/selection/preflight_case_manifest.jsonl \
  --mapping_path analysis/phase3b/selection/selected_video_mappings.jsonl \
  --output_dir <persistent-checkpoint-root>/preflight --shard_index 0

python scripts/analyze_phase3b.py \
  --manifest_path analysis/phase3b/selection/preflight_case_manifest.jsonl \
  --shards_root <persistent-checkpoint-root>/preflight \
  --output_dir analysis/phase3b/preflight_analysis
```

The `screening_progress.json` file lists the exact annotation/result paths for
case selection. Phase 3B analysis treats a base sample, not a decoder layer or
mirrored prompt, as the unit of inference. The selective rescue cohort and
small preflight are exploratory; bootstrap intervals are diagnostic rather
than population-level confirmation. Full Qwen3-VL execution requires the pinned
Colab runtime and a GPU; local unit tests do not validate model behavior.

### SURF VM: Two A10 GPUs

Use [notebooks/phase3b_vm.ipynb](notebooks/phase3b_vm.ipynb), not Colab's
`Run all`, on the VM. The VM entry point uses the frozen 50 rescue cases and
secondary controls without generating videos or re-screening candidates.
Use `--execution_mode model_parallel` on the two A10s: a single worker loads
one FP16 model across both GPUs with balanced weight placement. The default
10 GiB **weight budget per GPU** leaves room for eager-attention intermediates;
it is not a cap on total CUDA memory. Both GPUs must hold parameters, and
CPU/disk offloading is rejected. The VM notebook selects this mode by default.
No sampling, precision, attention-backend, mapping, or patch-grid changes are
made to reduce memory use. Actual module placement is recorded and checked
against preflight and across resumed/merged shards.

The opt-in `independent` scheduling mode remains available for hardware where
a full replica **and forward workspace** fit each GPU. It assigns disjoint
shards to isolated workers. The initial A10 independent-worker preflight
loaded the model successfully but exhausted VRAM during eager attention;
fitting weights alone does not establish that this mode is viable. Two-GPU
model parallelism runs shards sequentially, not as two concurrent replicas.

Keep the repository, input videos, model caches, logs, checkpoints and backups
on **`/data/yuxuanstorage`**, not `/mnt/scratch`. The VM runner pins the expected
core runtime to Transformers 5.9.0, PyTorch 2.11.0 and qwen-vl-utils 0.0.14;
use the same CUDA-enabled environment as the archived experiment. It does not
install packages or silently substitute library versions.

Before running, copy these files from Colab/Drive to persistent storage:

- `analysis/phase3b/selection_v2_controls` from `1001_phase3B_output.zip`, to
  `/data/yuxuanstorage/vlm_phase3b/analysis/selection_v2_controls`.
- The Drive `vlm_phase3b/rescue_pool` directory, including its original MP4s,
  to `/data/yuxuanstorage/vlm_phase3b/rescue_pool`.
- The existing `data/l5_feature_ablation_v1/L5_full/videos` MP4s to the same
  relative location in the VM repository. The ZIP does **not** contain videos.

The path map translates old Colab paths at read time; frozen manifests are
not rewritten. The CPU plan checks every referenced video and records its
SHA-256. Use a fresh A10 output root, never the A100 checkpoint directory.

```bash
python scripts/run_phase3b_vm.py --stage plan \
  --selection_dir /data/yuxuanstorage/vlm_phase3b/analysis/selection_v2_controls \
  --rescue_pool_root /data/yuxuanstorage/vlm_phase3b/rescue_pool \
  --output_root /data/yuxuanstorage/vlm_phase3b/a10_runs/run_v2_mp \
  --gpus 0,1 --execution_mode model_parallel --gpu_weight_budget_gib 10
```

Then repeat this command with `--stage preflight`. It runs the same two
frozen preflight cases using **one model spanning both A10s**, verifies
prediction/input/generation parity and technical controls, then runs the
matched temporal-relocation control with the same placement strategy. Review
the results before switching to `--stage full`. Independent mode instead
checks both replicas and compares their predictions; this cross-replica check
does not apply to model parallelism.
The old A100 preflight does not satisfy the VM gate. A10 rounding differences
or an out-of-memory failure must be diagnosed rather than relaxing parity or
changing the scientific settings automatically. Keep the failed `run_v1`
logs; the new execution mode/code must use a fresh output root. Frozen cases,
source videos and the persistent model cache are reused without re-screening.

`full` runs disjoint five-case shards, both patch directions, and a strict
CPU merge. `--stage analyze` re-runs only CPU analysis. GPU hardware, execution
mode and weight placement are part of shard provenance; mixed A100/A10,
independent/model-parallel or incompatible runtime shards are rejected.
Per-worker tracebacks are streamed and saved under `logs/`, with elapsed time
and checkpoints under `progress/`. Re-run the same command/configuration to
resume; successful capture and patch checkpoints are preserved. An exclusive
lock prevents duplicate orchestrators from writing to the same run root.

### Back Up After Every VM Run

The VM notebook's last cell creates a **full** backup, including saved `.pt`
activations, run configs, mapping/selection, logs, analyses and the referenced
source videos. The Colab notebook's last cell retains its reports-only download
on Colab, but dispatches to this full backup on the VM when
`PHASE3B_VM_OUTPUT_ROOT` is set. Do not confuse the historical reports ZIP,
which omitted `.pt` files, with a resumable checkpoint backup.

```bash
python scripts/backup_phase3b.py \
  --run_root /data/yuxuanstorage/vlm_phase3b/a10_runs/run_v2_mp \
  --backup_dir /data/yuxuanstorage/backups
```

Timestamped ZIP parts (approximately 4 GiB uncompressed per part, except a
single larger file), `backup_manifest.json`, and `SHA256SUMS` are written to
persistent storage. Download **all files** in that backup directory through
the VM file browser, or transfer them with scp/rsync. On your own computer:

```bash
python scripts/backup_phase3b.py --verify <downloaded-backup-directory>
```

Verification checks both archive and individual file SHA-256 hashes. Packaging
on the VM does not establish that a local backup exists. Keep the persistent
VM originals until the downloaded copy has been verified. If there is not
enough disk space for a second full copy, transfer the run directory and frozen
videos directly to your computer; do not delete checkpoints to create space.
`--reports_only` is an explicit lighter export, **not** a backup of activations.
Stop all workers before packaging; interrupted/failed outputs can also be
backed up and their completion status is retained in the manifest.

### Detached Runs and Optional SURF Pause

Use `scripts/launch_phase3b_vm.py` for the **next** long VM run. It launches
`run_phase3b_unattended.py` in a detached tmux session using the same Python
environment. Closing the browser or your local computer does not terminate
that session; pausing/rebooting the VM or killing the tmux server does.
Do **not** restart or duplicate a preflight that is already running.
The launcher does not change model placement, sampling, patching or the
scientific pipeline fingerprint. A completed A10 preflight in `run_v2_mp`
is still required before `full`.

Install tmux once on the VM:

```bash
sudo apt install tmux
```

Automatic Pause is **off by default**. To enable it, create a personal token
in the SURF portal under **Profile > API tokens**, with an expiration beyond
the planned run. Your account must be able to pause the workspace; ask its
owner if the read-only permission check fails. Never put the token in chat,
a notebook, a shell argument, Git, or an experiment backup.
In an activated VM terminal, run:

```bash
python scripts/surf_workspace.py setup \
  --workspace_id <workspace-UUID-from-SURF-Details> \
  --workspace_name YuxuanVM
python scripts/surf_workspace.py check
```

`setup` prompts for the token without echo, checks the exact workspace UUID,
name, running state and Pause permission, then creates
`/data/yuxuanstorage/.phase3b_private/surf.json` (file mode `600`, directory
mode `700`). It refuses to overwrite existing credentials. `check` is a GET
only: neither command pauses anything. Use the exact portal name if it has
changed. A custom `--config_path` must remain outside the repository, run,
backup and lifecycle directories. The unattended runner rejects unsafe
permissions/locations and never copies this file into backups.

Once the A10 preflight has passed and **no other person or job needs this
workspace**, launch the full run from the activated environment:

```bash
python scripts/launch_phase3b_vm.py \
  --stage full --session_name phase3b_full \
  --output_root /data/yuxuanstorage/vlm_phase3b/a10_runs/run_v2_mp \
  --pause_policy success --confirm_exclusive_workspace \
  -- \
  --selection_dir /data/yuxuanstorage/vlm_phase3b/analysis/selection_v2_controls \
  --rescue_pool_root /data/yuxuanstorage/vlm_phase3b/rescue_pool \
  --gpus 0,1 --execution_mode model_parallel --gpu_weight_budget_gib 10
```

`--dry_run` prints the launch command without tmux, API requests, file writes
or GPU work. For background execution **without** automatic Pause, omit
`--pause_policy success --confirm_exclusive_workspace`.
Stage/output/storage/project options belong before `--`; ordinary VM options
belong after it. The VM notebook has equivalent background/Pause switches.
In background mode its last cell displays status rather than trying to
archive an experiment that has just started.

Monitor from another terminal (both files are on persistent storage):

```bash
tmux attach -t phase3b_full
tail -f /data/yuxuanstorage/vlm_phase3b/a10_runs/run_v2_mp_lifecycle/job.log
python -m json.tool /data/yuxuanstorage/vlm_phase3b/a10_runs/run_v2_mp_lifecycle/job_status.json
```

After attaching, `Ctrl+B`, then `D` detaches without interrupting the job.
Do not use `Ctrl+C` or `tmux kill-session` just to disconnect. The session
can disappear after completion; logs and status remain. Check the initial
log/status for errors before leaving it unattended. Re-run the same launch
command only after the previous job has stopped; scientific checkpoints
resume without recomputing completed work. Duplicate sessions and run locks
are rejected.

The completion sequence is:

```text
experiment exits -> persist return code/completion status and log snapshot
-> full timestamped backup (including activations and frozen source videos)
-> verify archive and file SHA-256 checksums
-> confirm no remaining GPU processes or known research runners
-> request SURF Pause only if the explicit policy permits it
```

- `off`: always back up and verify, but never call Pause.
- `success`: pause only if the VM stage returned success and recorded complete.
  A failed run is still backed up, but the VM stays running for diagnosis.
- `finished`: after a verified full backup, also pause a failed/interrupted run
  once its workers have stopped. Use this explicit option to limit idle
  compute costs on failure; resume the workspace later to inspect checkpoints.

Backup failure, insufficient disk space, checksum failure, unsafe credentials,
other GPU jobs, or API errors prevent an unconfirmed Pause from being treated
as successful. Check `needs_attention` in the journal and the SURF portal:
the VM may still be running and charging. Unrelated CPU workloads cannot all
be detected automatically, so `--confirm_exclusive_workspace` is a real
operator confirmation, not a substitute for coordinating with colleagues.
An API timeout is ambiguous and is **not automatically retried**. The wrapper
writes `pause_request_pending` before submitting it, because the VM may be
paused before it can save an acknowledgement. `pause_requested` means the
request was accepted, not that the transition or billing stop is confirmed.
Check that the portal shows **paused**; storage may still incur charges.
Only Pause is implemented: no Delete, volume removal, or Linux shutdown.

The verified bundle is saved under `/data/yuxuanstorage/backups`.
**It is still a VM-side copy, not a local backup.** Download all bundle files
and verify them on your computer as described above. If the workspace has
already paused, resume briefly for transfer and pause again afterwards;
never delete originals before the local copy is verified.
The lifecycle journal is a sibling of the run directory so it does not
invalidate a fresh run's provenance. Its final Pause acknowledgement remains
in that journal; the backup includes the pre-Pause run status and log snapshot.

The API authentication and workspace actions follow the official
[SURF API guide](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/117178402/SRC%2BAPI)
and [workspace OpenAPI schema](https://gw.live.surfresearchcloud.nl/v1/workspace/swagger/schema/).
Session lifecycle follows the [tmux manual](https://man.openbsd.org/tmux.1).
