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
    generate_ladder_dataset.py
    run_eval.py
    analyze_results.py
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

For causal perturbation, create masked-video variants and evaluate them with the
same runner:

```bash
python scripts/make_roi_perturbation_dataset.py \
  --annotation_path data/l5_feature_ablation_v1/size_clear_contrast_pilot/L5_size_only_clear_small_many/annotations.jsonl \
  --output_root data/perturbations/l5_clear_small_many
```

For small-sample attention inspection, use the ROI probe. This is intentionally
separate from `run_eval.py` because `output_attentions=True` is memory-heavy.

```bash
python scripts/probe_attention_roi.py \
  --annotation_path data/l5_feature_ablation_v1/size_clear_contrast_pilot/L5_size_only_clear_small_many/annotations.jsonl \
  --output_path analysis/attention/l5_clear_small_many_attention.json \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --model_revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --max_samples 8
```

This produces feature-level accuracy, strict mirrored-pair accuracy, the accuracy-strict gap `d`, position-sensitive pair rates, paired boundary/feature differences, and swap-consistency diagnostics. Report plots include:

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
