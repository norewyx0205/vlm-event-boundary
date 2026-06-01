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
    ladder_v1/
      level_1_simple/
        videos/
        annotations.jsonl
      level_2_randomized/
        videos/
        annotations.jsonl
      level_3_static_distractors/
        videos/
        annotations.jsonl
      level_4_moving_distractors/
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
| 3 | `level_3_static_distractors` | Level 2 plus 1-2 static distractor objects. Tests visual object binding. |
| 4 | `level_4_moving_distractors` | Level 2 plus static/moving distractors and a later unrelated motion event. Hard setting. |

All levels include four boundary conditions:

- `low_boundary`
- `temporal_boundary`
- `visual_boundary`
- `audio_boundary`

## Generate Ladder Data

Default generation:

```bash
python scripts/generate_ladder_dataset.py \
  --dataset_version ladder_v1 \
  --samples_per_level 30 \
  --output_root data/ladder_v1 \
  --seed 42
```

Useful generation arguments:

```text
--samples_per_level
--fps
--level_durations          Comma-separated durations for levels 1-4, default 10,12,14,20
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

For a fixed `base_sample_id`, the two target objects keep the same color and shape across all four difficulty levels and all four boundary conditions. Across levels, only the difficulty manipulation changes: motion path, target order, distractors, and temporal complexity.

After generation, verify the dataset controls:

```bash
python scripts/check_ladder_dataset.py --root data/ladder_v1
```

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
  --annotation_path data/ladder_v1/level_1_simple/annotations.jsonl \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --dataset_name ladder_v1_level_1_simple \
  --output_dir results
```

Run Qwen3-VL:

```bash
python scripts/run_eval.py \
  --annotation_path data/ladder_v1/level_4_moving_distractors/annotations.jsonl \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --dataset_name ladder_v1_level_4_moving_distractors \
  --output_dir results
```

Quick smoke test:

```bash
python scripts/run_eval.py \
  --annotation_path data/ladder_v1/level_1_simple/annotations.jsonl \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --dataset_name smoke_ladder_v1_level_1_simple \
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
  --input results/Qwen_Qwen3-VL-8B-Instruct/ladder_v1_level_4_moving_distractors/<timestamp>/raw_results.jsonl \
  --output_dir analysis/ladder_v1_qwen3_level4 \
  --plots
```

Analyze a directory containing multiple run folders:

```bash
python scripts/analyze_results.py \
  --input results \
  --dataset_name_prefix ladder_v1_level_ \
  --output_dir analysis/ladder_v1_qwen3_all \
  --plots
```

The analyzer saves:

- `accuracy_by_difficulty.csv`
- `accuracy_by_difficulty_condition.csv`
- `accuracy_by_correct_option.csv`
- `accuracy_by_prompt_variant.csv`
- `prediction_distribution.csv`
- `swap_consistency_summary.csv`
- `swap_consistency_details.csv`
- `summary.json`
- optional `accuracy_by_difficulty_condition.png`

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
pip install torch transformers accelerate qwen-vl-utils decord
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
