# VLM Event Boundary Evaluation

This project generates synthetic 2D videos for testing whether video-text models understand event order under different boundary cues.

The current setup contains two datasets:

- `baseline_boundary_videos/`: easy baseline set. This preserves the original 20-video style where the model should be able to distinguish event order.
- `synthetic_boundary_videos/`: harder set with longer videos, randomized order/positions/directions, distractors, and counterbalanced prompts.

Both datasets use `before/after` prompt options because this wording is the target experimental manipulation.

## Dataset Structure

Each dataset directory contains:

```text
<dataset_dir>/
  annotations.jsonl
  videos/
    *.mp4
```

The annotation file has one JSON object per evaluation prompt. A single video appears twice:

- `prompt_original`
- `prompt_swapped`

These two rows use the same `video_id` but swap `option_A` and `option_B`. This balances whether the correct answer is A or B and supports analysis of A-bias.

## Generate Videos

Generate both datasets:

```bash
python3 generate_2d_boundary_videos.py --dataset all
```

Generate only the hard dataset:

```bash
python3 generate_2d_boundary_videos.py --dataset hard
```

Generate only the baseline dataset:

```bash
python3 generate_2d_boundary_videos.py --dataset baseline
```

The `--dataset` argument accepts:

- `all`: generate both hard and baseline datasets
- `hard`: generate `synthetic_boundary_videos`
- `baseline`: generate `baseline_boundary_videos`

## Dataset Sizes

Current generated sizes:

```text
baseline:
  20 videos
  40 evaluation rows
  5 videos per condition

hard:
  120 videos
  240 evaluation rows
  30 videos per condition
```

Conditions:

- `low_boundary`
- `temporal_boundary`
- `visual_boundary`
- `audio_boundary`

## Run Evaluation

Default evaluation uses Qwen2-VL:

```bash
python3 run_eval.py
```

Run the hard dataset with a specific model:

```bash
python3 run_eval.py \
  --model-name Qwen/Qwen2-VL-2B-Instruct \
  --annotation-path /content/vlm-event-boundary/synthetic_boundary_videos/annotations.jsonl
```

Run the baseline dataset:

```bash
python3 run_eval.py \
  --model-name Qwen/Qwen2-VL-2B-Instruct \
  --annotation-path /content/vlm-event-boundary/baseline_boundary_videos/annotations.jsonl
```

Run a Qwen3-VL model:

```bash
python3 run_eval.py \
  --model-name Qwen/Qwen3-VL-8B-Instruct \
  --annotation-path /content/vlm-event-boundary/synthetic_boundary_videos/annotations.jsonl \
  --experiment-version hard_v1
```

For a quick smoke test:

```bash
python3 run_eval.py \
  --annotation-path /content/vlm-event-boundary/baseline_boundary_videos/annotations.jsonl \
  --experiment-version baseline \
  --max-samples 4
```

## Evaluation Arguments

`run_eval.py` supports:

```text
--model-name       Hugging Face model id
--annotation-path  Path to annotations.jsonl
--result-dir       Root directory for timestamped result folders
--output-name      Optional raw result filename
--experiment-version Dataset or experiment version label
--max-samples      Optional limit for quick tests
```

The script prints:

- accuracy by boundary condition
- accuracy by correct option, useful for A-bias
- accuracy by prompt variant, useful for swapped-prompt checks

Each run is saved in a timestamped directory:

```text
<result-dir>/
  <experiment-version>/
    <model-name>/
      <YYYYMMDD_HHMMSS>/
        raw_results.jsonl
        summary.json
        summary.txt
        config.json
```

For example:

```text
results/
  hard_v1/
    Qwen_Qwen3-VL-8B-Instruct/
      20260518_172530/
```

This makes it easier to compare baseline runs and future hard-set iterations.

## Run OpenAI GPT Evaluation

OpenAI vision models currently receive image inputs through the API, so `run_eval_openai.py` samples frames from each video and sends the ordered frame sequence to GPT.

Set your API key first:

```bash
export OPENAI_API_KEY="your_api_key"
```

Run a baseline smoke test:

```bash
python3 run_eval_openai.py \
  --model-name gpt-4.1 \
  --annotation-path /content/vlm-event-boundary/baseline_boundary_videos/annotations.jsonl \
  --experiment-version baseline_gpt_smoke_test \
  --max-samples 4
```

Run the full baseline set:

```bash
python3 run_eval_openai.py \
  --model-name gpt-4.1 \
  --annotation-path /content/vlm-event-boundary/baseline_boundary_videos/annotations.jsonl \
  --experiment-version baseline_gpt
```

Run the hard set:

```bash
python3 run_eval_openai.py \
  --model-name gpt-4.1 \
  --annotation-path /content/vlm-event-boundary/synthetic_boundary_videos/annotations.jsonl \
  --experiment-version hard_v1_gpt
```

Useful frame sampling arguments:

```text
--frame-count       Number of frames sampled uniformly from each video
--max-frame-width   Resize sampled frames before sending
--image-detail      OpenAI image detail setting: low, high, or auto
```

Important limitation: this GPT evaluation uses sampled visual frames only. It does not pass the audio track, so the beep in `audio_boundary` is not available as audio input to the model.

## Dependencies

Video generation requires:

```bash
pip install opencv-python numpy imageio-ffmpeg
```

Model evaluation requires a working PyTorch/Transformers environment plus Qwen video utilities:

```bash
pip install torch transformers accelerate qwen-vl-utils
```

For Qwen3-VL, use a recent Transformers version that supports the selected Qwen3-VL model.

OpenAI GPT evaluation additionally requires:

```bash
pip install openai
```

## Notes

- Baseline videos are intentionally simple and should function as a sanity check.
- Hard videos are longer and include randomized target order, randomized motion directions, randomized positions, and distractor objects.
- The annotation rows are evaluation prompts, not unique videos. Use `eval_id` for prompt-level analysis and `video_id` for video-level pairing.
- The `before/after` relation and A/B correct option are balanced in both datasets.
