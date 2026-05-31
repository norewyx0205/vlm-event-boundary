# Ladder Dataset

Dataset version: `ladder_v1`

Each level contains unique videos plus mirrored evaluation rows in `annotations.jsonl`.

| Level | Name | Duration | Description |
| --- | --- | ---: | --- |
| 1 | `level_1_simple` | 10s | simple two-target baseline with no distractors |
| 2 | `level_2_randomized` | 12s | randomized target positions, directions, and order |
| 3 | `level_3_static_distractors` | 14s | randomized targets plus static distractors |
| 4 | `level_4_moving_distractors` | 20s | randomized targets plus static/moving distractors and later unrelated motion |

For a fixed `base_sample_id`, the two target objects keep the same color and shape across all difficulty levels and boundary conditions. This controls for possible color/shape response bias.

Every video is evaluated twice:

- `prompt_variant=original`: correct sentence in option A
- `prompt_variant=swapped`: correct sentence in option B
