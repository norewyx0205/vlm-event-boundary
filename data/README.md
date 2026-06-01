# Ladder Dataset

Dataset version: `ladder_v2`

Each level contains unique videos plus mirrored evaluation rows in `annotations.jsonl`.

| Level | Name | Duration | Description |
| --- | --- | ---: | --- |
| 1 | `level_1_simple` | 10s | fixed two targets, no distractors |
| 2 | `level_2_randomized` | 12s | random target positions, directions, and order |
| 3 | `level_3_non_target_static_distractors` | 14s | static distractors with colors/shapes distinct from targets |
| 4 | `level_4_target_like_static_distractors` | 16s | static distractors sharing color/shape with targets |
| 5 | `level_5_target_like_moving_distractors` | 18s | moving distractors sharing color/shape with targets near target events |
| 6 | `level_6_hard_temporal_interference` | 20s | target-like moving distractors near boundary plus unrelated later motion |

For a fixed `base_sample_id`, the two target objects keep the same color and shape across all difficulty levels and boundary conditions. This controls for possible color/shape response bias.

Every video is evaluated twice:

- `prompt_variant=original`: correct sentence in option A
- `prompt_variant=swapped`: correct sentence in option B
