# Level 5 Feature Ablation

Dataset version: `l5_feature_ablation_v1`

The four main variants use paired geometry, motion paths, event order, distractor timing, and boundary timing.
Only the object feature encoding changes.

| Variant | Feature encoding |
| --- | --- |
| `L5_full` | color and shape conjunction |
| `L5_shape_only` | all objects black; shape identifies each target |
| `L5_color_only` | all objects circles; color identifies each target |
| `L5_size_only` | all objects are black circles; target radius identifies small versus large |

Each main variant contains 30 base samples, four boundary conditions, and mirrored original/swapped prompts.

## Size-only 2x2 stress pilot

The `size_stress_pilot/` directory independently manipulates absolute target size and distractor count.

| Scene variant | Target radii | Moving distractors |
| --- | --- | ---: |
| `large_few` | 28, 50 | 1 |
| `large_many` | 28, 50 | 4 |
| `small_few` | 14, 28 | 1 |
| `small_many` | 14, 28 | 4 |

Every distractor radius lies strictly between the small and large target radii, so both target descriptions remain unique.
