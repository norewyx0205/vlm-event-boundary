# Level 5 Feature Ablation

Dataset version: `l5_feature_ablation_v1`

The three variants use paired geometry, motion paths, event order, distractor timing, and boundary timing.
Only the object feature encoding changes.

| Variant | Feature encoding |
| --- | --- |
| `L5_full` | color and shape conjunction |
| `L5_shape_only` | all objects black; shape identifies each target |
| `L5_color_only` | all objects circles; color identifies each target |

Each variant contains four boundary conditions and mirrored original/swapped prompts.
