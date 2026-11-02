# Thesis Integration: Frozen Evidence, Not New Experiments

The 26 October 2026 integration uses the author-confirmed Part 1 PDF and the
completed Phase 3B-3D results, including the L16-19 singleton supplement. No
new GPU inference, probe fitting, counterfactual panel or circuit search is
part of this work. Frozen measurement scripts and exports remain unchanged.

## CPU-Only Case Correspondence

```bash
MPLCONFIGDIR=/tmp/phase3d-thesis-matplotlib python3 scripts/analyze_phase3d_case_links.py \
  --source_evidence analysis/phase3d/full_run_v2_20261020/evidence \
  --singleton_evidence analysis/phase3d/single_layer_ko_run_v1_20261022/evidence \
  --output_dir analysis/phase3d/thesis_integration_20261026/case_analysis
```

Dependencies: NumPy, SciPy and Matplotlib, for local CPU analysis only.

The join verifies downloaded exports against the saved backup manifests,
requires exact baseline agreement and a one-to-one 50-base join, and keeps
the primary L20 readout and archived prompt view fixed. Probe observations
must remain held out by base. No feature arrays or model weights are loaded.

Three descriptive correlations are retained: patch margin gain versus L18
temporal damage, probe-score gain versus that damage, and patch margin gain
versus probe-score gain. Positive KO damage means minus the temporal margin
change. Uncertainty is a conditional case bootstrap, not inference about a
population or refitted probes. Fold-specific scales are examined using
within-fold standardized ranks and leave-one-fold-out summaries; this does
not establish calibration or remove lexical confounding.

Derived outputs include a one-row-per-base CSV, full sensitivity JSON,
source/output hashes, a scatter figure and an editable synthesis SVG with
PDF/PNG companions. The conceptual arrows express an interpretive hypothesis,
not observed mediation or a proven L16-to-L18 pathway.

```bash
python3 -m unittest discover -s tests -p test_phase3d_case_links.py -v
```

## Writing Artifacts

The local, gitignored package is under
`analysis/phase3d/thesis_integration_20261026/`:

- `evidence_rq_matrix.md`: original Part 1 wording for provenance, the adopted
  four-question architecture and current working SQ3-SQ4, an active
  evidence/chapter map, and explicit claim boundaries. SQ3 is behavioural;
  SQ4 examines bounded functional contributions using joint intervention and
  probe evidence, not a separate circuit-discovery or decodability question.
- `results_discussion_integrated.md`: a standalone, evidence-ordered writing
  draft, not an edit to the Part 1 PDF or a thesis source.
- `meeting_brief_20261029.md`: scope decisions for Albert, not an experiment
  authorization or an email sent on the user's behalf.
- `preservation_status.md`: full-backup and VM status kept separate from
  scientific completion.

The manuscript is organized by argument, not phase chronology. Earlier
attention calibration remains descriptive and must not be relabelled as a
completed causal analysis. No citations or literature claims are added
without checking their actual sources. Full off-VM preservation is a separate
checksum-verification task; lightweight evidence is not a complete backup.

## Reporting Convention

Follow `docs/reporting_conventions.md` in subsequent writing: report original
levels as well as differences, for example `before -> after (delta)`, with the
metric, paired base count and pooling made explicit. Distinguish probe evidence
from the model's answer margin, and calculate differences before rounding.
