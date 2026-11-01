# Rescue50 Missing-View Capture

Use `run_phase3e_rescue50_capture.py` only for the 100 missing pure A/B
permutations of the original 50 rescue bases. Reuse the 100 original prompts
from the verified CPU recovery. Do not regenerate videos or run KO/patching.

The portable input directory contains `plan.json`, `verification.json` and
the unchanged `recovery/` directory. Verify every resource checksum after
transfer. The live historical Phase 3D context must match the preparation's
source fingerprint, pair IDs, model/input settings and historical folds.

Four technical prompt cells cover two original bases with different first
movers and both boundary conditions. The preflight uses actual tokenizer
spans and processor tensor hashes, exact archived full-logit replay, exact
read-only hook parity, and standard first-token generation parity. Hooks
read only the two contextual reference tokens at L16/L17/L18/L20. They do
not modify outputs. Failure stops the job without dropping a case.

Full supplementation reuses the four preflight cells and acquires the other
96. Every new prompt uses baseline, capture and standard-generation checks:
100 prompt cells do not mean only 100 total technical forwards. Model weight
placement and precision follow the historical 2-A10 model-parallel setup.
Maintain a 12 GiB persistent reserve, plus space for a complete backup.

The full export combines 200 recovered original statement rows and 200 new
A/B-control rows into an isolated 400-row store. This is a new rescue50
schema, not the 40-family paired-video store. Probe fitting is a subsequent
CPU task and is not performed by the acquisition lifecycle.

`run_phase3e_rescue50_unattended.py --stage full` explicitly authorizes the
sequence preflight -> full -> export -> verification. It runs in tmux,
preserves checkpoints, writes a checksummed full backup, emails success or
failure, then requests SURF Pause only after a verified backup and GPU/CPU
quiescence checks. A failed or partial run is not reported as a completed
probe experiment. No automatic retry or deletion is performed.

```bash
python scripts/run_phase3e_rescue50_capture.py --action prepare \
  --source_root /data/yuxuanstorage/vlm_phase3d/run_v2 \
  --input_root /data/yuxuanstorage/vlm_phase3e/rescue50_inputs_v1 \
  --output_root /data/yuxuanstorage/vlm_phase3e/rescue50_reference_last_v1

python scripts/run_phase3e_rescue50_unattended.py --action launch --stage full \
  --output_root /data/yuxuanstorage/vlm_phase3e/rescue50_reference_last_v1 \
  --email_notify --pause_after --confirm_exclusive_workspace
```
