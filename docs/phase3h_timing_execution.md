# Phase 3H Execution: Timing-Cell Materials, Gated Forwards and Analysis

Implements `phase3h_timing_dissociation.md`. Behaviour only; no KO, patch or probe.

## Code

- `scripts/phase3h_timing_materials.py` (CPU): `generate` (fresh bases 1001–1050 via the rescue-pool
  generator, default seed and settings, batches of five), `fresh` / `rescue` (materials), `cache`
  (matched-encoder cache). For each base it first reconstructs the native Low and Temporal videos through its
  own render path and requires decoded-pixel equality with the existing videos. Only then does it render the
  new timing cells, each with a frame-differencing timing audit. At the native schedule the render spec is
  identical to the Phase 3E reconstruction spec (tested on all 100 rescue rows).
- `scripts/run_phase3h_timing_behaviour.py` (GPU): `prepare`, `run --stage preflight|full`, `verify`,
  `export`.
- `scripts/run_phase3h_timing_unattended.py`: tmux lifecycle (preflight → full → export → verify, checksummed
  backup, e-mail, idle check, Pause). It is derived from Phase 3G.
- `scripts/analyze_phase3h_timing.py` (CPU): S and b per cell, the five planned contrasts, the 2,000-draw
  base bootstrap and the pre-specified reading rule.
- `tests/test_phase3h_timing.py`.

## Forwards

| Block | Forwards |
| --- | ---: |
| Fresh: 50 bases × 5 cells × 2 layouts | 500 |
| Rescue new cells: 50 × 3 × 2 | 300 |
| Rescue native Low/Temporal: 50 × 2 × 2, each an exact replay of the archived logits | 200 |
| Total | 1,000 |

The protocol budgeted 8 native replays as a gate. The implementation instead re-runs all 200 rescue native
prompts and requires exact archived logits for each, so every margin in the analysis comes from one run.
The preflight is the full cell grid for two fresh bases (first fresh base of each first mover) and the two
Phase 3D preflight rescue bases: 40 forwards.

## Gates

1. Native reconstruction: decoded-pixel exact, one environment. The six early rescue bases (005, 011, 014,
   015, 017, 019) use the local OpenCV 4.13 cache. That cache reproduces their originals byte for byte, and
   it holds their new cells, rendered with the same encoder. The other 44 rescue bases and all fresh bases
   render on the VM.
2. Timing audit: decoded motion frames equal the scheduled motion frames for every video.
3. Rescue native prompts: logits equal the archived full-vocabulary logits (`torch.equal`), checked at run
   time and again at verify.
4. Within base × layout, all five cells have identical prompt token IDs and visual positions. The Event-1
   and Event-2 temporal-token bins shift relative to `low` by exactly the scheduled number of 1-s tokens.
   Under the default sampling (36 frames from 270, spacing ≈ 7.69 frames) the expected shifts hold for all
   cells; the preflight confirms them.
5. The full run requires the identical, re-verified preflight summary.

## VM Steps

Paths below are on the VM (`/data/yuxuanstorage`); `$PY` is the frozen Phase 3B venv Python.

```bash
cd /data/yuxuanstorage/vlm-event-boundary && git fetch origin && git merge --ff-only origin/main
```

Copy the local cache first (from this computer):

```bash
rsync -a analysis/phase3h/timing_cache_v1/ ywang12@145.38.205.113:/data/yuxuanstorage/vlm_phase3h/timing_cache_v1/
```

```bash
$PY scripts/phase3h_timing_materials.py --action generate --output_root /data/yuxuanstorage/vlm_phase3h/fresh_pool --base_ids 1001-1050
$PY scripts/phase3h_timing_materials.py --action fresh --annotations /data/yuxuanstorage/vlm_phase3h/fresh_pool/L5_full/annotations.jsonl --base_ids 1001-1050 --output_root /data/yuxuanstorage/vlm_phase3h/materials_fresh_v1
$PY scripts/phase3h_timing_materials.py --action rescue --pairs /data/yuxuanstorage/vlm_phase3d/run_v2/pairs.json --source_root /data/yuxuanstorage/vlm_phase3d/run_v2 --render_cache /data/yuxuanstorage/vlm_phase3h/timing_cache_v1 --output_root /data/yuxuanstorage/vlm_phase3h/materials_rescue_v1
$PY scripts/run_phase3h_timing_behaviour.py --action prepare --output_root /data/yuxuanstorage/vlm_phase3h/run_v1 --fresh_materials /data/yuxuanstorage/vlm_phase3h/materials_fresh_v1 --rescue_materials /data/yuxuanstorage/vlm_phase3h/materials_rescue_v1 --order_root <order_root from the Phase 3G run config>
$PY scripts/run_phase3h_timing_unattended.py --action launch --stage full --output_root /data/yuxuanstorage/vlm_phase3h/run_v1 --email_notify --pause_after --confirm_exclusive_workspace
```

Launch through `ssh … 'cd <repo>; exec $PY …'` so the idle guard does not match the SSH shell. After
the e-mail: download and verify the backup, then run locally:

```bash
python3 scripts/analyze_phase3h_timing.py --results <run>/results.json --output_root analysis/phase3h/analysis_v1
```

## Local Checks Already Run (3 November 2026)

- All 100 rescue rows: the native timing spec equals the Phase 3E reconstruction spec.
- Six early rescue bases: the cache reproduces originals byte for byte. A rescue-materials dry run on these
  bases gave 36 prompts; every reconstruction was exact and every timing audit passed.
- Fresh smoke (bases 1001–1005, local encoder): generation, exact native reconstruction and timing audits
  for all 25 videos passed. These local smoke renders are not used; fresh materials are rendered on the VM.
- No GPU forward has been run.
