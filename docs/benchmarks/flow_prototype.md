# Dense-flow mask propagation prototype (Modal, 8 s / 202-frame sample, 1920x1080)

Code: `backend/flow_prototype.py` (run with `backend/modal_flow.py`), CoralSCOP micro-benchmark: `backend/bench_coralscop.py`.
Raw results: `docs/benchmarks/raw/flow*_*.json`, `cs_*.json`. SAM2 pipeline reference: 96.7 s on a T4 (see `video_pipeline_gpu_bench.md`).
BioCLIP/YOLO classification is not part of the prototype (about 1.6 s async on a T4 in the app).

| config | GPU | wall (s) | e2e FPS | x real time | colonies seen |
|---|---|---:|---:|---:|---:|
| A: sync keyframes, fp32, medium flow | T4 | 21.0 | 9.6 | 0.38 | 37 |
| B: + guided prompts, fp16 enc+dec, half-res masks, half-res fast flow | T4 | 14.6 | 13.8 | 0.55 | 36 |
| C: B + async keyframes | T4 | 8.8 | 22.9 | 0.91 | 26* |
| C | H100 | 5.9 | 34.1 | 1.36 | 35 |

\* On the T4 a keyframe takes about 28 frames to finish, so keyframes that arrive while one is running are skipped (6 of 9 ran). Fewer anchors means fewer new IDs, not better tracking. Async results are timing dependent. The SAM2 pipeline found 27 colonies on this clip.

CoralSCOP keyframe latency (ms per keyframe, 854x480, 16x16 grid, mean mask agreement vs fp32 baseline in brackets):

| config | T4 | A10G | L40S | H100 |
|---|---:|---:|---:|---:|
| baseline fp32 | 1370 | 610 | 416 | 209 |
| fp16 encoder | 1144 (0.999) | 549 | 383 | 204 |
| fp16 encoder + decoder | 817 (0.99+) | 404 | 267 | 187 |
| + half-res masks | n/a (batch 128 fails on T4) | 338 (0.95) | 233 | 130 |
| coarser grid 12x12 / 8x8 | loses most masks (agreement 0.3-0.6 / 0.0-0.4) | | | |

Findings: the decoder (about 63% of T4 time) costs per prompt point and is compute-bound, so fp16 helps and batch size does not.
Thinning the uniform grid is not an option; guided prompts (one point per known colony plus grid points only where nothing is known) cut points without losing colonies.
The identity matching was also reworked (one-to-one greedy, containment match, stale colonies kept for 4 keyframes): 43 -> 37 colonies in config A.
Re-anchor IoU of warped masks against fresh detections after ~25 frames of pure flow is about 0.71-0.73.

## All Modal GPUs, async keyframes with queueing (config D; `raw/flowD_*.json`)

Keyframe jobs now queue (one running + one waiting) instead of being skipped: 0 skipped on every GPU, all 9 keyframes ran, and the colony count settled at 34-36 everywhere.

| GPU | wall (s) | FPS | x real time | CoralSCOP ms/keyframe | keyframe lag (frames) |
|---|---:|---:|---:|---:|---|
| T4 | 9.4 | 21.4 | 0.86 | 734 | about 21 |
| L4 | 7.9 | 25.5 | 1.02 | 552 | about 18 |
| A10G | 6.4 | 31.4 | 1.25 | 349 | about 14 |
| L40S | 5.7 | 35.2 | 1.41 | 341 | about 15 |
| A100-40GB | 6.9 | 29.2 | 1.17 | 248 | about 9 |
| A100-80GB | 8.0 | 25.3 | 1.01 | 278 | about 9 |
| H100 | 6.1 | 33.4 | 1.33 | 162 | about 7 |

Past an A10G the GPU no longer limits throughput: decode (about 6-10 ms), ego-motion (about 5 ms) and DIS flow (about 8-17 ms per frame) run on the host CPU, so the differences between the faster GPUs are mostly container CPU variance.

## With classification (config E = D + `--classify`; `raw/flowE_*.json`)

BioCLIP genus + YOLO bleaching run once per new colony in a background thread from a padded full-res crop (same functions the app uses),
and the genus / BLEACHED tag is drawn on the masked video. Every colony got a label on every GPU.

| GPU | wall (s) | FPS | x real time | wall without classification (s) | BioCLIP total (s) | YOLO total (s) |
|---|---:|---:|---:|---:|---:|---:|
| T4 | 9.9 | 20.5 | 0.82 | 9.4 | 0.43 | 0.09 |
| L4 | 8.0 | 25.3 | 1.01 | 7.9 | 0.16 | 0.06 |
| A10G | 6.3 | 32.2 | 1.29 | 6.4 | 0.15 | 0.05 |
| L40S | 6.7 | 30.1 | 1.21 | 5.7 | 0.09 | 0.04 |
| A100-40GB | 8.3 | 24.2 | 0.97 | 6.9 | 0.13 | 0.05 |
| A100-80GB | 7.3 | 27.7 | 1.11 | 8.0 | 0.11 | 0.05 |
| H100 | 5.7 | 35.7 | 1.43 | 6.1 | 0.10 | 0.04 |

Classification itself is 0.1-0.5 s of GPU time over the whole clip, so it is effectively free; wall-time differences against config D are run-to-run CPU noise (about +/-1 s).
Caveat: labels are not stable across runs. The genus mix on the same clip varied (e.g. Porites 5-10, Galaxea 2-6, 13-16 colonies flagged bleached) because crops are taken from the warped masks of the current frame, so slightly different mask extents and keyframe timing change the crop. There is no ground truth here, so accuracy is unverified.
