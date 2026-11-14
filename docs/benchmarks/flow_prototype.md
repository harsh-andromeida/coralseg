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
