# Flow pipeline: camera-motion mask propagation (Modal, 8 s / 202-frame sample, 1920x1080)

Code: `backend/core/video/flow_tracking.py` (used by the app's video engine, `core/video/engine.py`). The per-GPU table below was measured with the standalone script that preceded it (same logic); the app-level numbers are in the next section. Benchmark harness: `backend/bench_app_flow.py`, Modal runner: `backend/modal_flow.py`. SAM2 pipeline reference: `video_pipeline_gpu_bench.md` (96.7 s on a T4).

## How it works
Corals are static and the camera moves, so instead of tracking each colony with SAM2 (about 20 ms per colony per frame):
1. Every colony lives in one label image, warped each frame with dense optical flow (OpenCV DIS, computed at half resolution). About 1 ms per frame whatever the colony count.
2. CoralSCOP segments only the ego-motion keyframes, in a background thread. Its result is carried to the current frame through the flows recorded since the keyframe, then matched to the warped colonies (one-to-one, best IoU or containment first) to re-anchor them.
3. Keyframe prompts are guided: one point per known colony plus grid points only where nothing is known, instead of the full 16x16 grid.
4. CoralSCOP runs in fp16 (encoder and decoder) on CUDA, with masks decoded at half resolution.
5. Each new colony is classified once (BioCLIP genus, YOLO bleaching) in another thread from a padded full-resolution crop. The video is rendered and encoded in its own thread.

## Results (all stages, with classification)

| GPU | wall (s) | FPS | x real time | CoralSCOP keyframe lag (frames) | colonies | bleached |
|---|---:|---:|---:|---:|---:|---:|
| T4 | 10.3 | 19.7 | 0.79 | ~21 | 33 | 14 |
| L4 | 8.1 | 25.0 | 1.00 | ~19 | 36 | 15 |
| A10G | 6.2 | 32.6 | 1.30 | ~15 | 34 | 13 |
| L40S | 6.1 | 33.2 | 1.33 | ~12 | 36 | 16 |
| A100-40GB | 8.4 | 23.9 | 0.96 | ~8 | 36 | 16 |
| A100-80GB | 10.3 | 19.6 | 0.78 | ~6 | 35 | 16 |
| H100 | 5.5 | 36.5 | 1.46 | ~7 | 36 | 17 |

All 9 keyframes ran on every GPU (0 skipped). Past an A10G the GPU no longer limits throughput: decode (6-10 ms per frame), ego-motion (about 5 ms) and flow (8-17 ms) run on the host CPU, so differences between the faster GPUs are mostly container CPU variance. Classification itself is 0.1-0.5 s of GPU time over the whole clip.
For comparison the SAM2 pipeline ran at 1.8-3.9 FPS (0.07-0.16x real time) on the same clip.

## CoralSCOP keyframe latency (ms, 854x480, 16x16 grid; mask agreement with fp32 in brackets)

| config | T4 | A10G | L40S | H100 |
|---|---:|---:|---:|---:|
| fp32 baseline | 1370 | 610 | 416 | 209 |
| fp16 encoder + decoder | 817 (0.99+) | 404 | 267 | 187 |
| + half-resolution masks | n/a | 338 (0.95) | 233 | 130 |

The mask decoder (about 63% of T4 time) costs per prompt point and is compute-bound, so fp16 and fewer points help; decoder batch size does not. Thinning the uniform grid to 12x12 or 8x8 loses most masks (agreement 0.0-0.6), which is why prompts are guided instead.

## Trade-offs and caveats
- **Accuracy is unverified.** Warped masks overlap a fresh keyframe detection with IoU about 0.71 after ~25 frames of pure flow (a rough proxy, since the detections are noisy too). Edges drift between keyframes; tall colonies and newly visible area are handled worse than by a tracker.
- **Identity.** 34-36 colony IDs against 27 in the SAM2 run on this clip.
- **Labels are not stable across runs.** The genus mix and number flagged bleached vary a little between runs (crops come from warped masks and keyframe timing), and there is no ground truth.
- **Timing dependent.** Keyframe corrections arrive later on slower GPUs (lag column); results are not bit-for-bit reproducible.
- **Host CPU bound** from an A10G upward; a slow host CPU limits it.

## In the app (real `VideoProcessor`, Modal T4, same 8 s clip, MP4 + CSV + COCO exports included)

| | SAM2 pipeline (before) | flow pipeline (app engine) |
|---|---:|---:|
| wall time, end to end with exports | 96.7 s | 16.3 s |
| end-to-end FPS | 2.1 | 12.4 |
| processing loop only (what the job's FPS readout shows) | 2.1 | about 18 |
| colonies | 27 | 31 |

Main-thread cost per frame in the app: decode 8.5 ms, ego-motion 5 ms, flow + warp 13 ms, label-to-polygon geometry 16 ms; CoralSCOP keyframes (0.78 s each) run in the background. Geometry extraction is now the largest single per-frame cost and the next thing to optimise.
The annotated MP4 export used to be rejected as "unreadable or corrupted" on every run (ffmpeg was not given time to finalise the file); that is fixed.

## Reproduce
```bash
python backend/bench_app_flow.py --runs 3                       # local / Colab, drives the real engine
BENCH_GPU=T4 modal run backend/modal_flow.py                    # Modal
```
