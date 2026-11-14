# Video pipeline benchmark: Colab T4 and Modal GPU sweep

**What was measured.** The exact flow the app runs (`VideoProcessor.start_job` → `_run_job_pipeline`) with the options the UI sends (`UploadView.handleStartVideoAnalysis`):
sam2 tracker, 512 px, 1 worker, strides 10/25, overlap 0.70, 16×16 keyframe grid, MP4 export on, precision auto.
Clip: `demo_images/sample_reef_transect_25fps.mp4`, 1920×1080, 202 frames, 8.08 s (the "Load sample" clip).
Harness: `backend/bench_app_flow.py` (wraps hot calls from the outside, no app code touched). Modal runner: `backend/modal_bench.py`.
Each run is 1 cold + 2 warm jobs; the numbers below are the **warm median**. Raw data: `docs/benchmarks/raw/`.

## Headline: end-to-end speed

| GPU | wall (s) | e2e FPS | × real time | tracker FPS | SAM2 propagate (ms/frame) | keyframe segmentation (ms/call) | MP4 export (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Colab T4 | 114.1 | 1.77 | 0.071 | n/a¹ | 369 | 1595 | 9.4 |
| Modal T4 | 96.7 | 2.09 | 0.084 | 3.02 | 331 | 1371 | 5.1 |
| Modal L4 | 106.4 | 1.90 | 0.076 | 2.47 | 405 | 968 | 4.1 |
| Modal A10G | 64.7 | 3.12 | 0.125 | 4.17 | 240 | 554 | 3.4 |
| **Modal L40S** | **51.2** | **3.95** | **0.158** | **5.34** | **187** | 365 | 3.2 |
| Modal A100-40GB | 80.6 | 2.51 | 0.100 | 3.19 | 313 | 311 | 5.2 |
| Modal A100-80GB | 76.7 | 2.63 | 0.105 | 3.25 | 308 | 234 | 3.9 |
| Modal H100 | 65.3 | 3.09 | 0.124 | 3.80 | 263 | 179 | 3.2 |

¹ The Colab run predates a harness fix for the tracker-FPS metric; the app's own `tracker_fps` reads 0 at job completion (see Findings).
Cold start (first job after launch, includes tracker load and warm-up) adds about 5–25 s over the warm figure; model loading itself is 2–4 s on Modal and 13 s on Colab.

**The fastest GPU only reaches about 0.16× real time, i.e. 6× slower than the video.** Paying for a faster GPU barely helps:
the H100 is no faster than an A10G, and an A100 is slower than an A10G. The cause is below.

## Where the time goes (warm run, share of wall time)

| stage | Colab T4 | Modal L40S | Modal H100 |
|---|---:|---:|---:|
| SAM2 mask propagation | 66.8 % | 75.8 % | 83.2 % |
| Keyframe discovery (CoralSCOP, ONNX) | 14.0 % | 7.2 % | 2.7 % |
| Annotated MP4 export | 8.2 % | 6.3 % | 4.8 % |
| Decode and ego-motion prep | 3.9 % | 4.1 % | 3.0 % |
| Idle / unaccounted | 4.8 % | 3.7 % | 3.6 % |
| Geometry, transfer, persistence, CSV/COCO, classification | about 3.4 % | about 3 % | about 3 % |

Per-call latencies (T4, from `docs/benchmarks/raw/t4.md`): keyframe segmentation 1.6 s per call (10 calls), BioCLIP batch 142 ms, YOLO bleaching batch 18 ms,
mask registration 19 ms (144 calls), SAM2 per-frame propagation p50 353 ms / p95 613 ms / max 940 ms, chunk init about 1.1 s (2 chunks).

## Why propagation dominates and does not scale with the GPU

A synthetic scaling test on the T4 (SAM2 tiny, 512 px, fp16, 30 frames):

| tracked objects | ms/frame |
|---:|---:|
| 1 | 38 |
| 4 | 113 |
| 8 | 167 |
| 16 | 341 |
| 27 | 556 |

- Cost is about 20 ms per object per frame, linear. The sample clip ends with 27 tracked corals, which matches the 350 ms/frame seen in the real run.
- A torch profile at 16 objects shows 5.3 s of GPU kernel time for 10 s of wall time, and about 5,000 tiny kernels per frame. The GPU is idle about half the time: it is launch/Python bound.
  That is why a faster GPU barely helps and why the host CPU matters (the L40S container's CPU is likely the reason it wins).
- Stock SAM2 propagates each object separately (`batch_size=1` per object), so the work scales with object count and not with frame size.
- Keeping SAM2 state on the GPU (`offload_to_cpu=False`) was tested on the T4: about 6 % faster (wall 107 s vs 114 s, propagation 71.7 s vs 76.3 s), peak VRAM 1.2 GB vs 0.45 GB. It is not the bottleneck.
- Keyframe discovery, in contrast, does scale with the GPU: 1.4 s (T4) down to 0.18 s (H100). On the T4 it splits into ViT-B encoder about 550–600 ms (fp32) and decoder about 134 ms per 32-point batch (8 batches).

## Findings and fixes made during this benchmark

Fixed and pushed:
1. `psutil` is imported by `core/device.py` but was not in either requirements file, so a clean install failed at import (Colab only worked because it preinstalls psutil). Added to both.
2. ONNX Runtime silently falls back to CPU when the CUDA provider libraries cannot load. The loaders in `core/models.py` printed "successfully loaded on NVIDIA CUDA GPU (CPUExecutionProvider)" in that case. They now warn accurately. A first Modal T4 run went through this path (CPU inference, 1.23 FPS) before the image was fixed.
3. Modal image now installs `onnxruntime-gpu[cuda,cudnn]` so the CUDA provider loads (missing `libcurand`). A leftover diagnostic Modal app of mine that failed at startup repeatedly was the "crash loop" you saw; it is stopped.

Found, not fixed:
- `compile_tracker=true` (an API option, off by default) crashes on CUDA: "accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run". Needs `cudagraph_mark_step_begin()` or a non-CUDA-graph compile mode.
- The job's `tracker_fps` reads 0 once the job completes because timings are reset per chunk; the UI telemetry falls back to analysis FPS during the run.
- On macOS the annotated MP4 export of a 1 s trimmed clip failed with "moov atom not found" (export reported `partial`). It did not occur on Linux/CUDA runs; cause not investigated.

## Improvement plan (ranked by measured share of wall time)

Nothing below is implemented or validated yet except where noted. Items that change which masks are produced need an accuracy check on labelled clips before adoption.

1. **Cut SAM2 per-object cost (67–83 % of wall).**
   - Batch all tracked objects through one memory-attention / mask-decoder call per frame instead of one pass per object (custom propagate loop or a multi-object SAM2 fork). Expected to be the largest win and to make GPU choice matter again.
   - Reduce launch overhead with `torch.compile` / CUDA graphs. First fix the `compile_tracker` crash above.
   - Track fewer objects: drop tiny or low-IoU colonies at registration, merge duplicates. The sample ends with 27 corals.
   - Propagate every 2nd frame and interpolate masks (about 2× on propagation); lower `num_maskmem` (7 → 4). Quality trade-offs.
2. **Keyframe discovery (up to 14 % on T4/L4).**
   - Pass `discovery_mask` (already supported by `run_segmentation`) so the 16×16 grid only prompts regions not covered by existing tracks; fewer decoder batches.
   - fp16 encoder or the TensorRT execution provider (available on the Modal image); keep encoder and decoder batches on the GPU.
3. **MP4 export (3–9 s, 3–8 %).** Render overlays at proxy resolution or use hardware encoding (NVENC where present); the Colab CPU took 9.4 s against 3 s on Modal. Make it optional or lazy (render on download).
4. **Chunk init and idle gaps (about 4–7 %).** `sam2_init_video` takes 0.7–1.1 s per chunk (CPU resize and normalise of 128 frames). Do the resize and normalise on the GPU or in the prefetch thread (the `prefetch_executor` in the engine is currently unused).
5. **Telemetry.** Make `tracker_fps` survive chunk resets; report per-stage ms in the job timings (the harness already does this from outside).

Shortening the clip from 8 s to 4 s was not needed: per-frame costs are flat, so wall time scales about linearly with frames (about half), plus roughly 5–10 s of fixed startup. All eight GPU runs completed on the full 8 s clip.
