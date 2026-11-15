# Does the live results view cost processing FPS? (A/B, Colab T4)

Harness: `backend/bench_app_flow.py --runs 4 --poll-ui {off,live}`. `live` runs a thread that polls job status and drains frame deltas
(64 per page, every 400 ms, JSON-serialised) like `VideoAnalysisView`. Sample clip, 202 frames at 1920x1080, Colab T4 (2 vCPUs), order
off, live, off, live; the numbers are the warm median of 3 runs.

| | headless (2 sessions) | live view (2 sessions) |
|---|---:|---:|
| tracking loop FPS (job-reported) | 13.7 / 13.4 | 12.9 / 12.8 |
| loop finished at | 15.7 / 16.1 s | 16.6 / 16.7 s |
| end-to-end wall | 26.8 / 26.7 s | 27.7 / 28.0 s |
| end-to-end FPS | 7.5 / 7.6 | 7.3 / 7.2 |

**Live polling costs about 5% of loop FPS** (66 polls, 3.2 MiB serialised). Process-first would not change the FPS meaningfully.

## Where the time goes (headless, 26.5 s)

| stage | s | note |
|---|---:|---|
| video decode | 7.6 | main thread, 37 ms/frame on 2 vCPUs |
| ego-motion | 3.2 | main thread |
| flow + warp | 2.3 | main thread |
| CoralSCOP keyframes (9) | 8.5 | background thread, 0.92 s each; 0 s blocked on it |
| annotated MP4 export | 10.2 | runs after tracking, 38% of end-to-end |
| CSV + COCO export | 0.5 | |

The tracking loop ends at ~16 s but the job only reports complete at ~27 s. CPU is the limit (process 125% of 200%, GPU 23% mean).
The loop is decode-bound, not GPU-bound.

## Conclusions
1. Keep the live view; the process-first UI is not justified by throughput.
2. Biggest win: do not wait for the MP4 export. Mark results ready after the tracking loop and export in the background (saves ~10 s of 27).
3. Second: decode in a prefetch thread so it overlaps with ego-motion/flow (up to 7.6 s of the 16 s loop, bounded by the 2 vCPUs).

## After the changes (same setup, Colab T4, 2 vCPUs)

Jobs now report `completed` once every frame is persisted, and CSV/COCO/MP4 are written in a background thread (`export_status`
goes `pending` -> `ready`). Decoding runs on a prefetch thread (`_FramePrefetcher` in `core/video/engine.py`).

| | before | after |
|---|---:|---:|
| results usable (job `completed`) | 26.7 s | **14.9 s** (headless), 15.4 s (live view) |
| tracking loop FPS (job-reported) | 13.4-13.7 | 14.4 (headless), 14.0 (live view) |
| deliverables ready (end to end) | 26.7 s | 25.1 s (unchanged: the MP4 encode still takes ~10 s) |

Time to results dropped by about 44%, almost all from reporting completion before the exports. Prefetch decode adds only a few
percent of loop FPS on 2 vCPUs, where decode, ego-motion and flow compete for the same cores; expect more on a host with more cores.
The remaining ceiling is CPU: ego-motion and proxy resize/flow are still sequential on the main thread.
