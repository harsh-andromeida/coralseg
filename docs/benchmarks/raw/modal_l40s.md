### NVIDIA L40S  (sample_reef_transect_25fps.mp4, 202 frames, 8.08s)

- ORT providers: `['CUDAExecutionProvider', 'CPUExecutionProvider']`   tracker device: `cuda` dtype: `torch.bfloat16`
- cold run wall 58.408s  |  warm median wall 51.182s
- **e2e 3.95 FPS** | pipeline 3.96 FPS | analysis (no export) 4.24 FPS | tracker 5.34 FPS | realtime x0.158
- keyframes 9 | corals 28 | peak VRAM 434.7 MiB
- cold model load: {'coralscop': 1.275, 'bioclip': 0.847, 'bleaching': 0.074}

| stage | sec | % wall | ms/frame |
|---|---:|---:|---:|
| submit + 25fps standardise | 0.015 | 0.0 | 0.07 |
| decode / ego-motion prep | 2.084 | 4.1 | 10.32 |
| SAM2 propagation (GPU) | 38.758 | 75.7 | 191.87 |
| mask transfer GPU->CPU | 0.066 | 0.1 | 0.33 |
| geometry (contours) | 1.046 | 2.0 | 5.18 |
| keyframe discovery (CoralSCOP) | 3.652 | 7.1 | 18.08 |
| classification (async BioCLIP+YOLO) | 0.388 | 0.8 | 1.92 |
| persistence (SQLite) | 0.132 | 0.3 | 0.65 |
| export CSV+COCO | 0.254 | 0.5 | 1.26 |
| export annotated MP4 | 3.188 | 6.2 | 15.78 |
| unaccounted / idle | 1.860 | 3.6 | 9.21 |

| call | n | mean ms | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|
| bioclip_taxonomy_batch | 10 | 30.25 | 30.21 | 46.62 | 46.62 |
| crop_regions | 10 | 0.26 | 0.26 | 0.47 | 0.47 |
| export_all | 1 | 3442.16 | 3442.16 | 3442.16 | 3442.16 |
| keyframe_segmentation | 10 | 365.19 | 361.59 | 377.86 | 377.86 |
| sam2_init_video | 2 | 552.87 | 484.63 | 621.11 | 621.11 |
| sam2_load | 2 | 0.0 | 0.0 | 0.0 | 0.0 |
| sam2_propagate_per_frame | 207 | 187.25 | 173.64 | 456.92 | 666.44 |
| sam2_register_mask | 144 | 10.1 | 9.76 | 14.53 | 16.18 |
| sam2_retire_objects | 1 | 4.63 | 4.63 | 4.63 | 4.63 |
| yolo_bleaching_batch | 10 | 8.59 | 7.56 | 15.58 | 15.58 |
