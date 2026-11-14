### Tesla T4  (sample_reef_transect_25fps.mp4, 202 frames, 8.08s)

- ORT providers: `['CUDAExecutionProvider', 'CPUExecutionProvider']`   tracker device: `cuda` dtype: `torch.float16`
- cold run wall 115.167s  |  warm median wall 106.988s
- **e2e 1.89 FPS** | pipeline 1.89 FPS | analysis (no export) 2.07 FPS | tracker 2.89 FPS | realtime x0.076
- keyframes 9 | corals 27 | peak VRAM 1155.4 MiB
- cold model load: {'coralscop': 1.009, 'bioclip': 0.483, 'bleaching': 0.038}

| stage | sec | % wall | ms/frame |
|---|---:|---:|---:|
| submit + 25fps standardise | 0.015 | 0.0 | 0.07 |
| decode / ego-motion prep | 4.160 | 3.9 | 20.59 |
| SAM2 propagation (GPU) | 71.707 | 67.0 | 354.99 |
| mask transfer GPU->CPU | 0.149 | 0.1 | 0.74 |
| geometry (contours) | 1.321 | 1.2 | 6.54 |
| keyframe discovery (CoralSCOP) | 15.970 | 14.9 | 79.06 |
| classification (async BioCLIP+YOLO) | 1.622 | 1.5 | 8.03 |
| persistence (SQLite) | 0.251 | 0.2 | 1.24 |
| export CSV+COCO | 0.383 | 0.4 | 1.90 |
| export annotated MP4 | 8.881 | 8.3 | 43.97 |
| unaccounted / idle | 4.017 | 3.8 | 19.89 |

| call | n | mean ms | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|
| bioclip_taxonomy_batch | 10 | 143.36 | 114.94 | 288.66 | 288.66 |
| crop_regions | 10 | 0.33 | 0.23 | 0.82 | 0.82 |
| export_all | 1 | 9264.08 | 9264.08 | 9264.08 | 9264.08 |
| keyframe_segmentation | 10 | 1596.93 | 1577.38 | 1700.8 | 1700.8 |
| sam2_init_video | 2 | 999.6 | 814.46 | 1184.75 | 1184.75 |
| sam2_load | 2 | 0.0 | 0.0 | 0.0 | 0.0 |
| sam2_propagate_per_frame | 207 | 346.48 | 340.19 | 550.18 | 904.16 |
| sam2_register_mask | 144 | 12.98 | 11.31 | 21.26 | 33.25 |
| sam2_retire_objects | 1 | 5.54 | 5.54 | 5.54 | 5.54 |
| yolo_bleaching_batch | 10 | 18.86 | 15.9 | 43.41 | 43.41 |
