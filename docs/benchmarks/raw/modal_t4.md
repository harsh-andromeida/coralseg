### Tesla T4  (sample_reef_transect_25fps.mp4, 202 frames, 8.08s)

- ORT providers: `['CUDAExecutionProvider', 'CPUExecutionProvider']`   tracker device: `cuda` dtype: `torch.float16`
- cold run wall 101.774s  |  warm median wall 96.7s
- **e2e 2.09 FPS** | pipeline 2.09 FPS | analysis (no export) 2.22 FPS | tracker 3.02 FPS | realtime x0.084
- keyframes 9 | corals 27 | peak VRAM 907.7 MiB
- cold model load: {'coralscop': 2.032, 'bioclip': 0.689, 'bleaching': 0.042}

| stage | sec | % wall | ms/frame |
|---|---:|---:|---:|
| submit + 25fps standardise | 0.017 | 0.0 | 0.08 |
| decode / ego-motion prep | 2.270 | 2.3 | 11.24 |
| SAM2 propagation (GPU) | 68.286 | 70.6 | 338.05 |
| mask transfer GPU->CPU | 0.245 | 0.3 | 1.21 |
| geometry (contours) | 1.785 | 1.8 | 8.84 |
| keyframe discovery (CoralSCOP) | 13.709 | 14.2 | 67.87 |
| classification (async BioCLIP+YOLO) | 1.556 | 1.6 | 7.70 |
| persistence (SQLite) | 0.198 | 0.2 | 0.98 |
| export CSV+COCO | 0.448 | 0.5 | 2.22 |
| export annotated MP4 | 5.106 | 5.3 | 25.28 |
| unaccounted / idle | 4.412 | 4.6 | 21.84 |

| call | n | mean ms | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|
| bioclip_taxonomy_batch | 10 | 137.82 | 102.63 | 293.88 | 293.88 |
| crop_regions | 10 | 0.4 | 0.33 | 0.86 | 0.86 |
| export_all | 1 | 5554.17 | 5554.17 | 5554.17 | 5554.17 |
| keyframe_segmentation | 10 | 1370.85 | 1346.15 | 1516.71 | 1516.71 |
| sam2_init_video | 2 | 1414.62 | 1183.84 | 1645.4 | 1645.4 |
| sam2_load | 2 | 0.0 | 0.0 | 0.0 | 0.0 |
| sam2_propagate_per_frame | 207 | 330.78 | 323.29 | 573.17 | 1163.67 |
| sam2_register_mask | 144 | 19.27 | 17.94 | 28.57 | 60.41 |
| sam2_retire_objects | 1 | 4.34 | 4.34 | 4.34 | 4.34 |
| yolo_bleaching_batch | 10 | 17.73 | 14.36 | 45.66 | 45.66 |
