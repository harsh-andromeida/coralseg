### NVIDIA H100 80GB HBM3  (sample_reef_transect_25fps.mp4, 202 frames, 8.08s)

- ORT providers: `['CUDAExecutionProvider', 'CPUExecutionProvider']`   tracker device: `cuda` dtype: `torch.bfloat16`
- cold run wall 88.828s  |  warm median wall 65.308s
- **e2e 3.09 FPS** | pipeline 3.1 FPS | analysis (no export) 3.27 FPS | tracker 3.8 FPS | realtime x0.124
- keyframes 9 | corals 27 | peak VRAM 434.0 MiB
- cold model load: {'coralscop': 2.737, 'bioclip': 1.302, 'bleaching': 0.207}

| stage | sec | % wall | ms/frame |
|---|---:|---:|---:|
| submit + 25fps standardise | 0.016 | 0.0 | 0.08 |
| decode / ego-motion prep | 1.976 | 3.0 | 9.78 |
| SAM2 propagation (GPU) | 54.324 | 83.2 | 268.93 |
| mask transfer GPU->CPU | 0.173 | 0.3 | 0.86 |
| geometry (contours) | 0.960 | 1.5 | 4.75 |
| keyframe discovery (CoralSCOP) | 1.787 | 2.7 | 8.85 |
| classification (async BioCLIP+YOLO) | 0.495 | 0.8 | 2.45 |
| persistence (SQLite) | 0.123 | 0.2 | 0.61 |
| export CSV+COCO | 0.235 | 0.4 | 1.16 |
| export annotated MP4 | 3.158 | 4.8 | 15.63 |
| unaccounted / idle | 2.382 | 3.6 | 11.79 |

| call | n | mean ms | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|
| bioclip_taxonomy_batch | 10 | 41.62 | 28.47 | 100.53 | 100.53 |
| crop_regions | 10 | 0.28 | 0.29 | 0.48 | 0.48 |
| export_all | 1 | 3393.51 | 3393.51 | 3393.51 | 3393.51 |
| keyframe_segmentation | 10 | 178.73 | 171.67 | 197.29 | 197.29 |
| sam2_init_video | 2 | 681.82 | 580.91 | 782.72 | 782.72 |
| sam2_load | 2 | 0.0 | 0.0 | 0.0 | 0.0 |
| sam2_propagate_per_frame | 207 | 262.96 | 203.76 | 1153.27 | 1449.56 |
| sam2_register_mask | 143 | 11.22 | 10.27 | 16.31 | 30.59 |
| sam2_retire_objects | 1 | 5.67 | 5.67 | 5.67 | 5.67 |
| yolo_bleaching_batch | 10 | 7.91 | 7.41 | 14.86 | 14.86 |
