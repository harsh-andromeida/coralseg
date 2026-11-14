### NVIDIA A10  (sample_reef_transect_25fps.mp4, 202 frames, 8.08s)

- ORT providers: `['CUDAExecutionProvider', 'CPUExecutionProvider']`   tracker device: `cuda` dtype: `torch.bfloat16`
- cold run wall 72.45s  |  warm median wall 64.747s
- **e2e 3.12 FPS** | pipeline 3.13 FPS | analysis (no export) 3.32 FPS | tracker 4.17 FPS | realtime x0.125
- keyframes 9 | corals 27 | peak VRAM 434.7 MiB
- cold model load: {'coralscop': 1.161, 'bioclip': 0.589, 'bleaching': 0.181}

| stage | sec | % wall | ms/frame |
|---|---:|---:|---:|
| submit + 25fps standardise | 0.014 | 0.0 | 0.07 |
| decode / ego-motion prep | 1.914 | 3.0 | 9.48 |
| SAM2 propagation (GPU) | 49.769 | 76.9 | 246.38 |
| mask transfer GPU->CPU | 0.066 | 0.1 | 0.33 |
| geometry (contours) | 0.899 | 1.4 | 4.45 |
| keyframe discovery (CoralSCOP) | 5.539 | 8.6 | 27.42 |
| classification (async BioCLIP+YOLO) | 0.504 | 0.8 | 2.50 |
| persistence (SQLite) | 0.131 | 0.2 | 0.65 |
| export CSV+COCO | 0.291 | 0.4 | 1.44 |
| export annotated MP4 | 3.402 | 5.3 | 16.84 |
| unaccounted / idle | 2.599 | 4.0 | 12.87 |

| call | n | mean ms | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|
| bioclip_taxonomy_batch | 10 | 42.23 | 40.5 | 77.97 | 77.97 |
| crop_regions | 10 | 0.24 | 0.21 | 0.51 | 0.51 |
| export_all | 1 | 3693.07 | 3693.07 | 3693.07 | 3693.07 |
| keyframe_segmentation | 10 | 553.92 | 551.73 | 567.95 | 567.95 |
| sam2_init_video | 2 | 623.76 | 563.99 | 683.53 | 683.53 |
| sam2_load | 2 | 0.0 | 0.0 | 0.0 | 0.0 |
| sam2_propagate_per_frame | 207 | 239.9 | 226.63 | 584.8 | 800.01 |
| sam2_register_mask | 144 | 13.17 | 12.93 | 17.57 | 18.93 |
| sam2_retire_objects | 1 | 4.01 | 4.01 | 4.01 | 4.01 |
| yolo_bleaching_batch | 10 | 8.19 | 6.51 | 13.58 | 13.58 |
