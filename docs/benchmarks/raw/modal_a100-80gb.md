### NVIDIA A100 80GB PCIe  (sample_reef_transect_25fps.mp4, 202 frames, 8.08s)

- ORT providers: `['CUDAExecutionProvider', 'CPUExecutionProvider']`   tracker device: `cuda` dtype: `torch.bfloat16`
- cold run wall 86.755s  |  warm median wall 76.674s
- **e2e 2.63 FPS** | pipeline 2.64 FPS | analysis (no export) 2.79 FPS | tracker 3.25 FPS | realtime x0.105
- keyframes 9 | corals 28 | peak VRAM 434.7 MiB
- cold model load: {'coralscop': 2.038, 'bioclip': 0.837, 'bleaching': 0.205}

| stage | sec | % wall | ms/frame |
|---|---:|---:|---:|
| submit + 25fps standardise | 0.015 | 0.0 | 0.07 |
| decode / ego-motion prep | 2.273 | 3.0 | 11.25 |
| SAM2 propagation (GPU) | 63.871 | 83.3 | 316.19 |
| mask transfer GPU->CPU | 0.083 | 0.1 | 0.41 |
| geometry (contours) | 1.178 | 1.5 | 5.83 |
| keyframe discovery (CoralSCOP) | 2.343 | 3.1 | 11.60 |
| classification (async BioCLIP+YOLO) | 0.482 | 0.6 | 2.39 |
| persistence (SQLite) | 0.171 | 0.2 | 0.85 |
| export CSV+COCO | 0.343 | 0.4 | 1.70 |
| export annotated MP4 | 3.902 | 5.1 | 19.32 |
| unaccounted / idle | 2.397 | 3.1 | 11.87 |

| call | n | mean ms | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|
| bioclip_taxonomy_batch | 10 | 39.19 | 42.31 | 57.88 | 57.88 |
| crop_regions | 10 | 0.21 | 0.19 | 0.42 | 0.42 |
| export_all | 1 | 4244.64 | 4244.64 | 4244.64 | 4244.64 |
| keyframe_segmentation | 10 | 234.28 | 230.83 | 241.86 | 241.86 |
| sam2_init_video | 2 | 485.36 | 463.22 | 507.51 | 507.51 |
| sam2_load | 2 | 0.0 | 0.0 | 0.0 | 0.0 |
| sam2_propagate_per_frame | 207 | 307.8 | 288.88 | 728.56 | 1028.15 |
| sam2_register_mask | 144 | 15.12 | 14.85 | 21.11 | 23.42 |
| sam2_retire_objects | 1 | 6.29 | 6.29 | 6.29 | 6.29 |
| yolo_bleaching_batch | 10 | 9.01 | 8.09 | 15.8 | 15.8 |
