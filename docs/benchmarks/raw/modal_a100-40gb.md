### NVIDIA A100-SXM4-40GB  (sample_reef_transect_25fps.mp4, 202 frames, 8.08s)

- ORT providers: `['CUDAExecutionProvider', 'CPUExecutionProvider']`   tracker device: `cuda` dtype: `torch.bfloat16`
- cold run wall 90.96s  |  warm median wall 80.606s
- **e2e 2.51 FPS** | pipeline 2.51 FPS | analysis (no export) 2.7 FPS | tracker 3.19 FPS | realtime x0.1
- keyframes 9 | corals 28 | peak VRAM 434.7 MiB
- cold model load: {'coralscop': 2.336, 'bioclip': 0.81, 'bleaching': 0.106}

| stage | sec | % wall | ms/frame |
|---|---:|---:|---:|
| submit + 25fps standardise | 0.015 | 0.0 | 0.07 |
| decode / ego-motion prep | 2.161 | 2.7 | 10.70 |
| SAM2 propagation (GPU) | 64.774 | 80.4 | 320.66 |
| mask transfer GPU->CPU | 0.141 | 0.2 | 0.70 |
| geometry (contours) | 1.248 | 1.5 | 6.18 |
| keyframe discovery (CoralSCOP) | 3.110 | 3.9 | 15.40 |
| classification (async BioCLIP+YOLO) | 0.505 | 0.6 | 2.50 |
| persistence (SQLite) | 0.175 | 0.2 | 0.87 |
| export CSV+COCO | 0.391 | 0.5 | 1.94 |
| export annotated MP4 | 5.157 | 6.4 | 25.53 |
| unaccounted / idle | 3.313 | 4.1 | 16.40 |

| call | n | mean ms | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|
| bioclip_taxonomy_batch | 10 | 38.14 | 40.83 | 60.97 | 60.97 |
| crop_regions | 10 | 0.31 | 0.3 | 0.62 | 0.62 |
| export_all | 1 | 5547.87 | 5547.87 | 5547.87 | 5547.87 |
| keyframe_segmentation | 10 | 310.97 | 310.25 | 328.53 | 328.53 |
| sam2_init_video | 2 | 903.53 | 878.6 | 928.46 | 928.46 |
| sam2_load | 2 | 0.0 | 0.0 | 0.0 | 0.0 |
| sam2_propagate_per_frame | 207 | 313.17 | 294.3 | 714.43 | 1024.81 |
| sam2_register_mask | 144 | 16.12 | 15.68 | 21.53 | 24.67 |
| sam2_retire_objects | 1 | 5.46 | 5.46 | 5.46 | 5.46 |
| yolo_bleaching_batch | 10 | 12.38 | 10.66 | 31.79 | 31.79 |
