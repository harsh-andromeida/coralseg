### NVIDIA L4  (sample_reef_transect_25fps.mp4, 202 frames, 8.08s)

- ORT providers: `['CUDAExecutionProvider', 'CPUExecutionProvider']`   tracker device: `cuda` dtype: `torch.bfloat16`
- cold run wall 122.232s  |  warm median wall 106.446s
- **e2e 1.9 FPS** | pipeline 1.9 FPS | analysis (no export) 1.99 FPS | tracker 2.47 FPS | realtime x0.076
- keyframes 9 | corals 27 | peak VRAM 434.7 MiB
- cold model load: {'coralscop': 2.011, 'bioclip': 0.807, 'bleaching': 0.201}

| stage | sec | % wall | ms/frame |
|---|---:|---:|---:|
| submit + 25fps standardise | 0.018 | 0.0 | 0.09 |
| decode / ego-motion prep | 2.230 | 2.1 | 11.04 |
| SAM2 propagation (GPU) | 84.044 | 79.0 | 416.06 |
| mask transfer GPU->CPU | 0.147 | 0.1 | 0.73 |
| geometry (contours) | 1.228 | 1.2 | 6.08 |
| keyframe discovery (CoralSCOP) | 9.680 | 9.1 | 47.92 |
| classification (async BioCLIP+YOLO) | 0.660 | 0.6 | 3.27 |
| persistence (SQLite) | 0.206 | 0.2 | 1.02 |
| export CSV+COCO | 0.414 | 0.4 | 2.05 |
| export annotated MP4 | 4.149 | 3.9 | 20.54 |
| unaccounted / idle | 4.073 | 3.8 | 20.16 |

| call | n | mean ms | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|
| bioclip_taxonomy_batch | 10 | 51.56 | 52.11 | 84.82 | 84.82 |
| crop_regions | 10 | 0.37 | 0.3 | 0.82 | 0.82 |
| export_all | 1 | 4562.92 | 4562.92 | 4562.92 | 4562.92 |
| keyframe_segmentation | 10 | 967.95 | 964.14 | 982.44 | 982.44 |
| sam2_init_video | 2 | 995.86 | 920.97 | 1070.74 | 1070.74 |
| sam2_load | 2 | 0.0 | 0.0 | 0.0 | 0.0 |
| sam2_propagate_per_frame | 207 | 405.49 | 388.71 | 936.26 | 1458.72 |
| sam2_register_mask | 144 | 19.49 | 19.16 | 27.51 | 30.48 |
| sam2_retire_objects | 1 | 5.95 | 5.95 | 5.95 | 5.95 |
| yolo_bleaching_batch | 10 | 14.38 | 12.45 | 32.87 | 32.87 |
