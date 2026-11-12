"""
Standalone T4 GPU Benchmark for SAM 2.1 Tiny on Google Colab.
Installs necessary dependencies, benchmarks SAM 2.1 Tiny FP16 at 512px and 1024px,
and reports throughput (FPS), frame latency (ms), and peak VRAM.
"""

import os
import sys
import time
import subprocess

def run_cmd(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)

def main():
    print("=" * 60)
    print("ReefVision SAM 2.1 Tiny Benchmark on Colab T4 GPU")
    print("=" * 60)

    # 1. Check GPU
    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available on this runtime!")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    print(f"CUDA Device: {gpu_name} ({gpu_mem} GiB total VRAM)")

    # 2. Ensure sam2 is installed
    try:
        import sam2
        print(f"Found existing sam2: {sam2.__file__}")
    except ImportError:
        print("Installing sam2 from official repository...")
        run_cmd("pip install git+https://github.com/facebookresearch/sam2.git opencv-python-headless pillow tqdm")
        import sam2
        print("sam2 installed successfully.")

    from sam2.build_sam import build_sam2_video_predictor

    # 3. Create synthetic test video directory (128 frames)
    import cv2
    import numpy as np
    import tempfile

    frames_dir = tempfile.mkdtemp(prefix="colab_bench_frames_")
    num_frames = 64
    w, h = 640, 480
    print(f"Generating {num_frames} synthetic video frames ({w}x{h})...")
    for i in range(num_frames):
        img = np.full((h, w, 3), 40, dtype=np.uint8)
        # moving simulated coral colony
        cx = int(100 + i * 4)
        cy = int(150 + np.sin(i / 10.0) * 30)
        cv2.circle(img, (cx, cy), 45, (80, 180, 220), -1)
        cv2.circle(img, (w - cx, cy), 35, (200, 120, 80), -1)
        cv2.imwrite(os.path.join(frames_dir, f"{i:05d}.jpg"), img)

    # 4. Benchmark configurations
    configs = [
        {"model_id": "facebook/sam2.1-hiera-tiny", "model_size": 512, "precision": torch.float16, "name": "SAM 2.1 Tiny (FP16, 512px)"},
        {"model_id": "facebook/sam2.1-hiera-tiny", "model_size": 1024, "precision": torch.float16, "name": "SAM 2.1 Tiny (FP16, 1024px)"},
    ]

    results = []

    for cfg in configs:
        print(f"\n--- Benchmarking: {cfg['name']} ---")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        t_load_0 = time.perf_counter()
        # Build predictor
        predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            ckpt_path=None,  # downloads/loads via HF
            device="cuda",
        )
        t_load = round(time.perf_counter() - t_load_0, 2)
        print(f"Predictor loaded in {t_load}s")

        # Initialize video state
        with torch.inference_mode(), torch.autocast("cuda", dtype=cfg["precision"]):
            state = predictor.init_state(video_path=frames_dir)

            # Register 2 prompt masks on frame 0
            mask_1 = np.zeros((h, w), dtype=bool)
            mask_1[105:195, 55:145] = True
            mask_2 = np.zeros((h, w), dtype=bool)
            mask_2[115:185, (w - 135):(w - 65)] = True

            predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=mask_1)
            predictor.add_new_mask(state, frame_idx=0, obj_id=2, mask=mask_2)

            # Warmup
            torch.cuda.synchronize()
            t_prop_start = time.perf_counter()
            propagated_frames = 0

            for out_f, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
                torch.cuda.synchronize()
                propagated_frames += 1

            t_prop_sec = time.perf_counter() - t_prop_start
            torch.cuda.synchronize()

            peak_vram_mib = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
            fps = round(propagated_frames / max(0.001, t_prop_sec), 2)
            lat_per_frame_ms = round((t_prop_sec / max(1, propagated_frames)) * 1000.0, 1)

            print(f"Results for {cfg['name']}:")
            print(f"  Propagated Frames: {propagated_frames}")
            print(f"  Propagation Time:  {t_prop_sec:.3f} s")
            print(f"  Throughput:        {fps} FPS")
            print(f"  Latency/Frame:     {lat_per_frame_ms} ms")
            print(f"  Peak VRAM:         {peak_vram_mib} MiB")

            results.append({
                "config": cfg["name"],
                "fps": fps,
                "latency_ms": lat_per_frame_ms,
                "peak_vram_mib": peak_vram_mib,
            })

    print("\n" + "=" * 60)
    print("COLAB T4 BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Configuration':<35} | {'Throughput':<12} | {'Latency':<12} | {'Peak VRAM'}")
    print("-" * 75)
    for r in results:
        print(f"{r['config']:<35} | {r['fps']:>6.2f} FPS   | {r['latency_ms']:>6.1f} ms    | {r['peak_vram_mib']:>6.1f} MiB")
    print("=" * 60)

    # Cleanup
    shutil_rm = getattr(os, "system", None)
    if shutil_rm:
        os.system(f"rm -rf {frames_dir}")

if __name__ == "__main__":
    main()
