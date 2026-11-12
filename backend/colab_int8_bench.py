"""
Benchmark comparing FP16 vs INT8 (torchao) on Colab T4 GPU for SAM 2.1 Tiny.
"""

import os
import sys
import time
import subprocess
import torch

def run_cmd(cmd):
    print(f"Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)

def main():
    print("=" * 60)
    print("Testing INT8 Quantization (torchao) on Colab T4 GPU")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f"CUDA Device: {gpu_name}")

    # Install dependencies
    try:
        import sam2
    except ImportError:
        run_cmd("pip install git+https://github.com/facebookresearch/sam2.git opencv-python-headless pillow tqdm")
        import sam2

    try:
        import torchao
        from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight, int8_weight_only
        print("torchao is available.")
    except ImportError:
        print("Installing torchao...")
        run_cmd("pip install torchao")
        import torchao
        from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight, int8_weight_only

    from sam2.build_sam import build_sam2_video_predictor
    import cv2
    import numpy as np
    import tempfile

    # Create synthetic frames
    frames_dir = tempfile.mkdtemp(prefix="colab_int8_frames_")
    num_frames = 64
    w, h = 640, 480
    for i in range(num_frames):
        img = np.full((h, w, 3), 40, dtype=np.uint8)
        cx = int(100 + i * 4)
        cy = int(150 + np.sin(i / 10.0) * 30)
        cv2.circle(img, (cx, cy), 45, (80, 180, 220), -1)
        cv2.circle(img, (w - cx, cy), 35, (200, 120, 80), -1)
        cv2.imwrite(os.path.join(frames_dir, f"{i:05d}.jpg"), img)

    trials = [
        {"name": "SAM 2.1 Tiny (FP16 Baseline)", "quant": None},
        {"name": "SAM 2.1 Tiny (INT8 Weight-Only)", "quant": "int8_weight_only"},
        {"name": "SAM 2.1 Tiny (INT8 Dynamic W8A8)", "quant": "int8_dynamic"},
    ]

    results = []

    for trial in trials:
        print(f"\n--- Testing: {trial['name']} ---", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            ckpt_path=None,
            device="cuda",
        )

        if trial["quant"] == "int8_weight_only":
            try:
                quantize_(predictor, int8_weight_only())
                print("Applied int8_weight_only quantization.")
            except Exception as e:
                print(f"Failed int8_weight_only: {e}")
                continue
        elif trial["quant"] == "int8_dynamic":
            try:
                quantize_(predictor, int8_dynamic_activation_int8_weight())
                print("Applied int8_dynamic_activation_int8_weight quantization.")
            except Exception as e:
                print(f"Failed int8_dynamic: {e}")
                continue

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            state = predictor.init_state(video_path=frames_dir)

            mask_1 = np.zeros((h, w), dtype=bool)
            mask_1[105:195, 55:145] = True
            mask_2 = np.zeros((h, w), dtype=bool)
            mask_2[115:185, (w - 135):(w - 65)] = True

            predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=mask_1)
            predictor.add_new_mask(state, frame_idx=0, obj_id=2, mask=mask_2)

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

            print(f"Throughput: {fps} FPS | Latency: {lat_per_frame_ms} ms | Peak VRAM: {peak_vram_mib} MiB", flush=True)

            results.append({
                "config": trial["name"],
                "fps": fps,
                "latency_ms": lat_per_frame_ms,
                "peak_vram_mib": peak_vram_mib,
            })

    print("\n" + "=" * 65)
    print("FP16 vs INT8 COMPARISON ON TESLA T4")
    print("=" * 65)
    print(f"{'Configuration':<38} | {'Throughput':<12} | {'Latency':<10} | {'Peak VRAM'}")
    print("-" * 75)
    for r in results:
        print(f"{r['config']:<38} | {r['fps']:>6.2f} FPS   | {r['latency_ms']:>6.1f} ms | {r['peak_vram_mib']:>6.1f} MiB")
    print("=" * 65)

    os.system(f"rm -rf {frames_dir}")

if __name__ == "__main__":
    main()
