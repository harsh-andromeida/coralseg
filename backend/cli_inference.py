#!/usr/bin/env python3
"""
Andromeida Reef Vision Studio - Standalone CLI Inference Runner
Run CoralSCOP SAM segmentation, BioCLIP taxonomy, and NOAA bleaching assessment
from the terminal without GUI.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.device import get_optimal_device, get_device_info
from core.models import load_coralscop_model, load_bioclip_model, load_bleaching_model
from core.segmentation import run_segmentation
from core.taxonomy import enrich_masks_with_taxonomy_and_bleaching
from core.visualization import create_segmentation_overlay, generate_distinct_colors
from core.export import build_coco_json


def main():
    parser = argparse.ArgumentParser(description="ReefVision CLI Inference")
    parser.add_argument("--image", required=True, help="Path to input coral image")
    parser.add_argument("--output-dir", default="output", help="Directory to save output files")
    parser.add_argument("--points-per-side", type=int, default=16, help="Grid density (default: 16)")
    parser.add_argument("--iou-thresh", type=float, default=0.50, help="IoU threshold (default: 0.50)")
    parser.add_argument("--stability-thresh", type=float, default=0.50, help="Stability threshold (default: 0.50)")
    parser.add_argument("--min-area-px", type=int, default=100, help="Minimum mask area (default: 100)")
    parser.add_argument("--device", default="auto", help="Device preference (auto, gpu, multithread_cpu, cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Quick verification without full inference")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: File not found: {args.image}")
        sys.exit(1)

    print(f"[1/4] Loading image '{args.image}'...")
    image = Image.open(args.image).convert("RGB")
    print(f"      Resolution: {image.width}x{image.height}")

    if args.dry_run:
        print("[DRY RUN] Input verified successfully.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    dev_info = get_device_info(args.device)
    print(f"[2/4] Initializing inference models: {dev_info['mode_label']} (Provider: {dev_info['device']})...")
    sam_bundle = load_coralscop_model(device=args.device)
    bioclip_bundle = load_bioclip_model(device=args.device)
    bleaching_model = load_bleaching_model(device=args.device)

    print("[3/4] Running CoralSCOP instance segmentation...")
    masks_info, _ = run_segmentation(
        model=sam_bundle,
        image=image,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.iou_thresh,
        stability_score_thresh=args.stability_thresh,
        min_mask_region_area=0,
    )

    # Filter by min_area_px
    raw_filtered = [m for m in masks_info if m.get("area_px", m.get("area", 0)) >= args.min_area_px]
    colors = generate_distinct_colors(len(raw_filtered))
    for i, m in enumerate(raw_filtered):
        m["id"] = i + 1
        m["color_rgb"] = colors[i]
        m["color_hex"] = f"#{colors[i][0]:02x}{colors[i][1]:02x}{colors[i][2]:02x}"

    print(f"[4/4] Enriching {len(raw_filtered)} coral segments with BioCLIP & NOAA Bleaching...")
    enriched, full_eval, health = enrich_masks_with_taxonomy_and_bleaching(
        image=image,
        masks_info=raw_filtered,
        bioclip_bundle=bioclip_bundle,
        bleaching_model=bleaching_model,
    )

    stem = Path(args.image).stem
    # Save COCO JSON
    coco = build_coco_json(os.path.basename(args.image), image.width, image.height, enriched)
    coco["health_summary"] = health
    coco_path = os.path.join(args.output_dir, f"{stem}_coco.json")
    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)

    # Save Overlay Image
    overlay_res = create_segmentation_overlay(
        image=image,
        masks_info=enriched,
        alpha=0.45,
        color_mode="taxonomy_condition",
    )
    overlay_img = Image.fromarray(overlay_res)
    overlay_path = os.path.join(args.output_dir, f"{stem}_overlay.png")
    overlay_img.save(overlay_path)

    print(f"\nCompleted successfully!")
    print(f"Total Corals Detected: {len(enriched)}")
    print(f"Bleached Corals: {health.get('bleached_count', 0)}")
    print(f"COCO JSON: {coco_path}")
    print(f"Overlay PNG: {overlay_path}")


if __name__ == "__main__":
    main()
