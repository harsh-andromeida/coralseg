"""
COCO JSON and dataset annotation serialization for segmented coral instances.
Supports single-image and multi-image dataset archives (.zip) with standard COCO formatting,
tabular summaries, and dataset metadata.
"""

import io
import json
import zipfile
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image
import pycocotools.mask as mask_util


def build_coco_json(
    image_name: str,
    width: int,
    height: int,
    masks_info: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Serializes detections for a single image into standard COCO JSON format."""
    images = [{
        "id": 1,
        "file_name": image_name,
        "width": width,
        "height": height,
        "date_captured": datetime.datetime.now().strftime("%Y-%m-%d"),
    }]

    categories = [
        {"id": 1, "name": "Coral Instance", "supercategory": "Benthos"},
    ]

    annotations = []
    for ann_id, mask_dict in enumerate(masks_info, start=1):
        seg = mask_dict.get("mask", mask_dict.get("segmentation"))
        if seg is None:
            continue
        rle = mask_util.encode(np.asfortranarray(seg.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")

        x, y, bw, bh = mask_dict.get("bbox", [0, 0, 1, 1])
        area = mask_dict.get("area_px", mask_dict.get("area", int(np.sum(seg))))
        iou_val = mask_dict.get("predicted_iou", 0.0)

        ann = {
            "id": mask_dict.get("id", ann_id),
            "image_id": 1,
            "category_id": 1,
            "segmentation": rle,
            "area": area,
            "bbox": [x, y, bw, bh],
            "iscrowd": 0,
            "predicted_iou": iou_val,
        }
        if "taxonomy" in mask_dict:
            ann["taxonomy"] = mask_dict["taxonomy"].get("primary_label", "Coral")
            ann["genus"] = mask_dict["taxonomy"].get("genus", "Coral")
            ann["growth_form"] = mask_dict["taxonomy"].get("growth_form", "-")
        if "bleaching" in mask_dict:
            ann["bleaching_status"] = mask_dict["bleaching"].get("status", "Healthy")
            ann["bleached_prob"] = mask_dict["bleaching"].get("bleached_prob", 0.0)

        annotations.append(ann)

    return {
        "info": {
            "description": "Andromeida ReefVision Coral Instance Segmentation",
            "version": "2.0",
            "year": datetime.datetime.now().year,
            "contributor": "Andromeida Tech Reef Vision Studio",
            "date_created": datetime.datetime.now().isoformat(),
        },
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }


def build_multi_image_coco_dataset(
    processed_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Builds a unified multi-image COCO dataset JSON structure
    across all processed images in the batch.
    """
    images_list = []
    annotations_list = []
    category_map: Dict[str, int] = {
        "Coral Instance": 1,
    }
    categories_list = [
        {"id": 1, "name": "Coral Instance", "supercategory": "Benthos"},
    ]

    ann_global_id = 1

    for img_idx, item in enumerate(processed_items, start=1):
        img_name = item["image_name"]
        w = item.get("width", 1024)
        h = item.get("height", 768)

        images_list.append({
            "id": img_idx,
            "file_name": img_name,
            "width": w,
            "height": h,
            "date_captured": datetime.datetime.now().strftime("%Y-%m-%d"),
            "corals_detected": item.get("corals_count", 0),
            "coverage_pct": item.get("coverage_pct", 0.0),
            "bleaching_prevalence_pct": item.get("bleaching_prevalence_pct", 0.0),
        })

        masks_info = item.get("masks_info", [])
        for m in masks_info:
            seg = m.get("mask", m.get("segmentation"))
            if seg is None:
                continue

            rle = mask_util.encode(np.asfortranarray(seg.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")

            tax = m.get("taxonomy", {})
            bl = m.get("bleaching", {})
            genus = tax.get("genus", "Coral")
            growth_form = tax.get("growth_form", "-")
            cond = "Bleached" if bl.get("is_bleached") else "Healthy"

            # Register genus category if new
            if genus and genus not in category_map and genus != "Coral":
                new_cat_id = len(category_map) + 1
                category_map[genus] = new_cat_id
                categories_list.append({
                    "id": new_cat_id,
                    "name": genus,
                    "supercategory": "Coral",
                })

            cat_id = category_map.get(genus, 1)
            x, y, bw, bh = m.get("bbox", [0, 0, 1, 1])
            area = m.get("area_px", m.get("area", int(np.sum(seg))))

            ann = {
                "id": ann_global_id,
                "image_id": img_idx,
                "category_id": cat_id,
                "segmentation": rle,
                "area": area,
                "bbox": [x, y, bw, bh],
                "iscrowd": 0,
                "predicted_iou": round(float(m.get("predicted_iou", 0.0)), 3),
                "genus": genus,
                "growth_form": growth_form,
                "condition": cond,
                "taxon_confidence": round(float(tax.get("confidence_pct", 0.0)), 1),
                "condition_confidence": round(float(bl.get("confidence_pct", 0.0)), 1),
            }
            annotations_list.append(ann)
            ann_global_id += 1

    return {
        "info": {
            "description": "Andromeida ReefVision Multi-Image Coral Instance Segmentation Dataset",
            "version": "2.0",
            "year": datetime.datetime.now().year,
            "contributor": "Andromeida Tech Reef Vision Studio",
            "date_created": datetime.datetime.now().isoformat(),
        },
        "licenses": [
            {
                "id": 1,
                "name": "Creative Commons Attribution 4.0 International",
                "url": "https://creativecommons.org/licenses/by/4.0/",
            }
        ],
        "images": images_list,
        "categories": categories_list,
        "annotations": annotations_list,
    }


def create_coco_dataset_zip(
    processed_items: List[Dict[str, Any]],
    image_store: Dict[str, Image.Image],
) -> bytes:
    """
    Creates an in-memory ZIP package structured as a standard segmentation dataset:
      - annotations/instances_default.json (COCO format)
      - images/<filename> (All processed source images)
      - summary.csv (Full segments census spreadsheet)
      - dataset_summary.json (Survey-wide high-level metrics)
      - README.md (Documentation and quick start instructions)
    """
    zip_buffer = io.BytesIO()

    # 1. Build COCO annotations
    coco_data = build_multi_image_coco_dataset(processed_items)
    coco_json_bytes = json.dumps(coco_data, indent=2).encode("utf-8")

    # 2. Build Tabular CSV
    csv_rows = []
    for item in processed_items:
        img_name = item["image_name"]
        for seg in item.get("segments", []):
            csv_rows.append({
                "Image Name": img_name,
                "Segment ID": seg.get("id_str", f"#{seg.get('id', '')}"),
                "Taxon Genus": seg.get("genus", "Coral"),
                "Growth Form": seg.get("growth_form", "-"),
                "Taxon Confidence (%)": seg.get("taxon_conf", 0.0),
                "Condition": seg.get("condition", "Healthy"),
                "Condition Confidence (%)": seg.get("condition_conf", 0.0),
                "Area (%)": seg.get("area_pct", 0.0),
                "Area (px)": seg.get("area_px", 0),
                "IoU Confidence": seg.get("predicted_iou", 0.0),
                "BBox [x,y,w,h]": str(seg.get("bbox", [])),
                "Centroid [x,y]": str(seg.get("centroid", [])),
            })

    df = pd.DataFrame(csv_rows)
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    # 3. High-level dataset summary
    total_imgs = len(processed_items)
    total_corals = sum(item.get("corals_count", 0) for item in processed_items)
    avg_cov = round(
        float(np.mean([item.get("coverage_pct", 0.0) for item in processed_items])), 2
    ) if total_imgs > 0 else 0.0
    avg_bleach = round(
        float(np.mean([item.get("bleaching_prevalence_pct", 0.0) for item in processed_items])), 2
    ) if total_imgs > 0 else 0.0

    summary_obj = {
        "dataset_name": "Andromeida ReefVision Coral Survey Dataset",
        "generated_at": datetime.datetime.now().isoformat(),
        "total_images_processed": total_imgs,
        "total_corals_detected": total_corals,
        "mean_coral_coverage_pct": avg_cov,
        "mean_bleaching_prevalence_pct": avg_bleach,
        "models": {
            "segmentation": "SAM ViT-B (Dense Instance Segmentation)",
            "taxonomy": "BioCLIP Coral Taxonomy (Zero-Shot Benthic Hierarchy)",
            "bleaching": "NOAA YOLO11 Bleach Assessment",
        },
        "images_summary": [
            {
                "image_name": it["image_name"],
                "corals_detected": it.get("corals_count", 0),
                "coverage_pct": it.get("coverage_pct", 0.0),
                "bleaching_pct": it.get("bleaching_prevalence_pct", 0.0),
            }
            for it in processed_items
        ],
    }
    summary_json_bytes = json.dumps(summary_obj, indent=2).encode("utf-8")

    # 4. README documentation
    readme_text = f"""# Andromeida ReefVision Coral Dataset

This dataset was generated by **Andromeida Reef Vision Studio** on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.

## Dataset Structure
```
├── annotations/
│   └── instances_default.json   # Standard COCO instance segmentation annotations
├── images/                      # Original benthic survey imagery
├── summary.csv                  # Tabular census of all detected coral colonies
├── dataset_summary.json         # Aggregated ecological health and coverage statistics
└── README.md                    # Dataset documentation
```

## Quick Start with Python
```python
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

coco = COCO("annotations/instances_default.json")
img_ids = coco.getImgIds()
print(f"Loaded {{len(img_ids)}} images and {{len(coco.getAnnIds())}} coral annotations.")
```

## Survey Summary
- **Total Images Processed**: {total_imgs}
- **Total Corals Detected**: {total_corals}
- **Mean Coral Coverage**: {avg_cov}%
- **Mean Bleaching Prevalence**: {avg_bleach}%

Generated by Andromeida Tech (https://andromeida.com)
"""
    readme_bytes = readme_text.encode("utf-8")

    # Write files into ZIP
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("annotations/instances_default.json", coco_json_bytes)
        zf.writestr("summary.csv", csv_bytes)
        zf.writestr("dataset_summary.json", summary_json_bytes)
        zf.writestr("README.md", readme_bytes)

        # Include images
        for item in processed_items:
            img_name = item["image_name"]
            pil_img = image_store.get(img_name)
            if pil_img is not None:
                img_buf = io.BytesIO()
                ext = img_name.rsplit(".", 1)[-1].lower() if "." in img_name else "png"
                fmt = "JPEG" if ext in ("jpg", "jpeg") else "PNG"
                pil_img.save(img_buf, format=fmt)
                zf.writestr(f"images/{img_name}", img_buf.getvalue())

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def export_dataset_to_directory(
    processed_items: List[Dict[str, Any]],
    image_store: Dict[str, Image.Image],
    output_dir: Union[str, Path],
    include_zip: bool = False,
) -> Dict[str, Any]:
    """
    Exports the complete structured dataset directly into a user-specified custom folder:
      - <output_dir>/annotations/instances_default.json
      - <output_dir>/images/<filename>
      - <output_dir>/summary.csv
      - <output_dir>/dataset_summary.json
      - <output_dir>/README.md
      - (optional) <output_dir>/reefvision_coco_dataset_<timestamp>.zip
    """
    out_path = Path(output_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    annotations_dir = out_path / "annotations"
    images_dir = out_path / "images"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    written_files: List[str] = []

    # 1. COCO Annotations
    coco_data = build_multi_image_coco_dataset(processed_items)
    coco_file = annotations_dir / "instances_default.json"
    with open(coco_file, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2)
    written_files.append(str(coco_file))

    # 2. Build Tabular CSV
    csv_rows = []
    for item in processed_items:
        img_name = item["image_name"]
        for seg in item.get("segments", []):
            csv_rows.append({
                "Image Name": img_name,
                "Segment ID": seg.get("id_str", f"#{seg.get('id', '')}"),
                "Taxon Genus": seg.get("genus", "Coral"),
                "Growth Form": seg.get("growth_form", "-"),
                "Taxon Confidence (%)": seg.get("taxon_conf", 0.0),
                "Condition": seg.get("condition", "Healthy"),
                "Condition Confidence (%)": seg.get("condition_conf", 0.0),
                "Area (%)": seg.get("area_pct", 0.0),
                "Area (px)": seg.get("area_px", 0),
                "IoU Confidence": seg.get("predicted_iou", 0.0),
                "BBox [x,y,w,h]": str(seg.get("bbox", [])),
                "Centroid [x,y]": str(seg.get("centroid", [])),
            })

    df = pd.DataFrame(csv_rows)
    csv_file = out_path / "summary.csv"
    df.to_csv(csv_file, index=False, encoding="utf-8")
    written_files.append(str(csv_file))

    # 3. High-level dataset summary
    total_imgs = len(processed_items)
    total_corals = sum(item.get("corals_count", 0) for item in processed_items)
    avg_cov = round(
        float(np.mean([item.get("coverage_pct", 0.0) for item in processed_items])), 2
    ) if total_imgs > 0 else 0.0
    avg_bleach = round(
        float(np.mean([item.get("bleaching_prevalence_pct", 0.0) for item in processed_items])), 2
    ) if total_imgs > 0 else 0.0

    summary_obj = {
        "dataset_name": "Andromeida ReefVision Coral Survey Dataset",
        "generated_at": datetime.datetime.now().isoformat(),
        "total_images_processed": total_imgs,
        "total_corals_detected": total_corals,
        "mean_coral_coverage_pct": avg_cov,
        "mean_bleaching_prevalence_pct": avg_bleach,
        "models": {
            "segmentation": "SAM ViT-B (Dense Instance Segmentation)",
            "taxonomy": "BioCLIP Coral Taxonomy (Zero-Shot Benthic Hierarchy)",
            "bleaching": "NOAA YOLO11 Bleach Assessment",
        },
        "images_summary": [
            {
                "image_name": it["image_name"],
                "corals_detected": it.get("corals_count", 0),
                "coverage_pct": it.get("coverage_pct", 0.0),
                "bleaching_pct": it.get("bleaching_prevalence_pct", 0.0),
            }
            for it in processed_items
        ],
    }
    summary_file = out_path / "dataset_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, indent=2)
    written_files.append(str(summary_file))

    # 4. README documentation
    readme_text = f"""# Andromeida ReefVision Coral Dataset

This dataset was generated by **Andromeida Reef Vision Studio** on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.

## Dataset Structure
```
├── annotations/
│   └── instances_default.json   # Standard COCO instance segmentation annotations
├── images/                      # Original benthic survey imagery
├── summary.csv                  # Tabular census of all detected coral colonies
├── dataset_summary.json         # Aggregated ecological health and coverage statistics
└── README.md                    # Dataset documentation
```

## Quick Start with Python
```python
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

coco = COCO("annotations/instances_default.json")
img_ids = coco.getImgIds()
print(f"Loaded {{len(img_ids)}} images and {{len(coco.getAnnIds())}} coral annotations.")
```

## Survey Summary
- **Total Images Processed**: {total_imgs}
- **Total Corals Detected**: {total_corals}
- **Mean Coral Coverage**: {avg_cov}%
- **Mean Bleaching Prevalence**: {avg_bleach}%

Generated by Andromeida Tech (https://andromeida.com)
"""
    readme_file = out_path / "README.md"
    readme_file.write_text(readme_text, encoding="utf-8")
    written_files.append(str(readme_file))

    # 5. Save source images
    images_saved = 0
    for item in processed_items:
        img_name = item["image_name"]
        pil_img = image_store.get(img_name)
        if pil_img is not None:
            dest_img = images_dir / img_name
            ext = img_name.rsplit(".", 1)[-1].lower() if "." in img_name else "png"
            fmt = "JPEG" if ext in ("jpg", "jpeg") else "PNG"
            pil_img.save(dest_img, format=fmt)
            written_files.append(str(dest_img))
            images_saved += 1

    # 6. Optional ZIP archive inside directory
    zip_created = None
    if include_zip:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"reefvision_coco_dataset_{timestamp_str}.zip"
        zip_path = out_path / zip_filename
        zip_bytes = create_coco_dataset_zip(processed_items, image_store)
        zip_path.write_bytes(zip_bytes)
        written_files.append(str(zip_path))
        zip_created = str(zip_path)

    return {
        "success": True,
        "folder_path": str(out_path),
        "total_images": total_imgs,
        "images_saved": images_saved,
        "total_corals": total_corals,
        "written_files_count": len(written_files),
        "zip_file": zip_created,
    }

