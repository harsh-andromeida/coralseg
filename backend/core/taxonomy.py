"""
ReefNet BioCLIP zero-shot taxonomy classification and NOAA YOLO11n coral bleaching detection.
Pure NumPy and ONNX Runtime execution.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import onnxruntime as ort

from core.models import _INFERENCE_LOCK, load_bioclip_model, load_bleaching_model


GENUS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Acropora": (218, 112, 214),       # Orchid / Purple
    "Pocillopora": (255, 140, 150),    # Coral Pink
    "Porites": (255, 204, 0),          # Vibrant Golden Yellow
    "Montipora": (0, 204, 255),        # Electric Cyan
    "Stylophora": (255, 105, 180),     # Hot Pink
    "Favia / Favites": (230, 190, 0),  # Deep Amber Gold
    "Platygyra": (255, 160, 0),        # Orange
    "Turbinaria": (195, 245, 60),      # Lime Green
    "Millepora": (255, 127, 80),       # Coral Orange
    "Pavona": (72, 209, 204),          # Medium Turquoise
    "Fungia": (255, 99, 132),          # Rose Red
    "Goniopora": (147, 112, 219),      # Medium Slate Purple
    "Seriatopora": (255, 182, 193),    # Light Pink
    "Lobophyllia": (186, 85, 211),     # Medium Orchid
    "Galaxea": (64, 224, 208),         # Turquoise
    "Diploastrea": (240, 220, 130),    # Pale Gold
}


def get_genus_color(genus: str) -> Tuple[int, int, int]:
    """Returns the canonical consistent RGB color for a coral genus."""
    if genus in GENUS_COLORS:
        return GENUS_COLORS[genus]
    import hashlib
    h = int(hashlib.md5(genus.encode("utf-8")).hexdigest(), 16)
    r = 60 + (h % 170)
    g = 60 + ((h >> 8) % 170)
    b = 60 + ((h >> 16) % 170)
    return (r, g, b)


CORAL_TAXONOMY_CANDIDATES = [
    {"label": "Acropora (Staghorn Coral)", "genus": "Acropora", "common": "Staghorn Coral", "form": "Branching", "color": GENUS_COLORS["Acropora"]},
    {"label": "Acropora (Table Coral)", "genus": "Acropora", "common": "Table Coral", "form": "Tabular", "color": GENUS_COLORS["Acropora"]},
    {"label": "Pocillopora (Cauliflower Coral)", "genus": "Pocillopora", "common": "Cauliflower Coral", "form": "Branching", "color": GENUS_COLORS["Pocillopora"]},
    {"label": "Porites (Finger Coral)", "genus": "Porites", "common": "Finger Coral", "form": "Columnar", "color": GENUS_COLORS["Porites"]},
    {"label": "Porites (Massive / Lobe Coral)", "genus": "Porites", "common": "Massive / Lobe Coral", "form": "Massive", "color": GENUS_COLORS["Porites"]},
    {"label": "Montipora (Plate / Encrusting Coral)", "genus": "Montipora", "common": "Plate / Encrusting Coral", "form": "Foliose / Encrusting", "color": GENUS_COLORS["Montipora"]},
    {"label": "Stylophora (Cat's Paw Coral)", "genus": "Stylophora", "common": "Cat's Paw Coral", "form": "Branching", "color": GENUS_COLORS["Stylophora"]},
    {"label": "Favia / Favites (Brain / Moon Coral)", "genus": "Favia / Favites", "common": "Brain / Moon Coral", "form": "Massive", "color": GENUS_COLORS["Favia / Favites"]},
    {"label": "Platygyra (Maze / Brain Coral)", "genus": "Platygyra", "common": "Maze Coral", "form": "Meandering", "color": GENUS_COLORS["Platygyra"]},
    {"label": "Turbinaria (Scroll / Disc Coral)", "genus": "Turbinaria", "common": "Scroll / Disc Coral", "form": "Foliose", "color": GENUS_COLORS["Turbinaria"]},
    {"label": "Millepora (Fire Coral)", "genus": "Millepora", "common": "Fire Coral (Hydrocoral)", "form": "Branching / Encrusting", "color": GENUS_COLORS["Millepora"]},
    {"label": "Pavona (Cactus / Leaf Coral)", "genus": "Pavona", "common": "Cactus Coral", "form": "Foliose", "color": GENUS_COLORS["Pavona"]},
    {"label": "Fungia (Mushroom Coral)", "genus": "Fungia", "common": "Mushroom Coral", "form": "Solitary", "color": GENUS_COLORS["Fungia"]},
    {"label": "Goniopora (Flowerpot Coral)", "genus": "Goniopora", "common": "Flowerpot Coral", "form": "Massive", "color": GENUS_COLORS["Goniopora"]},
    {"label": "Seriatopora (Bird's Nest Coral)", "genus": "Seriatopora", "common": "Bird's Nest Coral", "form": "Branching", "color": GENUS_COLORS["Seriatopora"]},
    {"label": "Lobophyllia (Lobed Brain Coral)", "genus": "Lobophyllia", "common": "Lobed Brain Coral", "form": "Flabello-meandroid", "color": GENUS_COLORS["Lobophyllia"]},
    {"label": "Galaxea (Star Coral)", "genus": "Galaxea", "common": "Star Coral", "form": "Massive", "color": GENUS_COLORS["Galaxea"]},
    {"label": "Diploastrea (Honeycomb Coral)", "genus": "Diploastrea", "common": "Honeycomb Coral", "form": "Massive", "color": GENUS_COLORS["Diploastrea"]},
]


def classify_coral_taxonomy(
    images: List[Image.Image],
    bioclip_bundle: Optional[Dict[str, Any]] = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Classifies coral taxonomy for a list of PIL Images using ReefNet BioCLIP ONNX.
    Uses pre-computed text feature projection via pure NumPy matrix multiplication.
    """
    if not images:
        return []

    if bioclip_bundle is None:
        bioclip_bundle = load_bioclip_model()

    session = bioclip_bundle["session"]
    text_features = bioclip_bundle["text_features"]  # (18, 512)

    # Standard OpenCLIP image preprocessing in pure NumPy
    clip_mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    clip_std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    batch_tensors = []
    for img in images:
        resized = img.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        arr = np.array(resized, dtype=np.float32) / 255.0
        norm = (arr - clip_mean) / clip_std
        batch_tensors.append(norm.transpose(2, 0, 1))

    batch_np = np.stack(batch_tensors, axis=0).astype(np.float32)

    with _INFERENCE_LOCK:
        image_features = session.run(None, {"image": batch_np})[0]  # (B, 512)

    # Normalization
    norms = np.linalg.norm(image_features, axis=-1, keepdims=True)
    image_features = image_features / np.maximum(norms, 1e-6)

    # Cosine similarity matrix dot product (B, 18)
    with np.errstate(all="ignore"):
        logits = (image_features @ text_features.T) * 100.0
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    results = []
    for i in range(len(images)):
        prob_row = probs[i]
        ranked_indices = np.argsort(prob_row)[::-1][:top_k]

        candidates_ranked = []
        for idx in ranked_indices:
            cand = CORAL_TAXONOMY_CANDIDATES[idx]
            conf = float(prob_row[idx])
            candidates_ranked.append({
                "label": cand["label"],
                "genus": cand["genus"],
                "common": cand["common"],
                "form": cand["form"],
                "color": cand["color"],
                "confidence": round(conf, 4),
                "confidence_pct": round(conf * 100.0, 1),
            })

        top1 = candidates_ranked[0]
        results.append({
            "primary_label": top1["label"],
            "genus": top1["genus"],
            "growth_form": top1["form"],
            "confidence": top1["confidence"],
            "confidence_pct": top1["confidence_pct"],
            "color": top1["color"],
            "all_candidates": candidates_ranked,
        })

    return results


def detect_coral_bleaching(
    images: List[Image.Image],
    bleaching_model: Optional[ort.InferenceSession] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluates coral health and bleaching for a list of PIL Images using NOAA YOLO11n ONNX.
    Class 0: Healthy, Class 1: Bleached (CORAL_BL).
    """
    if not images:
        return []

    if bleaching_model is None:
        bleaching_model = load_bleaching_model()

    batch_tensors = []
    for img in images:
        resized = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        arr = (np.array(resized, dtype=np.float32) / 255.0).transpose(2, 0, 1)
        batch_tensors.append(arr)

    batch_np = np.stack(batch_tensors, axis=0).astype(np.float32)

    with _INFERENCE_LOCK:
        inp_name = bleaching_model.get_inputs()[0].name
        logits = bleaching_model.run(None, {inp_name: batch_np})[0]  # (B, 2)

    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    results = []
    for i in range(len(images)):
        p_healthy = float(probs[i, 0])
        p_bleached = float(probs[i, 1])

        is_bleached = p_bleached > p_healthy
        top_conf = p_bleached if is_bleached else p_healthy
        status = "Bleached" if is_bleached else "Healthy"

        results.append({
            "status": status,
            "is_bleached": is_bleached,
            "confidence": round(top_conf, 4),
            "confidence_pct": round(top_conf * 100.0, 1),
            "bleached_prob": round(p_bleached, 4),
            "healthy_prob": round(p_healthy, 4),
            "bleached_pct": round(p_bleached * 100.0, 1),
            "healthy_pct": round(p_healthy * 100.0, 1),
        })

    return results


def enrich_masks_with_taxonomy_and_bleaching(
    image: Union[np.ndarray, Image.Image],
    masks_info: List[Dict[str, Any]],
    bioclip_bundle: Optional[Dict[str, Any]] = None,
    bleaching_model: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Crops each segmented coral instance and evaluates taxonomy and bleaching severity.
    Also evaluates the entire coral reef scene.
    """
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image).convert("RGB")
    else:
        pil_img = image.convert("RGB")

    w, h = pil_img.size

    # 1. Full Image Evaluation
    full_tax = classify_coral_taxonomy([pil_img], bioclip_bundle=bioclip_bundle, top_k=5)[0]
    full_bleach = detect_coral_bleaching([pil_img], bleaching_model=bleaching_model)[0]
    full_image_eval = {
        "taxonomy": full_tax,
        "bleaching": full_bleach,
    }

    if not masks_info:
        health_summary = {
            "total_corals": 0,
            "bleached_count": 0,
            "healthy_count": 0,
            "bleaching_prevalence_pct": 0.0,
            "bleached_area_pct": 0.0,
            "dominant_taxon": full_tax["primary_label"],
            "dominant_genus": full_tax["genus"],
            "risk_level": "Severe Bleaching Event" if full_bleach["is_bleached"] else "Reef Healthy / Low Risk",
            "risk_color": "#ef4444" if full_bleach["is_bleached"] else "#10b981",
            "risk_icon": "error" if full_bleach["is_bleached"] else "check_circle",
            "taxa_distribution": {},
        }
        return masks_info, full_image_eval, health_summary

    # 2. Extract crops for each segmented instance
    crop_images = []
    for mask_dict in masks_info:
        x, y, bw, bh = mask_dict["bbox"]
        pad_x = int(bw * 0.10)
        pad_y = int(bh * 0.10)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + bw + pad_x)
        y1 = min(h, y + bh + pad_y)

        if (x1 - x0) < 10 or (y1 - y0) < 10:
            crop = pil_img.crop((x, y, max(x + 10, x + bw), max(y + 10, y + bh)))
        else:
            crop = pil_img.crop((x0, y0, x1, y1))
        crop_images.append(crop)

    tax_results = classify_coral_taxonomy(crop_images, bioclip_bundle=bioclip_bundle, top_k=3)
    bleach_results = detect_coral_bleaching(crop_images, bleaching_model=bleaching_model)

    enriched_masks = []
    bleached_count = 0
    bleached_area = 0
    total_area = 0
    taxa_counts: Dict[str, int] = {}

    for i, mask_dict in enumerate(masks_info):
        updated = dict(mask_dict)
        tax = dict(tax_results[i])
        bleach = bleach_results[i]

        genus = tax.get("genus", "Coral")
        tax_color = get_genus_color(genus)
        tax["color"] = tax_color

        updated["taxonomy"] = tax
        updated["bleaching"] = bleach

        # Preserve instance colors for Colony Instances mode
        inst_color = mask_dict.get("instance_color_rgb", mask_dict.get("color_rgb", (0, 255, 200)))
        inst_hex = mask_dict.get("instance_color_hex", mask_dict.get("color_hex", f"#{inst_color[0]:02x}{inst_color[1]:02x}{inst_color[2]:02x}"))
        updated["instance_color_rgb"] = inst_color
        updated["instance_color_hex"] = inst_hex

        # Taxonomy color (consistent per coral genus)
        updated["taxonomy_color_rgb"] = tax_color
        updated["taxonomy_color_hex"] = f"#{tax_color[0]:02x}{tax_color[1]:02x}{tax_color[2]:02x}"

        # Default color matches instance color so Colony Instances mode stays distinct
        updated["color"] = inst_color
        updated["color_rgb"] = inst_color
        updated["color_hex"] = inst_hex

        mask_area = updated.get("area_px", updated.get("area", 0))
        if bleach["is_bleached"]:
            bleached_count += 1
            bleached_area += mask_area
        total_area += mask_area

        taxa_counts[genus] = taxa_counts.get(genus, 0) + 1

        enriched_masks.append(updated)

    total_corals = len(enriched_masks)
    healthy_count = total_corals - bleached_count
    prevalence_pct = round((bleached_count / max(total_corals, 1)) * 100.0, 1)
    bleached_area_pct = round((bleached_area / max(total_area, 1)) * 100.0, 1)

    dominant_genus = max(taxa_counts.items(), key=lambda item: item[1])[0] if taxa_counts else full_tax["genus"]

    if prevalence_pct >= 50.0:
        risk_level = "Severe Bleaching Crisis"
        risk_color = "#ef4444"
        risk_icon = "error"
    elif prevalence_pct >= 20.0 or full_bleach["is_bleached"]:
        risk_level = "Moderate Bleaching Alert"
        risk_color = "#f59e0b"
        risk_icon = "warning"
    else:
        risk_level = "Reef Healthy / Low Risk"
        risk_color = "#10b981"
        risk_icon = "check_circle"

    health_summary = {
        "total_corals": total_corals,
        "bleached_count": bleached_count,
        "healthy_count": healthy_count,
        "bleaching_prevalence_pct": prevalence_pct,
        "bleached_area_pct": bleached_area_pct,
        "dominant_genus": dominant_genus,
        "dominant_taxon": full_tax["primary_label"],
        "taxa_distribution": taxa_counts,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_icon": risk_icon,
    }

    return enriched_masks, full_image_eval, health_summary
