#!/usr/bin/env python3
"""Build feature_bank.npy for each image domain.

Extracts feature vectors from each domain's model over a random subset 
of dataset images. The resulting .npy files enable meaningful PCA and 
KMeans concept extraction strategies.

Usage:
    python -m scripts.build_feature_banks [--domains birds medical vision] [--max-samples 300]
"""
from __future__ import annotations

import argparse
import base64
import logging
import os
import random
import sys
import time

import numpy as np

# Ensure backend/ is on sys.path when run as a module or directly
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from config import PROBE_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# ── Dataset image collectors ─────────────────────────────────────────

_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)


def _collect_birds(max_samples: int) -> list[str]:
    """Collect random bird image paths from CUB-200-2011."""
    root = os.path.join(_PROJECT_ROOT, "datasets", "CUB_200_2011", "images")
    if not os.path.isdir(root):
        logger.warning("CUB dataset not found at %s", root)
        return []
    all_imgs: list[str] = []
    for species_dir in sorted(os.listdir(root)):
        species_path = os.path.join(root, species_dir)
        if not os.path.isdir(species_path):
            continue
        for fname in os.listdir(species_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                all_imgs.append(os.path.join(species_path, fname))
    random.shuffle(all_imgs)
    return all_imgs[:max_samples]


def _collect_medical(max_samples: int) -> list[str]:
    """Collect random medical X-ray paths from Lung Segmentation Data."""
    root = os.path.join(
        _PROJECT_ROOT, "datasets", "Lung Segmentation Data",
        "Lung Segmentation Data",
    )
    if not os.path.isdir(root):
        logger.warning("Medical dataset not found at %s", root)
        return []
    all_imgs: list[str] = []
    for split in ("Train", "Val", "Test"):
        for cls in ("COVID-19", "Non-COVID", "Normal"):
            img_dir = os.path.join(root, split, cls, "images")
            if not os.path.isdir(img_dir):
                continue
            for fname in os.listdir(img_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    all_imgs.append(os.path.join(img_dir, fname))
    random.shuffle(all_imgs)
    return all_imgs[:max_samples]


def _collect_vision(max_samples: int) -> list[str]:
    """Collect images for CLIP vision — reuse CUB + medical for diversity."""
    # Mix bird and medical images to build a diverse feature distribution
    birds = _collect_birds(max_samples // 2 + 1)
    medical = _collect_medical(max_samples // 2 + 1)
    combined = birds + medical
    random.shuffle(combined)
    return combined[:max_samples]


_COLLECTORS = {
    "birds": _collect_birds,
    "medical": _collect_medical,
    "vision": _collect_vision,
}


# ── Model loaders ────────────────────────────────────────────────────

def _load_model(domain: str):
    """Load the domain model and return the model instance."""
    if domain == "birds":
        from models.bird_model import BirdModel
        cfg = {"model_path": os.path.join(_BACKEND_DIR, "data", "models", "bird_resnet50.pth")}
        model = BirdModel(domain, cfg)
        model.load()
        return model

    if domain == "medical":
        from models.medical_model import MedicalModel
        model = MedicalModel(domain, {})
        model.load()
        return model

    if domain == "vision":
        from models.vision_model import VisionModel
        model = VisionModel(domain, {"model_id": "openai/clip-vit-base-patch32"})
        model.load()
        return model

    raise ValueError(f"Unknown domain: {domain}")


# ── Main extraction loop ─────────────────────────────────────────────

def build_feature_bank(domain: str, max_samples: int = 300) -> np.ndarray | None:
    """Build and save a feature bank for the given domain."""
    collector = _COLLECTORS.get(domain)
    if collector is None:
        logger.error("No image collector for domain %r", domain)
        return None

    logger.info("Collecting up to %d images for %s…", max_samples, domain)
    image_paths = collector(max_samples)
    if not image_paths:
        logger.error("No images found for domain %s", domain)
        return None
    logger.info("Found %d images for %s", len(image_paths), domain)

    logger.info("Loading %s model…", domain)
    model = _load_model(domain)

    features_list: list[np.ndarray] = []
    t0 = time.time()
    for i, img_path in enumerate(image_paths):
        try:
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            feat = model.extract_features(b64)
            features_list.append(feat)
        except Exception as e:
            logger.debug("Skipping %s: %s", img_path, e)
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            logger.info(
                "  [%s] %d/%d extracted (%.1f img/s)",
                domain, i + 1, len(image_paths), rate,
            )

    if not features_list:
        logger.error("No features extracted for %s", domain)
        return None

    bank = np.stack(features_list)
    logger.info(
        "Feature bank for %s: shape %s, dtype %s",
        domain, bank.shape, bank.dtype,
    )

    # Save
    out_dir = os.path.join(PROBE_DATA_DIR, domain)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "feature_bank.npy")
    np.save(out_path, bank)
    logger.info("Saved → %s", out_path)

    return bank


def main():
    parser = argparse.ArgumentParser(description="Build feature banks for PCA/KMeans concept extraction")
    parser.add_argument(
        "--domains", nargs="+", default=["birds", "medical", "vision"],
        choices=["birds", "medical", "vision"],
        help="Domains to build feature banks for",
    )
    parser.add_argument(
        "--max-samples", type=int, default=300,
        help="Max images per domain (default: 300)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    for domain in args.domains:
        logger.info("=" * 60)
        logger.info("Building feature bank for: %s", domain)
        logger.info("=" * 60)
        bank = build_feature_bank(domain, args.max_samples)
        if bank is not None:
            logger.info("✓ %s done — %d vectors of dim %d", domain, bank.shape[0], bank.shape[1])
        else:
            logger.error("✗ %s FAILED", domain)
        logger.info("")


if __name__ == "__main__":
    main()
