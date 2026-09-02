#!/usr/bin/env python3
"""Train real logistic probes for birds and medical domains.

Usage
-----
    cd backend
    python -m scripts.train_probes_real [--domain birds|medical|all]
                                        [--max-samples 2000]
                                        [--batch-size 32]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
_SAVE_DIR = os.path.join(_BACKEND_DIR, "data")

# ═══════════════════════════════════════════════════════════════════════
# CUB-200 attribute → concept mapping (built from attributes.txt)
# ═══════════════════════════════════════════════════════════════════════

# CUB attribute IDs for "blue" within each body-part color group,
# used as the base offset for other colors.
_COLOR_BLUE_IDS = [10, 25, 40, 59, 80, 106, 121, 136, 153, 168, 183, 198,
                   249, 264, 279, 294]

# Color offsets within each 15-color body-part group
_COLOR_OFFSET = {
    "blue": 0, "brown": 1, "iridescent": 2, "purple": 3, "rufous": 4,
    "grey": 5, "yellow": 6, "olive": 7, "green": 8, "pink": 9,
    "orange": 10, "black": 11, "white": 12, "red": 13, "buff": 14,
}


def _color_attr_ids(color: str) -> list[int]:
    """Return all CUB attribute IDs (1-indexed) for a given color."""
    offset = _COLOR_OFFSET[color]
    return [base + offset for base in _COLOR_BLUE_IDS]


# Mapping: our 28 bird concept names → list of CUB attribute IDs (1-indexed)
_BIRD_CONCEPT_TO_CUB = {
    # Colors — aggregate across all body parts
    "has_red":        _color_attr_ids("red"),
    "has_blue":       _color_attr_ids("blue"),
    "has_yellow":     _color_attr_ids("yellow"),
    "has_orange":     _color_attr_ids("orange"),
    "has_black":      _color_attr_ids("black"),
    "has_white":      _color_attr_ids("white"),
    "has_brown":      _color_attr_ids("brown"),
    "has_grey":       _color_attr_ids("grey"),
    "has_green":      _color_attr_ids("green"),
    "has_iridescent": _color_attr_ids("iridescent"),
    # Bill shape (attrs 1-9)
    "curved_bill":  [1],        # curved_(up_or_down)
    "hooked_bill":  [3, 5],     # hooked, hooked_seabird
    "dagger_bill":  [2],        # dagger
    "cone_bill":    [8],        # cone
    # Bill length
    "long_bill":    [151],      # longer_than_head
    # Patterns (aggregate across breast, back, tail, belly, wing patterns)
    "spotted_pattern":       [56, 95, 238, 242, 246, 310],
    "striped_pattern":       [57, 104, 239, 243, 247, 311],
    "multi_colored_pattern": [58, 240, 244, 248, 312],
    # Head features
    "has_crest":      [97],     # crested
    "has_mask":       [98],     # masked
    "has_eyering":    [101],    # eyering
    "has_eye_stripe": [100, 103],  # eyebrow, eyeline
    "has_cap":        [105],    # capped
    # Tail shape
    "forked_tail":    [74],     # forked_tail
    # Wing shape
    "broad_wings":    [215],    # broad-wings
    "long_wings":     [217],    # long-wings
    # Size
    "large_bird":     [218, 220],  # large + very_large
    "small_bird":     [219, 222],  # small + very_small
}


# ═══════════════════════════════════════════════════════════════════════
# Medical concept → class membership heuristic
# ═══════════════════════════════════════════════════════════════════════

# For each concept, define P(concept=1 | class) probabilities used to
# generate noisy labels.  This approximates real annotation when we only
# have class-level supervision.
_MEDICAL_CONCEPT_CLASS_PRIORS = {
    # concept_name: {class_name: probability}
    "Consolidation":           {"COVID-19": 0.75, "Non-COVID": 0.30, "Normal": 0.02},
    "Ground Glass Opacity":    {"COVID-19": 0.85, "Non-COVID": 0.15, "Normal": 0.02},
    "Lung Opacity":            {"COVID-19": 0.80, "Non-COVID": 0.60, "Normal": 0.05},
    "Pleural Effusion":        {"COVID-19": 0.15, "Non-COVID": 0.50, "Normal": 0.02},
    "Cardiomegaly":            {"COVID-19": 0.10, "Non-COVID": 0.40, "Normal": 0.05},
    "Edema":                   {"COVID-19": 0.20, "Non-COVID": 0.45, "Normal": 0.02},
    "Atelectasis":             {"COVID-19": 0.25, "Non-COVID": 0.35, "Normal": 0.03},
    "Bilateral Involvement":   {"COVID-19": 0.80, "Non-COVID": 0.20, "Normal": 0.01},
    "Peripheral Distribution": {"COVID-19": 0.70, "Non-COVID": 0.10, "Normal": 0.01},
    "Air Bronchogram":         {"COVID-19": 0.50, "Non-COVID": 0.15, "Normal": 0.01},
    "Lung Volume Loss":        {"COVID-19": 0.30, "Non-COVID": 0.35, "Normal": 0.03},
    "Clear Lung Fields":       {"COVID-19": 0.05, "Non-COVID": 0.10, "Normal": 0.90},
}


# ═══════════════════════════════════════════════════════════════════════
# Feature extraction helpers
# ═══════════════════════════════════════════════════════════════════════

def _extract_bird_features(image_paths: list[str], batch_size: int = 32
                           ) -> np.ndarray:
    """Extract ResNet-50 layer4 features (2048-d) from bird images."""
    from torchvision import models, transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Load fine-tuned weights if available
    weight_path = os.path.join(_SAVE_DIR, "models", "bird_resnet50.pth")
    if os.path.isfile(weight_path):
        import torch.nn as nn
        from models.bird_model import NUM_CLASSES
        model.fc = nn.Linear(2048, NUM_CLASSES)
        model.load_state_dict(torch.load(weight_path, map_location=device,
                                         weights_only=True))
        logger.info("Using fine-tuned bird model for feature extraction")
    else:
        logger.warning("Fine-tuned bird model not found — using ImageNet weights")

    # Hook layer4 for 2048-d features
    captured = {}
    def _hook(_m, _i, o):
        captured["feats"] = o
    model.layer4.register_forward_hook(_hook)
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    all_feats = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            tensors.append(transform(img))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            _ = model(batch)
        feat = captured["feats"]  # (B, 2048, 7, 7)
        pooled = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
        all_feats.append(pooled.cpu().numpy())
        if (i // batch_size) % 20 == 0:
            logger.info("  bird features: %d / %d", min(i + batch_size, len(image_paths)),
                        len(image_paths))
    return np.concatenate(all_feats, axis=0)


def _extract_medical_features(image_paths: list[str], batch_size: int = 32
                              ) -> np.ndarray:
    """Extract DenseNet-121 features (1024-d) from lung X-ray images."""
    import torchxrayvision as xrv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    model.to(device).eval()

    captured = {}
    def _hook(_m, _i, o):
        captured["feats"] = o
    model.features.register_forward_hook(_hook)

    all_feats = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("L")
            arr = np.array(img, dtype=np.float32)
            arr = xrv.datasets.normalize(arr, maxval=255, reshape=True)
            t = torch.from_numpy(arr)
            t = F.interpolate(t.unsqueeze(0), size=(224, 224),
                              mode="bilinear", align_corners=False).squeeze(0)
            tensors.append(t)
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            _ = model(batch)
        feat = captured["feats"]  # (B, 1024, 7, 7)
        pooled = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
        all_feats.append(pooled.cpu().numpy())
        if (i // batch_size) % 20 == 0:
            logger.info("  medical features: %d / %d",
                        min(i + batch_size, len(image_paths)), len(image_paths))
    return np.concatenate(all_feats, axis=0)


# ═══════════════════════════════════════════════════════════════════════
# Training: Birds
# ═══════════════════════════════════════════════════════════════════════

def train_bird_probes(max_samples: int = 2000, batch_size: int = 32):
    """Train concept probes for the birds domain on CUB-200 data."""
    from cbm.concept_bank import ConceptBank
    from cbm.probe import ProbeBank
    from models.bird_model import CUB_SELECTED, CUB_ID_TO_LOCAL

    cub_dir = os.path.join(_PROJECT_ROOT, "datasets", "CUB_200_2011")
    logger.info("=== Training BIRD probes ===")

    # ── 1. Load CUB metadata ─────────────────────────────────────
    images = {}
    with open(os.path.join(cub_dir, "images.txt")) as f:
        for line in f:
            img_id, path = line.strip().split()
            images[int(img_id)] = path

    labels = {}
    with open(os.path.join(cub_dir, "image_class_labels.txt")) as f:
        for line in f:
            img_id, cls = line.strip().split()
            labels[int(img_id)] = int(cls)

    splits = {}
    with open(os.path.join(cub_dir, "train_test_split.txt")) as f:
        for line in f:
            img_id, is_train = line.strip().split()
            splits[int(img_id)] = int(is_train)

    # Filter to selected classes, train split
    train_ids = [img_id for img_id, rel_path in images.items()
                 if labels.get(img_id) in CUB_ID_TO_LOCAL and splits.get(img_id) == 1]
    train_ids.sort()

    if len(train_ids) > max_samples:
        rng = np.random.RandomState(42)
        train_ids = list(rng.choice(train_ids, size=max_samples, replace=False))
        train_ids.sort()

    logger.info("Using %d training images from %d bird classes",
                len(train_ids), len(CUB_SELECTED))

    # ── 2. Load per-image attribute labels ────────────────────────
    # Only load attributes for selected images (efficient)
    selected_set = set(train_ids)
    # attr_data[img_id] = {attr_id: (is_present, certainty)}
    attr_data: dict[int, dict[int, tuple[int, int]]] = {i: {} for i in train_ids}

    logger.info("Loading image attribute labels (this may take a moment)...")
    attr_file = os.path.join(cub_dir, "attributes", "image_attribute_labels.txt")
    with open(attr_file) as f:
        for line in f:
            parts = line.strip().split()
            img_id = int(parts[0])
            if img_id not in selected_set:
                continue
            attr_id = int(parts[1])
            is_present = int(parts[2])
            certainty = int(parts[3])
            attr_data[img_id][attr_id] = (is_present, certainty)

    logger.info("Loaded attributes for %d images", len(attr_data))

    # ── 3. Derive concept labels from CUB attributes ─────────────
    concept_bank = ConceptBank("birds")
    concept_names = concept_bank.concepts
    n = len(train_ids)

    concept_labels = {}  # concept_name → (n,) binary array
    for concept_name in concept_names:
        cub_attr_ids = _BIRD_CONCEPT_TO_CUB.get(concept_name, [])
        if not cub_attr_ids:
            logger.warning("No CUB attribute mapping for %r — skipping", concept_name)
            continue

        labels_arr = np.zeros(n, dtype=np.int32)
        for j, img_id in enumerate(train_ids):
            img_attrs = attr_data.get(img_id, {})
            # Concept is positive if ANY mapped attribute is present
            # with certainty >= 3 (probably or definitely)
            for attr_id in cub_attr_ids:
                entry = img_attrs.get(attr_id)
                if entry is not None:
                    is_present, certainty = entry
                    if is_present == 1 and certainty >= 3:
                        labels_arr[j] = 1
                        break

        pos_rate = labels_arr.mean()
        concept_labels[concept_name] = labels_arr
        logger.info("  %s: %.1f%% positive (%d/%d)",
                    concept_name, pos_rate * 100, labels_arr.sum(), n)

    # ── 4. Extract features ───────────────────────────────────────
    image_paths = [os.path.join(cub_dir, "images", images[img_id])
                   for img_id in train_ids]
    logger.info("Extracting ResNet-50 features...")
    features = _extract_bird_features(image_paths, batch_size=batch_size)
    logger.info("Feature matrix: %s", features.shape)

    # ── 5. Train probes ───────────────────────────────────────────
    probe_bank = ProbeBank(concept_bank, feature_dim=2048, probe_type="logistic")
    accs = probe_bank.train_all(features, concept_labels)
    for name, acc in sorted(accs.items()):
        logger.info("  Probe %s: accuracy=%.3f", name, acc)

    # ── 6. Train surrogate (concept activations → species) ────────
    logger.info("Training surrogate model...")
    # Build concept matrix from trained probes
    concept_matrix = np.zeros((n, len(concept_names)), dtype=np.float32)
    for i, name in enumerate(concept_names):
        if name in accs:  # probe was trained
            concept_matrix[:, i] = probe_bank._probes[name].predict_batch(features)
        else:
            concept_matrix[:, i] = 0.5

    species_labels = np.array([CUB_ID_TO_LOCAL[labels[img_id]] for img_id in train_ids])
    from models.bird_model import LOCAL_TO_NAME
    surrogate = LogisticRegression(max_iter=200, solver="saga", C=1.0,
                                   class_weight="balanced", multi_class="multinomial")
    surrogate.fit(concept_matrix, species_labels)
    surr_acc = surrogate.score(concept_matrix, species_labels)
    logger.info("  Surrogate accuracy: %.3f", surr_acc)

    # ── 7. Save ───────────────────────────────────────────────────
    save_dir = os.path.join(_SAVE_DIR, "cbm", "birds")
    probe_bank.save(os.path.join(save_dir, "probes"))

    import pickle
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "surrogate.pkl"), "wb") as f:
        pickle.dump({"model": surrogate, "classes": LOCAL_TO_NAME}, f)

    logger.info("Bird probes and surrogate saved to %s", save_dir)
    return accs


# ═══════════════════════════════════════════════════════════════════════
# Training: Medical
# ═══════════════════════════════════════════════════════════════════════

def train_medical_probes(max_samples: int = 2000, batch_size: int = 32):
    """Train concept probes for the medical domain on lung X-ray data."""
    from cbm.concept_bank import ConceptBank
    from cbm.probe import ProbeBank

    lung_dir = os.path.join(_PROJECT_ROOT, "datasets",
                            "Lung Segmentation Data", "Lung Segmentation Data", "Train")
    logger.info("=== Training MEDICAL probes ===")

    # ── 1. Collect image paths and class labels ───────────────────
    class_names = ["COVID-19", "Non-COVID", "Normal"]
    image_paths = []
    class_labels = []

    for cls_name in class_names:
        # Images are under Train/<class>/images/
        cls_dir = os.path.join(lung_dir, cls_name, "images")
        if not os.path.isdir(cls_dir):
            # Fallback: images directly in class dir
            cls_dir = os.path.join(lung_dir, cls_name)
        if not os.path.isdir(cls_dir):
            logger.warning("Missing class directory: %s", cls_dir)
            continue
        files = sorted(f for f in os.listdir(cls_dir)
                       if f.lower().endswith((".png", ".jpg", ".jpeg")))
        for fname in files:
            image_paths.append(os.path.join(cls_dir, fname))
            class_labels.append(cls_name)

    # Subsample if needed (balanced across classes)
    if len(image_paths) > max_samples:
        rng = np.random.RandomState(42)
        per_class = max_samples // len(class_names)
        selected = []
        for cls_name in class_names:
            indices = [i for i, c in enumerate(class_labels) if c == cls_name]
            chosen = rng.choice(indices, size=min(per_class, len(indices)), replace=False)
            selected.extend(chosen)
        selected.sort()
        image_paths = [image_paths[i] for i in selected]
        class_labels = [class_labels[i] for i in selected]

    n = len(image_paths)
    logger.info("Using %d lung X-ray training images", n)

    # ── 2. Derive concept labels from class membership ────────────
    concept_bank = ConceptBank("medical")
    concept_names = concept_bank.concepts

    rng = np.random.RandomState(123)
    concept_labels = {}
    for concept_name in concept_names:
        priors = _MEDICAL_CONCEPT_CLASS_PRIORS.get(concept_name)
        if priors is None:
            logger.warning("No class prior for %r — skipping", concept_name)
            continue

        labels_arr = np.zeros(n, dtype=np.int32)
        for j, cls_name in enumerate(class_labels):
            p = priors.get(cls_name, 0.5)
            labels_arr[j] = 1 if rng.random() < p else 0

        pos_rate = labels_arr.mean()
        concept_labels[concept_name] = labels_arr
        logger.info("  %s: %.1f%% positive (%d/%d)",
                    concept_name, pos_rate * 100, labels_arr.sum(), n)

    # ── 3. Extract features ───────────────────────────────────────
    logger.info("Extracting DenseNet features...")
    features = _extract_medical_features(image_paths, batch_size=batch_size)
    logger.info("Feature matrix: %s", features.shape)

    # ── 4. Train probes ───────────────────────────────────────────
    probe_bank = ProbeBank(concept_bank, feature_dim=1024, probe_type="logistic")
    accs = probe_bank.train_all(features, concept_labels)
    for name, acc in sorted(accs.items()):
        logger.info("  Probe %s: accuracy=%.3f", name, acc)

    # ── 5. Train surrogate (concept activations → class) ──────────
    logger.info("Training surrogate model...")
    concept_matrix = np.zeros((n, len(concept_names)), dtype=np.float32)
    for i, name in enumerate(concept_names):
        if name in accs:
            concept_matrix[:, i] = probe_bank._probes[name].predict_batch(features)
        else:
            concept_matrix[:, i] = 0.5

    # Encode class labels as integers
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    y = np.array([cls_to_idx[c] for c in class_labels])

    surrogate = LogisticRegression(max_iter=200, solver="saga", C=1.0,
                                   class_weight="balanced", multi_class="multinomial")
    surrogate.fit(concept_matrix, y)
    surr_acc = surrogate.score(concept_matrix, y)
    logger.info("  Surrogate accuracy: %.3f", surr_acc)

    # ── 6. Save ───────────────────────────────────────────────────
    save_dir = os.path.join(_SAVE_DIR, "cbm", "medical")
    probe_bank.save(os.path.join(save_dir, "probes"))

    import pickle
    os.makedirs(save_dir, exist_ok=True)
    labels_lower = [c.lower() for c in class_names]
    with open(os.path.join(save_dir, "surrogate.pkl"), "wb") as f:
        pickle.dump({"model": surrogate, "classes": labels_lower}, f)

    logger.info("Medical probes and surrogate saved to %s", save_dir)
    return accs


# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train real concept probes")
    parser.add_argument("--domain", choices=["birds", "medical", "all"],
                        default="all")
    parser.add_argument("--max-samples", type=int, default=2000,
                        help="Max training images per domain")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.domain in ("birds", "all"):
        train_bird_probes(max_samples=args.max_samples, batch_size=args.batch_size)

    if args.domain in ("medical", "all"):
        train_medical_probes(max_samples=args.max_samples, batch_size=args.batch_size)

    logger.info("All done.")


if __name__ == "__main__":
    main()
