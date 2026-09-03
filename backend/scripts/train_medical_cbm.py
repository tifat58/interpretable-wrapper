#!/usr/bin/env python3
"""Train the medical CBM pipeline with weak (non-random) concept labels,
a learned task head, calibration, and honest train/val/test evaluation.

Replaces the Bernoulli-prior label sampling in ``train_probes_real.py`` for
the medical domain.

Label sources
-------------
1. Pathology-aligned concepts — weak labels from an *independent*
   torchxrayvision checkpoint (``densenet121-res224-all``), distinct from
   the wrapped ``-chex`` model used for features, to reduce circularity.
2. Geometric concepts — derived from the COVID-QU-Ex lung masks and
   intensity statistics, z-scored against the Normal-class train set.
3. Air Bronchogram — conjunction proxy (consolidation AND pneumonia).

Artifacts (backend/data/cbm/medical/)
-------------------------------------
- probes/*.pkl        temperature-calibrated logistic probes
- surrogate.pkl       concept→label surrogate (trained on head predictions)
- task_head.pkl       calibrated logistic task head (features → 3 classes)
- metrics.json        full evaluation report
- cache/feat_*.npz    cached features / labeler scores / mask stats

Usage
-----
    cd backend
    python -m scripts.train_medical_cbm [--max-train 6000] [--max-eval 2000]
                                        [--batch-size 64] [--full]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
_DATA_ROOT = os.path.join(_PROJECT_ROOT, "datasets",
                          "Lung Segmentation Data", "Lung Segmentation Data")
_SAVE_DIR = os.path.join(_BACKEND_DIR, "data", "cbm", "medical")
_CACHE_DIR = os.path.join(_SAVE_DIR, "cache")

CLASS_NAMES = ["COVID-19", "Non-COVID", "Normal"]

# Concepts scored by the independent labeler model (concept → txv pathology)
_LABELER_WEIGHTS = "densenet121-res224-all"
_CONCEPT_TO_LABELER_PATHOLOGY = {
    "Consolidation": "Consolidation",
    "Ground Glass Opacity": "Infiltration",   # closest available proxy
    "Lung Opacity": "Lung Opacity",
    "Pleural Effusion": "Effusion",
    "Cardiomegaly": "Cardiomegaly",
    "Edema": "Edema",
    "Atelectasis": "Atelectasis",
    "Pneumothorax": "Pneumothorax",
    "Nodule": "Nodule",
    "Mass": "Mass",
    "Fibrosis": "Fibrosis",
}
# Geometric concepts derived from lung masks + intensity statistics
_MASK_CONCEPTS = ["Bilateral Involvement", "Peripheral Distribution",
                  "Lung Volume Loss", "Clear Lung Fields"]

_IMG_SIZE = 224


# ═══════════════════════════════════════════════════════════════════════
# Dataset indexing
# ═══════════════════════════════════════════════════════════════════════

def index_split(split: str, max_samples: int | None, seed: int = 42
                ) -> tuple[list[str], list[str], np.ndarray]:
    """Return (image_paths, mask_paths, class_indices) for a split."""
    image_paths, mask_paths, y = [], [], []
    for ci, cls in enumerate(CLASS_NAMES):
        img_dir = os.path.join(_DATA_ROOT, split, cls, "images")
        msk_dir = os.path.join(_DATA_ROOT, split, cls, "lung masks")
        files = sorted(f for f in os.listdir(img_dir)
                       if f.lower().endswith((".png", ".jpg", ".jpeg")))
        for fname in files:
            image_paths.append(os.path.join(img_dir, fname))
            mask_paths.append(os.path.join(msk_dir, fname))
            y.append(ci)
    y = np.array(y, dtype=np.int64)

    if max_samples is not None and len(image_paths) > max_samples:
        rng = np.random.RandomState(seed)
        per_class = max_samples // len(CLASS_NAMES)
        keep = []
        for ci in range(len(CLASS_NAMES)):
            idx = np.where(y == ci)[0]
            keep.extend(rng.choice(idx, size=min(per_class, len(idx)),
                                   replace=False))
        keep = sorted(keep)
        image_paths = [image_paths[i] for i in keep]
        mask_paths = [mask_paths[i] for i in keep]
        y = y[keep]

    logger.info("%s split: %d images (%s)", split, len(image_paths),
                np.bincount(y, minlength=3).tolist())
    return image_paths, mask_paths, y


# ═══════════════════════════════════════════════════════════════════════
# Feature / labeler-score / mask-statistic extraction (cached)
# ═══════════════════════════════════════════════════════════════════════

def _load_xrv_batch(paths: list[str], device: torch.device) -> torch.Tensor:
    import torchxrayvision as xrv
    tensors = []
    for p in paths:
        img = Image.open(p).convert("L")
        arr = np.array(img, dtype=np.float32)
        arr = xrv.datasets.normalize(arr, maxval=255, reshape=True)
        t = torch.from_numpy(arr)
        t = F.interpolate(t.unsqueeze(0), size=(_IMG_SIZE, _IMG_SIZE),
                          mode="bilinear", align_corners=False).squeeze(0)
        tensors.append(t)
    return torch.stack(tensors).to(device)


def _mask_stats(image_path: str, mask_path: str) -> np.ndarray:
    """Return [left_int, right_int, periph_int, core_int, area_frac].

    Intensities are mean grayscale values (0-1) inside lung-mask regions;
    higher intensity inside the lung field = more opacity.
    """
    img = np.array(Image.open(image_path).convert("L")
                   .resize((_IMG_SIZE, _IMG_SIZE)), dtype=np.float32) / 255.0
    mask = np.array(Image.open(mask_path).convert("L")
                    .resize((_IMG_SIZE, _IMG_SIZE), Image.NEAREST)) > 127

    half = _IMG_SIZE // 2
    lm, rm = mask[:, :half], mask[:, half:]
    left_int = float(img[:, :half][lm].mean()) if lm.any() else 0.0
    right_int = float(img[:, half:][rm].mean()) if rm.any() else 0.0

    core = binary_erosion(mask, iterations=14)
    periph = mask & ~core
    periph_int = float(img[periph].mean()) if periph.any() else 0.0
    core_int = float(img[core].mean()) if core.any() else 0.0

    area_frac = float(mask.mean())
    return np.array([left_int, right_int, periph_int, core_int, area_frac],
                    dtype=np.float32)


def extract_split(split: str, image_paths: list[str], mask_paths: list[str],
                  y: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    """Extract chex features, labeler pathology scores, and mask stats."""
    cache_path = os.path.join(_CACHE_DIR, f"feat_{split}.npz")
    if os.path.isfile(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        if list(cached["paths"]) == image_paths:
            logger.info("%s: using cached features (%s)", split, cache_path)
            return {k: cached[k] for k in cached.files}
        logger.info("%s: cache stale — recomputing", split)

    import torchxrayvision as xrv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat_model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    feat_model.to(device).eval()
    captured = {}
    feat_model.features.register_forward_hook(
        lambda _m, _i, o: captured.__setitem__("f", o))

    labeler = xrv.models.DenseNet(weights=_LABELER_WEIGHTS)
    labeler.to(device).eval()
    labeler_pathologies = list(labeler.pathologies)

    feats, labeler_scores, mstats = [], [], []
    n = len(image_paths)
    for i in range(0, n, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch = _load_xrv_batch(batch_paths, device)
        with torch.no_grad():
            _ = feat_model(batch)
            pooled = F.adaptive_avg_pool2d(captured["f"], 1).flatten(1)
            probs = torch.sigmoid(labeler(batch))
        feats.append(pooled.cpu().numpy())
        labeler_scores.append(probs.cpu().numpy())
        for ip, mp in zip(batch_paths, mask_paths[i:i + batch_size]):
            mstats.append(_mask_stats(ip, mp))
        if (i // batch_size) % 20 == 0:
            logger.info("  %s: %d / %d", split, min(i + batch_size, n), n)

    out = {
        "paths": np.array(image_paths, dtype=object),
        "y": y,
        "features": np.concatenate(feats, axis=0),
        "labeler_scores": np.concatenate(labeler_scores, axis=0),
        "labeler_pathologies": np.array(labeler_pathologies, dtype=object),
        "mask_stats": np.stack(mstats, axis=0),
    }
    os.makedirs(_CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache_path, **out)
    logger.info("%s: cached to %s", split, cache_path)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Weak concept labels
# ═══════════════════════════════════════════════════════════════════════

def _adaptive_threshold(scores: np.ndarray, default: float = 0.5) -> float:
    """0.5 operating point, falling back to the 80th percentile when the
    labeler almost never (or always) crosses it on this dataset."""
    pos_rate = float((scores >= default).mean())
    if 0.02 <= pos_rate <= 0.98:
        return default
    return float(np.quantile(scores, 0.8))


def build_concept_labels(split_data: dict[str, np.ndarray],
                         normal_ref: dict[str, float],
                         thresholds: dict[str, float] | None = None,
                         ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Weak binary labels for every concept.  Returns (labels, thresholds)."""
    scores = split_data["labeler_scores"]
    pathologies = list(split_data["labeler_pathologies"])
    ms = split_data["mask_stats"]
    labels: dict[str, np.ndarray] = {}
    thresholds = dict(thresholds) if thresholds else {}

    def _pscore(name: str) -> np.ndarray:
        return scores[:, pathologies.index(name)]

    # 1. Labeler-aligned concepts
    for concept, pathology in _CONCEPT_TO_LABELER_PATHOLOGY.items():
        s = _pscore(pathology)
        if concept not in thresholds:
            thresholds[concept] = _adaptive_threshold(s)
        labels[concept] = (s >= thresholds[concept]).astype(np.int32)

    # 2. Air Bronchogram — conjunction proxy (weakest label)
    cons, pneu = _pscore("Consolidation"), _pscore("Pneumonia")
    if "Air Bronchogram" not in thresholds:
        thresholds["Air Bronchogram"] = _adaptive_threshold(cons)
    ab_thr = thresholds["Air Bronchogram"]
    labels["Air Bronchogram"] = ((cons >= ab_thr) &
                                 (pneu >= np.quantile(pneu, 0.5))).astype(np.int32)

    # 3. Mask/intensity-derived concepts (z-scored vs Normal train stats)
    zl = (ms[:, 0] - normal_ref["int_mean"]) / normal_ref["int_std"]
    zr = (ms[:, 1] - normal_ref["int_mean"]) / normal_ref["int_std"]
    periph_ratio = ms[:, 2] / np.maximum(ms[:, 3], 1e-6)
    # "any finding" = any mapped pathology above its own adaptive threshold
    any_finding = np.zeros(len(ms), dtype=bool)
    for concept, pathology in _CONCEPT_TO_LABELER_PATHOLOGY.items():
        any_finding |= _pscore(pathology) >= thresholds[concept]

    labels["Bilateral Involvement"] = ((zl > 1.0) & (zr > 1.0)).astype(np.int32)
    labels["Peripheral Distribution"] = (
        (periph_ratio > normal_ref["periph_q75"]) & ((zl > 0.5) | (zr > 0.5))
    ).astype(np.int32)
    labels["Lung Volume Loss"] = (ms[:, 4] < normal_ref["area_q10"]).astype(np.int32)
    labels["Clear Lung Fields"] = (
        (zl < 0.5) & (zr < 0.5) & ~any_finding
    ).astype(np.int32)

    return labels, thresholds


def normal_reference_stats(train_data: dict[str, np.ndarray]) -> dict[str, float]:
    """Intensity / geometry reference distribution from Normal train images."""
    ms, y = train_data["mask_stats"], train_data["y"]
    normal = ms[y == CLASS_NAMES.index("Normal")]
    both_int = np.concatenate([normal[:, 0], normal[:, 1]])
    periph_ratio = normal[:, 2] / np.maximum(normal[:, 3], 1e-6)
    return {
        "int_mean": float(both_int.mean()),
        "int_std": float(max(both_int.std(), 1e-6)),
        "periph_q75": float(np.quantile(periph_ratio, 0.75)),
        "area_q10": float(np.quantile(normal[:, 4], 0.1)),
    }


# ═══════════════════════════════════════════════════════════════════════
# Calibration helpers
# ═══════════════════════════════════════════════════════════════════════

def _fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Scalar temperature minimizing NLL.  Works for (n,) binary logits
    or (n, k) multiclass logits."""
    def nll(log_t: float) -> float:
        t = np.exp(log_t)
        z = logits / t
        if z.ndim == 1:
            p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(y)), y].mean())

    res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def _apply_temperature(model: LogisticRegression, t: float) -> None:
    """Fold temperature into LR weights so pickles stay drop-in compatible."""
    model.coef_ = model.coef_ / t
    model.intercept_ = model.intercept_ / t


def _ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error (top-label for multiclass)."""
    if probs.ndim == 2:
        conf, correct = probs.max(axis=1), (probs.argmax(axis=1) == y)
    else:
        conf = np.where(probs >= 0.5, probs, 1 - probs)
        correct = (probs >= 0.5).astype(int) == y
    ece = 0.0
    for lo in np.linspace(0, 1, n_bins, endpoint=False):
        m = (conf >= lo) & (conf < lo + 1 / n_bins)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


# ═══════════════════════════════════════════════════════════════════════
# Training stages
# ═══════════════════════════════════════════════════════════════════════

def train_task_head(train, val, test) -> tuple[LogisticRegression, dict]:
    logger.info("=== Task head (features → 3 classes) ===")
    head = LogisticRegression(max_iter=500, solver="saga", C=1.0,
                              class_weight="balanced")
    head.fit(train["features"], train["y"])

    val_logits = head.decision_function(val["features"])
    metrics = {"val_ece_before": _ece(_softmax(val_logits), val["y"])}
    t = _fit_temperature(val_logits, val["y"])
    _apply_temperature(head, t)
    metrics["temperature"] = round(t, 4)
    metrics["val_ece_after"] = _ece(head.predict_proba(val["features"]), val["y"])

    for name, d in (("val", val), ("test", test)):
        probs = head.predict_proba(d["features"])
        pred = probs.argmax(axis=1)
        metrics[f"{name}_accuracy"] = round(float((pred == d["y"]).mean()), 4)
        metrics[f"{name}_macro_auc"] = round(float(
            roc_auc_score(d["y"], probs, multi_class="ovr", average="macro")), 4)
    per_class = {}
    test_probs = head.predict_proba(test["features"])
    for ci, cls in enumerate(CLASS_NAMES):
        per_class[cls] = round(float(
            roc_auc_score((test["y"] == ci).astype(int), test_probs[:, ci])), 4)
    metrics["test_per_class_auc"] = per_class
    logger.info("Task head: %s", json.dumps(metrics, indent=2))
    return head, metrics


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_probes(train, val, test, train_labels, val_labels, test_labels):
    from cbm.concept_bank import ConceptBank
    from cbm.probe import ProbeBank

    logger.info("=== Concept probes ===")
    concept_bank = ConceptBank("medical")
    probe_bank = ProbeBank(concept_bank, feature_dim=train["features"].shape[1],
                           probe_type="logistic")
    probe_bank.train_all(train["features"], train_labels)

    metrics: dict[str, dict] = {}
    for name in concept_bank.concepts:
        if name not in train_labels:
            continue
        probe = probe_bank._probes[name]
        m: dict[str, float] = {
            "train_pos_rate": round(float(train_labels[name].mean()), 4),
        }
        # calibrate on val (needs both classes present)
        yv = val_labels[name]
        if 0 < yv.sum() < len(yv):
            logits = probe.model.decision_function(val["features"])
            t = _fit_temperature(logits, yv)
            _apply_temperature(probe.model, t)
            m["temperature"] = round(t, 4)
            m["val_auc"] = round(float(
                roc_auc_score(yv, probe.predict_batch(val["features"]))), 4)
        yt = test_labels[name]
        if 0 < yt.sum() < len(yt):
            p = probe.predict_batch(test["features"])
            m["test_auc"] = round(float(roc_auc_score(yt, p)), 4)
            m["test_accuracy"] = round(float(((p >= 0.5) == yt).mean()), 4)
            m["test_ece"] = round(_ece(p, yt), 4)
        metrics[name] = m
        logger.info("  %-24s %s", name, m)

    probe_bank.save(os.path.join(_SAVE_DIR, "probes"))
    return probe_bank, metrics


def train_surrogate(probe_bank, head, train, val, test):
    """Concept activations → label surrogate, trained to mimic the head."""
    from cbm.concept_bank import ConceptBank
    logger.info("=== Surrogate (concepts → label) ===")
    concept_names = ConceptBank("medical").concepts

    def concept_matrix(d):
        cols = []
        for name in concept_names:
            probe = probe_bank._probes.get(name)
            if probe is not None and probe.is_trained:
                cols.append(probe.predict_batch(d["features"]))
            else:
                cols.append(np.full(len(d["y"]), 0.5))
        return np.stack(cols, axis=1)

    head_pred = {k: head.predict_proba(d["features"]).argmax(axis=1)
                 for k, d in (("train", train), ("val", val), ("test", test))}

    surrogate = LogisticRegression(max_iter=500, solver="saga", C=1.0,
                                   class_weight="balanced")
    surrogate.fit(concept_matrix(train), head_pred["train"])

    metrics = {}
    for name, d in (("val", val), ("test", test)):
        pred = surrogate.predict(concept_matrix(d))
        metrics[f"{name}_fidelity"] = round(float((pred == head_pred[name]).mean()), 4)
        metrics[f"{name}_gt_accuracy"] = round(float((pred == d["y"]).mean()), 4)
    logger.info("Surrogate: %s", metrics)

    with open(os.path.join(_SAVE_DIR, "surrogate.pkl"), "wb") as f:
        pickle.dump({"model": surrogate, "classes": CLASS_NAMES,
                     "metrics": metrics}, f)
    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-train", type=int, default=6000)
    ap.add_argument("--max-eval", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--full", action="store_true",
                    help="use every image in each split")
    args = ap.parse_args()

    max_train = None if args.full else args.max_train
    max_eval = None if args.full else args.max_eval

    splits = {}
    for split, cap in (("Train", max_train), ("Val", max_eval), ("Test", max_eval)):
        paths, masks, y = index_split(split, cap)
        splits[split] = extract_split(split, paths, masks, y, args.batch_size)
    train, val, test = splits["Train"], splits["Val"], splits["Test"]

    normal_ref = normal_reference_stats(train)
    train_labels, thresholds = build_concept_labels(train, normal_ref)
    val_labels, _ = build_concept_labels(val, normal_ref, thresholds)
    test_labels, _ = build_concept_labels(test, normal_ref, thresholds)

    os.makedirs(_SAVE_DIR, exist_ok=True)

    head, head_metrics = train_task_head(train, val, test)
    with open(os.path.join(_SAVE_DIR, "task_head.pkl"), "wb") as f:
        pickle.dump({"model": head, "classes": CLASS_NAMES,
                     "metrics": head_metrics}, f)

    probe_bank, probe_metrics = train_probes(
        train, val, test, train_labels, val_labels, test_labels)
    surrogate_metrics = train_surrogate(probe_bank, head, train, val, test)

    report = {
        "dataset": "COVID-QU-Ex (Lung Segmentation Data)",
        "n_train": len(train["y"]), "n_val": len(val["y"]),
        "n_test": len(test["y"]),
        "labeler_model": _LABELER_WEIGHTS,
        "feature_model": "densenet121-res224-chex",
        "concept_thresholds": {k: round(v, 4) for k, v in thresholds.items()},
        "normal_reference": {k: round(v, 4) for k, v in normal_ref.items()},
        "task_head": head_metrics,
        "probes": probe_metrics,
        "surrogate": surrogate_metrics,
    }
    with open(os.path.join(_SAVE_DIR, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved metrics to %s", os.path.join(_SAVE_DIR, "metrics.json"))


if __name__ == "__main__":
    main()
