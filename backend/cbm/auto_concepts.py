"""Automatic concept discovery via PCA/NMF and K-Means clustering.

These extractors discover concept axes from the model's feature space
without requiring labeled concept data.  Components / clusters are
auto-labeled via CLIP text-image similarity (for CLIP models) or by
correlating discovered axes with known concept probes.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Candidate descriptor words for auto-labeling discovered components.
_CANDIDATE_DESCRIPTORS = [
    "color", "texture", "shape", "edge", "pattern", "size", "position",
    "brightness", "contrast", "roughness", "smoothness", "curvature",
    "symmetry", "orientation", "complexity", "density", "sharpness",
    "stripes", "spots", "fur", "feathers", "scales", "skin",
    "round", "elongated", "angular", "flat", "curved",
    "large", "small", "medium-sized",
    "red", "blue", "green", "yellow", "black", "white", "brown", "gray",
    "natural", "artificial", "organic", "geometric",
    "eyes", "face", "limbs", "body", "head", "tail",
    "background", "foreground", "center", "periphery",
    "water", "land", "sky", "vegetation", "indoor", "outdoor",
]


def _clip_label_directions(
    directions: np.ndarray,
    vision_model: Any,
    candidates: list[str] | None = None,
) -> list[str]:
    """Label embedding-space directions using CLIP text similarity.

    For each direction vector, encode candidate descriptor texts via CLIP,
    compute cosine similarity, and pick the best unique match.

    Parameters
    ----------
    directions : (K, D) array of direction vectors (components or centroids)
    vision_model : a loaded VisionModel with ``score_concepts_clip``
    candidates : optional override list; defaults to ``_CANDIDATE_DESCRIPTORS``

    Returns
    -------
    list of K unique label strings
    """
    import torch
    import torch.nn.functional as F

    if candidates is None:
        candidates = list(_CANDIDATE_DESCRIPTORS)

    model = vision_model._model
    processor = vision_model._processor

    # Encode all candidate texts → (C, D)
    prompts = [f"a photo showing {c}" for c in candidates]
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
    text_inputs = {
        k: v.to(next(model.parameters()).device)
        for k, v in text_inputs.items()
        if k in ("input_ids", "attention_mask")
    }
    with torch.no_grad():
        text_embeds = vision_model._to_tensor(model.get_text_features(**text_inputs))
        text_embeds = F.normalize(text_embeds, dim=-1).cpu().numpy()  # (C, D)

    # Normalize directions
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    dirs_normed = directions / norms  # (K, D)

    # Cosine similarity matrix (K, C)
    sims = dirs_normed @ text_embeds.T

    # Greedy unique assignment: for each component pick the best unused label
    used: set[int] = set()
    labels: list[str] = []
    for i in range(directions.shape[0]):
        row = sims[i].copy()
        for u in used:
            row[u] = -np.inf
        best_idx = int(np.argmax(row))
        used.add(best_idx)
        labels.append(candidates[best_idx])

    return labels


def _probe_label_directions(
    directions: np.ndarray,
    feature_bank: np.ndarray,
    domain: str,
) -> list[str]:
    """Label directions by correlating projections with trained concept probes.

    For non-CLIP models, project the feature bank onto each direction and
    correlate those projections with the concept probe activations from the
    domain's trained probes. The concept most correlated with a direction
    becomes its label.

    Parameters
    ----------
    directions : (K, D) array
    feature_bank : (N, D) array of features used to fit PCA/KMeans
    domain : domain name to load concept probes

    Returns
    -------
    list of K label strings
    """
    from config import ALT_PROBE_DATA_DIR, PROBE_DATA_DIR
    from cbm.probe import ConceptProbe
    from cbm.concept_bank import ConceptBank

    try:
        cb = ConceptBank(domain)
    except KeyError:
        return [f"component_{i}" for i in range(directions.shape[0])]

    # Search multiple potential probe directories
    probe_dirs = [
        os.path.join(ALT_PROBE_DATA_DIR, domain),
        os.path.join(ALT_PROBE_DATA_DIR, domain, "probes"),
        os.path.join(PROBE_DATA_DIR, domain),
        os.path.join(PROBE_DATA_DIR, domain, "probes"),
    ]

    # Load all trained probes and score the feature bank
    concept_scores: dict[str, np.ndarray] = {}
    for cname in cb.concepts:
        safe = cname.replace(" ", "_").replace("/", "_")
        for probe_dir in probe_dirs:
            path = os.path.join(probe_dir, f"{safe}.pkl")
            if os.path.isfile(path):
                probe = ConceptProbe(cname, feature_bank.shape[1])
                try:
                    probe.load(path)
                    if probe.is_trained:
                        concept_scores[cname] = probe.predict_batch(feature_bank)
                except Exception:
                    pass
                break

    if not concept_scores:
        # No trained probes — fall back to numbered labels
        return [f"component_{i}" for i in range(directions.shape[0])]

    concept_names = list(concept_scores.keys())
    # (N, C) matrix of concept activations
    score_mat = np.column_stack([concept_scores[c] for c in concept_names])

    # Project feature bank onto each direction → (N, K)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    dirs_normed = directions / norms
    projections = feature_bank @ dirs_normed.T  # (N, K)

    # Pearson correlation between each direction's projections and each concept
    # Normalize columns to zero-mean unit-variance
    def _standardize(a: np.ndarray) -> np.ndarray:
        m = a.mean(axis=0, keepdims=True)
        s = a.std(axis=0, keepdims=True)
        s = np.where(s == 0, 1, s)
        return (a - m) / s

    proj_z = _standardize(projections)  # (N, K)
    score_z = _standardize(score_mat)   # (N, C)
    corr = (proj_z.T @ score_z) / feature_bank.shape[0]  # (K, C)

    # Use absolute correlation (direction could be negatively correlated)
    abs_corr = np.abs(corr)

    # Greedy unique assignment
    used: set[int] = set()
    labels: list[str] = []
    for i in range(directions.shape[0]):
        row = abs_corr[i].copy()
        for u in used:
            row[u] = -np.inf
        best_idx = int(np.argmax(row))
        best_val = abs_corr[i, best_idx]
        used.add(best_idx)
        if best_val > 0.05:
            labels.append(concept_names[best_idx])
        else:
            labels.append(f"component_{i}")

    return labels


class PCAConceptExtractor:
    """Discover concept axes via PCA or NMF on the feature space.

    Decomposes features into principal components, then labels each
    component by finding the highest CLIP similarity between the
    component direction and candidate descriptor texts.
    """

    def __init__(self, n_concepts: int = 10, method: str = "pca"):
        self._n_concepts = n_concepts
        self._method = method  # "pca" or "nmf"
        self._components = None      # (n_concepts, feature_dim)
        self._labels: list[str] = []
        self._mean: np.ndarray | None = None
        self._feature_bank: np.ndarray | None = None
        self._domain: str | None = None

    def fit(self, feature_bank: np.ndarray, domain: str | None = None) -> None:
        """Fit PCA/NMF on the feature bank."""
        n = min(self._n_concepts, feature_bank.shape[0], feature_bank.shape[1])
        self._feature_bank = feature_bank
        self._domain = domain

        if self._method == "nmf":
            from sklearn.decomposition import NMF
            # NMF requires non-negative input; shift if needed
            shifted = feature_bank - feature_bank.min(axis=0, keepdims=True)
            model = NMF(n_components=n, max_iter=300, random_state=42)
            model.fit(shifted)
            self._components = model.components_
            self._mean = feature_bank.min(axis=0)
        else:
            from sklearn.decomposition import PCA
            model = PCA(n_components=n, random_state=42)
            model.fit(feature_bank)
            self._components = model.components_
            self._mean = model.mean_

        # Generate default labels
        self._labels = [f"component_{i}" for i in range(n)]

    def label_with_clip(self, vision_model: Any) -> None:
        """Auto-label components using CLIP or probe correlation."""
        if self._components is None:
            return
        try:
            if vision_model is not None and hasattr(vision_model, "score_concepts_clip"):
                # CLIP-capable model: match component directions to text descriptors
                self._labels = _clip_label_directions(
                    self._components, vision_model,
                )
            elif self._feature_bank is not None and self._domain:
                # Non-CLIP model: correlate with trained concept probes
                self._labels = _probe_label_directions(
                    self._components, self._feature_bank, self._domain,
                )
            else:
                logger.debug("No labeling method available, keeping default labels")
        except Exception:
            logger.debug("Auto-labeling failed, keeping default labels", exc_info=True)

    def extract(self, features: np.ndarray) -> dict[str, float]:
        """Project a single feature vector onto the discovered components."""
        if self._components is None:
            return {}
        x = features.reshape(1, -1)
        if self._method == "nmf" and self._mean is not None:
            x = x - self._mean
            x = np.maximum(x, 0)

        # Project onto components
        projections = (x @ self._components.T).squeeze()

        # Normalize to [0, 1] via sigmoid
        activations = 1.0 / (1.0 + np.exp(-projections))

        result = {}
        for i, label in enumerate(self._labels):
            if i < len(activations):
                result[label] = round(float(activations[i]), 4)
        return result


class KMeansConceptExtractor:
    """Discover concepts by clustering the feature space.

    Each cluster represents a concept.  Concept activation for a new
    sample is the softmax of negative distances to cluster centroids.
    """

    def __init__(self, n_concepts: int = 10):
        self._n_concepts = n_concepts
        self._centroids: np.ndarray | None = None  # (n_concepts, feature_dim)
        self._labels: list[str] = []
        self._feature_bank: np.ndarray | None = None
        self._domain: str | None = None

    def fit(self, feature_bank: np.ndarray, domain: str | None = None) -> None:
        """Fit K-Means on the feature bank."""
        from sklearn.cluster import KMeans
        self._feature_bank = feature_bank
        self._domain = domain
        n = min(self._n_concepts, feature_bank.shape[0])
        model = KMeans(n_clusters=n, random_state=42, n_init=10, max_iter=300)
        model.fit(feature_bank)
        self._centroids = model.cluster_centers_
        self._labels = [f"cluster_{i}" for i in range(n)]

    def label_with_clip(self, vision_model: Any) -> None:
        """Auto-label clusters using CLIP or probe correlation."""
        if self._centroids is None:
            return
        try:
            if vision_model is not None and hasattr(vision_model, "score_concepts_clip"):
                self._labels = _clip_label_directions(
                    self._centroids, vision_model,
                )
            elif self._feature_bank is not None and self._domain:
                self._labels = _probe_label_directions(
                    self._centroids, self._feature_bank, self._domain,
                )
            else:
                logger.debug("No labeling method available, keeping default labels")
        except Exception:
            logger.debug("Auto-labeling failed, keeping default labels", exc_info=True)

    def extract(self, features: np.ndarray) -> dict[str, float]:
        """Compute concept activations as softmax of negative distances."""
        if self._centroids is None:
            return {}
        x = features.reshape(1, -1)
        # Euclidean distances to each centroid
        dists = np.linalg.norm(self._centroids - x, axis=1)
        # Softmax of negative distances → closer = higher activation
        neg_dists = -dists
        exp_vals = np.exp(neg_dists - neg_dists.max())
        activations = exp_vals / exp_vals.sum()

        result = {}
        for i, label in enumerate(self._labels):
            if i < len(activations):
                result[label] = round(float(activations[i]), 4)
        return result


def load_feature_bank(domain: str) -> np.ndarray | None:
    """Load pre-computed feature bank for a domain.

    Looks for ``data/cbm/{domain}/feature_bank.npy``.
    """
    from config import PROBE_DATA_DIR
    path = os.path.join(PROBE_DATA_DIR, domain, "feature_bank.npy")
    if os.path.isfile(path):
        return np.load(path)
    return None


def build_feature_bank(model: Any, domain: str, max_samples: int = 200) -> np.ndarray | None:
    """Build a feature bank from sample images/texts on the fly."""
    from config import SAMPLE_DIR
    import base64

    sample_dir = os.path.join(SAMPLE_DIR, domain)
    if not os.path.isdir(sample_dir):
        return None

    features_list = []
    for fname in sorted(os.listdir(sample_dir)):
        if len(features_list) >= max_samples:
            break
        fpath = os.path.join(sample_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                with open(fpath, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                feat = model.extract_features(b64)
                features_list.append(feat)
        except Exception:
            continue

    if not features_list:
        return None
    return np.stack(features_list)
