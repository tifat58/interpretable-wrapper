"""Concept probes: sklearn logistic regression + CLIP zero-shot scorer.

Two complementary strategies for mapping features → concept activations:

1. ``ConceptProbe`` — supervised logistic regression (needs labeled data)
2. ``CLIPConceptScorer`` — zero-shot via CLIP text-image similarity
3. ``ProbeBank`` — manages a collection of probes for one domain
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.linear_model import LogisticRegression

if TYPE_CHECKING:
    from cbm.concept_bank import ConceptBank

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Supervised logistic regression probe
# ═══════════════════════════════════════════════════════════════════════
class ConceptProbe:
    """Binary logistic-regression probe for a single concept."""

    def __init__(self, concept_name: str, feature_dim: int):
        self.concept_name = concept_name
        self.feature_dim = feature_dim
        self.model = LogisticRegression(
            max_iter=200, solver="saga", C=1.0, class_weight="balanced",
        )
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Fit the probe on (features, binary_labels).  Returns accuracy."""
        self.model.fit(features, labels)
        self._trained = True
        acc = float(self.model.score(features, labels))
        logger.debug("Probe %r trained — accuracy %.3f", self.concept_name, acc)
        return acc

    def predict(self, features: np.ndarray) -> float:
        """Return concept activation probability for a single sample.

        Parameters
        ----------
        features : array of shape ``(feature_dim,)`` or ``(1, feature_dim)``

        Returns
        -------
        float in [0, 1] — concept activation score.
        """
        if not self._trained:
            return 0.5  # uninformative prior
        x = features.reshape(1, -1) if features.ndim == 1 else features
        if x.shape[1] != self.feature_dim:
            return 0.5  # incompatible feature dimension
        proba = self.model.predict_proba(x)
        # positive class is index 1
        col = 1 if proba.shape[1] > 1 else 0
        return float(proba[0, col])

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """Return concept activations for a batch of samples."""
        if not self._trained:
            return np.full(features.shape[0], 0.5)
        proba = self.model.predict_proba(features)
        col = 1 if proba.shape[1] > 1 else 0
        return proba[:, col]

    # ── serialization ────────────────────────────────────────────────
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "concept": self.concept_name,
                         "feature_dim": self.feature_dim, "trained": self._trained}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301 — trusted local files only
        self.model = data["model"]
        self.concept_name = data["concept"]
        self.feature_dim = data["feature_dim"]
        self._trained = data["trained"]


# ═══════════════════════════════════════════════════════════════════════
# CLIP zero-shot concept scorer (no labeled data needed)
# ═══════════════════════════════════════════════════════════════════════
class CLIPConceptScorer:
    """Score concept presence via CLIP text-image cosine similarity.

    This bypasses the need for labeled concept data entirely: concept
    descriptions are encoded as CLIP text embeddings and compared to the
    image embedding.
    """

    def __init__(self, concept_descriptions: dict[str, str]):
        self._descriptions = concept_descriptions
        self._concept_names = list(concept_descriptions.keys())

    def score(self, data: Any, vision_model: Any) -> dict[str, float]:
        """Score all concepts for the given image.

        Parameters
        ----------
        data : raw image input (path, base64, PIL Image)
        vision_model : a loaded ``VisionModel`` instance with
                       ``score_concepts_clip()``

        Returns
        -------
        dict of concept_name → activation in [0, 1]
        """
        return vision_model.score_concepts_clip(data, self._descriptions)


# ═══════════════════════════════════════════════════════════════════════
# Probe bank — collection of probes for one domain
# ═══════════════════════════════════════════════════════════════════════
class ProbeBank:
    """Manages concept probes (supervised and/or CLIP-based) for a domain."""

    def __init__(self, concept_bank: ConceptBank, feature_dim: int,
                 probe_type: str = "logistic"):
        """
        Parameters
        ----------
        concept_bank : domain concept definitions
        feature_dim :  dimensionality of model feature vectors
        probe_type :   "logistic" | "clip" | "hybrid"
        """
        self._concept_bank = concept_bank
        self._feature_dim = feature_dim
        self._probe_type = probe_type

        # Create supervised probes (even for clip/hybrid — they're cheap)
        self._probes: dict[str, ConceptProbe] = {
            name: ConceptProbe(name, feature_dim)
            for name in concept_bank.concepts
        }

        # CLIP scorer (created when needed)
        self._clip_scorer: CLIPConceptScorer | None = None
        if probe_type in ("clip", "hybrid"):
            self._clip_scorer = CLIPConceptScorer(concept_bank.descriptions)

    @property
    def probe_type(self) -> str:
        return self._probe_type

    @property
    def has_trained_probes(self) -> bool:
        return any(p.is_trained for p in self._probes.values())

    # ── scoring ──────────────────────────────────────────────────────
    def predict_concepts(self, features: np.ndarray,
                         raw_input: Any = None,
                         vision_model: Any = None,
                         strategy: str | None = None,
                         custom_concepts: list[str] | None = None) -> dict[str, float]:
        """Predict concept activations.

        Parameters
        ----------
        features : feature vector from the domain model
        raw_input : original input (needed for CLIP scoring)
        vision_model : loaded VisionModel (needed for CLIP scoring)
        strategy : override probe strategy ("predefined"|"clip"|"hybrid"|
                   "pca"|"kmeans"|"custom"|"token_aggregation")
        custom_concepts : user-defined concept descriptions (for strategy="custom")
        """
        effective_strategy = strategy or self._probe_type

        # ── Custom user-defined concepts via CLIP ────────────────────
        if effective_strategy == "custom" and custom_concepts and vision_model:
            if not hasattr(vision_model, "score_concepts_clip"):
                logger.warning("Model %s does not support CLIP concept scoring", type(vision_model).__name__)
                return {c: 0.5 for c in custom_concepts}
            descriptions = {c: f"an image showing {c}" for c in custom_concepts}
            scorer = CLIPConceptScorer(descriptions)
            return scorer.score(raw_input, vision_model)

        # ── PCA auto-discovery ───────────────────────────────────────
        if effective_strategy == "pca":
            return self._predict_pca(features, vision_model)

        # ── K-Means auto-discovery ───────────────────────────────────
        if effective_strategy == "kmeans":
            return self._predict_kmeans(features, vision_model)

        if effective_strategy == "label_free":
            return self._predict_label_free(features, vision_model, raw_input)

        # ── Token-attribution aggregation (text domains) ─────────────
        if effective_strategy == "token_aggregation" and vision_model is not None:
            if hasattr(vision_model, "aggregate_token_attribution"):
                return vision_model.aggregate_token_attribution(raw_input)
            # Fallback: use predefined probes
            logger.warning("Model does not support aggregate_token_attribution, falling back")

        # ── Original strategies ──────────────────────────────────────
        results: dict[str, float] = {}

        # Check feature dimension compatibility with trained probes
        probes_compatible = True
        if features.ndim == 1:
            feat_dim = features.shape[0]
        else:
            feat_dim = features.shape[-1]

        if feat_dim != self._feature_dim:
            logger.warning(
                "Feature dim mismatch: model=%d, probes=%d — probes unavailable",
                feat_dim, self._feature_dim,
            )
            probes_compatible = False

        if effective_strategy == "clip" and self._clip_scorer and vision_model:
            if not hasattr(vision_model, "score_concepts_clip"):
                logger.warning("Model %s does not support CLIP — falling back to probes", type(vision_model).__name__)
            else:
                return self._clip_scorer.score(raw_input, vision_model)

        if effective_strategy in ("logistic", "hybrid", "predefined"):
            # Supervised probes
            if probes_compatible:
                for name, probe in self._probes.items():
                    results[name] = round(probe.predict(features), 4)
            else:
                for name in self._probes:
                    results[name] = 0.5

        if effective_strategy == "hybrid" and self._clip_scorer and vision_model and hasattr(vision_model, "score_concepts_clip"):
            # Average with CLIP scores
            clip_scores = self._clip_scorer.score(raw_input, vision_model)
            for name in results:
                if name in clip_scores:
                    probe_trained = self._probes[name].is_trained
                    if probe_trained:
                        # Weighted average: 0.7 trained + 0.3 CLIP
                        results[name] = round(
                            0.7 * results[name] + 0.3 * clip_scores[name], 4,
                        )
                    else:
                        # Only CLIP available
                        results[name] = clip_scores[name]

        if not results:
            # Fallback: uniform 0.5 for all concepts
            results = {name: 0.5 for name in self._concept_bank.concepts}

        return results

    def _predict_pca(self, features: np.ndarray, vision_model: Any = None) -> dict[str, float]:
        """Use PCA-based auto-discovery for concept extraction."""
        from cbm.auto_concepts import PCAConceptExtractor, load_feature_bank, build_feature_bank
        domain = self._concept_bank.domain

        if not hasattr(self, "_pca_extractor") or self._pca_extractor is None:
            feature_bank = load_feature_bank(domain)
            if feature_bank is None and vision_model is not None:
                feature_bank = build_feature_bank(vision_model, domain)
            if feature_bank is None:
                # Fallback: create a minimal feature bank from the current features
                feature_bank = features.reshape(1, -1) + np.random.normal(0, 0.01, (50, features.shape[-1] if features.ndim > 1 else features.shape[0]))
                feature_bank = np.vstack([features.reshape(1, -1), feature_bank])
            self._pca_extractor = PCAConceptExtractor(n_concepts=10)
            self._pca_extractor.fit(feature_bank, domain=domain)
            self._pca_extractor.label_with_clip(vision_model)

        return self._pca_extractor.extract(features)

    def _predict_kmeans(self, features: np.ndarray, vision_model: Any = None) -> dict[str, float]:
        """Use K-Means-based auto-discovery for concept extraction."""
        from cbm.auto_concepts import KMeansConceptExtractor, load_feature_bank, build_feature_bank
        domain = self._concept_bank.domain

        if not hasattr(self, "_kmeans_extractor") or self._kmeans_extractor is None:
            feature_bank = load_feature_bank(domain)
            if feature_bank is None and vision_model is not None:
                feature_bank = build_feature_bank(vision_model, domain)
            if feature_bank is None:
                feature_bank = features.reshape(1, -1) + np.random.normal(0, 0.01, (50, features.shape[-1] if features.ndim > 1 else features.shape[0]))
                feature_bank = np.vstack([features.reshape(1, -1), feature_bank])
            self._kmeans_extractor = KMeansConceptExtractor(n_concepts=10)
            self._kmeans_extractor.fit(feature_bank, domain=domain)
            self._kmeans_extractor.label_with_clip(vision_model)

        return self._kmeans_extractor.extract(features)

    # ── training ─────────────────────────────────────────────────────
    def train_all(self, features: np.ndarray,
                  concept_labels: dict[str, np.ndarray]) -> dict[str, float]:
        """Train all supervised probes.

        Parameters
        ----------
        features : (N, feature_dim) array
        concept_labels : concept_name → (N,) binary array

        Returns
        -------
        dict of concept_name → training accuracy
        """
        accuracies = {}
        for name, probe in self._probes.items():
            if name in concept_labels:
                labels = concept_labels[name]
                if len(np.unique(labels)) < 2:
                    logger.warning("Skipping probe %r — only one class present", name)
                    continue
                acc = probe.train(features, labels)
                accuracies[name] = acc
        return accuracies

    # ── serialization ────────────────────────────────────────────────
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        for name, probe in self._probes.items():
            if probe.is_trained:
                safe_name = name.replace(" ", "_").replace("/", "_")
                probe.save(os.path.join(directory, f"{safe_name}.pkl"))
        logger.info("Saved %d trained probes to %s",
                     sum(1 for p in self._probes.values() if p.is_trained),
                     directory)

    def load(self, directory: str) -> int:
        """Load trained probes from directory.  Returns count loaded."""
        if not os.path.isdir(directory):
            logger.warning("Probe directory %s does not exist", directory)
            return 0
        loaded = 0
        skipped = 0
        for name, probe in self._probes.items():
            safe_name = name.replace(" ", "_").replace("/", "_")
            path = os.path.join(directory, f"{safe_name}.pkl")
            if os.path.isfile(path):
                probe.load(path)
                if probe.feature_dim != self._feature_dim:
                    probe._trained = False
                    skipped += 1
                    continue
                loaded += 1
        if skipped:
            logger.info("Skipped %d probes (dim %d != %d)", skipped, probe.feature_dim, self._feature_dim)
        logger.info("Loaded %d probes from %s", loaded, directory)
        return loaded


# ═══════════════════════════════════════════════════════════════════════
# Label-Free CBM — curated visual concept candidates
# Broad class-agnostic concepts following the Label-Free CBM approach
# (Oikarinen et al., 2023). Scored via CLIP at inference time.
# ═══════════════════════════════════════════════════════════════════════
_LABEL_FREE_CONCEPTS = {
    # Colors
    "red coloring": "an image showing red coloring on the subject",
    "blue coloring": "an image showing blue coloring on the subject",
    "yellow coloring": "an image showing yellow coloring on the subject",
    "orange coloring": "an image showing orange coloring on the subject",
    "green coloring": "an image showing green coloring on the subject",
    "black coloring": "an image showing black coloring on the subject",
    "white coloring": "an image showing white coloring on the subject",
    "brown coloring": "an image showing brown coloring on the subject",
    "grey coloring": "an image showing grey coloring on the subject",
    "iridescent sheen": "an image showing iridescent or shimmering coloring",
    # Textures & patterns
    "spotted pattern": "an image of a subject with spotted or speckled markings",
    "striped pattern": "an image of a subject with stripes or streaked markings",
    "solid color": "an image of a subject with uniform solid coloring",
    "multi-colored": "an image of a subject with multiple distinct colors",
    "smooth texture": "an image showing a smooth surface texture",
    "rough texture": "an image showing a rough or coarse texture",
    "feathered texture": "an image showing feather-like texture",
    "furry texture": "an image showing fur or hair-like texture",
    # Shapes & morphology
    "round shape": "an image of something round or circular in shape",
    "elongated shape": "an image of something long and slender",
    "pointed features": "an image showing sharp or pointed features",
    "curved features": "an image showing curved or rounded features",
    "compact body": "an image of a compact or stocky body shape",
    "slender body": "an image of a slender or thin body shape",
    "large size": "an image of a large subject relative to surroundings",
    "small size": "an image of a small subject relative to surroundings",
    # Body parts (general & bird-specific)
    "prominent beak": "an image showing a prominent beak or bill",
    "long beak": "an image of a bird with a long beak",
    "short thick beak": "an image of a bird with a short thick beak",
    "hooked beak": "an image of a bird with a hooked or curved beak",
    "crest on head": "an image showing a crest or tuft on the head",
    "long tail": "an image showing a long tail",
    "forked tail": "an image showing a forked or notched tail",
    "broad wings": "an image showing broad spread wings",
    "long legs": "an image showing long legs",
    "webbed feet": "an image showing webbed feet",
    "eye markings": "an image showing distinctive eye markings or eye ring",
    "facial mask": "an image showing a dark mask pattern on the face",
    "breast markings": "an image showing markings on the breast or chest",
    "wing bars": "an image showing distinct bars or bands on the wings",
    # Habitat & background
    "water background": "an image with water in the background",
    "sky background": "an image with sky in the background",
    "forest background": "an image with trees or forest in the background",
    "ground perched": "an image of a subject on the ground",
    "branch perched": "an image of a subject perched on a branch or wire",
    "in flight": "an image of a subject in flight or flying",
    # Behavior & posture
    "feeding posture": "an image showing a feeding or foraging posture",
    "alert posture": "an image showing an alert or upright posture",
    "resting posture": "an image showing a resting or relaxed posture",
    "wings spread": "an image showing spread or outstretched wings",
}
