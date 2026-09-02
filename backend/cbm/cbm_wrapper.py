"""Post-hoc Concept Bottleneck Model orchestrator.

Brings together a domain model, concept bank, probe bank, attribution
engine, and learned surrogate into one coherent pipeline.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from cbm.attribution import AttributionEngine
from cbm.concept_bank import ConceptBank
from cbm.probe import ProbeBank

logger = logging.getLogger(__name__)


class PostHocCBM:
    """Full post-hoc CBM pipeline for a single domain.

    Usage
    -----
    >>> cbm = PostHocCBM(domain="medical", model=model,
    ...                   concept_bank=cb, probe_bank=pb)
    >>> result = cbm.predict(image_data)
    """

    def __init__(self, domain: str, model: Any, concept_bank: ConceptBank,
                 probe_bank: ProbeBank):
        self._domain = domain
        self._model = model
        self._concept_bank = concept_bank
        self._probe_bank = probe_bank
        self._attribution = AttributionEngine(model)

        # Learned surrogate: concept activations → label
        self._surrogate: LogisticRegression | None = None
        self._surrogate_classes: list[str] = []

    # ── properties ───────────────────────────────────────────────────
    @property
    def domain(self) -> str:
        return self._domain

    @property
    def concept_bank(self) -> ConceptBank:
        return self._concept_bank

    @property
    def has_surrogate(self) -> bool:
        return self._surrogate is not None

    @property
    def model(self) -> Any:
        return self._model

    # ── predict ──────────────────────────────────────────────────────
    def predict(self, data: Any,
                concept_strategy: str | None = None,
                custom_concepts: list[str] | None = None) -> dict:
        """Full pipeline: raw prediction + concept activations.

        Parameters
        ----------
        concept_strategy : override probe strategy (pca, kmeans, custom, …)
        custom_concepts  : user-defined concept descriptions (for strategy="custom")

        Returns
        -------
        dict with keys: ``label``, ``confidence``, ``concepts``, ``domain``,
        ``concept_strategy``.
        """
        # 1. Raw black-box prediction
        raw = self._model.predict_raw(data)
        label = raw["label"]
        confidence = raw["confidence"]

        # 2. Feature extraction
        features = self._model.extract_features(data)

        # 3. Concept activations via probe bank
        needs_model = (
            self._probe_bank.probe_type in ("clip", "hybrid")
            or concept_strategy in ("clip", "custom", "pca", "kmeans", "token_aggregation", "label_free")
        )
        vision_model = self._model if needs_model else None
        concepts = self._probe_bank.predict_concepts(
            features, raw_input=data, vision_model=vision_model,
            strategy=concept_strategy, custom_concepts=custom_concepts,
        )

        return {
            "label": label,
            "confidence": confidence,
            "concepts": concepts,
            "domain": self._domain,
            "concept_strategy": concept_strategy or self._probe_bank.probe_type,
        }

    # ── attribution ──────────────────────────────────────────────────
    def get_concept_attribution(self, data: Any,
                                concept_name: str | None = None) -> dict:
        """Compute saliency / attribution map for a concept."""
        return self._attribution.compute(data, concept_name=concept_name)

    # ── counterfactual ───────────────────────────────────────────────
    def counterfactual(self, original_concepts: dict[str, float],
                       modified_concepts: dict[str, float],
                       original_confidence: float) -> dict:
        """Compute counterfactual prediction from modified concept activations.

        If a surrogate model is available, use its learned weights.
        Otherwise, use a linear interpolation fallback.
        """
        concept_names = self._concept_bank.concepts
        concept_deltas = {}
        for c in concept_names:
            orig = original_concepts.get(c, 0.5)
            new = modified_concepts.get(c, orig)
            concept_deltas[c] = round(new - orig, 4)

        if self._surrogate is not None:
            return self._counterfactual_surrogate(
                modified_concepts, concept_names, concept_deltas,
            )
        return self._counterfactual_linear(
            original_concepts, modified_concepts, original_confidence,
            concept_names, concept_deltas,
        )

    def _counterfactual_surrogate(self, modified_concepts: dict[str, float],
                                  concept_names: list[str],
                                  concept_deltas: dict[str, float]) -> dict:
        """Use the learned surrogate for counterfactual prediction."""
        vec = np.array([modified_concepts.get(c, 0.5) for c in concept_names]).reshape(1, -1)
        proba = self._surrogate.predict_proba(vec)[0]
        pred_idx = int(proba.argmax())
        label = self._surrogate_classes[pred_idx]
        confidence = round(float(proba[pred_idx]), 4)
        return {"label": label, "confidence": confidence, "concept_deltas": concept_deltas}

    @staticmethod
    def _counterfactual_linear(original_concepts: dict[str, float],
                               modified_concepts: dict[str, float],
                               original_confidence: float,
                               concept_names: list[str],
                               concept_deltas: dict[str, float]) -> dict:
        """Linear fallback when no surrogate is trained."""
        # Equal weights
        n = max(len(concept_names), 1)
        total_delta = sum(concept_deltas.values()) / n
        new_conf = max(0.01, min(0.99, round(original_confidence + total_delta, 4)))
        label = "positive" if new_conf >= 0.5 else "negative"
        return {"label": label, "confidence": new_conf, "concept_deltas": concept_deltas}

    # ── surrogate training ───────────────────────────────────────────
    def train_surrogate(self, concept_matrix: np.ndarray,
                        labels: np.ndarray,
                        class_names: list[str] | None = None) -> float:
        """Train a logistic-regression surrogate on concept activations → label.

        Returns training accuracy.
        """
        self._surrogate = LogisticRegression(
            max_iter=200, solver="saga", C=1.0, class_weight="balanced",
        )
        self._surrogate.fit(concept_matrix, labels)
        acc = float(self._surrogate.score(concept_matrix, labels))

        if class_names is not None:
            self._surrogate_classes = class_names
        else:
            self._surrogate_classes = [str(c) for c in self._surrogate.classes_]

        logger.info("Surrogate trained for %r — accuracy %.3f", self._domain, acc)
        return acc

    def fit_local_surrogate(self, data: Any,
                            concept_activations: dict[str, float],
                            surrogate_type: str = "logistic",
                            n_perturbations: int = 200) -> dict:
        """Fit a LIME-style local surrogate for a specific input instance.

        Returns serialised surrogate info (weights, fidelity, etc.)
        """
        from cbm.local_surrogate import LocalSurrogate

        needs_model = (
            self._probe_bank.probe_type in ("clip", "hybrid")
            or True  # always pass for flexibility
        )
        vision_model = self._model if needs_model else None

        surrogate = LocalSurrogate()
        surrogate.fit(
            model=self._model,
            data=data,
            probe_bank=self._probe_bank,
            concept_bank=self._concept_bank,
            n_perturbations=n_perturbations,
            surrogate_type=surrogate_type,
            vision_model=vision_model,
        )
        return surrogate.to_dict()

    # ── serialization ────────────────────────────────────────────────
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        # Save probes
        self._probe_bank.save(os.path.join(directory, "probes"))
        # Save surrogate
        if self._surrogate is not None:
            with open(os.path.join(directory, "surrogate.pkl"), "wb") as f:
                pickle.dump({"model": self._surrogate,
                             "classes": self._surrogate_classes}, f)
        logger.info("CBM state saved to %s", directory)

    def load(self, directory: str) -> None:
        # Load probes
        probe_dir = os.path.join(directory, "probes")
        self._probe_bank.load(probe_dir)
        # Load surrogate
        surr_path = os.path.join(directory, "surrogate.pkl")
        if os.path.isfile(surr_path):
            with open(surr_path, "rb") as f:
                data = pickle.load(f)  # noqa: S301
            self._surrogate = data["model"]
            self._surrogate_classes = data["classes"]
            logger.info("Loaded surrogate for %r", self._domain)

    # ── extension hooks ────────────────────────────────────────────────
    def explain(self, label: str, confidence: float,
                concepts: dict[str, float], evidence: bool = False) -> dict:
        """Generate explanation using RAG engine (falls back to templates)."""
        try:
            from rag_engine import get_rag_engine
            engine = get_rag_engine()
            return engine.generate_explanation(
                self._domain, label, confidence, concepts, evidence,
            )
        except Exception:
            logger.debug("RAG unavailable, falling back to template", exc_info=True)
            from explanations import generate_explanation
            return generate_explanation(
                self._domain, label, confidence, concepts, evidence,
            )

    def edit_input(self, data: Any, edit_spec: dict) -> dict:
        """Edit the raw input and re-run prediction for true counterfactual."""
        from input_editing import apply_edit
        return apply_edit(self._domain, data, edit_spec, self)
