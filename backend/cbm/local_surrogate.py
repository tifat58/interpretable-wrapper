"""LIME-style local surrogate for per-instance explanations.

Generates perturbations around a single input, queries the black-box
model for each, computes concept vectors, and trains a lightweight
local surrogate to approximate the model's behavior in that neighborhood.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class LocalSurrogate:
    """Per-instance local surrogate trained on perturbations."""

    def __init__(self):
        self._model = None
        self._type: str = "logistic"
        self._concept_names: list[str] = []
        self._importance_weights: dict[str, float] = {}
        self._fidelity_score: float = 0.0
        self._n_perturbations: int = 0
        self._classes: list[str] = []

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(
        self,
        model: Any,
        data: Any,
        probe_bank: Any,
        concept_bank: Any,
        n_perturbations: int = 100,
        surrogate_type: str = "logistic",
        vision_model: Any = None,
    ) -> None:
        """Fit a local surrogate around the given input instance.

        Parameters
        ----------
        model : domain model (BaseModel subclass)
        data : raw input (text string or base64 image)
        probe_bank : ProbeBank to compute concept vectors
        concept_bank : ConceptBank for concept names
        n_perturbations : number of perturbations to generate
        surrogate_type : "logistic" | "ridge" | "tree"
        vision_model : VisionModel for CLIP scoring (if needed)
        """
        self._type = surrogate_type
        self._n_perturbations = n_perturbations
        self._concept_names = concept_bank.concepts

        # 1. Get original prediction and concepts
        orig_pred = model.predict_raw(data)
        orig_features = model.extract_features(data)
        orig_concepts = probe_bank.predict_concepts(
            orig_features, raw_input=data, vision_model=vision_model,
        )

        # 2. Generate perturbations and collect concept vectors + predictions
        concept_matrix = []
        predictions = []
        input_type = getattr(model, "input_type", None) or model._config.get("input_type", "text")

        # Add original
        concept_vec = [orig_concepts.get(c, 0.5) for c in self._concept_names]
        concept_matrix.append(concept_vec)
        predictions.append(orig_pred["label"])

        for i in range(n_perturbations):
            try:
                perturbed = self._perturb(data, input_type, i, n_perturbations)
                pred = model.predict_raw(perturbed)
                features = model.extract_features(perturbed)
                concepts = probe_bank.predict_concepts(
                    features, raw_input=perturbed, vision_model=vision_model,
                )
                vec = [concepts.get(c, 0.5) for c in self._concept_names]
                concept_matrix.append(vec)
                predictions.append(pred["label"])
            except Exception:
                continue

        if len(concept_matrix) < 5:
            logger.warning("Too few successful perturbations (%d)", len(concept_matrix))
            return

        X = np.array(concept_matrix)
        y = np.array(predictions)

        # Get unique labels
        unique_labels = sorted(set(y))
        self._classes = list(unique_labels)

        if len(unique_labels) < 2:
            # If all perturbations predict the same class, duplicate with noise
            logger.info("All perturbations same class; adding noise samples")
            for _ in range(min(10, n_perturbations)):
                noisy = X[0] + np.random.normal(0, 0.1, X.shape[1])
                noisy = np.clip(noisy, 0, 1)
                X = np.vstack([X, noisy.reshape(1, -1)])
                y = np.append(y, f"not_{unique_labels[0]}")
            self._classes = sorted(set(y))

        # 3. Train local surrogate
        self._model = self._create_model(surrogate_type)
        self._model.fit(X, y)

        # 4. Compute fidelity
        y_pred = self._model.predict(X)
        self._fidelity_score = float(np.mean(y_pred == y))

        # 5. Extract importance weights
        self._importance_weights = self._extract_weights()

        logger.info(
            "Local surrogate (%s) fitted — %d samples, fidelity %.3f",
            surrogate_type, len(X), self._fidelity_score,
        )

    @staticmethod
    def _create_model(surrogate_type: str):
        if surrogate_type == "ridge":
            from sklearn.linear_model import RidgeClassifier
            return RidgeClassifier(alpha=1.0)
        elif surrogate_type == "tree":
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier(max_depth=5, random_state=42)
        else:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                max_iter=200, solver="saga", C=1.0, class_weight="balanced",
            )

    def _extract_weights(self) -> dict[str, float]:
        """Extract feature importance from the fitted model."""
        weights = {}
        if self._model is None:
            return weights

        try:
            if hasattr(self._model, "coef_"):
                coef = self._model.coef_
                if coef.ndim > 1:
                    # Multi-class: use mean absolute across classes
                    importance = np.mean(np.abs(coef), axis=0)
                    # Also provide signed weights for the primary class
                    signed = coef[0] if coef.shape[0] == 1 else np.mean(coef, axis=0)
                else:
                    importance = np.abs(coef)
                    signed = coef

                for i, name in enumerate(self._concept_names):
                    if i < len(signed):
                        weights[name] = round(float(signed[i]), 4)

            elif hasattr(self._model, "feature_importances_"):
                fi = self._model.feature_importances_
                for i, name in enumerate(self._concept_names):
                    if i < len(fi):
                        weights[name] = round(float(fi[i]), 4)
        except Exception:
            logger.debug("Failed to extract weights", exc_info=True)

        return weights

    def _perturb(self, data: Any, input_type: str, idx: int, total: int) -> Any:
        """Generate a perturbation of the input."""
        if input_type == "text":
            return self._perturb_text(data, idx, total)
        else:
            return self._perturb_image(data, idx, total)

    @staticmethod
    def _perturb_text(text: str, idx: int, total: int) -> str:
        """Perturb text by randomly dropping/masking tokens."""
        words = text.split()
        if not words:
            return text
        rng = np.random.RandomState(idx)
        # Drop fraction of words (10-50%)
        drop_rate = 0.1 + 0.4 * (idx / max(total, 1))
        mask = rng.random(len(words)) > drop_rate
        if not mask.any():
            mask[0] = True  # keep at least one word
        return " ".join(w for w, keep in zip(words, mask) if keep)

    @staticmethod
    def _perturb_image(data: Any, idx: int, total: int) -> Any:
        """Perturb image by applying random occlusion masks."""
        import base64
        import io
        from PIL import Image, ImageFilter

        # Decode image
        if isinstance(data, str):
            if "," in data[:80]:
                data = data.split(",", 1)[1]
            raw = base64.b64decode(data)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        elif isinstance(data, Image.Image):
            img = data.convert("RGB")
        else:
            return data

        w, h = img.size
        rng = np.random.RandomState(idx)
        img_arr = np.array(img)

        # Random rectangular occlusion
        n_rects = rng.randint(1, 4)
        for _ in range(n_rects):
            rx = rng.randint(0, max(w - 20, 1))
            ry = rng.randint(0, max(h - 20, 1))
            rw = rng.randint(20, max(w // 3, 21))
            rh = rng.randint(20, max(h // 3, 21))
            # Randomly choose occlusion type
            occ_type = rng.randint(0, 3)
            if occ_type == 0:
                img_arr[ry:ry+rh, rx:rx+rw] = 0  # black out
            elif occ_type == 1:
                img_arr[ry:ry+rh, rx:rx+rw] = 128  # gray out
            else:
                # Add noise
                noise = rng.randint(0, 256, (min(rh, h-ry), min(rw, w-rx), 3), dtype=np.uint8)
                img_arr[ry:min(ry+rh, h), rx:min(rx+rw, w)] = noise

        result = Image.fromarray(img_arr)
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def get_importance_weights(self) -> dict[str, float]:
        """Return concept importance weights from the local surrogate."""
        return dict(self._importance_weights)

    def get_fidelity_score(self) -> float:
        """Return fidelity score (agreement with black-box on neighborhood)."""
        return self._fidelity_score

    def predict(self, concept_vector: dict[str, float]) -> dict:
        """Predict using the local surrogate from modified concepts."""
        if self._model is None:
            return {"label": "unknown", "confidence": 0.5}

        vec = np.array([concept_vector.get(c, 0.5) for c in self._concept_names]).reshape(1, -1)

        label = str(self._model.predict(vec)[0])

        # Get confidence if available
        confidence = 0.5
        if hasattr(self._model, "predict_proba"):
            try:
                proba = self._model.predict_proba(vec)[0]
                confidence = round(float(proba.max()), 4)
            except Exception:
                pass
        elif hasattr(self._model, "decision_function"):
            try:
                dec = self._model.decision_function(vec)
                # Sigmoid for confidence
                if dec.ndim > 1:
                    confidence = round(float(1.0 / (1.0 + np.exp(-dec.max()))), 4)
                else:
                    confidence = round(float(1.0 / (1.0 + np.exp(-float(dec)))), 4)
            except Exception:
                pass

        return {"label": label, "confidence": confidence}

    def to_dict(self) -> dict:
        """Serialize surrogate info for API response."""
        return {
            "importance_weights": self._importance_weights,
            "fidelity_score": self._fidelity_score,
            "surrogate_type": self._type,
            "n_perturbations": self._n_perturbations,
            "n_concepts": len(self._concept_names),
            "classes": self._classes,
        }
