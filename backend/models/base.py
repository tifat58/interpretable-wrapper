"""Abstract base class for all domain models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseModel(ABC):
    """Interface every domain model must implement.

    Subclasses own their domain-specific preprocessing, inference, feature
    extraction, and attribution logic.  The CBM layer only interacts
    through this interface.
    """

    def __init__(self, domain: str, config: dict):
        self._domain = domain
        self._config = config
        self._loaded = False

    # ── properties ───────────────────────────────────────────────────
    @property
    def domain(self) -> str:
        return self._domain

    @property
    def input_type(self) -> str:
        return self._config["input_type"]

    @property
    def feature_dim(self) -> int:
        return self._config["feature_dim"]

    @property
    def device(self) -> str:
        from config import DEVICE
        return DEVICE

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── abstract methods (must override) ─────────────────────────────
    @abstractmethod
    def load(self) -> None:
        """Load model weights onto the configured device."""

    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """Domain-specific input preprocessing."""

    @abstractmethod
    def predict_raw(self, data: Any) -> dict:
        """Run black-box inference.

        Returns
        -------
        dict with at least ``label``, ``confidence``, and optionally
        ``raw_scores`` (full logit / probability vector).
        """

    @abstractmethod
    def extract_features(self, data: Any) -> np.ndarray:
        """Extract penultimate-layer activations for concept probing.

        Returns
        -------
        1-D numpy array of shape ``(feature_dim,)``.
        """

    # ── optional overrides ───────────────────────────────────────────
    def get_attribution(self, data: Any, target_concept: str | None = None) -> dict:
        """Compute an attribution / saliency map.

        Subclasses should override this.  Default returns an empty result.

        Returns
        -------
        dict with keys ``method``, ``type`` ("heatmap" | "tokens"),
        ``data``, and ``concept``.
        """
        return {"method": "none", "type": "none", "data": None, "concept": target_concept}
