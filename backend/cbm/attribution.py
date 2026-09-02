"""Attribution / saliency orchestration layer.

Thin wrapper around each model's ``get_attribution()`` that adds
normalisation, caching, and a uniform response format.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from models.base import BaseModel

logger = logging.getLogger(__name__)


class AttributionEngine:
    """Computes and post-processes attribution maps for any domain model."""

    def __init__(self, model: BaseModel):
        self._model = model

    def compute(self, data: Any, concept_name: str | None = None,
                method: str = "auto") -> dict:
        """Compute attribution for a given input and optional concept.

        Parameters
        ----------
        data : raw input data (text string or image)
        concept_name : concept to attribute to; ``None`` → model default
        method : "auto" (pick best per domain), "gradcam", "attention",
                 "clip_spatial"

        Returns
        -------
        dict with keys: ``method``, ``type``, ``data``, ``concept``,
        and optionally ``shape``.
        """
        if method == "auto":
            method = self._auto_method()

        result = self._model.get_attribution(data, target_concept=concept_name)

        # Ensure standardised keys are present
        result.setdefault("method", method)
        result.setdefault("type", "none")
        result.setdefault("data", None)
        result.setdefault("concept", concept_name)

        return result

    # ── internals ────────────────────────────────────────────────────
    def _auto_method(self) -> str:
        input_type = self._model.input_type
        domain = self._model.domain
        if domain == "vision":
            return "clip_spatial"
        if input_type == "image":
            return "gradcam"
        return "attention"
