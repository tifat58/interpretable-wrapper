"""Model registry with lazy loading."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.base import BaseModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Lazy-loading registry that instantiates domain models on first access.

    Supports multiple models per domain keyed by ``(domain, model_id)``.
    """

    def __init__(self, domain_config: dict):
        self._config = domain_config
        self._instances: dict[tuple[str, str], BaseModel] = {}

    # ── public API ───────────────────────────────────────────────────
    def get(self, domain: str, model_id: str | None = None) -> BaseModel:
        """Return the model for *domain*, loading it on first call.

        Parameters
        ----------
        domain : domain name
        model_id : specific model variant id.  If ``None``, use the
                   default model for the domain.
        """
        from config import get_default_model, get_model_by_id

        if model_id is None:
            model_spec = get_default_model(domain)
        else:
            model_spec = get_model_by_id(domain, model_id)

        if model_spec is None:
            raise KeyError(f"Unknown model {model_id!r} for domain {domain!r}")

        effective_id = model_spec["id"]
        cache_key = (domain, effective_id)

        if cache_key in self._instances:
            return self._instances[cache_key]

        cfg = self._config.get(domain)
        if cfg is None:
            raise KeyError(f"Unknown domain: {domain!r}")

        # Build an effective config merging domain-level and model-level settings
        effective_cfg = {**cfg, **model_spec}

        model = self._import_and_instantiate(domain, effective_cfg)
        model.load()
        self._instances[cache_key] = model
        logger.info("Loaded model %r for domain %r on %s",
                     effective_id, domain, model.device)
        return model

    def available_domains(self) -> list[str]:
        return list(self._config.keys())

    def is_loaded(self, domain: str, model_id: str | None = None) -> bool:
        if model_id:
            return (domain, model_id) in self._instances
        return any(k[0] == domain for k in self._instances)

    def unload(self, domain: str, model_id: str | None = None) -> None:
        """Free GPU memory for a specific domain/model."""
        if model_id:
            inst = self._instances.pop((domain, model_id), None)
            if inst is not None:
                del inst
                logger.info("Unloaded model %r for domain %r", model_id, domain)
        else:
            keys = [k for k in self._instances if k[0] == domain]
            for k in keys:
                del self._instances[k]
            if keys:
                logger.info("Unloaded all models for domain %r", domain)

    # ── internals ────────────────────────────────────────────────────
    @staticmethod
    def _import_and_instantiate(domain: str, cfg: dict) -> BaseModel:
        module_path, class_name = cfg["model_class"].rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(domain=domain, config=cfg)
