"""Central configuration for the Interpretable Wrapper backend."""

from __future__ import annotations

import json
import os
import torch

# ── Device ───────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Paths ────────────────────────────────────────────────────────────
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROBE_DATA_DIR = os.path.join(_BACKEND_DIR, "data", "cbm")
ALT_PROBE_DATA_DIR = os.path.join(_BACKEND_DIR, "data", "probes")
SAMPLE_DIR = os.path.join(_BACKEND_DIR, "static", "samples")
_DOMAIN_SETTINGS_PATH = os.path.join(_BACKEND_DIR, "data", "domain_settings.json")

# ── Enabled domains ──────────────────────────────────────────────────
# Domains listed here are exposed via the API.  Comment out or remove
# entries to disable a domain.  Runtime changes via POST /domains/toggle
# are persisted to data/domain_settings.json.
_DEFAULT_ENABLED_DOMAINS = ["medical", "toxicity", "vision"]


def _load_enabled_domains() -> list[str]:
    """Load enabled domains from persisted settings, falling back to default."""
    if os.path.isfile(_DOMAIN_SETTINGS_PATH):
        try:
            with open(_DOMAIN_SETTINGS_PATH) as f:
                data = json.load(f)
            return data.get("enabled_domains", list(_DEFAULT_ENABLED_DOMAINS))
        except Exception:
            pass
    return list(_DEFAULT_ENABLED_DOMAINS)


def _save_enabled_domains(domains: list[str]) -> None:
    """Persist enabled domains to disk."""
    os.makedirs(os.path.dirname(_DOMAIN_SETTINGS_PATH), exist_ok=True)
    with open(_DOMAIN_SETTINGS_PATH, "w") as f:
        json.dump({"enabled_domains": domains}, f, indent=2)


ENABLED_DOMAINS: list[str] = _load_enabled_domains()


def set_domain_enabled(domain: str, enabled: bool) -> list[str]:
    """Enable or disable a domain at runtime and persist the change."""
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"Unknown domain: {domain!r}")
    if enabled and domain not in ENABLED_DOMAINS:
        ENABLED_DOMAINS.append(domain)
    elif not enabled and domain in ENABLED_DOMAINS:
        ENABLED_DOMAINS.remove(domain)
    _save_enabled_domains(ENABLED_DOMAINS)
    return list(ENABLED_DOMAINS)


def get_all_domains() -> list[str]:
    """Return all known domain names (enabled or not)."""
    return list(DOMAIN_CONFIG.keys())

# ── Domain configuration ─────────────────────────────────────────────
# Each domain entry defines everything needed to instantiate a model +
# CBM pipeline.  Adding a new domain only requires a new entry here
# plus the corresponding model file in backend/models/.
#
# ``models`` is a list of model variants available for the domain.
# The first entry with ``default: True`` (or the first entry if none
# is marked) is loaded on startup.
DOMAIN_CONFIG = {
    "medical": {
        "input_type": "image",
        "probe_type": "logistic",
        "description": "Lung X-ray classification (COVID-19 / Non-COVID / Normal)",
        "labels": ["COVID-19", "Non-COVID", "Normal"],
        "primary_target": None,
        "models": [
            {
                "id": "densenet121-chex",
                "name": "DenseNet-121 (CheXpert)",
                "model_class": "models.medical_model.MedicalModel",
                "model_id": "densenet121-res224-chex",
                "feature_dim": 1024,
                "default": True,
            },
            {
                "id": "densenet121-mimic",
                "name": "DenseNet-121 (MIMIC-CH)",
                "model_class": "models.medical_model.MedicalModel",
                "model_id": "densenet121-res224-mimic_ch",
                "feature_dim": 1024,
                "default": False,
            },
        ],
        # Backward-compatibility flat keys (default model)
        "model_class": "models.medical_model.MedicalModel",
        "model_id": "densenet121-res224-chex",
        "feature_dim": 1024,
    },
    "toxicity": {
        "input_type": "text",
        "probe_type": "logistic",
        "description": "Toxic language detection (toxic-bert)",
        "labels": ["toxic", "not toxic"],
        "primary_target": "toxic",
        "models": [
            {
                "id": "toxic-bert",
                "name": "Toxic-BERT",
                "model_class": "models.toxicity_model.ToxicityModel",
                "model_id": "unitary/toxic-bert",
                "feature_dim": 768,
                "default": True,
            },
            {
                "id": "detoxify-unbiased",
                "name": "Detoxify Unbiased",
                "model_class": "models.toxicity_model.ToxicityModel",
                "model_id": "unitary/unbiased-toxic-roberta",
                "feature_dim": 768,
                "default": False,
            },
        ],
        "model_class": "models.toxicity_model.ToxicityModel",
        "model_id": "unitary/toxic-bert",
        "feature_dim": 768,
    },
    "vision": {
        "input_type": "image",
        "probe_type": "logistic",
        "description": "Bird species classification (CUB-200 dataset)",
        "labels": [
            "Indigo Bunting", "Painted Bunting", "Cardinal", "American Crow",
            "Northern Flicker", "Frigatebird", "American Goldfinch", "Western Grebe",
            "Herring Gull", "Anna's Hummingbird", "Ruby-throated Hummingbird",
            "Blue Jay", "Dark-eyed Junco", "Belted Kingfisher", "Mallard",
            "Mockingbird", "Baltimore Oriole", "Brown Pelican", "Horned Puffin",
            "Common Raven", "American Redstart", "House Sparrow", "Barn Swallow",
            "Scarlet Tanager", "Brown Thrasher", "Yellow Warbler", "Cedar Waxwing",
            "Pileated Woodpecker", "Red-headed Woodpecker", "Common Yellowthroat",
        ],
        "primary_target": None,
        "models": [
            {
                "id": "resnet50-cub",
                "name": "ResNet-50 (CUB-200)",
                "model_class": "models.bird_model.BirdModel",
                "model_id": "resnet50-cub",
                "feature_dim": 2048,
                "default": True,
            },
            {
                "id": "clip-vit-b32",
                "name": "CLIP ViT-B/32",
                "model_class": "models.vision_model.VisionModel",
                "model_id": "openai/clip-vit-base-patch32",
                "feature_dim": 512,
                "default": False,
            },
        ],
        "model_class": "models.bird_model.BirdModel",
        "model_id": "resnet50-cub",
        "feature_dim": 2048,
    },
}

# ── Runtime-registered custom models ────────────────────────────────
# Populated via POST /models/register at runtime.
_custom_models: dict[str, list[dict]] = {}


def get_domain_models(domain: str) -> list[dict]:
    """Return all model variants for a domain (built-in + custom)."""
    cfg = DOMAIN_CONFIG.get(domain, {})
    models = list(cfg.get("models", []))
    models.extend(_custom_models.get(domain, []))
    return models


def get_default_model(domain: str) -> dict | None:
    """Return the default model config for a domain."""
    for m in get_domain_models(domain):
        if m.get("default"):
            return m
    models = get_domain_models(domain)
    return models[0] if models else None


def get_model_by_id(domain: str, model_id: str) -> dict | None:
    """Return a specific model config by its id."""
    for m in get_domain_models(domain):
        if m["id"] == model_id:
            return m
    return None


def register_custom_model(domain: str, model_spec: dict) -> str:
    """Register a custom model for a domain.  Returns the model id."""
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"Unknown domain: {domain!r}")
    model_spec.setdefault("default", False)
    model_spec.setdefault("custom", True)
    if domain not in _custom_models:
        _custom_models[domain] = []
    _custom_models[domain].append(model_spec)
    return model_spec["id"]


def unregister_custom_model(domain: str, model_id: str) -> bool:
    """Remove a custom model.  Returns True if found and removed."""
    if domain not in _custom_models:
        return False
    before = len(_custom_models[domain])
    _custom_models[domain] = [m for m in _custom_models[domain] if m["id"] != model_id]
    return len(_custom_models[domain]) < before

# ── Backward-compatibility mapping ──────────────────────────────────
# Old API used input_type="text"|"image"; map to canonical domain names.
INPUT_TYPE_TO_DOMAIN = {
    "text": "toxicity",
    "image": "medical",
}
