"""Surrogate counterfactual logic.

Given original and user-modified concept activations, compute a new
predicted label and confidence by treating concept deltas as a weighted
shift on the original confidence.

When a trained PostHocCBM surrogate is available the CBM layer calls its
own ``counterfactual()`` instead of this module.  This module is the
fallback for domains without a surrogate.
"""

from __future__ import annotations

# Concept importance weights per domain (higher = more impact on prediction)
_CONCEPT_WEIGHTS = {
    # Legacy / fallback keys (text → toxicity, image → medical)
    "text": {"insult": 0.35, "threat": 0.25, "obscene": 0.25, "identity_attack": 0.15},
    "image": {"opacity": 0.35, "cardiomegaly": 0.15, "effusion": 0.30, "consolidation": 0.20},
    # Domain-keyed
    "toxicity": {"insult": 0.30, "threat": 0.20, "obscene": 0.20, "identity_hate": 0.10,
                 "severe_toxic": 0.35, "sexually_explicit": 0.05, "flirtation": 0.02,
                 "profanity_score": 0.15, "caps_ratio": 0.05, "negative_sentiment": 0.10},
    "medical": {"Pneumonia": 0.30, "Consolidation": 0.20, "Lung Opacity": 0.15,
                "Pleural Effusion": 0.15, "Edema": 0.10, "Cardiomegaly": 0.08,
                "Atelectasis": 0.08, "Pneumothorax": 0.05},
    "vision": {},  # equal-weight fallback for bird concepts
}

_LABEL_PAIRS = {
    "text": ("toxic", "not toxic"),
    "image": ("covid-19", "normal"),
    "toxicity": ("toxic", "not toxic"),
    "medical": ("COVID", "Normal"),
    "vision": ("identified", "unknown"),
}


def compute_counterfactual(
    domain_or_type: str,
    original_concepts: dict[str, float],
    modified_concepts: dict[str, float],
    original_confidence: float,
    surrogate=None,
):
    """Return a counterfactual prediction dict after concept manipulation.

    Parameters
    ----------
    domain_or_type : domain name ("medical", "toxicity", "vision") or
                     legacy input_type ("text" | "image")
    original_concepts : concept name → original activation (0–1)
    modified_concepts : concept name → user-adjusted activation (0–1)
    original_confidence : the model's original prediction confidence
    surrogate : optional trained sklearn model (from PostHocCBM)

    Returns
    -------
    dict with keys: label, confidence, concept_deltas
    """
    weights = _CONCEPT_WEIGHTS.get(domain_or_type, {})
    positive_label, negative_label = _LABEL_PAIRS.get(
        domain_or_type, ("positive", "negative"),
    )

    # Compute weighted delta  (positive delta → user increased a risky concept)
    total_delta = 0.0
    concept_deltas = {}
    default_weight = 1.0 / max(len(original_concepts), 1)

    for concept, orig_val in original_concepts.items():
        new_val = modified_concepts.get(concept, orig_val)
        delta = new_val - orig_val
        concept_deltas[concept] = round(delta, 3)
        w = weights.get(concept, default_weight)
        total_delta += delta * w

    new_confidence = max(0.01, min(0.99, round(original_confidence + total_delta, 3)))

    if new_confidence >= 0.5:
        label = positive_label
    else:
        label = negative_label

    return {
        "label": label,
        "confidence": new_confidence,
        "concept_deltas": concept_deltas,
    }
