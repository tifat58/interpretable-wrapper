"""Surrogate counterfactual logic.

Given original and user-modified concept activations, compute a new
predicted label and confidence by treating concept deltas as a weighted
shift on the original confidence.
"""

from __future__ import annotations

# Concept importance weights per modality (higher = more impact on prediction)
_CONCEPT_WEIGHTS = {
    "text": {"insult": 0.35, "threat": 0.25, "obscene": 0.25, "identity_attack": 0.15},
    "image": {"opacity": 0.35, "cardiomegaly": 0.15, "effusion": 0.30, "consolidation": 0.20},
}

_LABEL_PAIRS = {
    "text": ("toxic", "not toxic"),
    "image": ("pneumonia", "normal"),
}


def compute_counterfactual(
    input_type: str,
    original_concepts: dict[str, float],
    modified_concepts: dict[str, float],
    original_confidence: float,
):
    """Return a counterfactual prediction dict after concept manipulation.

    Parameters
    ----------
    input_type : "text" | "image"
    original_concepts : concept name → original activation (0–1)
    modified_concepts : concept name → user-adjusted activation (0–1)
    original_confidence : the model's original prediction confidence

    Returns
    -------
    dict with keys: label, confidence, concept_deltas
    """
    weights = _CONCEPT_WEIGHTS.get(input_type, {})
    positive_label, negative_label = _LABEL_PAIRS.get(input_type, ("positive", "negative"))

    # Compute weighted delta  (positive delta → user increased a risky concept)
    total_delta = 0.0
    concept_deltas = {}
    for concept, orig_val in original_concepts.items():
        new_val = modified_concepts.get(concept, orig_val)
        delta = new_val - orig_val
        concept_deltas[concept] = round(delta, 3)
        w = weights.get(concept, 0.2)
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
