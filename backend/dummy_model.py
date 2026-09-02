from __future__ import annotations

import random


# Base concept scores for each modality / scenario
_TEXT_CONCEPTS = {
    "toxic": {"insult": 0.70, "threat": 0.30, "obscene": 0.50, "identity_attack": 0.10},
    "not toxic": {"insult": 0.10, "threat": 0.05, "obscene": 0.08, "identity_attack": 0.02},
}

_MEDICAL_CONCEPTS = {
    "COVID": {"Consolidation": 0.80, "Ground Glass Opacity": 0.75, "Bilateral Involvement": 0.65, "Lung Opacity": 0.60},
    "Pneumonia": {"Pleural Effusion": 0.60, "Cardiomegaly": 0.45, "Edema": 0.40, "Atelectasis": 0.35},
    "Normal": {"Clear Lung Fields": 0.90, "Consolidation": 0.05, "Lung Opacity": 0.08, "Edema": 0.03},
}

_BIRD_CONCEPTS = {
    "bird_positive": {"has_red": 0.60, "has_blue": 0.40, "curved_bill": 0.30, "small_bird": 0.70},
    "bird_default": {"has_brown": 0.50, "has_grey": 0.45, "cone_bill": 0.35, "small_bird": 0.60},
}

_VISION_CONCEPTS = {
    "animal_positive": {"furry": 0.70, "quadrapedal": 0.65, "big": 0.50, "hunter": 0.40},
    "animal_default": {"furry": 0.55, "quadrapedal": 0.50, "small": 0.45, "domestic": 0.35},
}

# Domain-specific label sets
_DOMAIN_LABELS = {
    "medical": ["COVID", "Pneumonia", "Normal"],
    "toxicity": ["toxic", "not toxic"],
    "vision": [
        "Indigo Bunting", "Cardinal", "Blue Jay", "American Crow",
        "Mallard", "House Sparrow", "Brown Pelican", "Common Raven",
    ],
}


def _add_noise(concepts, scale=0.05):
    """Add small uniform noise to concept scores, clamped to [0, 1]."""
    return {k: round(max(0.0, min(1.0, v + random.uniform(-scale, scale))), 3)
            for k, v in concepts.items()}


def _text_looks_toxic(text: str) -> bool:
    """Very simple keyword heuristic for the dummy model."""
    keywords = {"terrible", "hate", "stupid", "idiot", "kill", "die", "ugly",
                "dumb", "moron", "loser", "shut up", "worst", "disgusting",
                "insult", "threat", "obscene", "toxic"}
    lower = text.lower()
    return any(kw in lower for kw in keywords)


class DummyModel:
    """Simulated black-box model that returns predictions + concept activations.

    Used as fallback when the real domain model fails to load.
    Returns domain-appropriate labels and concepts.
    """

    def predict(self, input_type: str, data: str | None = None,
                domain: str | None = None):
        if input_type == "text":
            return self._predict_text(data or "")
        return self._predict_image(domain or "medical")

    # ------------------------------------------------------------------
    def _predict_text(self, text: str):
        is_toxic = _text_looks_toxic(text)
        label = "toxic" if is_toxic else "not toxic"
        confidence = round(random.uniform(0.75, 0.95) if is_toxic else random.uniform(0.70, 0.90), 3)
        concepts = _add_noise(_TEXT_CONCEPTS[label])
        return {"label": label, "confidence": confidence, "concepts": concepts}

    def _predict_image(self, domain: str):
        labels = _DOMAIN_LABELS.get(domain, ["positive", "negative"])
        label = random.choice(labels)
        confidence = round(random.uniform(0.55, 0.85), 3)

        if domain == "medical":
            concept_key = label if label in _MEDICAL_CONCEPTS else "Normal"
            concepts = _add_noise(_MEDICAL_CONCEPTS[concept_key])
        elif domain == "vision":
            concepts = _add_noise(_BIRD_CONCEPTS["bird_default"])
        else:
            concepts = {"feature_1": round(random.random(), 3), "feature_2": round(random.random(), 3)}

        return {"label": label, "confidence": confidence, "concepts": concepts}
