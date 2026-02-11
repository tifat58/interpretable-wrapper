from __future__ import annotations

import random


# Base concept scores for each modality / scenario
_TEXT_CONCEPTS = {
    "toxic": {"insult": 0.70, "threat": 0.30, "obscene": 0.50, "identity_attack": 0.10},
    "not toxic": {"insult": 0.10, "threat": 0.05, "obscene": 0.08, "identity_attack": 0.02},
}

_IMAGE_CONCEPTS = {
    "pneumonia": {"opacity": 0.80, "cardiomegaly": 0.20, "effusion": 0.60, "consolidation": 0.40},
    "normal": {"opacity": 0.10, "cardiomegaly": 0.05, "effusion": 0.08, "consolidation": 0.06},
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
    """Simulated black-box model that returns predictions + concept activations."""

    def predict(self, input_type: str, data: str | None = None):
        if input_type == "text":
            return self._predict_text(data or "")
        return self._predict_image()

    # ------------------------------------------------------------------
    def _predict_text(self, text: str):
        is_toxic = _text_looks_toxic(text)
        label = "toxic" if is_toxic else "not toxic"
        confidence = round(random.uniform(0.75, 0.95) if is_toxic else random.uniform(0.70, 0.90), 3)
        concepts = _add_noise(_TEXT_CONCEPTS[label])
        return {"label": label, "confidence": confidence, "concepts": concepts}

    def _predict_image(self):
        is_positive = random.random() > 0.4  # biased toward pneumonia for demo
        label = "pneumonia" if is_positive else "normal"
        confidence = round(random.uniform(0.70, 0.92), 3)
        concepts = _add_noise(_IMAGE_CONCEPTS[label])
        return {"label": label, "confidence": confidence, "concepts": concepts}
