"""Template-based natural-language explanation generator."""

from __future__ import annotations

# Evidence snippets grounded in each concept (2-3 sentences each)
_EVIDENCE = {
    # ---- text / toxicity ----
    "insult": (
        "The input contains language that demeans or belittles the target. "
        "Research shows that direct insults are among the strongest predictors "
        "of perceived toxicity in online comments (Wulczyn et al., 2017)."
    ),
    "threat": (
        "The input includes expressions that may be interpreted as threats of "
        "harm. Threatening language is a key factor in content moderation "
        "systems and is penalised heavily in toxicity classifiers."
    ),
    "obscene": (
        "The input uses profane or obscene language. Swear words and vulgar "
        "expressions raise toxicity scores even when used non-aggressively "
        "(Perspective API documentation, 2023)."
    ),
    "identity_attack": (
        "The input targets a person or group based on a protected identity "
        "attribute. Identity-based attacks are weighted heavily in hate-speech "
        "detection models."
    ),
    # ---- image / radiology ----
    "opacity": (
        "Increased lung opacity on the chest X-ray suggests fluid, "
        "consolidation, or mass in the pulmonary parenchyma. Ground-glass "
        "and dense opacities are hallmarks of pneumonia."
    ),
    "cardiomegaly": (
        "The cardiothoracic ratio exceeds the 0.5 threshold, indicating "
        "an enlarged cardiac silhouette. Cardiomegaly may co-occur with "
        "pulmonary congestion and effusion."
    ),
    "effusion": (
        "Blunting of the costophrenic angle suggests pleural effusion. "
        "Effusions often accompany infectious and cardiac pathologies and "
        "contribute to the model's pneumonia prediction."
    ),
    "consolidation": (
        "Dense consolidation in the lung fields indicates alveolar filling, "
        "a classic radiographic sign of bacterial pneumonia (Franquet, 2018)."
    ),
}


def generate_explanation(
    input_type: str,
    label: str,
    confidence: float,
    concepts: dict[str, float],
    evidence: bool = False,
):
    """Return a natural-language explanation dict.

    Parameters
    ----------
    input_type : "text" | "image"
    label : predicted label string
    confidence : model confidence (0–1)
    concepts : concept name → activation score
    evidence : whether to include grounded evidence snippets

    Returns
    -------
    dict with keys: explanation_text, evidence_snippets (list of dicts)
    """
    # Sort concepts by activation (descending) for ranking
    ranked = sorted(concepts.items(), key=lambda x: x[1], reverse=True)
    top_concept, top_value = ranked[0]

    # ---- Main explanation sentence ----
    explanation = (
        f"The model predicts \"{label}\" (confidence {confidence:.0%}) "
        f"primarily because the concept \"{top_concept}\" is activated at "
        f"{top_value:.0%}."
    )

    # Add secondary concepts
    if len(ranked) > 1:
        secondary = ", ".join(
            f"\"{c}\" ({v:.0%})" for c, v in ranked[1:]
        )
        explanation += f" Other contributing concepts are: {secondary}."

    # ---- Evidence snippets (optional) ----
    evidence_snippets = []
    if evidence:
        for concept, value in ranked:
            snippet = _EVIDENCE.get(concept)
            if snippet:
                evidence_snippets.append({
                    "concept": concept,
                    "activation": value,
                    "text": snippet,
                })

    return {
        "explanation_text": explanation,
        "evidence_snippets": evidence_snippets,
    }
