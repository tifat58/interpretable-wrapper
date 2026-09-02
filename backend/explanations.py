"""Template-based natural-language explanation generator.

Phase 1A: uses concept descriptions from ConceptBank when available,
falls back to hardcoded evidence snippets.  Phase 1B will swap in
a RAG engine.
"""

from __future__ import annotations

# Evidence snippets grounded in each concept (2-3 sentences each)
_EVIDENCE = {
    # ── toxicity ──────────────────────────────────────────────────────
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
    "identity_hate": (
        "The input targets a person or group based on a protected identity "
        "attribute. Identity-based attacks are weighted heavily in hate-speech "
        "detection models."
    ),
    "severe_toxic": (
        "The input is flagged as severely toxic, indicating extreme hostility, "
        "aggression, or dehumanizing language that goes beyond casual rudeness."
    ),
    "sexually_explicit": (
        "The input contains sexually explicit or graphic content. Such content "
        "is penalized in toxicity scoring even when not directed at a person."
    ),
    "flirtation": (
        "The input contains unwanted flirtatious or suggestive language that "
        "may be perceived as harassment in context."
    ),
    "profanity_score": (
        "A high profanity score indicates heavy use of explicit or vulgar "
        "language throughout the text."
    ),
    "caps_ratio": (
        "A high proportion of ALL-CAPS text often signals shouting or "
        "aggressive intent, which correlates with perceived toxicity."
    ),
    "negative_sentiment": (
        "The text has an overall negative emotional tone, which contributes "
        "to the perception of hostility even without explicit toxic words."
    ),
    # ── medical (lung X-ray — COVID-19 / Non-COVID / Normal) ────────
    "Consolidation": (
        "Dense consolidation in the lung fields indicates alveolar filling, "
        "a hallmark of bacterial pneumonia and severe COVID-19 infection."
    ),
    "Ground Glass Opacity": (
        "Ground-glass opacities (GGOs) are hazy areas without obscuration "
        "of underlying structures, commonly seen in early COVID-19 and "
        "viral pneumonias (Shi et al., 2020)."
    ),
    "Lung Opacity": (
        "Increased lung opacity suggests fluid, consolidation, or "
        "inflammatory infiltrate in the pulmonary parenchyma."
    ),
    "Pleural Effusion": (
        "Blunting of the costophrenic angle suggests pleural effusion. "
        "Effusions are more common in non-COVID pathologies like heart "
        "failure and bacterial infections."
    ),
    "Cardiomegaly": (
        "The cardiothoracic ratio exceeds 0.5, indicating an enlarged "
        "cardiac silhouette. Cardiomegaly suggests underlying cardiac "
        "disease rather than primary pulmonary infection."
    ),
    "Edema": (
        "Pulmonary edema with increased interstitial markings and possible "
        "alveolar flooding, suggesting fluid overload or cardiac failure."
    ),
    "Atelectasis": (
        "Partial or complete collapse of lung tissue, which may indicate "
        "post-operative changes, mucus plugging, or compression."
    ),
    "Bilateral Involvement": (
        "Abnormalities are present in both lung fields, a pattern strongly "
        "associated with COVID-19 pneumonia, especially in moderate-to-severe "
        "cases (Bernheim et al., 2020)."
    ),
    "Peripheral Distribution": (
        "Opacities are predominantly in the peripheral lung zones, a "
        "characteristic distribution pattern of COVID-19 pneumonia."
    ),
    "Air Bronchogram": (
        "Air-filled bronchi are visible within opacified lung parenchyma, "
        "indicating consolidation surrounding patent airways."
    ),
    "Lung Volume Loss": (
        "Decreased lung volume suggesting a restrictive process, atelectasis, "
        "or post-inflammatory fibrotic changes."
    ),
    "Clear Lung Fields": (
        "Lung fields appear clear with no significant opacities or "
        "infiltrates, consistent with a normal chest X-ray."
    ),
    # ── birds (CUB-200 visual attributes) ────────────────────────────
    "has_red": "Red plumage is visible, commonly found in cardinals, tanagers, and finches.",
    "has_blue": "Blue coloring is present, characteristic of jays, bluebirds, and buntings.",
    "has_yellow": "Yellow plumage is visible, typical of goldfinches, warblers, and orioles.",
    "has_orange": "Orange coloring is present, seen in orioles, robins, and some warblers.",
    "has_black": "Black plumage is visible, common in crows, blackbirds, and many raptors.",
    "has_white": "White coloring is present, found in egrets, gulls, and some woodpeckers.",
    "has_brown": "Brown plumage is visible, the most common bird color across many species.",
    "has_grey": "Grey coloring is present, typical of mockingbirds, chickadees, and some flycatchers.",
    "has_green": "Green plumage is visible, common in hummingbirds, parakeets, and vireos.",
    "has_iridescent": "Iridescent plumage with a shimmering quality, found in hummingbirds, grackles, and starlings.",
    "curved_bill": "The bill is curved (up or down), an adaptation for probing flowers or mud.",
    "hooked_bill": "The bill is hooked like a raptor's, used for tearing prey or tough food.",
    "dagger_bill": "A long straight dagger-shaped bill, typical of herons and kingfishers.",
    "cone_bill": "A short thick cone-shaped bill, adapted for cracking seeds, found in sparrows and finches.",
    "long_bill": "The bill is longer than the head, an adaptation for probing or catching insects in flight.",
    "spotted_pattern": "Spotted or speckled markings on the body, providing camouflage or species identification.",
    "striped_pattern": "Striped or streaked markings, common in sparrows, thrushes, and some warblers.",
    "multi_colored_pattern": "Multi-colored patterning on the body, indicating a species with complex plumage.",
    "has_crest": "A raised crest or tuft of feathers on the head, found in jays, cardinals, and kingfishers.",
    "has_mask": "A dark mask pattern around the eyes or face, seen in waxwings, shrikes, and some warblers.",
    "has_eyering": "A visible ring of color around the eye, a key field mark in vireos and some flycatchers.",
    "has_eye_stripe": "A stripe above or through the eye (supercilium or eye-line), common in sparrows and warblers.",
    "has_cap": "A distinctly colored cap on top of the head, found in chickadees, sparrows, and some warblers.",
    "forked_tail": "A forked or deeply notched tail, characteristic of swallows, terns, and some flycatchers.",
    "broad_wings": "Broad rounded wings adapted for soaring, found in hawks, eagles, and vultures.",
    "long_wings": "Long narrow wings for sustained flight, typical of swallows, terns, and albatrosses.",
    "large_bird": "A large bird (16-72 inches), such as herons, hawks, or pelicans.",
    "small_bird": "A small bird (3-9 inches), such as warblers, chickadees, or hummingbirds.",
    # ── vision (AwA2 attributes) ─────────────────────────────────────
    "furry": "The animal has fur, a key visual feature distinguishing mammals from reptiles and amphibians.",
    "stripes": "Striped markings on the animal, characteristic of zebras, tigers, and some fish.",
    "big": "The animal is large in size, typical of megafauna like elephants, whales, and bears.",
    "claws": "Visible claws suggest a predatory or climbing animal such as a cat, bear, or raptor.",
    "swims": "The animal is adapted for swimming, found in marine mammals, fish, and semi-aquatic species.",
    "domestic": "A domesticated animal, commonly kept as a pet or farm animal such as dogs, cats, or horses.",
    "fierce": "The animal displays aggressive or fierce behavior, typical of large predators.",
    "ocean": "The animal lives in the ocean, characteristic of whales, dolphins, and seals.",
    "meat": "The animal is a carnivore that eats meat, indicating predatory behavior.",
    "tusks": "The animal has tusks, large elongated teeth found in elephants, walruses, and boars.",
    "horns": "The animal has horns or antlers, found in cattle, deer, rhinoceroses, and antelopes.",
    "hooves": "The animal has hooves, characteristic of ungulates like horses, cattle, and deer.",
    "quadrapedal": "The animal walks on four legs, the most common locomotion for terrestrial mammals.",
    "hunter": "The animal actively hunts prey, indicating a carnivorous or omnivorous predator.",
    "fast": "The animal is known for speed, like cheetahs, horses, and antelopes.",
    "nocturnal": "The animal is primarily active at night, adapting to low-light conditions.",
    "group": "The animal lives in social groups, herds, or packs for protection or hunting.",
    "smart": "The animal is considered intelligent, such as primates, dolphins, and elephants.",
    "solitary": "The animal is primarily solitary, living and hunting alone.",
    "strong": "The animal is physically strong, capable of exerting great force.",
    "flippers": "The animal has flippers for swimming, found in seals, whales, and sea turtles.",
}


def _get_concept_evidence(concept: str, domain: str | None = None) -> str | None:
    """Look up evidence for a concept, trying ConceptBank descriptions as fallback."""
    # Direct match in _EVIDENCE
    ev = _EVIDENCE.get(concept)
    if ev:
        return ev

    # Try ConceptBank description
    if domain:
        try:
            from cbm.concept_bank import ConceptBank
            cb = ConceptBank(domain)
            desc = cb.get_description(concept)
            if desc and desc != concept:
                return desc
        except Exception:
            pass

    return None


def generate_explanation(
    domain_or_type: str,
    label: str,
    confidence: float,
    concepts: dict[str, float],
    evidence: bool = False,
):
    """Return a natural-language explanation dict.

    Parameters
    ----------
    domain_or_type : domain name or legacy input_type
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
    top_concept, top_value = ranked[0] if ranked else ("N/A", 0)

    # ── Main explanation sentence ────
    explanation = (
        f"The model predicts \"{label}\" (confidence {confidence:.0%}) "
        f"primarily because the concept \"{top_concept}\" is activated at "
        f"{top_value:.0%}."
    )

    # Add secondary concepts (show top 5 to avoid overwhelming for 85-concept domains)
    if len(ranked) > 1:
        secondary = ", ".join(
            f"\"{c}\" ({v:.0%})" for c, v in ranked[1:6]
        )
        explanation += f" Other contributing concepts are: {secondary}."
        if len(ranked) > 6:
            explanation += f" ({len(ranked) - 6} additional concepts omitted.)"

    # ── Highlighted segments for rich rendering ────
    segments = []
    segments.append({"text": "The model predicts ", "type": "text"})
    segments.append({"text": f"\"{label}\"", "type": "decision"})
    segments.append({"text": " (confidence ", "type": "text"})
    segments.append({"text": f"{confidence:.0%}", "type": "percentage", "value": confidence})
    segments.append({"text": ") primarily because the concept ", "type": "text"})
    segments.append({"text": f"\"{top_concept}\"", "type": "concept", "concept": top_concept, "value": top_value})
    segments.append({"text": " is activated at ", "type": "text"})
    segments.append({"text": f"{top_value:.0%}", "type": "percentage", "value": top_value})
    segments.append({"text": ".", "type": "text"})

    if len(ranked) > 1:
        segments.append({"text": " Other contributing concepts are: ", "type": "text"})
        for i, (c, v) in enumerate(ranked[1:6]):
            if i > 0:
                segments.append({"text": ", ", "type": "text"})
            segments.append({"text": f"\"{c}\"", "type": "concept", "concept": c, "value": v})
            segments.append({"text": " (", "type": "text"})
            segments.append({"text": f"{v:.0%}", "type": "percentage", "value": v})
            segments.append({"text": ")", "type": "text"})
        segments.append({"text": ".", "type": "text"})
        if len(ranked) > 6:
            segments.append({"text": f" ({len(ranked) - 6} additional concepts omitted.)", "type": "text"})

    # ── Evidence snippets (optional) ────
    evidence_snippets = []
    if evidence:
        for concept, value in ranked:
            snippet = _get_concept_evidence(concept, domain=domain_or_type)
            if snippet:
                evidence_snippets.append({
                    "concept": concept,
                    "activation": value,
                    "text": snippet,
                })

    return {
        "explanation_text": explanation,
        "highlighted_segments": segments,
        "evidence_snippets": evidence_snippets,
    }
