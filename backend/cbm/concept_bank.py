"""Concept bank definitions for each domain.

Each domain has a fixed set of human-interpretable concepts with natural-
language descriptions.  The descriptions are used by the CLIP concept scorer
(zero-shot) and by the explanation / RAG engine.
"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════════
# MEDICAL — Lung X-ray concepts (COVID-19 / Non-COVID / Normal)
# ═══════════════════════════════════════════════════════════════════════
_MEDICAL_CONCEPTS = {
    "Consolidation": "Dense consolidation in the lung fields indicating alveolar filling",
    "Ground Glass Opacity": "Hazy increased lung opacity without obscuration of underlying structures",
    "Lung Opacity": "There is increased opacity in the lung fields suggesting fluid or consolidation",
    "Pleural Effusion": "Fluid collection in the pleural space with blunting of the costophrenic angle",
    "Cardiomegaly": "The heart appears enlarged with a cardiothoracic ratio exceeding 0.5",
    "Edema": "Pulmonary edema with increased interstitial markings and possible alveolar flooding",
    "Atelectasis": "Partial or complete collapse of lung tissue visible on the X-ray",
    "Bilateral Involvement": "Abnormalities present in both left and right lung fields",
    "Peripheral Distribution": "Opacities predominantly located in the peripheral lung zones",
    "Air Bronchogram": "Air-filled bronchi visible within opacified lung parenchyma",
    "Lung Volume Loss": "Decreased lung volume suggesting restrictive process or collapse",
    "Clear Lung Fields": "Lung fields appear clear with no significant opacities or infiltrates",
    "Pneumothorax": "Air in the pleural space causing partial or complete lung collapse",
    "Nodule": "A small round or oval opacity within the lung parenchyma",
    "Mass": "A large soft-tissue opacity greater than 3cm within the lung field",
    "Fibrosis": "Chronic scarring of lung tissue producing reticular or linear opacities",
}

# ═══════════════════════════════════════════════════════════════════════
# TOXICITY — Jigsaw categories + linguistic markers
# ═══════════════════════════════════════════════════════════════════════
_TOXICITY_CONCEPTS = {
    "severe_toxic": "The text contains extremely toxic or harmful language",
    "obscene": "The text includes obscene profane or vulgar language",
    "threat": "The text contains language that threatens harm or violence",
    "insult": "The text demeans belittles or insults a person or group",
    "identity_hate": "The text targets a person or group based on a protected identity attribute",
    "sexually_explicit": "The text contains sexually explicit or graphic content",
    "flirtation": "The text contains unwanted flirtatious or suggestive language",
    "profanity_score": "The overall profanity intensity level of the text",
    "caps_ratio": "The proportion of text written in ALL CAPS indicating shouting or aggression",
    "negative_sentiment": "The overall negative emotional tone or sentiment of the text",
}

# ═══════════════════════════════════════════════════════════════════════
# BIRDS — CUB-200 visual attributes (28 interpretable concepts)
# ═══════════════════════════════════════════════════════════════════════
_BIRD_CONCEPTS = {
    # Colors (aggregated across all plumage body parts)
    "has_red": "A bird with red coloring on any body part",
    "has_blue": "A bird with blue coloring on any body part",
    "has_yellow": "A bird with yellow coloring on any body part",
    "has_orange": "A bird with orange coloring on any body part",
    "has_black": "A bird with black coloring on any body part",
    "has_white": "A bird with white coloring on any body part",
    "has_brown": "A bird with brown coloring on any body part",
    "has_grey": "A bird with grey coloring on any body part",
    "has_green": "A bird with green coloring on any body part",
    "has_iridescent": "A bird with iridescent or shimmering plumage",
    # Bill shape
    "curved_bill": "A bird with a curved bill (upward or downward)",
    "hooked_bill": "A bird with a hooked bill like a raptor or seabird",
    "dagger_bill": "A bird with a long straight dagger-shaped bill",
    "cone_bill": "A bird with a short thick cone-shaped bill for cracking seeds",
    # Bill length
    "long_bill": "A bird whose bill is longer than its head",
    # Patterns
    "spotted_pattern": "A bird with spotted or speckled markings on its body",
    "striped_pattern": "A bird with striped or streaked markings on its body",
    "multi_colored_pattern": "A bird with multi-colored patterning on its body",
    # Head features
    "has_crest": "A bird with a raised crest or tuft of feathers on its head",
    "has_mask": "A bird with a dark mask pattern around its eyes or face",
    "has_eyering": "A bird with a visible ring of color around its eye",
    "has_eye_stripe": "A bird with a stripe above or through the eye",
    "has_cap": "A bird with a distinctly colored cap on top of its head",
    # Tail shape
    "forked_tail": "A bird with a forked or deeply notched tail",
    # Wing shape
    "broad_wings": "A bird with broad rounded wings for soaring",
    "long_wings": "A bird with long narrow wings for sustained flight",
    # Size
    "large_bird": "A large or very large bird (16-72 inches)",
    "small_bird": "A small or very small bird (3-9 inches)",
}

# ═══════════════════════════════════════════════════════════════════════
# VISION — Animals with Attributes 2 (AwA2) — 85 semantic attributes
# ═══════════════════════════════════════════════════════════════════════
_VISION_CONCEPTS = {
    # Colors
    "black": "The animal has black coloring on its body",
    "white": "The animal has white coloring on its body",
    "blue": "The animal has blue coloring on its body",
    "brown": "The animal has brown coloring on its body",
    "gray": "The animal has gray coloring on its body",
    "orange": "The animal has orange coloring on its body",
    "red": "The animal has red coloring on its body",
    "yellow": "The animal has yellow coloring on its body",
    # Patterns
    "patches": "The animal has patches or irregular markings on its body",
    "spots": "The animal has spots or dotted markings on its body",
    "stripes": "The animal has stripes or banded markings on its body",
    # Body features
    "furry": "The animal has fur covering its body",
    "hairless": "The animal has little or no hair on its body",
    "toughskin": "The animal has tough thick or armored skin",
    "big": "The animal is large in overall body size",
    "small": "The animal is small in overall body size",
    "bulbous": "The animal has a bulbous or rounded body shape",
    "lean": "The animal has a lean slender body build",
    "muscle": "The animal has a visibly muscular build",
    "tail": "The animal has a prominent tail",
    "longleg": "The animal has long legs relative to its body",
    "longneck": "The animal has a long neck",
    "tusks": "The animal has tusks or elongated teeth",
    "horns": "The animal has horns or antlers",
    "claws": "The animal has visible claws",
    "flippers": "The animal has flippers instead of legs",
    "hands": "The animal has hands or hand-like appendages",
    "hooves": "The animal has hooves",
    "pads": "The animal has padded paws or feet",
    "paws": "The animal has paws",
    "buckteeth": "The animal has prominent buck teeth or front incisors",
    "meatteeth": "The animal has sharp teeth for tearing meat",
    "chewteeth": "The animal has flat teeth for chewing vegetation",
    "strainteeth": "The animal has baleen or straining teeth for filter feeding",
    # Movement and behavior
    "flys": "The animal can fly",
    "hops": "The animal moves by hopping",
    "swims": "The animal can swim",
    "walks": "The animal walks on land",
    "fast": "The animal is fast-moving",
    "slow": "The animal is slow-moving",
    "bipedal": "The animal walks on two legs",
    "quadrapedal": "The animal walks on four legs",
    "active": "The animal is generally active and energetic",
    "inactive": "The animal is generally inactive or sedentary",
    "agility": "The animal is agile and nimble in movement",
    "hibernate": "The animal hibernates during cold seasons",
    "nocturnal": "The animal is primarily active at night",
    # Diet and feeding
    "fish": "The animal eats fish",
    "meat": "The animal eats meat",
    "plankton": "The animal feeds on plankton",
    "vegetation": "The animal eats plants and vegetation",
    "insects": "The animal eats insects",
    "forager": "The animal forages for food",
    "grazer": "The animal grazes on grass and ground vegetation",
    "hunter": "The animal actively hunts prey",
    "scavenger": "The animal scavenges for food",
    "skimmer": "The animal skims water surface to feed",
    "stalker": "The animal stalks its prey before attacking",
    # Habitat
    "arctic": "The animal lives in arctic or polar regions",
    "coastal": "The animal lives in coastal areas",
    "desert": "The animal lives in desert environments",
    "bush": "The animal lives in bushland or scrub",
    "plains": "The animal lives on open plains or grasslands",
    "forest": "The animal lives in forests",
    "fields": "The animal is found in fields or farmland",
    "jungle": "The animal lives in tropical jungle",
    "mountains": "The animal lives in mountainous terrain",
    "ocean": "The animal lives in the ocean",
    "ground": "The animal spends most of its time on the ground",
    "water": "The animal spends significant time in water",
    "tree": "The animal climbs or lives in trees",
    "cave": "The animal uses caves for shelter",
    "tunnels": "The animal digs or lives in tunnels or burrows",
    # Social and temperament
    "group": "The animal lives in groups or herds",
    "solitary": "The animal is primarily solitary",
    "nestspot": "The animal builds nests or uses specific nesting spots",
    "domestic": "The animal is commonly domesticated",
    "fierce": "The animal is fierce or aggressive",
    "timid": "The animal is timid or easily frightened",
    "smart": "The animal is considered intelligent",
    "strong": "The animal is physically strong",
    "weak": "The animal is physically weak",
    "smelly": "The animal is known for a strong odor",
    # Biogeography
    "newworld": "The animal is native to the New World (Americas)",
    "oldworld": "The animal is native to the Old World (Africa, Asia, Europe)",
}

# ═══════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════
_CONCEPT_BANKS = {
    "medical": _MEDICAL_CONCEPTS,
    "toxicity": _TOXICITY_CONCEPTS,
    "birds": _BIRD_CONCEPTS,
    "vision": _BIRD_CONCEPTS,
}


class ConceptBank:
    """Immutable set of concepts for a single domain."""

    def __init__(self, domain: str):
        bank = _CONCEPT_BANKS.get(domain)
        if bank is None:
            raise KeyError(f"No concept bank for domain {domain!r}")
        self._domain = domain
        self._concepts: dict[str, str] = dict(bank)

    # ── public API ───────────────────────────────────────────────────
    @property
    def domain(self) -> str:
        return self._domain

    @property
    def concepts(self) -> list[str]:
        """Ordered concept names."""
        return list(self._concepts.keys())

    @property
    def descriptions(self) -> dict[str, str]:
        """concept_name → human-readable description."""
        return dict(self._concepts)

    @property
    def num_concepts(self) -> int:
        return len(self._concepts)

    def get_description(self, concept: str) -> str:
        return self._concepts.get(concept, concept)

    def __repr__(self) -> str:
        return f"ConceptBank(domain={self._domain!r}, n={self.num_concepts})"
