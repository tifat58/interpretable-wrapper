"""Toxicity domain model — Unitary toxic-bert."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from models.base import BaseModel

logger = logging.getLogger(__name__)


class ToxicityModel(BaseModel):
    """Wraps ``unitary/toxic-bert`` for binary toxicity classification."""

    def __init__(self, domain: str, config: dict):
        super().__init__(domain, config)
        self._model = None
        self._tokenizer = None
        self._last_hidden: torch.Tensor | None = None
        self._last_attentions: tuple | None = None

    # ── loading ──────────────────────────────────────────────────────
    def load(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_id = self._config["model_id"]
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_id, output_hidden_states=True, output_attentions=True,
        )
        self._model.to(self.device).eval()
        self._loaded = True
        logger.info("ToxicityModel loaded (%s) — %s", self.device, model_id)

    # ── preprocessing ────────────────────────────────────────────────
    def preprocess(self, data: Any) -> dict:
        """Tokenize text input (max 512 tokens)."""
        text = str(data) if data is not None else ""
        encoding = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        return {k: v.to(self.device) for k, v in encoding.items()}

    # ── inference ────────────────────────────────────────────────────
    def predict_raw(self, data: Any) -> dict:
        inputs = self.preprocess(data)
        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits  # (1, num_labels)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

        # toxic-bert: single logit → sigmoid
        # unbiased-toxic-roberta: 16 labels, first is 'toxicity'
        confidence = float(probs[0]) if len(probs) > 0 else 0.5
        label = "toxic" if confidence >= 0.5 else "not toxic"

        # Cache hidden states and attentions for feature extraction / attribution
        self._last_hidden = outputs.hidden_states[-1]   # (1, seq_len, hidden)
        self._last_attentions = outputs.attentions       # tuple of (1, heads, seq, seq)

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "raw_scores": {"toxic": float(confidence)},
        }

    # ── feature extraction ───────────────────────────────────────────
    def extract_features(self, data: Any) -> np.ndarray:
        inputs = self.preprocess(data)
        with torch.no_grad():
            outputs = self._model(**inputs)

        # CLS token from last hidden layer → (768,)
        hidden = outputs.hidden_states[-1]  # (1, seq_len, 768)
        cls_embedding = hidden[:, 0, :].squeeze()
        self._last_hidden = hidden
        self._last_attentions = outputs.attentions
        return cls_embedding.cpu().numpy()

    # ── attribution (attention-based) ────────────────────────────────
    def get_attribution(self, data: Any, target_concept: str | None = None) -> dict:
        inputs = self.preprocess(data)
        with torch.no_grad():
            outputs = self._model(**inputs)

        attentions = outputs.attentions  # tuple of (1, num_heads, seq_len, seq_len)
        if not attentions:
            return {"method": "none", "type": "tokens", "data": [], "concept": target_concept}

        # Average attention from last layer across all heads → (seq_len, seq_len)
        last_attn = attentions[-1].squeeze(0).mean(dim=0)  # (seq_len, seq_len)
        # CLS row: how much CLS attends to each token
        cls_attn = last_attn[0].cpu().numpy()

        # Decode tokens
        input_ids = inputs["input_ids"].squeeze(0).cpu().tolist()
        tokens = self._tokenizer.convert_ids_to_tokens(input_ids)

        # Normalize scores to [0, 1]
        attn_min, attn_max = cls_attn.min(), cls_attn.max()
        if attn_max - attn_min > 1e-8:
            scores = (cls_attn - attn_min) / (attn_max - attn_min)
        else:
            scores = np.zeros_like(cls_attn)

        # Concept-specific weighting: boost tokens relevant to target concept
        # Resolve probe concept name to token category key(s)
        resolved_categories = []
        if target_concept:
            if target_concept in _DEFAULT_TOKEN_CATEGORIES:
                resolved_categories = [target_concept]
            elif target_concept in _PROBE_TO_TOKEN_CATEGORY:
                resolved_categories = _PROBE_TO_TOKEN_CATEGORY[target_concept]
        if resolved_categories:
            # Merge keywords from all resolved categories
            keywords: list[str] = []
            for cat in resolved_categories:
                keywords.extend(_DEFAULT_TOKEN_CATEGORIES.get(cat, []))
            concept_mask = np.zeros_like(scores)
            for j, tok in enumerate(tokens):
                tok_lower = tok.lower().lstrip("##").lstrip("\u0120")
                for kw in keywords:
                    if kw in tok_lower or tok_lower in kw:
                        concept_mask[j] = 1.0
                        break
            if concept_mask.any():
                # Emphasize concept-relevant tokens, dampen others
                scores = scores * (0.3 + 0.7 * concept_mask)
                # Re-normalize
                s_min, s_max = scores.min(), scores.max()
                if s_max - s_min > 1e-8:
                    scores = (scores - s_min) / (s_max - s_min)

        # Filter out special tokens (BERT: [CLS]/[SEP]/[PAD], RoBERTa: <s>/</s>/<pad>)
        special = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}
        token_scores = [
            {"token": tok, "score": round(float(s), 4)}
            for tok, s in zip(tokens, scores)
            if tok not in special
        ]

        return {
            "method": "attention",
            "type": "tokens",
            "data": token_scores,
            "concept": target_concept,
        }

    # ── token-level attribution aggregation ──────────────────────────
    def aggregate_token_attribution(self, data: Any,
                                    concept_categories: dict[str, list[str]] | None = None
                                    ) -> dict[str, float]:
        """Aggregate token-level attention scores into semantic concept categories.

        Parameters
        ----------
        data : raw text input
        concept_categories : mapping of concept name → list of keywords.
            If None, uses built-in categories.

        Returns
        -------
        dict mapping concept name → aggregated activation score
        """
        if concept_categories is None:
            concept_categories = _DEFAULT_TOKEN_CATEGORIES

        # Get token-level attribution
        attr = self.get_attribution(data)
        token_scores = attr.get("data", [])
        if not token_scores:
            return {cat: 0.5 for cat in concept_categories}

        results = {}
        for category, keywords in concept_categories.items():
            matched_scores = []
            for ts in token_scores:
                token_lower = ts["token"].lower().lstrip("##").lstrip("Ġ")
                for kw in keywords:
                    if kw in token_lower or token_lower in kw:
                        matched_scores.append(ts["score"])
                        break
            if matched_scores:
                # Max-over-matches activation (as per paper)
                results[category] = round(max(matched_scores), 4)
            else:
                results[category] = 0.0
        return results


# ── Default semantic categories for token aggregation ────────────────
_DEFAULT_TOKEN_CATEGORIES = {
    "profanity": ["fuck", "shit", "damn", "hell", "ass", "crap", "dick", "bitch"],
    "slur": ["slur", "racial", "ethnic", "homophob", "transphob"],
    "threat": ["kill", "die", "death", "attack", "destroy", "bomb", "shoot", "hurt", "harm", "stab"],
    "insult": ["stupid", "idiot", "moron", "dumb", "ugly", "loser", "pathetic", "worthless"],
    "sexual_content": ["sex", "porn", "nude", "naked", "erotic", "explicit"],
    "identity_attack": ["hate", "racist", "sexist", "bigot", "nazi", "supremac"],
    "negative_sentiment": ["bad", "terrible", "awful", "horrible", "worst", "hate", "disgusting"],
    "positive_sentiment": ["good", "great", "love", "awesome", "excellent", "best", "wonderful"],
    "aggression": ["fight", "punch", "beat", "smash", "crush", "slam", "rage", "furious", "angry"],
    "sarcasm_markers": ["obviously", "clearly", "sure", "right", "totally", "definitely", "wow"],
}

# Maps probe concept names → token category keys used above
_PROBE_TO_TOKEN_CATEGORY = {
    "severe_toxic": ["profanity", "threat", "slur"],
    "obscene": ["profanity", "sexual_content"],
    "identity_hate": ["identity_attack", "slur"],
    "sexually_explicit": ["sexual_content"],
    "flirtation": ["sexual_content", "positive_sentiment"],
    "profanity_score": ["profanity"],
    "caps_ratio": ["aggression"],
}
