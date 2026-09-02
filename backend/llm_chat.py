"""LLM-powered chat agent via Ollama.

Replaces the template-based ``_chat_response()`` from Phase 1A with
a real conversational agent backed by a local LLM (Ollama).  Falls
back to the template engine when Ollama is unavailable.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Generator

logger = logging.getLogger(__name__)


class ChatAgent:
    """Conversational explanation agent backed by Ollama."""

    def __init__(self):
        self._available: bool | None = None  # lazy-checked
        self._model: str | None = None
        self._history: list[dict] = []       # conversation history
        self._max_history = 20               # keep last N messages

    # ── availability ─────────────────────────────────────────────────
    def is_available(self) -> bool:
        """Check if Ollama is reachable."""
        if self._available is not None:
            return self._available
        try:
            import ollama
            ollama.list()
            self._available = True
            self._model = _get_best_model()
            logger.info("ChatAgent: Ollama available, using %s", self._model)
        except Exception as e:
            logger.warning("ChatAgent: Ollama not available (%s), using fallback", e)
            self._available = False
        return self._available

    # ── chat (non-streaming) ─────────────────────────────────────────
    def chat(self, message: str, context: dict) -> str:
        """Generate a chat response.

        Parameters
        ----------
        message : user message
        context : dict with keys: domain, label, confidence, concepts

        Returns
        -------
        Reply string
        """
        if not self.is_available():
            return _template_fallback(message, context)

        system_prompt = _build_system_prompt(context)

        # Add user message to history
        self._history.append({"role": "user", "content": message})

        # Trim history
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        messages = [
            {"role": "system", "content": system_prompt},
            *self._history,
        ]

        try:
            import ollama
            response = ollama.chat(
                model=self._model,
                messages=messages,
                options={"temperature": 0.5, "num_predict": 400},
            )
            reply = response["message"]["content"].strip()

            # Add assistant reply to history
            self._history.append({"role": "assistant", "content": reply})

            return reply
        except Exception as e:
            logger.warning("Ollama chat failed: %s", e)
            return _template_fallback(message, context)

    # ── streaming chat ───────────────────────────────────────────────
    def chat_stream(self, message: str, context: dict) -> Generator[str, None, None]:
        """Generate a streaming chat response (yields text chunks).

        Falls back to single-shot if Ollama is unavailable.
        """
        if not self.is_available():
            yield _template_fallback(message, context)
            return

        system_prompt = _build_system_prompt(context)
        self._history.append({"role": "user", "content": message})

        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        messages = [
            {"role": "system", "content": system_prompt},
            *self._history,
        ]

        full_reply = []
        try:
            import ollama
            stream = ollama.chat(
                model=self._model,
                messages=messages,
                stream=True,
                options={"temperature": 0.5, "num_predict": 400},
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                full_reply.append(token)
                yield token

            self._history.append({"role": "assistant", "content": "".join(full_reply)})
        except Exception as e:
            logger.warning("Ollama stream failed: %s", e)
            fallback = _template_fallback(message, context)
            yield fallback

    # ── reset conversation ───────────────────────────────────────────
    def reset(self) -> None:
        """Clear conversation history."""
        self._history.clear()


# ═══════════════════════════════════════════════════════════════════════
# System prompt construction
# ═══════════════════════════════════════════════════════════════════════

def _build_system_prompt(context: dict) -> str:
    domain = context.get("domain", "unknown")
    label = context.get("label", "unknown")
    confidence = context.get("confidence", 0)
    concepts = context.get("concepts", {})

    ranked = sorted(concepts.items(), key=lambda x: x[1], reverse=True) if concepts else []
    concept_str = ", ".join(f"{c}: {v:.0%}" for c, v in ranked[:8])

    # Retrieve RAG context if available
    rag_context = ""
    try:
        from rag_engine import get_rag_engine
        engine = get_rag_engine()
        if ranked:
            docs = engine.retrieve(ranked[0][0], domain=domain, n_results=3)
            if docs:
                rag_context = "\n\nRelevant documentation:\n" + "\n".join(
                    f"- {d['document']}" for d in docs[:3]
                )
    except Exception:
        pass

    return (
        f"You are an AI interpretability assistant explaining predictions from a "
        f"black-box {domain} model.\n\n"
        f"Current prediction context:\n"
        f"- Domain: {domain}\n"
        f"- Predicted label: {label}\n"
        f"- Confidence: {confidence:.0%}\n"
        f"- Top concept activations: {concept_str}\n"
        f"{rag_context}\n\n"
        f"Guidelines:\n"
        f"- Answer questions about why the model made this prediction\n"
        f"- Explain concepts and their influence on the prediction\n"
        f"- Suggest counterfactual edits the user can try\n"
        f"- Be concise and precise (2-4 sentences per response)\n"
        f"- Ground explanations in specific concept activations\n"
        f"- If asked about something unrelated to the model, redirect to the prediction"
    )


# ═══════════════════════════════════════════════════════════════════════
# Template fallback (from Phase 1A)
# ═══════════════════════════════════════════════════════════════════════

def _template_fallback(message: str, context: dict) -> str:
    """Rule-based fallback when Ollama is not available."""
    msg = message.lower().strip()
    label = context.get("label", "unknown")
    confidence = context.get("confidence", 0)
    concepts = context.get("concepts", {})

    ranked = sorted(concepts.items(), key=lambda x: x[1], reverse=True) if concepts else []
    top = ranked[0] if ranked else ("N/A", 0)

    if any(kw in msg for kw in ("why", "reason", "how come", "explain")):
        return (
            f"The model predicts \"{label}\" mainly because the concept "
            f"\"{top[0]}\" has a high activation of {top[1]:.0%}. "
            f"Overall confidence is {confidence:.0%}."
        )
    if any(kw in msg for kw in ("what if", "change", "adjust", "reduce", "increase")):
        return (
            f"Try adjusting the concept sliders in the Concept Panel. "
            f"Reducing \"{top[0]}\" below 0.5 would likely lower confidence "
            f"and could flip the prediction."
        )
    if any(kw in msg for kw in ("what is", "define", "meaning")):
        for concept in concepts:
            if concept in msg:
                return (
                    f"\"{concept}\" is a concept the model uses internally. "
                    f"Its current activation is {concepts[concept]:.0%}."
                )
        return "Could you specify which concept you'd like to know about?"
    if any(kw in msg for kw in ("confident", "confidence", "sure", "certain")):
        return f"The model is {confidence:.0%} confident that the input is \"{label}\"."
    return (
        f"I can help explain the current prediction (\"{label}\", "
        f"{confidence:.0%} confidence). Try asking \"why this prediction?\" "
        f"or \"what if I change a concept?\""
    )


# ═══════════════════════════════════════════════════════════════════════
# Model selection
# ═══════════════════════════════════════════════════════════════════════

def _get_best_model() -> str:
    """Select best available Ollama model for chat."""
    preferred = ["gemma3:12b", "phi4:latest", "gpt-oss:20b", "llama3.2-vision:latest"]
    try:
        import ollama
        models = ollama.list()
        available = {m.model for m in models.models} if hasattr(models, 'models') else set()
        for m in preferred:
            if m in available:
                return m
        if available:
            return next(iter(available))
    except Exception:
        pass
    return "gemma3:12b"


# ── Singleton ────────────────────────────────────────────────────────
_agent: ChatAgent | None = None

def get_chat_agent() -> ChatAgent:
    """Get or create the singleton chat agent."""
    global _agent
    if _agent is None:
        _agent = ChatAgent()
    return _agent
