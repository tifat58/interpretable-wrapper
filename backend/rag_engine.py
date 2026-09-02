"""RAG-powered explanation engine.

Uses ChromaDB + sentence-transformers to store and retrieve concept
documentation, then optionally generates richer explanations via an LLM
(Ollama).  Falls back to template-based explanations when Ollama is not
available.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_CHROMA_DIR = os.path.join(_BACKEND_DIR, "data", "chroma_db")


class RAGEngine:
    """Retrieval-Augmented Generation engine for concept explanations."""

    def __init__(self):
        self._collection = None
        self._embed_fn = None
        self._indexed = False

    # ── lazy init ────────────────────────────────────────────────────
    def _ensure_ready(self) -> None:
        """Initialize ChromaDB collection and embedding function on first use."""
        if self._collection is not None:
            return

        # ChromaDB requires sqlite3 >= 3.35.0; use pysqlite3 if available
        try:
            __import__("pysqlite3")
            import sys
            sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
        except ImportError:
            pass

        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self._embed_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
        )

        client = chromadb.PersistentClient(path=_CHROMA_DIR)
        self._collection = client.get_or_create_collection(
            name="concept_docs",
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info("RAG engine ready — ChromaDB at %s, %d docs",
                     _CHROMA_DIR, self._collection.count())

    # ── indexing ─────────────────────────────────────────────────────
    def index_concepts(self, domain: str, force: bool = False) -> int:
        """Index concept descriptions + evidence for a domain.

        Returns number of documents indexed.
        """
        self._ensure_ready()

        # Check if already indexed for this domain
        existing = self._collection.get(where={"domain": domain})
        if existing["ids"] and not force:
            logger.info("Domain %r already indexed (%d docs)", domain, len(existing["ids"]))
            return len(existing["ids"])

        # If forcing, delete old docs for this domain
        if force and existing["ids"]:
            self._collection.delete(ids=existing["ids"])

        documents = []
        metadatas = []
        ids = []

        # Source 1: ConceptBank descriptions
        from cbm.concept_bank import ConceptBank
        cb = ConceptBank(domain)
        for concept in cb.concepts:
            desc = cb.get_description(concept)
            if desc and desc != concept:
                doc_id = f"{domain}__desc__{concept}"
                documents.append(
                    f"Concept: {concept}\nDomain: {domain}\n"
                    f"Description: {desc}"
                )
                metadatas.append({
                    "domain": domain,
                    "concept": concept,
                    "source": "concept_bank",
                })
                ids.append(doc_id)

        # Source 2: Evidence snippets from explanations.py
        from explanations import _EVIDENCE
        for concept, evidence in _EVIDENCE.items():
            if isinstance(evidence, str) and evidence:
                doc_id = f"{domain}__evidence__{concept}"
                # Avoid duplicates
                if doc_id not in ids:
                    documents.append(
                        f"Concept: {concept}\nDomain: {domain}\n"
                        f"Evidence: {evidence}"
                    )
                    metadatas.append({
                        "domain": domain,
                        "concept": concept,
                        "source": "evidence",
                    })
                    ids.append(doc_id)

        if documents:
            self._collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

        logger.info("Indexed %d documents for domain %r", len(documents), domain)
        return len(documents)

    def index_all_domains(self) -> int:
        """Index concepts for all configured domains."""
        from config import DOMAIN_CONFIG
        total = 0
        for domain in DOMAIN_CONFIG:
            total += self.index_concepts(domain)
        return total

    # ── retrieval ────────────────────────────────────────────────────
    def retrieve(self, query: str, domain: str | None = None,
                 n_results: int = 5) -> list[dict]:
        """Retrieve relevant concept documents for a query.

        Parameters
        ----------
        query : natural language query or concept name
        domain : optional filter by domain
        n_results : max results to return

        Returns
        -------
        list of dicts with keys: document, concept, domain, distance
        """
        self._ensure_ready()

        # Ensure at least this domain is indexed
        if domain:
            self.index_concepts(domain)

        where = {"domain": domain} if domain else None
        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )

        docs = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i] if results.get("distances") else None
            docs.append({
                "document": doc,
                "concept": meta.get("concept", ""),
                "domain": meta.get("domain", ""),
                "source": meta.get("source", ""),
                "distance": dist,
            })

        return docs

    # ── RAG-augmented explanation ────────────────────────────────────
    def generate_explanation(self, domain: str, label: str,
                             confidence: float, concepts: dict[str, float],
                             evidence: bool = False,
                             use_llm: bool = True) -> dict:
        """Generate explanation using retrieved context + optional LLM.

        Parameters
        ----------
        domain : domain name
        label : predicted label
        confidence : model confidence
        concepts : concept name → activation
        evidence : whether to include evidence snippets
        use_llm : try LLM generation (falls back to template if unavailable)

        Returns
        -------
        dict with: explanation_text, evidence_snippets, rag_sources
        """
        ranked = sorted(concepts.items(), key=lambda x: x[1], reverse=True)
        top_concepts = ranked[:5]

        # Retrieve relevant documents for top concepts
        rag_docs = []
        for concept, _ in top_concepts:
            docs = self.retrieve(concept, domain=domain, n_results=2)
            rag_docs.extend(docs)

        # Deduplicate
        seen = set()
        unique_docs = []
        for d in rag_docs:
            key = d["concept"]
            if key not in seen:
                seen.add(key)
                unique_docs.append(d)

        # Build context for LLM
        context_text = self._build_context(domain, label, confidence, concepts, unique_docs)

        # Try LLM generation
        explanation_text = None
        if use_llm:
            explanation_text = self._generate_with_llm(context_text)

        # Fallback to template
        if not explanation_text:
            from explanations import generate_explanation
            result = generate_explanation(domain, label, confidence, concepts, evidence)
            explanation_text = result["explanation_text"]

        # Build evidence snippets
        evidence_snippets = []
        if evidence:
            for doc in unique_docs:
                concept = doc["concept"]
                activation = concepts.get(concept, 0)
                evidence_snippets.append({
                    "concept": concept,
                    "activation": activation,
                    "text": doc["document"].split("Evidence: ")[-1]
                           if "Evidence:" in doc["document"]
                           else doc["document"].split("Description: ")[-1],
                    "source": doc.get("source", "rag"),
                })

            # Fill in any missing top concepts from static evidence
            covered = {s["concept"] for s in evidence_snippets}
            for concept, activation in ranked:
                if concept not in covered:
                    from explanations import _get_concept_evidence
                    text = _get_concept_evidence(concept, domain)
                    if text:
                        evidence_snippets.append({
                            "concept": concept,
                            "activation": activation,
                            "text": text,
                            "source": "static",
                        })

        # Build highlighted segments for rich rendering
        segments = self._build_highlighted_segments(
            label, confidence, ranked, explanation_text,
        )

        return {
            "explanation_text": explanation_text,
            "highlighted_segments": segments,
            "evidence_snippets": evidence_snippets,
            "rag_sources": len(unique_docs),
        }

    # ── LLM generation ───────────────────────────────────────────────
    def _generate_with_llm(self, context: str) -> str | None:
        """Generate explanation text using Ollama. Returns None on failure."""
        try:
            import ollama
            response = ollama.chat(
                model=_get_ollama_model(),
                messages=[
                    {"role": "system", "content": _EXPLAIN_SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                options={"temperature": 0.3, "num_predict": 300},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            logger.warning("LLM generation failed: %s", e)
            return None

    @staticmethod
    def _build_highlighted_segments(
        label: str, confidence: float,
        ranked: list[tuple[str, float]],
        explanation_text: str,
    ) -> list[dict]:
        """Build highlighted segments by scanning the explanation text.

        Scans the explanation text for concept names, percentages, and the
        predicted label, then wraps matched spans in typed segments for
        coloured rendering in the frontend.
        """
        import re

        text = explanation_text
        # Collect all concept names (longest first to avoid partial matches)
        concept_names = {c for c, _ in ranked}
        concept_vals = {c: v for c, v in ranked}

        # Build a pattern for all concepts (escaped, longest first)
        sorted_concepts = sorted(concept_names, key=len, reverse=True)
        # Also match quoted or underscored variants
        concept_patterns = []
        for c in sorted_concepts:
            # Match the concept with optional surrounding quotes, underscores→spaces
            readable = c.replace("_", " ")
            variants = {re.escape(c), re.escape(readable)}
            if c != readable:
                variants.add(re.escape(readable))
            concept_patterns.append("|".join(variants))

        # Unified regex: concepts | percentages | label
        parts = []
        if concept_patterns:
            parts.append(f'(?P<concept>["\']?(?:{"|".join(concept_patterns)})["\']?)')
        parts.append(r'(?P<pct>\d{1,3}(?:\.\d+)?%)')
        parts.append(f'(?P<label>["\']?{re.escape(label)}["\']?)')
        pattern = re.compile("|".join(parts), re.IGNORECASE)

        segments: list[dict] = []
        pos = 0
        for m in pattern.finditer(text):
            # Add preceding plain text
            if m.start() > pos:
                segments.append({"text": text[pos:m.start()], "type": "text"})

            matched = m.group()
            if m.group("label"):
                segments.append({"text": matched, "type": "decision"})
            elif m.group("pct"):
                segments.append({"text": matched, "type": "percentage"})
            elif m.group("concept"):
                # Find which concept was matched
                raw = matched.strip("\"'")
                canon = None
                for c in sorted_concepts:
                    if raw.lower() in (c.lower(), c.replace("_", " ").lower()):
                        canon = c
                        break
                seg = {"text": matched, "type": "concept"}
                if canon:
                    seg["concept"] = canon
                    seg["value"] = concept_vals.get(canon, 0)
                segments.append(seg)

            pos = m.end()

        # Trailing text
        if pos < len(text):
            segments.append({"text": text[pos:], "type": "text"})

        return segments if segments else [{"text": text, "type": "text"}]

    @staticmethod
    def _build_context(domain: str, label: str, confidence: float,
                       concepts: dict[str, float], docs: list[dict]) -> str:
        ranked = sorted(concepts.items(), key=lambda x: x[1], reverse=True)
        top_str = ", ".join(f"{c} ({v:.0%})" for c, v in ranked[:5])

        docs_text = ""
        if docs:
            docs_text = "\n\nRelevant concept documentation:\n"
            for d in docs[:5]:
                docs_text += f"- {d['document']}\n"

        return (
            f"Domain: {domain}\n"
            f"Prediction: {label} (confidence: {confidence:.0%})\n"
            f"Top concept activations: {top_str}\n"
            f"Total concepts: {len(concepts)}"
            f"{docs_text}\n\n"
            f"Generate a clear, concise explanation of why the model made "
            f"this prediction, grounding it in the concept activations and "
            f"documentation above."
        )


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

_EXPLAIN_SYSTEM_PROMPT = (
    "You are an AI model interpretability assistant. You explain predictions "
    "made by black-box AI models using concept-based explanations. Your "
    "explanations should be clear, concise, and grounded in the concept "
    "activations provided. Refer to specific concepts and their activation "
    "levels. Keep your explanation to 2-4 sentences. Do not use bullet points."
)

def _get_ollama_model() -> str:
    """Return the preferred Ollama model name."""
    preferred = ["gemma3:12b", "phi4:latest", "llama3.2-vision:latest"]
    try:
        import ollama
        models = ollama.list()
        available = {m.model for m in models.models} if hasattr(models, 'models') else set()
        for m in preferred:
            if m in available:
                return m
        # Return first available
        if available:
            return next(iter(available))
    except Exception:
        pass
    return "gemma3:12b"


# ── Singleton ────────────────────────────────────────────────────────
_engine: RAGEngine | None = None

def get_rag_engine() -> RAGEngine:
    """Get or create the singleton RAG engine."""
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine
