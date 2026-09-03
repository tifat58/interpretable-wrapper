"""Flask REST API for the Interpretable Wrapper.

Phase 1A+1B: domain-dispatching via DomainManager with lazy-loaded
PostHocCBM instances, RAG explanations, LLM chat, input editing.
Falls back to DummyModel when a real model cannot be loaded.
"""

from __future__ import annotations

import logging
import os

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

app = Flask(__name__, static_folder="static")
CORS(app)


# ═══════════════════════════════════════════════════════════════════════
# Domain manager — lazy-loads PostHocCBM per domain, falls back to dummy
# ═══════════════════════════════════════════════════════════════════════
class DomainManager:
    """Central registry that creates PostHocCBM instances on demand."""

    def __init__(self):
        from config import DOMAIN_CONFIG, INPUT_TYPE_TO_DOMAIN
        self._config = DOMAIN_CONFIG
        self._input_type_map = INPUT_TYPE_TO_DOMAIN
        self._cbm_cache: dict = {}       # (domain, model_id) → PostHocCBM
        self._dummy_cache: dict = {}     # domain → True (if falling back)
        self._registry = None            # lazy ModelRegistry

    @property
    def enabled_config(self) -> dict:
        """Return only the domains that are currently enabled."""
        from config import ENABLED_DOMAINS
        return {k: v for k, v in self._config.items() if k in ENABLED_DOMAINS}

    def _get_registry(self):
        if self._registry is None:
            from models import ModelRegistry
            self._registry = ModelRegistry(self._config)
        return self._registry

    # ── resolve domain from body ─────────────────────────────────────
    def resolve_domain(self, body: dict) -> str:
        """Extract domain from request body with backward compat."""
        domain = body.get("domain")
        if domain and domain in self._config:
            return domain
        # Backward compat: input_type → domain
        input_type = body.get("input_type", "text")
        return self._input_type_map.get(input_type, "toxicity")

    # ── get CBM or dummy ─────────────────────────────────────────────
    def get_cbm(self, domain: str, model_id: str | None = None):
        """Return a PostHocCBM for the domain/model, or None to use dummy."""
        from config import get_default_model
        effective_id = model_id
        if effective_id is None:
            dm = get_default_model(domain)
            effective_id = dm["id"] if dm else "default"

        cache_key = (domain, effective_id)
        if cache_key in self._cbm_cache:
            return self._cbm_cache[cache_key]
        if domain in self._dummy_cache:
            return None  # already known to fail

        try:
            cbm = self._build_cbm(domain, model_id)
            self._cbm_cache[cache_key] = cbm
            return cbm
        except Exception:
            logger.exception("Failed to load real model for %r/%r — falling back to DummyModel",
                             domain, effective_id)
            self._dummy_cache[domain] = True
            return None

    def _build_cbm(self, domain: str, model_id: str | None = None):
        from config import PROBE_DATA_DIR, ALT_PROBE_DATA_DIR, get_model_by_id, get_default_model
        from cbm.concept_bank import ConceptBank
        from cbm.probe import ProbeBank
        from cbm.cbm_wrapper import PostHocCBM

        cfg = self._config[domain]
        registry = self._get_registry()
        model = registry.get(domain, model_id)

        # Get the effective model spec for feature_dim
        model_spec = None
        if model_id:
            model_spec = get_model_by_id(domain, model_id)
        if model_spec is None:
            model_spec = get_default_model(domain)
        feature_dim = model_spec["feature_dim"] if model_spec else cfg["feature_dim"]

        concept_bank = ConceptBank(domain)
        probe_bank = ProbeBank(
            concept_bank=concept_bank,
            feature_dim=feature_dim,
            probe_type=cfg["probe_type"],
        )

        # Try loading pre-trained probes — check primary dir then alt dir
        primary_dir = os.path.join(PROBE_DATA_DIR, domain)
        probe_dir = primary_dir
        loaded = probe_bank.load(probe_dir)
        if loaded == 0:
            alt_dir = os.path.join(ALT_PROBE_DATA_DIR, domain)
            if os.path.isdir(alt_dir):
                loaded = probe_bank.load(alt_dir)

        cbm = PostHocCBM(
            domain=domain, model=model,
            concept_bank=concept_bank, probe_bank=probe_bank,
        )

        # Try loading surrogate from primary dir (has probes/ subfolder + surrogate.pkl)
        cbm.load(primary_dir)

        return cbm

    # ── domain metadata ──────────────────────────────────────────────
    def domain_info(self) -> list[dict]:
        """Return metadata for enabled domains only."""
        from config import get_domain_models
        from cbm.concept_bank import ConceptBank
        result = []
        for name, cfg in self.enabled_config.items():
            cb = ConceptBank(name)
            models = get_domain_models(name)
            result.append({
                "name": name,
                "input_type": cfg["input_type"],
                "description": cfg["description"],
                "num_concepts": cb.num_concepts,
                "labels": cfg["labels"],
                "models": [
                    {"id": m["id"], "name": m["name"],
                     "default": m.get("default", False),
                     "custom": m.get("custom", False)}
                    for m in models
                ],
            })
        return result

    def all_domain_info(self) -> list[dict]:
        """Return metadata for all domains (enabled and disabled)."""
        from config import get_domain_models, ENABLED_DOMAINS
        from cbm.concept_bank import ConceptBank
        result = []
        for name, cfg in self._config.items():
            cb = ConceptBank(name)
            models = get_domain_models(name)
            result.append({
                "name": name,
                "input_type": cfg["input_type"],
                "description": cfg["description"],
                "num_concepts": cb.num_concepts,
                "labels": cfg["labels"],
                "enabled": name in ENABLED_DOMAINS,
                "models": [
                    {"id": m["id"], "name": m["name"],
                     "default": m.get("default", False),
                     "custom": m.get("custom", False)}
                    for m in models
                ],
            })
        return result


dm = DomainManager()

# Keep DummyModel for fallback
from dummy_model import DummyModel
_dummy = DummyModel()


# ── helper: get prediction via CBM or dummy ──────────────────────────
def _predict_with_fallback(domain: str, data, model_id: str | None = None,
                           concept_strategy: str | None = None,
                           custom_concepts: list[str] | None = None):
    cbm = dm.get_cbm(domain, model_id)
    if cbm is not None:
        return cbm.predict(data, concept_strategy=concept_strategy,
                           custom_concepts=custom_concepts)
    # Fallback to dummy — pass domain for correct labels
    cfg = dm._config.get(domain, {})
    input_type = cfg.get("input_type", "text")
    result = _dummy.predict(input_type, data, domain=domain)
    result["domain"] = domain
    return result


# ═══════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════

@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    domain = dm.resolve_domain(body)
    data = body.get("data")
    model_id = body.get("model_id")
    concept_strategy = body.get("concept_strategy")
    custom_concepts = body.get("custom_concepts")
    result = _predict_with_fallback(domain, data, model_id,
                                    concept_strategy=concept_strategy,
                                    custom_concepts=custom_concepts)
    return jsonify(result)


@app.route("/counterfactual", methods=["POST"])
def counterfactual():
    body = request.get_json(force=True)
    domain = dm.resolve_domain(body)
    model_id = body.get("model_id")
    original_concepts = body.get("original_concepts", {})
    modified_concepts = body.get("modified_concepts", {})
    original_confidence = body.get("original_confidence", 0.5)

    cbm = dm.get_cbm(domain, model_id)
    if cbm is not None:
        result = cbm.counterfactual(original_concepts, modified_concepts, original_confidence)
    else:
        from counterfactuals import compute_counterfactual
        result = compute_counterfactual(
            domain, original_concepts, modified_concepts, original_confidence,
        )
    return jsonify(result)


@app.route("/explain", methods=["POST"])
def explain():
    body = request.get_json(force=True)
    domain = dm.resolve_domain(body)
    model_id = body.get("model_id")
    label = body.get("label", "")
    confidence = body.get("confidence", 0)
    concepts = body.get("concepts", {})
    evidence = body.get("evidence", False)

    cbm = dm.get_cbm(domain, model_id)
    if cbm is not None:
        result = cbm.explain(label, confidence, concepts, evidence)
    else:
        from explanations import generate_explanation
        result = generate_explanation(domain, label, confidence, concepts, evidence)
    return jsonify(result)


@app.route("/attribution", methods=["POST"])
def attribution():
    """Compute saliency / attribution map for a concept."""
    body = request.get_json(force=True)
    domain = dm.resolve_domain(body)
    model_id = body.get("model_id")
    data = body.get("data")
    concept = body.get("concept")

    cbm = dm.get_cbm(domain, model_id)
    if cbm is None:
        return jsonify({"method": "none", "type": "none", "data": None,
                        "concept": concept, "error": "Model not loaded"})

    result = cbm.get_concept_attribution(data, concept_name=concept)
    return jsonify(result)


@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True)
    message = body.get("message", "")
    context = body.get("context", {})
    stream = body.get("stream", False)

    from llm_chat import get_chat_agent
    agent = get_chat_agent()

    if stream:
        # SSE streaming response
        def generate():
            for chunk in agent.chat_stream(message, context):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})

    reply = agent.chat(message, context)
    return jsonify({"reply": reply})


@app.route("/chat/reset", methods=["POST"])
def chat_reset():
    """Reset conversation history."""
    from llm_chat import get_chat_agent
    get_chat_agent().reset()
    return jsonify({"status": "ok"})


@app.route("/edit_input", methods=["POST"])
def edit_input():
    """Edit raw input and re-run prediction for true counterfactual."""
    body = request.get_json(force=True)
    domain = dm.resolve_domain(body)
    model_id = body.get("model_id")
    data = body.get("data")
    edit_spec = body.get("edit_spec", {})

    cbm = dm.get_cbm(domain, model_id)
    if cbm is None:
        return jsonify({"error": "Model not loaded for input editing"}), 400

    result = cbm.edit_input(data, edit_spec)
    return jsonify(result)


@app.route("/domains", methods=["GET"])
def domains():
    """Return metadata for all enabled domains."""
    return jsonify({"domains": dm.domain_info()})


@app.route("/domains/all", methods=["GET"])
def domains_all():
    """Return metadata for all domains (enabled and disabled)."""
    return jsonify({"domains": dm.all_domain_info()})


@app.route("/domains/toggle", methods=["POST"])
def domains_toggle():
    """Enable or disable a domain at runtime."""
    body = request.get_json(force=True)
    domain = body.get("domain")
    enabled = body.get("enabled", True)
    if not domain:
        return jsonify({"error": "domain is required"}), 400
    try:
        from config import set_domain_enabled
        new_list = set_domain_enabled(domain, enabled)
        return jsonify({"status": "ok", "enabled_domains": new_list})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/concept_strategies", methods=["GET"])
def concept_strategies():
    """Return available concept extraction strategies for a domain."""
    domain = request.args.get("domain", "")
    cfg = dm._config.get(domain, {})
    input_type = cfg.get("input_type", "text")

    # Check if domain uses or has a CLIP-capable model
    model_class = cfg.get("model_class", "")
    models = cfg.get("models", [])
    has_clip_model = any("vision_model" in m.get("model_class", "").lower() for m in models)
    is_clip_capable = "vision_model" in model_class.lower() or cfg.get("probe_type") == "clip" or has_clip_model

    strategies = [
        {"id": "predefined", "name": "CBM (Trained Probes)",
         "description": "Concept Bottleneck Model with domain-expert defined concepts and trained linear probes"},
    ]

    if is_clip_capable:
        strategies.append(
            {"id": "custom", "name": "Custom Concepts",
             "description": "Define your own concepts — scored via CLIP similarity"},
        )

    if input_type == "image":
        if is_clip_capable:
            strategies.extend([
                {"id": "clip", "name": "CLIP Zero-Shot",
                 "description": "Score concepts via CLIP text-image similarity (no training data needed)"},
                {"id": "label_free", "name": "Label-Free CBM",
                 "description": "Auto-discovered concept bottleneck using CLIP with curated visual concepts"},
            ])
        # PCA/K-Means only need the domain model's own feature space — no
        # CLIP required — so offer them for every image domain, including
        # specialist models like the medical DenseNet.
        strategies.extend([
            {"id": "pca", "name": "PCA Auto-Discovery",
             "description": "Automatically discover concept axes from model feature space"},
            {"id": "kmeans", "name": "K-Means Auto-Discovery",
             "description": "Cluster feature space into interpretable groups"},
        ])

    if input_type == "text":
        strategies.append(
            {"id": "token_aggregation", "name": "Token Attribution",
             "description": "Aggregate token-level attention into semantic categories"},
        )

    return jsonify({"domain": domain, "strategies": strategies})


@app.route("/local_surrogate", methods=["POST"])
def local_surrogate():
    """Fit a LIME-style local surrogate for a single input."""
    body = request.get_json(force=True)
    domain = dm.resolve_domain(body)
    model_id = body.get("model_id")
    data = body.get("data")
    concepts = body.get("concepts", {})
    surrogate_type = body.get("surrogate_type", "logistic")
    n_perturbations = body.get("n_perturbations", 200)

    cbm = dm.get_cbm(domain, model_id)
    if cbm is None:
        return jsonify({"error": "Model not loaded"}), 400

    result = cbm.fit_local_surrogate(
        data, concepts,
        surrogate_type=surrogate_type,
        n_perturbations=n_perturbations,
    )
    return jsonify(result)


@app.route("/samples", methods=["GET"])
def samples():
    """Return sample data for a domain (default: toxicity)."""
    domain = request.args.get("domain", "toxicity")

    if domain == "toxicity":
        path = os.path.join(app.static_folder, "sample_text.txt")
        with open(path) as f:
            raw = f.read()
        texts = [t.strip() for t in raw.split("---") if t.strip()]
        return jsonify({"samples": texts, "domain": domain, "type": "text"})

    # Image domains: list files in the samples directory
    sample_dir = os.path.join(app.static_folder, "samples", domain)
    # Fallback: vision domain uses birds sample directory
    _has_images = os.path.isdir(sample_dir) and any(
        f.lower().endswith((".png", ".jpg", ".jpeg")) for f in os.listdir(sample_dir)
    )
    if not _has_images and domain == "vision":
        sample_dir = os.path.join(app.static_folder, "samples", "birds")
    if os.path.isdir(sample_dir):
        import base64
        items = []
        for fname in sorted(os.listdir(sample_dir)):
            fpath = os.path.join(sample_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith((".png", ".jpg", ".jpeg")):
                with open(fpath, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                items.append({"filename": fname, "data": f"data:image/png;base64,{b64}"})
        return jsonify({"samples": items, "domain": domain, "type": "image"})

    return jsonify({"samples": [], "domain": domain, "type": "unknown"})


@app.route("/rag/index", methods=["POST"])
def rag_index():
    """Manually trigger RAG indexing for a domain."""
    body = request.get_json(force=True)
    domain = body.get("domain")
    from rag_engine import get_rag_engine
    engine = get_rag_engine()
    if domain:
        count = engine.index_concepts(domain, force=body.get("force", False))
    else:
        count = engine.index_all_domains()
    return jsonify({"indexed": count})


# ── Model management routes ──────────────────────────────────────────

@app.route("/models/<domain>", methods=["GET"])
def list_models(domain: str):
    """List available models for a domain."""
    from config import get_domain_models
    models = get_domain_models(domain)
    return jsonify({"domain": domain, "models": [
        {"id": m["id"], "name": m["name"],
         "default": m.get("default", False),
         "custom": m.get("custom", False),
         "model_id": m.get("model_id", ""),
         "feature_dim": m.get("feature_dim", 0)}
        for m in models
    ]})


@app.route("/models/register", methods=["POST"])
def register_model():
    """Register a custom model at runtime."""
    body = request.get_json(force=True)
    domain = body.get("domain")
    if not domain:
        return jsonify({"error": "domain is required"}), 400

    required = ("id", "name", "model_class", "model_id", "feature_dim")
    missing = [k for k in required if k not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # Validate model_class is a known subclass path
    model_class = body["model_class"]
    allowed_classes = {
        "models.toxicity_model.ToxicityModel",
        "models.medical_model.MedicalModel",
        "models.vision_model.VisionModel",
        "models.bird_model.BirdModel",
    }
    if model_class not in allowed_classes:
        return jsonify({"error": f"model_class must be one of {sorted(allowed_classes)}"}), 400

    from config import register_custom_model, get_model_by_id
    # Check for duplicate id
    if get_model_by_id(domain, body["id"]):
        return jsonify({"error": f"Model id {body['id']!r} already exists"}), 409

    model_spec = {
        "id": body["id"],
        "name": body["name"],
        "model_class": model_class,
        "model_id": body["model_id"],
        "feature_dim": int(body["feature_dim"]),
    }
    mid = register_custom_model(domain, model_spec)
    return jsonify({"status": "ok", "model_id": mid})


@app.route("/models/<domain>/<model_id>", methods=["DELETE"])
def delete_model(domain: str, model_id: str):
    """Remove a custom-registered model."""
    from config import get_model_by_id, unregister_custom_model
    spec = get_model_by_id(domain, model_id)
    if spec is None:
        return jsonify({"error": "Model not found"}), 404
    if not spec.get("custom"):
        return jsonify({"error": "Cannot delete built-in models"}), 400

    unregister_custom_model(domain, model_id)
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
