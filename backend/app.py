"""Flask REST API for the Interpretable Wrapper."""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from dummy_model import DummyModel
from counterfactuals import compute_counterfactual
from explanations import generate_explanation

app = Flask(__name__, static_folder="static")
CORS(app)

model = DummyModel()


# ── Chat template logic ─────────────────────────────────────────────
def _chat_response(message: str, context: dict) -> str:
    """Return a template-generated reply based on keyword matching."""
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
        # Try to match a concept name in the message
        for concept in concepts:
            if concept in msg:
                return (
                    f"\"{concept}\" is a concept the model uses internally. "
                    f"Its current activation is {concepts[concept]:.0%}."
                )
        return "Could you specify which concept you'd like to know about?"
    if any(kw in msg for kw in ("confident", "confidence", "sure", "certain")):
        return f"The model is {confidence:.0%} confident that the input is \"{label}\"."
    # Fallback
    return (
        f"I can help explain the current prediction (\"{label}\", "
        f"{confidence:.0%} confidence). Try asking \"why this prediction?\" "
        f"or \"what if I change a concept?\""
    )


# ── Routes ───────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    input_type = body.get("input_type", "text")
    data = body.get("data")
    result = model.predict(input_type, data)
    return jsonify(result)


@app.route("/counterfactual", methods=["POST"])
def counterfactual():
    body = request.get_json(force=True)
    input_type = body.get("input_type", "text")
    original_concepts = body.get("original_concepts", {})
    modified_concepts = body.get("modified_concepts", {})
    original_confidence = body.get("original_confidence", 0.5)
    result = compute_counterfactual(
        input_type, original_concepts, modified_concepts, original_confidence
    )
    return jsonify(result)


@app.route("/explain", methods=["POST"])
def explain():
    body = request.get_json(force=True)
    input_type = body.get("input_type", "text")
    label = body.get("label", "")
    confidence = body.get("confidence", 0)
    concepts = body.get("concepts", {})
    evidence = body.get("evidence", False)
    result = generate_explanation(input_type, label, confidence, concepts, evidence)
    return jsonify(result)


@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True)
    message = body.get("message", "")
    context = body.get("context", {})
    reply = _chat_response(message, context)
    return jsonify({"reply": reply})


@app.route("/samples", methods=["GET"])
def samples():
    """Return the sample text file contents split by '---' separator."""
    path = app.static_folder + "/sample_text.txt"
    with open(path) as f:
        raw = f.read()
    texts = [t.strip() for t in raw.split("---") if t.strip()]
    return jsonify({"samples": texts})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
