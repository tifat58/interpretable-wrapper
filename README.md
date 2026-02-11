# Interpretable Wrapper — Interactive Interpretability for Black-Box AI Models

A model-agnostic interactive system that wraps around any black-box AI model (text or image) and makes it interpretable, controllable, and explainable through a concept-level user interface. Users can view predictions with concept activations, manipulate concepts via sliders, see counterfactual outputs, and generate evidence-grounded natural language explanations — all from a single dashboard. An optional conversational chat agent lets users ask questions about predictions in plain English.

> **MVP Note:** All ML components are simulated with dummy data and template logic so the full interaction pipeline can be demonstrated end-to-end without real model weights.

## Setup

### Backend (Flask)

```bash
# From the project root
pip install -r requirements.txt
python backend/app.py          # Starts on http://localhost:5000
```

### Frontend (React + Vite + Tailwind)

```bash
cd frontend
npm install
npm run dev                    # Starts on http://localhost:5173
```

The Vite dev server proxies API requests (`/predict`, `/counterfactual`, `/explain`, `/chat`, `/samples`) to `localhost:5000`.

## API Reference

### `POST /predict`

Classify an input and return concept activations.

**Request:**
```json
{ "input_type": "text", "data": "you are terrible" }
```

**Response:**
```json
{
  "label": "toxic",
  "confidence": 0.85,
  "concepts": { "insult": 0.72, "threat": 0.31, "obscene": 0.48, "identity_attack": 0.09 }
}
```

### `POST /counterfactual`

Simulate a new prediction after modifying concept values.

**Request:**
```json
{
  "input_type": "text",
  "original_concepts": { "insult": 0.72, "threat": 0.31, "obscene": 0.48, "identity_attack": 0.09 },
  "modified_concepts": { "insult": 0.20, "threat": 0.31, "obscene": 0.48, "identity_attack": 0.09 },
  "original_confidence": 0.85
}
```

**Response:**
```json
{ "label": "not toxic", "confidence": 0.668, "concept_deltas": { "insult": -0.52, "threat": 0.0, "obscene": 0.0, "identity_attack": 0.0 } }
```

### `POST /explain`

Generate a natural-language explanation, optionally with grounded evidence.

**Request:**
```json
{
  "input_type": "text",
  "label": "toxic",
  "confidence": 0.85,
  "concepts": { "insult": 0.72, "threat": 0.31, "obscene": 0.48, "identity_attack": 0.09 },
  "evidence": true
}
```

**Response:**
```json
{
  "explanation_text": "The model predicts \"toxic\" (confidence 85%) primarily because the concept \"insult\" is activated at 72%...",
  "evidence_snippets": [
    { "concept": "insult", "activation": 0.72, "text": "The input contains language that demeans..." }
  ]
}
```

### `POST /chat`

Conversational QA about the current prediction.

**Request:**
```json
{ "message": "why this prediction?", "context": { "label": "toxic", "confidence": 0.85, "concepts": { "insult": 0.72 } } }
```

**Response:**
```json
{ "reply": "The model predicts \"toxic\" mainly because the concept \"insult\" has a high activation of 72%..." }
```

## Architecture

```
frontend/  (React + Vite + Tailwind)
├── InputPanel       — text/image input selector
├── PredictionPanel  — label badge + confidence bar
├── ConceptPanel     — concept sliders + counterfactual trigger
├── ExplanationPanel — NL explanation + evidence toggle
└── ChatAgent        — conversational QA interface

backend/  (Flask)
├── app.py             — REST routes
├── dummy_model.py     — simulated black-box model
├── counterfactuals.py — concept-delta counterfactual logic
└── explanations.py    — template-based NL explanation generator
```
