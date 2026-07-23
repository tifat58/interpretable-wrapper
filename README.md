# ConceptLens: Interactive Concept-Guided Explanations for Black-Box Classifiers

**ConceptLens** is an interactive explainable AI prototype for turning static post-hoc explanations into a two-way, concept-guided interface. Instead of only showing feature attributions, heatmaps, pixels, or tokens, the system exposes semantically meaningful concepts that users can inspect, manipulate, and use for what-if analysis.

The project accompanies our ICMI 2026 demo paper:

> **ConceptLens: Interactive Concept Bottleneck Wrappers for Black-Box Models**  
> Demo paper: https://camps.aptaracorp.com/ACM_PMS/PMS/ACM/ICMI26/168/b5fa282a-8433-11f1-b513-16ffd757ba29/OUT/icmi26-168.html

---

## Motivation

Most interpretability tools explain model decisions after the fact, but offer limited support for user action. In high-stakes domains such as medicine, users often need more than a static explanation: they need to inspect model behavior, test alternatives, identify failure modes, and reason about how meaningful changes affect predictions.

**ConceptLens** is designed around **actionable interpretability**. It wraps a pretrained black-box classifier with a post-hoc concept interface, allowing users to move from one-way explanation toward two-way interaction with model behavior.

---

## What ConceptLens Enables

ConceptLens provides a unified interface for:

- **Concept inspection** — view concept activations associated with a prediction.
- **Local fidelity analysis** — evaluate how well a local concept-space approximation reflects the model near a selected input.
- **What-if interaction** — modify concepts with sliders, masking, or token edits.
- **Counterfactual comparison** — compare original and modified outcomes after concept changes.
- **Evidence-grounded explanation** — generate natural-language explanations supported by retrieved evidence or concept-specific snippets.
- **Feedback capture** — record concept-level user feedback that can support future refinement of concept probes or explanatory interfaces.

The current repository contains a lightweight research prototype with a simulated black-box backend. The architecture is designed to support integration with task-specific pretrained classifiers and concept extraction pipelines.

---

## Medical Demo Scenario

For medical audiences, ConceptLens demonstrates an interactive XAI workflow for **chest X-ray classification**. The system exposes clinically meaningful concepts such as opacity, effusion, consolidation, and cardiomegaly, allowing users to inspect which concepts are active, adjust them to test what-if scenarios, compare counterfactual predictions, and receive evidence-grounded explanations.

This shifts explanation from static heatmaps toward a more actionable workflow where clinicians, AI developers, and industry stakeholders can explore model behavior through semantic medical concepts.

---

## Supported Demonstration Tasks

The demo is designed around three classification scenarios spanning visual and textual modalities:

1. **Chest X-ray classification** — medical image setting with radiology-inspired concepts.
2. **Bird species recognition** — natural image setting with part- and appearance-based concepts.
3. **Toxic comment detection** — text classification setting with semantic concepts such as insult, threat, obscenity, and identity attack.

---

## System Overview

ConceptLens follows a post-hoc wrapper design:

```text
Input instance
   ↓
Pretrained black-box classifier
   ↓
Concept extraction / concept scoring
   ↓
Local concept-space approximation
   ↓
Interactive concept interface
   ↓
What-if analysis, counterfactual comparison, and grounded explanation
```

The original predictor remains fixed. The interactive layer operates externally by exposing concept activations and estimating how changes in those concepts affect the model output.

---

## Repository Structure

```text
frontend/  (React + Vite + Tailwind)
├── InputPanel       — text/image input selector
├── PredictionPanel  — label badge + confidence display
├── ConceptPanel     — concept sliders and counterfactual trigger
├── ExplanationPanel — natural-language explanation and evidence toggle
└── ChatAgent        — conversational QA about the current prediction

backend/  (Flask)
├── app.py             — REST API routes
├── dummy_model.py     — simulated black-box classifier for the prototype
├── counterfactuals.py — concept-delta counterfactual logic
└── explanations.py    — template-based natural-language explanations
```

---

## Setup

### Backend: Flask

From the project root:

```bash
pip install -r requirements.txt
python backend/app.py
```

The backend starts at:

```text
http://localhost:5000
```

### Frontend: React + Vite + Tailwind

```bash
cd frontend
npm install
npm run dev
```

The frontend starts at:

```text
http://localhost:5173
```

The Vite development server proxies API requests to the Flask backend.

---

## API Reference

### `POST /predict`

Classify an input and return prediction confidence with concept activations.

**Request**

```json
{
  "input_type": "text",
  "data": "you are terrible"
}
```

**Response**

```json
{
  "label": "toxic",
  "confidence": 0.85,
  "concepts": {
    "insult": 0.72,
    "threat": 0.31,
    "obscene": 0.48,
    "identity_attack": 0.09
  }
}
```

---

### `POST /counterfactual`

Simulate a new prediction after modifying concept values.

**Request**

```json
{
  "input_type": "text",
  "original_concepts": {
    "insult": 0.72,
    "threat": 0.31,
    "obscene": 0.48,
    "identity_attack": 0.09
  },
  "modified_concepts": {
    "insult": 0.20,
    "threat": 0.31,
    "obscene": 0.48,
    "identity_attack": 0.09
  },
  "original_confidence": 0.85
}
```

**Response**

```json
{
  "label": "not toxic",
  "confidence": 0.668,
  "concept_deltas": {
    "insult": -0.52,
    "threat": 0.0,
    "obscene": 0.0,
    "identity_attack": 0.0
  }
}
```

---

### `POST /explain`

Generate a natural-language explanation, optionally with evidence snippets.

**Request**

```json
{
  "input_type": "text",
  "label": "toxic",
  "confidence": 0.85,
  "concepts": {
    "insult": 0.72,
    "threat": 0.31,
    "obscene": 0.48,
    "identity_attack": 0.09
  },
  "evidence": true
}
```

**Response**

```json
{
  "explanation_text": "The model predicts toxic with high confidence primarily because the concept insult is strongly activated.",
  "evidence_snippets": [
    {
      "concept": "insult",
      "activation": 0.72,
      "text": "The input contains language that demeans or attacks another person."
    }
  ]
}
```

---

### `POST /chat`

Ask conversational questions about the current prediction and concept profile.

**Request**

```json
{
  "message": "why this prediction?",
  "context": {
    "label": "toxic",
    "confidence": 0.85,
    "concepts": {
      "insult": 0.72
    }
  }
}
```

**Response**

```json
{
  "reply": "The model predicts toxic mainly because the concept insult has a high activation of 72%."
}
```

---

## Implementation Notes

- The current prototype uses a simulated black-box model to demonstrate the interaction workflow.
- The backend is implemented with Flask.
- The frontend is implemented with React, Vite, and Tailwind.
- Counterfactual outputs are computed from concept deltas in the prototype implementation.
- The design is intended to be extended with real pretrained classifiers, learned concept probes, vision-language concept scoring, and retrieval-backed explanation modules.

---

## Intended Use

ConceptLens is intended for research and demonstration purposes. It is not a clinical decision-support tool and should not be used for medical diagnosis. The medical scenario is included to demonstrate how concept-guided interaction can support transparent and controllable analysis of medical AI systems.

---

## Citation

If you use or refer to this demo, please cite the associated ICMI 2026 demo paper. A BibTeX entry will be added after the official ACM metadata is finalized.
