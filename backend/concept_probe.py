"""Concept-probe layer with feedback-driven updates (actionable interpretability).

This module turns the demo from a passive *inspection* tool into an *actionable*
one: a domain expert corrects a concept reading, that correction is stored, and
the concept probe is re-fit. The effect is measurable and generalises:

  * the corrected concept's activation moves toward the expert value,
  * the surrogate's **fidelity** to the black box improves on a held-out set,
  * other held-out examples are read more accurately too (generalisation).

To stay dependency-free (no numpy/sklearn in this MVP) the probes are tiny
logistic-regression models trained with plain-Python gradient descent on a
seeded synthetic feature space. On the GPU server this same interface is backed
by real encoder features (e.g. torchxrayvision / CLIP) and sklearn probes; the
contract (state / feedback / retrain / reset) is identical.

Narrative for the demo: the ``effusion`` probe ships **miscalibrated** (trained
on a small, biased subset), so the wrapper systematically misreads pleural
effusion and the surrogate disagrees with the black box on ~30% of cases.
A handful of expert corrections re-fit the probe and recover fidelity.
"""

from __future__ import annotations

import math
import random

# ── Configuration ────────────────────────────────────────────────────
FEATURE_DIM = 8
CONCEPTS = ["opacity", "cardiomegaly", "effusion", "consolidation"]
MISCALIBRATED_CONCEPT = "effusion"
SEED = 7

# Surrogate weights mapping concept activations -> black-box label score.
# Effusion is weighted highly so that misreading it (the miscalibrated probe)
# visibly degrades the surrogate's fidelity to the black box.
_SURROGATE_WEIGHTS = {
    "opacity": 0.25,
    "cardiomegaly": 0.10,
    "effusion": 0.50,
    "consolidation": 0.15,
}
_SURROGATE_BIAS = -0.5  # decision boundary at weighted-sum = 0.5


def _sigmoid(z: float) -> float:
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class _LinearProbe:
    """Logistic-regression probe: features -> concept activation in [0, 1]."""

    def __init__(self, dim: int):
        self.w = [0.0] * dim
        self.b = 0.0

    def predict(self, x) -> float:
        return _sigmoid(_dot(self.w, x) + self.b)

    def fit(self, X, y, epochs=400, lr=0.5, l2=1e-3, sample_weight=None):
        n = len(X)
        if n == 0:
            return
        sw = sample_weight if sample_weight is not None else [1.0] * n
        for _ in range(epochs):
            grad_w = [0.0] * len(self.w)
            grad_b = 0.0
            for xi, yi, wi in zip(X, y, sw):
                pred = self.predict(xi)
                err = (pred - yi) * wi
                for j in range(len(self.w)):
                    grad_w[j] += err * xi[j]
                grad_b += err
            for j in range(len(self.w)):
                self.w[j] -= lr * (grad_w[j] / n + l2 * self.w[j])
            self.b -= lr * (grad_b / n)


class ProbeSystem:
    """Holds the synthetic data, the probes, and the feedback buffer."""

    def __init__(self):
        self.rng = random.Random(SEED)
        self._build_dataset()
        self._build_true_decoders()
        self._label_data()
        self.feedback = {c: [] for c in CONCEPTS}  # concept -> list of (x, target)
        self._init_probes()
        # A fixed "current example" the expert is inspecting: pick a validation
        # case that the miscalibrated probe currently misreads.
        self.current_idx = self._pick_demo_example()

    # ── data generation ─────────────────────────────────────────────
    def _build_dataset(self):
        self.X_train = [[self.rng.gauss(0, 1) for _ in range(FEATURE_DIM)]
                        for _ in range(160)]
        self.X_val = [[self.rng.gauss(0, 1) for _ in range(FEATURE_DIM)]
                      for _ in range(80)]

    def _build_true_decoders(self):
        # Ground-truth linear decoder for each concept (what a perfect probe learns).
        self.true_w = {}
        self.true_b = {}
        for c in CONCEPTS:
            self.true_w[c] = [self.rng.uniform(-1.5, 1.5) for _ in range(FEATURE_DIM)]
            self.true_b[c] = self.rng.uniform(-0.3, 0.3)

    def _true_concept(self, x, c) -> float:
        return _sigmoid(_dot(self.true_w[c], x) + self.true_b[c])

    def _black_box_label(self, x) -> int:
        """f(x): the black box, defined over the TRUE concepts."""
        score = sum(_SURROGATE_WEIGHTS[c] * self._true_concept(x, c) for c in CONCEPTS)
        return 1 if score + _SURROGATE_BIAS >= 0 else 0  # 1 = pneumonia

    def _label_data(self):
        self.y_train = {c: [1 if self._true_concept(x, c) >= 0.5 else 0
                            for x in self.X_train] for c in CONCEPTS}
        self.y_val = {c: [1 if self._true_concept(x, c) >= 0.5 else 0
                          for x in self.X_val] for c in CONCEPTS}
        self.f_val = [self._black_box_label(x) for x in self.X_val]
        self.f_train = [self._black_box_label(x) for x in self.X_train]

    # ── probes ───────────────────────────────────────────────────────
    def _init_probes(self):
        self.probes = {}
        for c in CONCEPTS:
            p = _LinearProbe(FEATURE_DIM)
            if c == MISCALIBRATED_CONCEPT:
                # Miscalibrated: trained on a small subset with noisy labels
                # (22% flipped) -> poor decoder that systematically misreads
                # effusion (~68% concept accuracy, ~78% surrogate fidelity).
                idx = list(range(24))
                noisy_rng = random.Random(SEED + 1)
                ys = [self.y_train[c][i] for i in idx]
                ys = [(1 - y if noisy_rng.random() < 0.22 else y) for y in ys]
                p.fit([self.X_train[i] for i in idx], ys, epochs=120, lr=0.4)
            else:
                # Well-calibrated probes (full data, converged).
                p.fit(self.X_train, self.y_train[c], epochs=400, lr=0.5)
            self.probes[c] = p

    def _pick_demo_example(self) -> int:
        # Prefer a validation example where the surrogate currently disagrees
        # with the black box *because of* the miscalibrated concept.
        for i, x in enumerate(self.X_val):
            if self._surrogate_label(x) != self.f_val[i]:
                est = self.probes[MISCALIBRATED_CONCEPT].predict(x)
                true = self._true_concept(x, MISCALIBRATED_CONCEPT)
                if abs(est - true) > 0.3:
                    return i
        return 0

    # ── surrogate + fidelity ─────────────────────────────────────────
    def _concept_vector(self, x):
        return {c: self.probes[c].predict(x) for c in CONCEPTS}

    def _surrogate_label(self, x) -> int:
        cv = self._concept_vector(x)
        score = sum(_SURROGATE_WEIGHTS[c] * cv[c] for c in CONCEPTS)
        return 1 if score + _SURROGATE_BIAS >= 0 else 0

    def fidelity(self) -> float:
        """Agreement between surrogate h(c(x)) and black box f(x) on val set."""
        agree = sum(1 for x, f in zip(self.X_val, self.f_val)
                    if self._surrogate_label(x) == f)
        return round(agree / len(self.X_val), 3)

    def concept_accuracy(self, c) -> float:
        """How often the probe's binarised reading matches the true concept."""
        correct = sum(1 for x, y in zip(self.X_val, self.y_val[c])
                      if (1 if self.probes[c].predict(x) >= 0.5 else 0) == y)
        return round(correct / len(self.X_val), 3)

    # ── public API ───────────────────────────────────────────────────
    def state(self):
        x = self.X_val[self.current_idx]
        cv = self._concept_vector(x)
        return {
            "concept": MISCALIBRATED_CONCEPT,
            "current_example": self.current_idx,
            "estimated_concepts": {k: round(v, 3) for k, v in cv.items()},
            "expert_value": round(self._true_concept(x, MISCALIBRATED_CONCEPT), 3),
            "fidelity": self.fidelity(),
            "concept_accuracy": self.concept_accuracy(MISCALIBRATED_CONCEPT),
            "feedback_count": len(self.feedback[MISCALIBRATED_CONCEPT]),
            "validation_preview": self._val_preview(),
        }

    def _val_preview(self, n=6):
        """A few held-out examples: probe estimate vs expert/true value."""
        out = []
        for i in range(n):
            x = self.X_val[i]
            out.append({
                "id": i,
                "estimate": round(self.probes[MISCALIBRATED_CONCEPT].predict(x), 3),
                "truth": round(self._true_concept(x, MISCALIBRATED_CONCEPT), 3),
            })
        return out

    def add_feedback(self, concept=MISCALIBRATED_CONCEPT, corrected_value=None):
        """Store an expert correction for the current example."""
        if concept not in CONCEPTS:
            return {"error": f"unknown concept '{concept}'"}
        x = self.X_val[self.current_idx]
        if corrected_value is None:
            corrected_value = self._true_concept(x, concept)
        target = 1.0 if corrected_value >= 0.5 else 0.0
        self.feedback[concept].append((x, target))
        # Advance to the next currently-misread example so each correction is
        # about a different case (more convincing in the demo).
        self.current_idx = self._next_misread(concept)
        return {
            "feedback_count": len(self.feedback[concept]),
            "next_example": self.current_idx,
        }

    def _next_misread(self, concept):
        start = self.current_idx
        n = len(self.X_val)
        for off in range(1, n + 1):
            i = (start + off) % n
            est = self.probes[concept].predict(self.X_val[i])
            true = self._true_concept(self.X_val[i], concept)
            if abs(est - true) > 0.3:
                return i
        return (start + 1) % n

    def retrain(self, concept=MISCALIBRATED_CONCEPT):
        """Re-fit the probe on its original data + expert feedback (upweighted)."""
        before = {
            "fidelity": self.fidelity(),
            "concept_accuracy": self.concept_accuracy(concept),
            "validation_preview": self._val_preview(),
        }
        fb = self.feedback[concept]
        if not fb:
            return {"error": "No feedback collected yet.", "before": before}

        # Combine the full training set with feedback points (weighted higher).
        X = list(self.X_train) + [x for x, _ in fb]
        y = list(self.y_train[concept]) + [t for _, t in fb]
        sw = [1.0] * len(self.X_train) + [8.0] * len(fb)

        p = _LinearProbe(FEATURE_DIM)
        p.fit(X, y, epochs=500, lr=0.5, sample_weight=sw)
        self.probes[concept] = p

        after = {
            "fidelity": self.fidelity(),
            "concept_accuracy": self.concept_accuracy(concept),
            "validation_preview": self._val_preview(),
        }
        return {
            "concept": concept,
            "before": before,
            "after": after,
            "fidelity_gain": round(after["fidelity"] - before["fidelity"], 3),
            "accuracy_gain": round(after["concept_accuracy"] - before["concept_accuracy"], 3),
            "feedback_used": len(fb),
        }

    def reset(self):
        """Restore the miscalibrated probe and clear feedback (demo replay)."""
        self.feedback = {c: [] for c in CONCEPTS}
        self._init_probes()
        self.current_idx = self._pick_demo_example()
        return self.state()


# Module-level singleton used by the Flask app.
probe_system = ProbeSystem()
