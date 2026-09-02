#!/usr/bin/env python3
"""Train concept probes and surrogate for a domain.

Usage
-----
    # Synthetic mode (random data — for testing the pipeline):
    python -m scripts.train_probes --domain toxicity --synthetic

    # Synthetic mode for all domains:
    python -m scripts.train_probes --all --synthetic

    # Real mode (requires dataset — not yet implemented):
    python -m scripts.train_probes --domain medical
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

# Ensure backend/ is on the path
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from config import DOMAIN_CONFIG, PROBE_DATA_DIR
from cbm.concept_bank import ConceptBank
from cbm.probe import ProbeBank

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)


def train_synthetic(domain: str, n_samples: int = 500) -> None:
    """Generate random features + labels, train probes + surrogate, save."""
    cfg = DOMAIN_CONFIG[domain]
    feature_dim = cfg["feature_dim"]

    cb = ConceptBank(domain)
    pb = ProbeBank(concept_bank=cb, feature_dim=feature_dim, probe_type="logistic")

    logger.info("Training synthetic probes for %r (%d concepts, dim=%d, N=%d)",
                domain, cb.num_concepts, feature_dim, n_samples)

    # Generate random features
    rng = np.random.default_rng(42)
    features = rng.standard_normal((n_samples, feature_dim)).astype(np.float32)

    # Generate random binary labels for each concept
    concept_labels = {}
    for name in cb.concepts:
        concept_labels[name] = rng.integers(0, 2, size=n_samples)

    # Train concept probes
    accuracies = pb.train_all(features, concept_labels)
    logger.info("Probe accuracies: %s",
                {k: f"{v:.3f}" for k, v in sorted(accuracies.items())})

    # Save probes
    probe_dir = os.path.join(PROBE_DATA_DIR, domain)
    pb.save(probe_dir)

    # Train surrogate (concept activations → binary label)
    logger.info("Training surrogate for %r ...", domain)
    concept_matrix = np.column_stack([
        concept_labels[name].astype(np.float32) for name in cb.concepts
    ])
    binary_labels = rng.integers(0, 2, size=n_samples)

    from cbm.cbm_wrapper import PostHocCBM

    # Create a minimal stub model for surrogate training
    class _StubModel:
        pass

    stub = _StubModel()
    stub.domain = domain
    stub.input_type = cfg["input_type"]
    stub.device = "cpu"
    stub.feature_dim = feature_dim
    stub.is_loaded = False
    stub.load = lambda: None
    stub.preprocess = lambda d: None
    stub.predict_raw = lambda d: {"label": "x", "confidence": 0.5}
    stub.extract_features = lambda d: np.zeros(feature_dim)
    stub.get_attribution = lambda d, **kw: {}

    cbm = PostHocCBM(domain=domain, model=stub,
                     concept_bank=cb, probe_bank=pb)

    labels = cfg.get("labels", ["positive", "negative"])
    if len(labels) >= 2:
        class_names = labels[:2]
    else:
        class_names = ["positive", "negative"]

    acc = cbm.train_surrogate(concept_matrix, binary_labels, class_names=class_names)
    logger.info("Surrogate accuracy: %.3f", acc)

    # Save CBM state (includes surrogate)
    cbm.save(probe_dir)
    logger.info("Saved all to %s", probe_dir)


def main():
    parser = argparse.ArgumentParser(description="Train concept probes")
    parser.add_argument("--domain", type=str, choices=list(DOMAIN_CONFIG.keys()),
                        help="Domain to train probes for")
    parser.add_argument("--all", action="store_true",
                        help="Train probes for all domains")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use random synthetic data (for testing)")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of synthetic samples (default: 500)")
    args = parser.parse_args()

    if not args.domain and not args.all:
        parser.error("Specify --domain DOMAIN or --all")

    domains = list(DOMAIN_CONFIG.keys()) if args.all else [args.domain]

    for domain in domains:
        if args.synthetic:
            train_synthetic(domain, n_samples=args.n_samples)
        else:
            logger.error("Real training not yet implemented for %r. Use --synthetic.", domain)
            sys.exit(1)


if __name__ == "__main__":
    main()
