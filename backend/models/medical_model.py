"""Medical domain model — torchxrayvision DenseNet on chest X-rays."""

from __future__ import annotations

import base64
import io
import logging
import os
import pickle
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.base import BaseModel

logger = logging.getLogger(__name__)

_TASK_HEAD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "cbm", "medical", "task_head.pkl",
)
_PROBE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "cbm", "medical", "probes",
)

# torchxrayvision pathology labels (order matters — matches model output)
_TXV_PATHOLOGIES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
]


class MedicalModel(BaseModel):
    """Wraps torchxrayvision DenseNet121 trained on CheXpert / NIH data."""

    def __init__(self, domain: str, config: dict):
        super().__init__(domain, config)
        self._model = None
        self._features_hook = None
        self._last_features: torch.Tensor | None = None
        self._task_head = None
        self._task_head_classes: list[str] = []

    # ── loading ──────────────────────────────────────────────────────
    def load(self) -> None:
        import torchxrayvision as xrv

        self._model = xrv.models.DenseNet(weights="densenet121-res224-chex")
        self._model.to(self.device).eval()

        # Register a forward hook on the final feature block to capture
        # penultimate activations for concept probing.
        def _hook(_module, _input, output):
            self._last_features = output

        self._features_hook = self._model.features.register_forward_hook(_hook)

        # Trained task head (features → COVID-19/Non-COVID/Normal), produced
        # by scripts/train_medical_cbm.py.  Falls back to the pathology
        # heuristic in predict_raw when absent.
        if os.path.isfile(_TASK_HEAD_PATH):
            with open(_TASK_HEAD_PATH, "rb") as f:
                data = pickle.load(f)  # noqa: S301 — trusted local file
            self._task_head = data["model"]
            self._task_head_classes = data["classes"]
            logger.info("MedicalModel: trained task head loaded (%s)",
                        self._task_head_classes)
        else:
            logger.warning("MedicalModel: no task head found — using "
                           "pathology-score heuristic")

        self._loaded = True
        logger.info("MedicalModel loaded (%s)", self.device)

    # ── preprocessing ────────────────────────────────────────────────
    def preprocess(self, data: Any) -> torch.Tensor:
        """Accept file path, base64 string, or PIL Image → (1,1,224,224) tensor."""
        import torchxrayvision as xrv

        img = self._to_pil(data).convert("L")  # grayscale
        img_np = np.array(img, dtype=np.float32)

        # torchxrayvision expects shape (C, H, W) with values in [0, 1024]
        img_np = xrv.datasets.normalize(img_np, maxval=255, reshape=True)

        # Resize to 224×224
        tensor = torch.from_numpy(img_np).unsqueeze(0)  # (1, 1, H, W)
        tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        return tensor.to(self.device)

    # ── inference ────────────────────────────────────────────────────
    def predict_raw(self, data: Any) -> dict:
        tensor = self.preprocess(data)
        with torch.no_grad():
            logits = self._model(tensor)  # (1, 14)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

        # Build per-pathology scores
        raw_scores = {}
        for i, name in enumerate(_TXV_PATHOLOGIES):
            if i < len(probs):
                raw_scores[name] = float(probs[i])

        # Preferred path: calibrated logistic head on penultimate features.
        if self._task_head is not None and self._last_features is not None:
            pooled = F.adaptive_avg_pool2d(self._last_features, 1).flatten(1)
            head_probs = self._task_head.predict_proba(pooled.cpu().numpy())[0]
            idx = int(head_probs.argmax())
            return {
                "label": self._task_head_classes[idx],
                "confidence": round(float(head_probs[idx]), 4),
                "raw_scores": raw_scores,
                "class_probs": {c: round(float(p), 4)
                                for c, p in zip(self._task_head_classes, head_probs)},
            }

        # Fallback heuristic: derive 3-class prediction from pathology scores.
        pneumonia_score = raw_scores.get("Pneumonia", 0.0)
        consolidation_score = raw_scores.get("Consolidation", 0.0)
        infiltration_score = raw_scores.get("Infiltration", 0.0)
        # COVID-19 signal: high consolidation + infiltration pattern
        covid_score = 0.4 * pneumonia_score + 0.3 * consolidation_score + 0.3 * infiltration_score
        # Non-COVID abnormal: any significant pathology
        abnormal_score = max(raw_scores.get("Effusion", 0.0),
                             raw_scores.get("Edema", 0.0),
                             raw_scores.get("Cardiomegaly", 0.0),
                             raw_scores.get("Atelectasis", 0.0))
        non_covid_score = max(abnormal_score, pneumonia_score * 0.5)
        # Normal: absence of pathology
        normal_score = 1.0 - max(covid_score, non_covid_score)

        scores_3c = {"COVID-19": covid_score, "Non-COVID": non_covid_score,
                     "Normal": normal_score}
        label = max(scores_3c, key=scores_3c.get)
        confidence = scores_3c[label]

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "raw_scores": raw_scores,
        }

    # ── feature extraction ───────────────────────────────────────────
    def extract_features(self, data: Any) -> np.ndarray:
        tensor = self.preprocess(data)
        with torch.no_grad():
            _ = self._model(tensor)

        # _last_features captured by the hook — shape (1, 1024, 7, 7)
        feat = self._last_features
        if feat is None:
            raise RuntimeError("Feature hook did not fire")
        # Global average pooling → (1024,)
        pooled = F.adaptive_avg_pool2d(feat, 1).squeeze()
        return pooled.cpu().numpy()

    # ── attribution (GradCAM) ────────────────────────────────────────
    def get_attribution(self, data: Any, target_concept: str | None = None) -> dict:
        tensor = self.preprocess(data).requires_grad_(True)

        if target_concept and self._probe_coefficients(target_concept) is not None:
            cam = self._probe_attribution(tensor, target_concept)
            return self._format_attribution(data, cam, target_concept, "probe_gradcam")

        # Forward with gradient tracking — capture feature activations
        # and retain their grads for true GradCAM.
        captured = {}

        def _fwd_hook(_module, _input, output):
            captured["feats"] = output
            output.retain_grad()

        hook = self._model.features.register_forward_hook(_fwd_hook)

        self._model.zero_grad()
        logits = self._model(tensor)
        probs = torch.sigmoid(logits)

        # Pick target index — map concept names to pathology indices.
        # Concept names from the concept bank may differ from _TXV_PATHOLOGIES
        # names, so try direct match first, then fuzzy match.
        target = target_concept or "Pneumonia"
        _CONCEPT_TO_PATHOLOGY = {
            "Consolidation": "Consolidation",
            "Ground Glass Opacity": "Infiltration",  # closest proxy
            "Lung Opacity": "Infiltration",
            "Pleural Effusion": "Effusion",
            "Cardiomegaly": "Cardiomegaly",
            "Edema": "Edema",
            "Atelectasis": "Atelectasis",
            "Bilateral Involvement": "Pneumonia",
            "Peripheral Distribution": "Infiltration",
            "Air Bronchogram": "Consolidation",
            "Lung Volume Loss": "Atelectasis",
            "Clear Lung Fields": "Pneumonia",  # inverse
        }
        pathology_name = _CONCEPT_TO_PATHOLOGY.get(target, target)
        idx = _TXV_PATHOLOGIES.index(pathology_name) if pathology_name in _TXV_PATHOLOGIES else 8
        if idx >= probs.shape[1]:
            idx = 0

        # Backward for GradCAM
        score = probs[0, idx]
        score.backward(retain_graph=False)
        hook.remove()

        feats = captured.get("feats")  # (1, C, H, W)

        # GradCAM: gradient-weighted sum of feature maps
        if feats is not None and feats.grad is not None:
            weights = feats.grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * feats).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
            cam = cam.squeeze().detach().cpu().numpy()
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)
        else:
            cam = np.zeros((224, 224), dtype=np.float32)

        # Raw heatmap
        heatmap_b64 = self._array_to_base64_png(cam)

        # Blended overlay on original image
        overlay_b64 = self._make_overlay(data, cam)
        original_b64 = self._encode_original(data)
        heatmap_colored_b64 = self._encode_colored_heatmap(cam)

        return {
            "method": "gradcam",
            "type": "heatmap",
            "data": heatmap_b64,
            "overlay": overlay_b64,
            "original_image": original_b64,
            "heatmap_colored": heatmap_colored_b64,
            "shape": [224, 224],
            "concept": target,
        }

    def _probe_coefficients(self, concept: str) -> np.ndarray | None:
        path = os.path.join(_PROBE_DIR, f"{concept}.pkl")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "rb") as f:
                probe = pickle.load(f)  # noqa: S301 — trusted local artifact
            model = probe.get("model")
            coefficients = getattr(model, "coef_", None)
            if coefficients is None or coefficients.shape[0] != 1:
                return None
            return coefficients[0].astype(np.float32)
        except (OSError, KeyError, AttributeError, ValueError):
            logger.exception("Could not load concept probe %r", concept)
            return None

    def _probe_attribution(self, tensor: torch.Tensor, concept: str) -> np.ndarray:
        coefficients = self._probe_coefficients(concept)
        if coefficients is None:
            raise RuntimeError(f"No binary probe available for {concept!r}")

        captured: dict[str, torch.Tensor] = {}

        def _fwd_hook(_module, _input, output):
            captured["features"] = output

        hook = self._model.features.register_forward_hook(_fwd_hook)
        self._model.zero_grad()
        self._model(tensor)
        hook.remove()
        features = captured.get("features")
        if features is None:
            raise RuntimeError("Feature hook did not fire")

        # A linear probe over global-average-pooled features has a spatial
        # equivalent: coefficient-weighted feature maps before pooling.
        weights = torch.as_tensor(coefficients, device=features.device)
        if weights.numel() != features.shape[1]:
            raise RuntimeError("Probe and DenseNet feature dimensions differ")
        cam = F.relu((features * weights.view(1, -1, 1, 1)).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        return (cam - cam_min) / (cam_max - cam_min) if cam_max - cam_min > 1e-8 else np.zeros_like(cam)

    def _format_attribution(self, data: Any, cam: np.ndarray,
                            concept: str, method: str) -> dict:
        return {
            "method": method,
            "type": "heatmap",
            "data": self._array_to_base64_png(cam),
            "overlay": self._make_overlay(data, cam),
            "original_image": self._encode_original(data),
            "heatmap_colored": self._encode_colored_heatmap(cam),
            "shape": [224, 224],
            "concept": concept,
        }

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _to_pil(data: Any) -> Image.Image:
        if isinstance(data, Image.Image):
            return data
        if isinstance(data, str):
            # Try base64 first
            if len(data) > 260 or ";" in data[:30]:
                # Strip data-URI prefix if present
                if "," in data[:80]:
                    data = data.split(",", 1)[1]
                raw = base64.b64decode(data)
                return Image.open(io.BytesIO(raw))
            # Otherwise treat as file path
            return Image.open(data)
        if isinstance(data, (bytes, bytearray)):
            return Image.open(io.BytesIO(data))
        raise ValueError(f"Cannot convert {type(data).__name__} to PIL Image")

    @staticmethod
    def _array_to_base64_png(arr: np.ndarray) -> str:
        from PIL import Image as PILImage
        arr_uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
        img = PILImage.fromarray(arr_uint8, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _encode_original(self, data: Any) -> str:
        """Return the input image resized to 224x224 as base64 PNG."""
        from PIL import Image as PILImage
        img = self._to_pil(data).convert("RGB").resize((224, 224))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _encode_colored_heatmap(cam: np.ndarray) -> str:
        """Return a JET-colored heatmap as a standalone base64 PNG."""
        from PIL import Image as PILImage
        import matplotlib.cm as cm
        colored = (cm.jet(cam)[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
        buf = io.BytesIO()
        PILImage.fromarray(colored).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _make_overlay(self, data: Any, cam: np.ndarray, alpha: float = 0.4) -> str:
        """Blend a [0,1] CAM heatmap onto the original image as a JET colormap overlay."""
        from PIL import Image as PILImage
        import matplotlib.cm as cm

        img = self._to_pil(data).convert("RGB").resize((224, 224))
        img_arr = np.array(img, dtype=np.float32) / 255.0

        # Apply JET colormap on the CAM
        colored = cm.jet(cam)[:, :, :3]  # drop alpha channel

        # Blend: original * (1-alpha) + heatmap * alpha
        blended = img_arr * (1 - alpha) + colored * alpha
        blended = (blended * 255).clip(0, 255).astype(np.uint8)

        buf = io.BytesIO()
        PILImage.fromarray(blended).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
