"""Bird domain model — ResNet-50 fine-tuned on CUB-200-2011."""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from models.base import BaseModel

logger = logging.getLogger(__name__)

# 30 selected CUB species (class_id is 1-indexed in CUB)
CUB_SELECTED = [
    (14,  "014.Indigo_Bunting",            "Indigo Bunting"),
    (16,  "016.Painted_Bunting",           "Painted Bunting"),
    (17,  "017.Cardinal",                  "Cardinal"),
    (29,  "029.American_Crow",             "American Crow"),
    (36,  "036.Northern_Flicker",          "Northern Flicker"),
    (44,  "044.Frigatebird",               "Frigatebird"),
    (47,  "047.American_Goldfinch",        "American Goldfinch"),
    (53,  "053.Western_Grebe",             "Western Grebe"),
    (62,  "062.Herring_Gull",              "Herring Gull"),
    (67,  "067.Anna_Hummingbird",          "Anna's Hummingbird"),
    (68,  "068.Ruby_throated_Hummingbird", "Ruby-throated Hummingbird"),
    (73,  "073.Blue_Jay",                  "Blue Jay"),
    (76,  "076.Dark_eyed_Junco",           "Dark-eyed Junco"),
    (79,  "079.Belted_Kingfisher",         "Belted Kingfisher"),
    (87,  "087.Mallard",                   "Mallard"),
    (91,  "091.Mockingbird",               "Mockingbird"),
    (95,  "095.Baltimore_Oriole",          "Baltimore Oriole"),
    (100, "100.Brown_Pelican",             "Brown Pelican"),
    (106, "106.Horned_Puffin",             "Horned Puffin"),
    (107, "107.Common_Raven",              "Common Raven"),
    (109, "109.American_Redstart",         "American Redstart"),
    (118, "118.House_Sparrow",             "House Sparrow"),
    (136, "136.Barn_Swallow",              "Barn Swallow"),
    (139, "139.Scarlet_Tanager",           "Scarlet Tanager"),
    (149, "149.Brown_Thrasher",            "Brown Thrasher"),
    (182, "182.Yellow_Warbler",            "Yellow Warbler"),
    (186, "186.Cedar_Waxwing",             "Cedar Waxwing"),
    (188, "188.Pileated_Woodpecker",       "Pileated Woodpecker"),
    (191, "191.Red_headed_Woodpecker",     "Red-headed Woodpecker"),
    (200, "200.Common_Yellowthroat",       "Common Yellowthroat"),
]

# Maps: original CUB class_id → local 0-indexed label
CUB_ID_TO_LOCAL = {cid: i for i, (cid, _, _) in enumerate(CUB_SELECTED)}
LOCAL_TO_NAME = [name for _, _, name in CUB_SELECTED]
NUM_CLASSES = len(CUB_SELECTED)

# ImageNet normalization
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class BirdModel(BaseModel):
    """ResNet-50 fine-tuned on a 30-class CUB-200 subset."""

    def __init__(self, domain: str, config: dict):
        super().__init__(domain, config)
        self._model: nn.Module | None = None
        self._last_features: torch.Tensor | None = None
        self._eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    # ── loading ──────────────────────────────────────────────────────
    def load(self) -> None:
        self._model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self._model.fc = nn.Linear(2048, NUM_CLASSES)

        # Load fine-tuned weights if available
        weights_path = self._config.get("weights_path")
        if not weights_path:
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            weights_path = os.path.join(backend_dir, "data", "models", "bird_resnet50.pth")

        if os.path.isfile(weights_path):
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(state)
            logger.info("Loaded fine-tuned bird weights from %s", weights_path)
        else:
            logger.warning("No fine-tuned weights at %s — using ImageNet init", weights_path)

        self._model.to(self.device).eval()

        # Hook on layer4 for feature extraction + GradCAM
        def _hook(_module, _input, output):
            self._last_features = output

        self._model.layer4.register_forward_hook(_hook)
        self._loaded = True
        logger.info("BirdModel loaded (%s) — %d classes", self.device, NUM_CLASSES)

    # ── preprocessing ────────────────────────────────────────────────
    def preprocess(self, data: Any) -> torch.Tensor:
        img = self._to_pil(data).convert("RGB")
        tensor = self._eval_transform(img).unsqueeze(0)  # (1, 3, 224, 224)
        return tensor.to(self.device)

    # ── inference ────────────────────────────────────────────────────
    def predict_raw(self, data: Any) -> dict:
        tensor = self.preprocess(data)
        with torch.no_grad():
            logits = self._model(tensor)  # (1, NUM_CLASSES)
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
        top_idx = int(probs.argmax())
        label = LOCAL_TO_NAME[top_idx]
        confidence = float(probs[top_idx])
        raw_scores = {LOCAL_TO_NAME[i]: float(probs[i]) for i in range(NUM_CLASSES)}
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
        feat = self._last_features  # (1, 2048, 7, 7)
        if feat is None:
            raise RuntimeError("Feature hook did not fire")
        pooled = F.adaptive_avg_pool2d(feat, 1).squeeze()  # (2048,)
        return pooled.cpu().numpy()

    # ── attribution (GradCAM) ────────────────────────────────────────
    def get_attribution(self, data: Any, target_concept: str | None = None) -> dict:
        tensor = self.preprocess(data).requires_grad_(True)

        captured = {}

        def _fwd_hook(_module, _input, output):
            captured["feats"] = output
            output.retain_grad()

        hook = self._model.layer4.register_forward_hook(_fwd_hook)
        self._model.zero_grad()
        logits = self._model(tensor)  # (1, NUM_CLASSES)

        # Determine gradient target based on concept
        if target_concept and target_concept in LOCAL_TO_NAME:
            # Concept matches a class name — use that class directly
            target_idx = LOCAL_TO_NAME.index(target_concept)
        elif target_concept:
            # Concept is a visual attribute (e.g., "has_red", "curved_bill").
            # Create a concept-specific gradient target by weighting class
            # logits by how related each class is to the concept.
            # We use the concept keyword to match against class names and
            # generate distinct gradients per concept.
            concept_weights = self._concept_class_weights(target_concept)
            score = (logits[0] * concept_weights).sum()
            score.backward(retain_graph=False)
            hook.remove()
            return self._build_gradcam_result(data, captured, target_concept)
        else:
            target_idx = int(logits.argmax(dim=1).item())

        # Backward on the target logit (pre-softmax for sharper gradients)
        score = logits[0, target_idx]
        score.backward(retain_graph=False)
        hook.remove()

        concept_label = target_concept or LOCAL_TO_NAME[target_idx]
        return self._build_gradcam_result(data, captured, concept_label)

    def _concept_class_weights(self, concept: str) -> torch.Tensor:
        """Generate per-class weights based on concept–species associations."""
        # Map visual attribute concepts to species that typically exhibit them
        _CONCEPT_SPECIES = {
            "has_red": ["Cardinal", "Scarlet Tanager", "Red-headed Woodpecker", "American Redstart", "Ruby-throated Hummingbird", "Northern Flicker"],
            "has_blue": ["Indigo Bunting", "Blue Jay", "Painted Bunting"],
            "has_yellow": ["American Goldfinch", "Yellow Warbler", "Common Yellowthroat", "Baltimore Oriole", "Cedar Waxwing", "Northern Flicker"],
            "has_orange": ["Baltimore Oriole", "Northern Flicker", "American Redstart", "Painted Bunting"],
            "has_black": ["American Crow", "Common Raven", "Dark-eyed Junco", "Brown Pelican", "Pileated Woodpecker"],
            "has_white": ["Herring Gull", "Western Grebe", "Horned Puffin", "Mockingbird", "Brown Pelican"],
            "has_brown": ["Brown Thrasher", "House Sparrow", "Brown Pelican", "Mallard"],
            "has_grey": ["Mockingbird", "Dark-eyed Junco", "Herring Gull"],
            "has_green": ["Anna's Hummingbird", "Ruby-throated Hummingbird", "Painted Bunting", "Common Yellowthroat"],
            "has_iridescent": ["Anna's Hummingbird", "Ruby-throated Hummingbird", "American Crow", "Common Raven"],
            "curved_bill": ["Anna's Hummingbird", "Ruby-throated Hummingbird", "Brown Thrasher"],
            "hooked_bill": ["Frigatebird", "Brown Pelican", "Horned Puffin"],
            "dagger_bill": ["Western Grebe", "Belted Kingfisher", "Herring Gull"],
            "cone_bill": ["House Sparrow", "Indigo Bunting", "Painted Bunting", "Cardinal", "American Goldfinch"],
            "long_bill": ["Anna's Hummingbird", "Ruby-throated Hummingbird", "Brown Pelican"],
            "spotted_pattern": ["Brown Thrasher", "Northern Flicker"],
            "striped_pattern": ["House Sparrow", "Yellow Warbler"],
            "multi_colored_pattern": ["Painted Bunting", "Scarlet Tanager", "Baltimore Oriole", "Red-headed Woodpecker"],
            "has_crest": ["Cardinal", "Blue Jay", "Pileated Woodpecker", "Belted Kingfisher", "Cedar Waxwing"],
            "has_mask": ["Common Yellowthroat", "Cedar Waxwing"],
            "has_eyering": ["Common Yellowthroat", "Mockingbird"],
            "has_eye_stripe": ["House Sparrow", "Brown Thrasher", "Yellow Warbler"],
            "has_cap": ["Dark-eyed Junco", "American Goldfinch", "Cardinal"],
            "forked_tail": ["Barn Swallow", "Frigatebird"],
            "broad_wings": ["Frigatebird", "Brown Pelican"],
            "long_wings": ["Frigatebird", "Barn Swallow", "Herring Gull"],
            "large_bird": ["Brown Pelican", "Western Grebe", "Frigatebird", "Herring Gull", "Common Raven", "Mallard"],
            "small_bird": ["Anna's Hummingbird", "Ruby-throated Hummingbird", "American Goldfinch", "House Sparrow", "Dark-eyed Junco", "Common Yellowthroat", "Yellow Warbler"],
        }
        species = _CONCEPT_SPECIES.get(concept, [])
        weights = torch.zeros(NUM_CLASSES, device=self.device)
        if species:
            for s in species:
                if s in LOCAL_TO_NAME:
                    weights[LOCAL_TO_NAME.index(s)] = 1.0
            weights = weights / max(weights.sum(), 1.0)
        else:
            # Fallback: uniform weights (reduces to predicted-class behaviour)
            weights = torch.ones(NUM_CLASSES, device=self.device) / NUM_CLASSES
        return weights

    def _build_gradcam_result(self, data: Any, captured: dict, concept_label: str) -> dict:
        """Build the GradCAM result dict from captured features."""
        from scipy.ndimage import gaussian_filter

        feats = captured.get("feats")  # (1, 2048, 7, 7)

        if feats is not None and feats.grad is not None:
            # Grad-CAM++ style: use positive gradient weighting for
            # better spatial localization
            grads = feats.grad                                    # (1, 2048, 7, 7)
            alpha = F.relu(grads)                                 # keep positive gradients
            alpha = alpha / (alpha.sum(dim=(2, 3), keepdim=True) + 1e-7)
            weights = (alpha * F.relu(feats)).sum(dim=(2, 3), keepdim=True)
            cam = (weights * feats).sum(dim=1, keepdim=True)      # (1, 1, 7, 7)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=(224, 224), mode="bicubic", align_corners=False)
            cam = cam.squeeze().detach().cpu().numpy()
            cam = np.clip(cam, 0, None)
            # Gaussian smoothing to remove 7×7 grid artefacts
            cam = gaussian_filter(cam, sigma=8)
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
                # Power transform to increase contrast / spread
                cam = np.power(cam, 0.6)
            else:
                cam = np.zeros_like(cam)
        else:
            cam = np.zeros((224, 224), dtype=np.float32)

        heatmap_b64 = self._array_to_base64_png(cam)
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
            "concept": concept_label,
        }

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _to_pil(data: Any) -> Image.Image:
        if isinstance(data, Image.Image):
            return data
        if isinstance(data, str):
            if len(data) > 260 or ";" in data[:30]:
                if "," in data[:80]:
                    data = data.split(",", 1)[1]
                raw = base64.b64decode(data)
                return Image.open(io.BytesIO(raw))
            return Image.open(data)
        if isinstance(data, (bytes, bytearray)):
            return Image.open(io.BytesIO(data))
        raise ValueError(f"Cannot convert {type(data).__name__} to PIL Image")

    @staticmethod
    def _array_to_base64_png(arr: np.ndarray) -> str:
        arr_uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr_uint8, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _encode_original(self, data: Any) -> str:
        """Return the input image resized to 224x224 as base64 PNG."""
        img = self._to_pil(data).convert("RGB").resize((224, 224))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _encode_colored_heatmap(cam: np.ndarray) -> str:
        """Return a JET-colored heatmap as a standalone base64 PNG."""
        import matplotlib.cm as cm
        colored = (cm.jet(cam)[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(colored).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _make_overlay(self, data: Any, cam: np.ndarray, alpha: float = 0.4) -> str:
        import matplotlib.cm as cm

        img = self._to_pil(data).convert("RGB").resize((224, 224))
        img_arr = np.array(img, dtype=np.float32) / 255.0
        colored = cm.jet(cam)[:, :, :3]
        blended = img_arr * (1 - alpha) + colored * alpha
        blended = (blended * 255).clip(0, 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
