"""Vision domain model — CLIP zero-shot on CUB-200 bird species."""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.base import BaseModel

logger = logging.getLogger(__name__)

# CUB-200 — 30 selected bird species (same list as BirdModel)
CUB_CLASSES = [
    "Indigo Bunting", "Painted Bunting", "Cardinal", "American Crow",
    "Northern Flicker", "Frigatebird", "American Goldfinch", "Western Grebe",
    "Herring Gull", "Anna's Hummingbird", "Ruby-throated Hummingbird",
    "Blue Jay", "Dark-eyed Junco", "Belted Kingfisher", "Mallard",
    "Mockingbird", "Baltimore Oriole", "Brown Pelican", "Horned Puffin",
    "Common Raven", "American Redstart", "House Sparrow", "Barn Swallow",
    "Scarlet Tanager", "Brown Thrasher", "Yellow Warbler", "Cedar Waxwing",
    "Pileated Woodpecker", "Red-headed Woodpecker", "Common Yellowthroat",
]


class VisionModel(BaseModel):
    """Wraps CLIP ViT-B/32 for zero-shot bird species classification."""

    def __init__(self, domain: str, config: dict):
        super().__init__(domain, config)
        self._model = None
        self._processor = None
        self._text_embeds: torch.Tensor | None = None  # cached class text embeddings

    @staticmethod
    def _to_tensor(out):
        """Extract tensor from model output (transformers >=5 returns objects)."""
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0]

    # ── loading ──────────────────────────────────────────────────────
    def load(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        model_id = self._config["model_id"]
        self._processor = CLIPProcessor.from_pretrained(model_id)
        # Force safetensors to avoid torch.load CVE restriction in transformers >=5
        self._model = CLIPModel.from_pretrained(model_id, use_safetensors=True)
        self._model.to(self.device).eval()

        # Pre-compute text embeddings for all 30 bird species
        prompts = [f"a photo of a {c}" for c in CUB_CLASSES]
        text_inputs = self._processor(text=prompts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()
                       if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            self._text_embeds = self._to_tensor(self._model.get_text_features(**text_inputs))
            self._text_embeds = F.normalize(self._text_embeds, dim=-1)

        self._loaded = True
        logger.info("VisionModel loaded (%s) — %s", self.device, model_id)

    # ── preprocessing ────────────────────────────────────────────────
    def preprocess(self, data: Any) -> dict:
        img = self._to_pil(data).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    # ── inference ────────────────────────────────────────────────────
    def predict_raw(self, data: Any) -> dict:
        inputs = self.preprocess(data)
        with torch.no_grad():
            img_embeds = self._to_tensor(self._model.get_image_features(**inputs))
            img_embeds = F.normalize(img_embeds, dim=-1)

        # Cosine similarity → softmax over 30 classes
        sims = (img_embeds @ self._text_embeds.T).squeeze(0)  # (30,)
        probs = F.softmax(sims * 100.0, dim=0).cpu().numpy()  # CLIP logit_scale ~ 100

        top_idx = int(probs.argmax())
        label = CUB_CLASSES[top_idx]
        confidence = float(probs[top_idx])

        raw_scores = {CUB_CLASSES[i]: float(probs[i]) for i in range(len(CUB_CLASSES))}

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "raw_scores": raw_scores,
        }

    # ── feature extraction ───────────────────────────────────────────
    def extract_features(self, data: Any) -> np.ndarray:
        inputs = self.preprocess(data)
        with torch.no_grad():
            img_embeds = self._to_tensor(self._model.get_image_features(**inputs))
        return img_embeds.squeeze(0).cpu().numpy()  # (512,)

    # ── attribution (CLIP spatial similarity) ────────────────────────
    def get_attribution(self, data: Any, target_concept: str | None = None) -> dict:
        """Compute patch-level CLIP similarity with a target concept text."""
        img = self._to_pil(data).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        # Get patch embeddings from the vision encoder (before final pooling)
        with torch.no_grad():
            vision_outputs = self._model.vision_model(pixel_values=pixel_values)
            # last_hidden_state: (1, num_patches+1, hidden_dim)
            # First token is CLS, rest are patches
            patch_embeds = vision_outputs.last_hidden_state[:, 1:, :]  # (1, N, D)

            # Project to CLIP joint space
            patch_embeds = self._model.visual_projection(patch_embeds)
            patch_embeds = F.normalize(patch_embeds, dim=-1)

        # Get concept text embedding
        concept_text = target_concept or "bird"
        concept_prompt = f"a photo showing {concept_text}"
        text_inputs = self._processor(text=[concept_prompt], return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()
                       if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            text_embed = self._to_tensor(self._model.get_text_features(**text_inputs))
            text_embed = F.normalize(text_embed, dim=-1)

        # Patch-concept similarity → spatial heatmap
        sims = (patch_embeds @ text_embed.T).squeeze()  # (N,)
        sims = sims.cpu().numpy()

        # Reshape to spatial grid (ViT-B/32: 224/32 = 7 → 7×7 = 49 patches)
        grid_size = int(np.sqrt(len(sims)))
        if grid_size * grid_size != len(sims):
            grid_size = int(np.ceil(np.sqrt(len(sims))))
            sims = np.pad(sims, (0, grid_size * grid_size - len(sims)))
        heatmap = sims.reshape(grid_size, grid_size)

        # Normalize to [0, 1]
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max - h_min > 1e-8:
            heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap = np.zeros_like(heatmap)

        # Upscale to 224×224 and encode
        from PIL import Image as PILImage
        heatmap_uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)
        heatmap_img = PILImage.fromarray(heatmap_uint8, mode="L").resize(
            (224, 224), PILImage.BILINEAR,
        )
        buf = io.BytesIO()
        heatmap_img.save(buf, format="PNG")
        heatmap_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        # Blended overlay on original image
        overlay_b64 = self._make_overlay(data, heatmap)

        # Original image and colored heatmap for layered display
        original_b64 = self._encode_original(data)
        heatmap_colored_b64 = self._encode_colored_heatmap(heatmap)

        return {
            "method": "clip_spatial",
            "type": "heatmap",
            "data": heatmap_b64,
            "overlay": overlay_b64,
            "original_image": original_b64,
            "heatmap_colored": heatmap_colored_b64,
            "shape": [224, 224],
            "concept": concept_text,
        }

    # ── CLIP concept scoring (used by CLIPConceptScorer in probe.py) ─
    def score_concepts_clip(self, data: Any, concept_descriptions: dict[str, str]) -> dict[str, float]:
        """Zero-shot concept scoring via CLIP text-image similarity.

        Parameters
        ----------
        data : image input
        concept_descriptions : concept_name → natural language description

        Returns
        -------
        dict of concept_name → activation score in [0, 1]
        """
        inputs = self.preprocess(data)
        with torch.no_grad():
            img_embeds = self._to_tensor(self._model.get_image_features(**inputs))
            img_embeds = F.normalize(img_embeds, dim=-1)

        names = list(concept_descriptions.keys())
        prompts = [concept_descriptions[n] for n in names]
        text_inputs = self._processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()
                       if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            text_embeds = self._to_tensor(self._model.get_text_features(**text_inputs))
            text_embeds = F.normalize(text_embeds, dim=-1)

        # Cosine similarities → sigmoid to get [0, 1] activations
        sims = (img_embeds @ text_embeds.T).squeeze(0)  # (num_concepts,)
        # Scale and sigmoid: raw cosine is in [-1, 1]; shift to get meaningful [0, 1]
        activations = torch.sigmoid(sims * 5.0).cpu().numpy()

        return {name: round(float(act), 4) for name, act in zip(names, activations)}

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

    def _make_overlay(self, data: Any, cam: np.ndarray, alpha: float = 0.4) -> str:
        """Blend a [0,1] heatmap onto the original image as a JET colormap overlay."""
        from PIL import Image as PILImage
        import matplotlib.cm as cm

        img = self._to_pil(data).convert("RGB").resize((224, 224))
        img_arr = np.array(img, dtype=np.float32) / 255.0

        # Upscale heatmap if needed
        if cam.shape[0] != 224 or cam.shape[1] != 224:
            hm_img = PILImage.fromarray((cam * 255).clip(0, 255).astype(np.uint8), mode="L")
            hm_img = hm_img.resize((224, 224), PILImage.BILINEAR)
            cam = np.array(hm_img, dtype=np.float32) / 255.0

        colored = cm.jet(cam)[:, :, :3]
        blended = img_arr * (1 - alpha) + colored * alpha
        blended = (blended * 255).clip(0, 255).astype(np.uint8)

        buf = io.BytesIO()
        PILImage.fromarray(blended).save(buf, format="PNG")
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
        if cam.shape[0] != 224 or cam.shape[1] != 224:
            hm_img = PILImage.fromarray((cam * 255).clip(0, 255).astype(np.uint8), mode="L")
            hm_img = hm_img.resize((224, 224), PILImage.BILINEAR)
            cam = np.array(hm_img, dtype=np.float32) / 255.0
        colored = (cm.jet(cam)[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
        buf = io.BytesIO()
        PILImage.fromarray(colored).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
