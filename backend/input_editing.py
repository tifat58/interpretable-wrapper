"""Input-editing counterfactuals.

Edit the raw input (mask tokens in text, mask regions in images) and
re-run the model to produce a true counterfactual prediction — as
opposed to the surrogate-based counterfactual from Phase 1A.

Supported edit operations
-------------------------
Text:
  - ``mask_tokens``  : replace specified token indices with [MASK] / [UNK]
  - ``replace_tokens``: replace specified spans with user-supplied text
  - ``remove_tokens`` : delete specified token indices

Image:
  - ``mask_region``  : zero-out a bounding box [x1, y1, x2, y2] (normalised 0-1)
  - ``blur_region``  : Gaussian-blur a bounding box
  - ``grayscale_region``: desaturate a bounding box
"""

from __future__ import annotations

import base64
import io
import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Text editing
# ═══════════════════════════════════════════════════════════════════════

def edit_text(original: str, edit_spec: dict) -> str:
    """Apply edits to a text string and return the modified version.

    Parameters
    ----------
    original : the original text string
    edit_spec : dict with keys:
        action : "mask_tokens" | "replace_tokens" | "remove_tokens"
        indices : list of word indices to edit (0-based)
        replacement : replacement string (for replace_tokens)

    Returns
    -------
    Modified text string
    """
    action = edit_spec.get("action", "mask_tokens")
    indices = set(edit_spec.get("indices", []))
    replacement = edit_spec.get("replacement", "[MASK]")

    words = original.split()

    if action == "mask_tokens":
        edited = [
            "[MASK]" if i in indices else w
            for i, w in enumerate(words)
        ]
    elif action == "replace_tokens":
        edited = [
            replacement if i in indices else w
            for i, w in enumerate(words)
        ]
    elif action == "remove_tokens":
        edited = [
            w for i, w in enumerate(words)
            if i not in indices
        ]
    else:
        edited = words

    return " ".join(edited)


# ═══════════════════════════════════════════════════════════════════════
# Image editing
# ═══════════════════════════════════════════════════════════════════════

def edit_image(original: Any, edit_spec: dict) -> str:
    """Apply edits to an image and return modified image as base64 PNG.

    Parameters
    ----------
    original : base64 string, file path, or PIL Image
    edit_spec : dict with keys:
        action : "mask_region" | "blur_region" | "grayscale_region"
        region : [x1, y1, x2, y2] normalised coordinates (0-1)

    Returns
    -------
    Base64-encoded PNG string of the edited image
    """
    from PIL import Image, ImageFilter

    img = _to_pil(original).copy()
    action = edit_spec.get("action", "mask_region")
    region = edit_spec.get("region", [0, 0, 1, 1])

    # Support both {x,y,w,h} dict and [x1,y1,x2,y2] list formats
    if isinstance(region, dict):
        rx, ry = region.get("x", 0), region.get("y", 0)
        rw, rh = region.get("w", 1), region.get("h", 1)
        region = [rx, ry, rx + rw, ry + rh]

    w, h = img.size
    x1 = int(region[0] * w)
    y1 = int(region[1] * h)
    x2 = int(region[2] * w)
    y2 = int(region[3] * h)

    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if action == "mask_region":
        # Zero-out (black) the region
        pixels = img.load()
        for px in range(x1, x2):
            for py in range(y1, y2):
                if img.mode == "L":
                    pixels[px, py] = 0
                elif img.mode == "RGB":
                    pixels[px, py] = (0, 0, 0)
                elif img.mode == "RGBA":
                    pixels[px, py] = (0, 0, 0, 255)

    elif action == "blur_region":
        crop = img.crop((x1, y1, x2, y2))
        blur_radius = max(5, min(x2 - x1, y2 - y1) // 4)
        blurred = crop.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        img.paste(blurred, (x1, y1))

    elif action == "grayscale_region":
        crop = img.crop((x1, y1, x2, y2)).convert("L")
        if img.mode == "RGB":
            crop = crop.convert("RGB")
        elif img.mode == "RGBA":
            crop = crop.convert("RGBA")
        img.paste(crop, (x1, y1))

    return _pil_to_base64(img)


# ═══════════════════════════════════════════════════════════════════════
# Combined entry point
# ═══════════════════════════════════════════════════════════════════════

def apply_edit(domain: str, original_data: Any, edit_spec: dict,
               model: Any) -> dict:
    """Edit input, re-run the model, and return counterfactual result.

    Parameters
    ----------
    domain : domain name
    original_data : original raw input
    edit_spec : edit specification (see edit_text / edit_image)
    model : the PostHocCBM instance to re-run prediction

    Returns
    -------
    dict with: edited_data, original_prediction, edited_prediction,
               prediction_delta
    """
    from config import DOMAIN_CONFIG
    cfg = DOMAIN_CONFIG.get(domain, {})
    input_type = cfg.get("input_type", "text")

    # Get original prediction
    original_result = model.predict(original_data)

    # Apply edit
    if input_type == "text":
        edited_data = edit_text(str(original_data), edit_spec)
    else:
        edited_data = edit_image(original_data, edit_spec)

    # Re-run prediction on edited input
    edited_result = model.predict(edited_data)

    # Compute delta
    concept_deltas = {}
    for c in original_result.get("concepts", {}):
        orig_v = original_result["concepts"].get(c, 0)
        edit_v = edited_result["concepts"].get(c, 0)
        concept_deltas[c] = round(edit_v - orig_v, 4)

    return {
        "edited_data": edited_data if input_type == "text" else f"data:image/png;base64,{edited_data}",
        "original_prediction": {
            "label": original_result["label"],
            "confidence": original_result["confidence"],
        },
        "edited_prediction": {
            "label": edited_result["label"],
            "confidence": edited_result["confidence"],
        },
        "prediction_delta": round(
            edited_result["confidence"] - original_result["confidence"], 4
        ),
        "concept_deltas": concept_deltas,
        "edit_spec": edit_spec,
    }


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _to_pil(data: Any):
    from PIL import Image
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


def _pil_to_base64(img) -> str:
    buf = io.BytesIO()
    fmt = "PNG" if img.mode in ("RGBA", "L") else "PNG"
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")
