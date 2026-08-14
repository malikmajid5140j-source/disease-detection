"""
SpecialistRouter v3 — Smart OOD Detection
──────────────────────────────────────────
Data-driven thresholds from OOD analysis:

CORRECT crop:  Combined Score ~0.58
WRONG crop:    Combined Score ~0.30
Threshold:     0.44 (midpoint with safety margin)

Combined Score = Confidence × Top2_Gap × (1 - Entropy)
High = model sure, Low = model confused/wrong crop
"""

from __future__ import annotations
import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, models

# ── Paths ─────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "specialist_models"
MODELS_DIR.mkdir(exist_ok=True)

WHEAT_PTH  = MODELS_DIR / "wheat_efficientnetv2s_best.pth"
CHILLI_PTH = MODELS_DIR / "chilli_efficientnetv2s_best.pth"

HF_BASE    = os.getenv("HF_MODEL_URL",
             "https://huggingface.co/majid-agriscan/models/resolve/main")
WHEAT_URL  = f"{HF_BASE}/wheat_efficientnetv2s_best.pth"
CHILLI_URL = f"{HF_BASE}/chilli_efficientnetv2s_best.pth"

# ── Transforms ────────────────────────────────────────────────
EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Class Lists ───────────────────────────────────────────────
WHEAT_CLASSES = [
    'Aphid', 'Black_Rust', 'Brown_Rust', 'Common_Root_Rot',
    'Fusarium_Head_Blight', 'Healthy', 'Leaf_Blight', 'Mite',
    'Powdery_Mildew', 'Septoria', 'Smut', 'Stem_Fly',
    'Tan_Spot', 'Wheat_Blast', 'Yellow_Rust'
]

CHILLI_CLASSES = [
    'Anthracnose', 'Damping_Off', 'Healthy', 'Leaf_Curl_Virus',
    'Leaf_Spot', 'Veinal_Mottle_Virus', 'Whitefly', 'Yellowish'
]

# ── OOD Thresholds (Data-Driven from Analysis) ────────────────
# Combined Score = Confidence × Top2_Gap × (1 - Entropy)
# Correct crop avg: ~0.577
# Wrong crop avg:   ~0.287
# Threshold at midpoint + safety:
COMBINED_SCORE_THRESHOLD = 0.55   # below this → wrong crop / unclear

# Individual safety nets
MIN_CONFIDENCE = 0.82   # below 82% → definitely wrong
MAX_ENTROPY    = 0.28   # above 0.28 → model confused
MIN_TOP2_GAP   = 0.65   # below 0.65 → two classes too close


def _download_if_needed(url: str, dest: Path) -> bool:
    import urllib.request
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        return True
    print(f"[specialist] Downloading {dest.name}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[specialist] ✅ {dest.name} ({dest.stat().st_size/1e6:.1f}MB)")
        return True
    except Exception as e:
        print(f"[specialist] ❌ {dest.name}: {e}")
        return False


def _load_efficientnet(pth_path: Path, num_classes: int, device: str):
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes
    )
    ckpt = torch.load(pth_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    acc = ckpt.get('val_acc', 0)
    print(f"[specialist] ✅ {pth_path.name} — val_acc={acc:.1f}%")
    return model


class SpecialistRouter:

    def __init__(self):
        self.device        = "cuda" if torch.cuda.is_available() else "cpu"
        self._wheat_model  = None
        self._chilli_model = None
        self._wheat_ok     = False
        self._chilli_ok    = False

    def load(self):
        if _download_if_needed(WHEAT_URL, WHEAT_PTH):
            try:
                self._wheat_model = _load_efficientnet(
                    WHEAT_PTH, len(WHEAT_CLASSES), self.device)
                self._wheat_ok = True
            except Exception as e:
                print(f"[specialist] Wheat load failed: {e}")

        if _download_if_needed(CHILLI_URL, CHILLI_PTH):
            try:
                self._chilli_model = _load_efficientnet(
                    CHILLI_PTH, len(CHILLI_CLASSES), self.device)
                self._chilli_ok = True
            except Exception as e:
                print(f"[specialist] Chilli load failed: {e}")

    def status(self) -> dict:
        return {
            "wheat":  "ready" if self._wheat_ok  else "not loaded",
            "chilli": "ready" if self._chilli_ok else "not loaded",
        }

    def wheat_classes(self):  return WHEAT_CLASSES
    def chilli_classes(self): return CHILLI_CLASSES

    def _infer(self, model, classes: list, image: Image.Image) -> dict:
        """
        Full inference with OOD metrics:
        - confidence: max softmax probability
        - entropy: normalized entropy (0=sure, 1=confused)
        - top2_gap: gap between top 2 classes
        - combined_score: confidence × top2_gap × (1 - entropy)
        - is_valid_crop: True if combined_score > threshold
        """
        img_t = EVAL_TF(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = F.softmax(model(img_t)[0], dim=0).cpu().numpy()

        idx          = int(probs.argmax())
        confidence   = float(probs[idx])
        sorted_p     = np.sort(probs)[::-1]
        top2_gap     = float(sorted_p[0] - sorted_p[1])
        entropy      = float(-np.sum(probs * np.log(probs + 1e-8)))
        norm_entropy = entropy / np.log(len(probs))

        # Combined OOD score
        combined = confidence * top2_gap * (1 - norm_entropy)

        # Is this actually this crop?
        is_valid = (
            combined   >= COMBINED_SCORE_THRESHOLD and
            confidence >= MIN_CONFIDENCE           and
            norm_entropy <= MAX_ENTROPY            and
            top2_gap   >= MIN_TOP2_GAP
        )

        return {
            "label":          classes[idx],
            "confidence":     confidence,
            "entropy":        norm_entropy,
            "top2_gap":       top2_gap,
            "combined_score": combined,
            "is_valid_crop":  is_valid,
        }

    def smart_predict(self, image: Image.Image,
                      general_model=None) -> dict:
        """
        Fully automatic — kisan sirf photo upload kare.

        Flow:
        1. Wheat + Chilli specialist models chalao
        2. Jo valid crop hai (OOD check pass) uska combined_score dekho
        3. Highest combined_score WINS
        4. Agar koi bhi valid nahi → General PlantVillage CNN
        5. General bhi low confidence → "Unclear Image"

        Returns complete prediction dict.
        """
        specialist_results = []

        # ── Wheat specialist ──────────────────────────────────
        if self._wheat_ok:
            r = self._infer(self._wheat_model, WHEAT_CLASSES, image)
            if r["is_valid_crop"]:
                specialist_results.append({
                    "label":          r["label"],
                    "confidence":     r["confidence"],
                    "combined_score": r["combined_score"],
                    "crop_type":      "Wheat",
                    "model_used":     "wheat_specialist",
                    "ood_metrics":    {
                        "entropy":    r["entropy"],
                        "top2_gap":   r["top2_gap"],
                        "combined":   r["combined_score"],
                    }
                })
                print(f"[specialist] Wheat valid ✅ "
                      f"conf={r['confidence']:.2f} "
                      f"combined={r['combined_score']:.3f}")
            else:
                print(f"[specialist] Wheat OOD ❌ "
                      f"conf={r['confidence']:.2f} "
                      f"combined={r['combined_score']:.3f} "
                      f"(below {COMBINED_SCORE_THRESHOLD})")

        # ── Chilli specialist ─────────────────────────────────
        if self._chilli_ok:
            r = self._infer(self._chilli_model, CHILLI_CLASSES, image)
            if r["is_valid_crop"]:
                specialist_results.append({
                    "label":          r["label"],
                    "confidence":     r["confidence"],
                    "combined_score": r["combined_score"],
                    "crop_type":      "Chilli",
                    "model_used":     "chilli_specialist",
                    "ood_metrics":    {
                        "entropy":    r["entropy"],
                        "top2_gap":   r["top2_gap"],
                        "combined":   r["combined_score"],
                    }
                })
                print(f"[specialist] Chilli valid ✅ "
                      f"conf={r['confidence']:.2f} "
                      f"combined={r['combined_score']:.3f}")
            else:
                print(f"[specialist] Chilli OOD ❌ "
                      f"conf={r['confidence']:.2f} "
                      f"combined={r['combined_score']:.3f} "
                      f"(below {COMBINED_SCORE_THRESHOLD})")

        # ── Pick best specialist ──────────────────────────────
        if specialist_results:
            best = max(specialist_results,
                       key=lambda x: x["combined_score"])
            return {
                "label":      best["label"],
                "confidence": best["confidence"],
                "crop_type":  best["crop_type"],
                "model_used": best["model_used"],
                "ood_metrics": best.get("ood_metrics", {}),
            }

        # ── Fallback: General PlantVillage CNN ────────────────
        if general_model is not None:
            try:
                preds = general_model.predict(image, top_k=1)
                if preds:
                    top   = preds[0]
                    label = top["label"]
                    conf  = top["confidence"]
                    crop  = label.split("___")[0].replace("_", " ") \
                            if "___" in label else "Unknown"

                    # General model bhi low confidence → unclear
                    if conf < 0.75:
                        print(f"[specialist] General also uncertain "
                              f"({conf:.2f}) → Unclear")
                        return self._unclear()

                    print(f"[specialist] General model: "
                          f"{label} ({conf:.2f})")
                    return {
                        "label":      label,
                        "confidence": conf,
                        "crop_type":  crop,
                        "model_used": "general_plantvillage",
                        "ood_metrics": {},
                    }
            except Exception as e:
                print(f"[specialist] General model error: {e}")

        return self._unclear()

    @staticmethod
    def _unclear() -> dict:
        return {
            "label":      "Unclear_Image",
            "confidence": 0.0,
            "crop_type":  "Unknown",
            "model_used": "none",
            "ood_metrics": {},
        }

    # ── Direct methods ────────────────────────────────────────
    def predict_wheat(self, image: Image.Image) -> dict:
        r = self._infer(self._wheat_model, WHEAT_CLASSES, image)
        return {"label": r["label"], "confidence": r["confidence"],
                "crop_type": "Wheat", "model_used": "wheat_specialist"}

    def predict_chilli(self, image: Image.Image) -> dict:
        r = self._infer(self._chilli_model, CHILLI_CLASSES, image)
        return {"label": r["label"], "confidence": r["confidence"],
                "crop_type": "Chilli", "model_used": "chilli_specialist"}
