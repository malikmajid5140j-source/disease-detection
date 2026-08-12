"""
SpecialistRouter v2 — Smart Auto Detection
────────────────────────────────────────────
Kisan sirf photo upload kare — AI khud detect kare:
  1. Wheat model chalao
  2. Chilli model chalao  
  3. General CNN chalao
  4. Sabse confident WINS
  5. Sab uncertain → unclear image
"""

from __future__ import annotations
import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

MODELS_DIR = Path(__file__).parent / "specialist_models"
MODELS_DIR.mkdir(exist_ok=True)

WHEAT_PTH  = MODELS_DIR / "wheat_efficientnetv2s_best.pth"
CHILLI_PTH = MODELS_DIR / "chilli_efficientnetv2s_best.pth"

HF_BASE    = os.getenv("HF_MODEL_URL", "https://huggingface.co/majid-agriscan/models/resolve/main")
WHEAT_URL  = f"{HF_BASE}/wheat_efficientnetv2s_best.pth"
CHILLI_URL = f"{HF_BASE}/chilli_efficientnetv2s_best.pth"

EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

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

MIN_CONFIDENCE   = 0.55   # 55% se kam → unclear
SPECIALIST_BOOST = 1.15   # specialist ko 15% boost


def _download_if_needed(url: str, dest: Path) -> bool:
    import urllib.request
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        return True
    print(f"[specialist] Downloading {dest.name}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[specialist] Done: {dest.stat().st_size/1e6:.1f}MB")
        return True
    except Exception as e:
        print(f"[specialist] Failed: {e}")
        return False


def _load_efficientnet(pth_path: Path, num_classes: int, device: str):
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    ckpt = torch.load(pth_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    print(f"[specialist] Loaded {pth_path.name} — {ckpt.get('val_acc',0):.1f}%")
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
                self._wheat_model = _load_efficientnet(WHEAT_PTH, len(WHEAT_CLASSES), self.device)
                self._wheat_ok    = True
            except Exception as e:
                print(f"[specialist] Wheat error: {e}")

        if _download_if_needed(CHILLI_URL, CHILLI_PTH):
            try:
                self._chilli_model = _load_efficientnet(CHILLI_PTH, len(CHILLI_CLASSES), self.device)
                self._chilli_ok    = True
            except Exception as e:
                print(f"[specialist] Chilli error: {e}")

    def status(self) -> dict:
        return {
            "wheat":  "ready" if self._wheat_ok  else "not loaded",
            "chilli": "ready" if self._chilli_ok else "not loaded",
        }

    def wheat_classes(self):  return WHEAT_CLASSES
    def chilli_classes(self): return CHILLI_CLASSES

    def _infer(self, model, classes: list, image: Image.Image) -> dict:
        img_t = EVAL_TF(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = F.softmax(model(img_t)[0], dim=0).cpu().numpy()
        idx = int(probs.argmax())
        return {"label": classes[idx], "confidence": float(probs[idx])}

    def smart_predict(self, image: Image.Image, general_model=None) -> dict:
        """
        Fully automatic — kisan ko kuch select nahi karna.
        Teen models chalte hain, sabse confident WINS.
        """
        candidates = []

        # 1. Wheat specialist
        if self._wheat_ok:
            r = self._infer(self._wheat_model, WHEAT_CLASSES, image)
            candidates.append({
                "label":      r["label"],
                "confidence": r["confidence"],
                "score":      r["confidence"] * SPECIALIST_BOOST,
                "crop_type":  "Wheat",
                "model_used": "wheat_specialist",
            })

        # 2. Chilli specialist
        if self._chilli_ok:
            r = self._infer(self._chilli_model, CHILLI_CLASSES, image)
            candidates.append({
                "label":      r["label"],
                "confidence": r["confidence"],
                "score":      r["confidence"] * SPECIALIST_BOOST,
                "crop_type":  "Chilli",
                "model_used": "chilli_specialist",
            })

        # 3. General PlantVillage CNN (no boost)
        if general_model is not None:
            try:
                preds = general_model.predict(image, top_k=1)
                if preds:
                    top   = preds[0]
                    label = top["label"]
                    conf  = top["confidence"]
                    crop  = label.split("___")[0].replace("_"," ") if "___" in label else "Unknown"
                    candidates.append({
                        "label":      label,
                        "confidence": conf,
                        "score":      conf,
                        "crop_type":  crop,
                        "model_used": "general_plantvillage",
                    })
            except Exception as e:
                print(f"[specialist] General model error: {e}")

        if not candidates:
            return self._unclear()

        # Pick highest score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        winner = candidates[0]

        # Agar winner bhi 55% se kam confident
        if winner["confidence"] < MIN_CONFIDENCE:
            return self._unclear(candidates)

        return {
            "label":          winner["label"],
            "confidence":     winner["confidence"],
            "crop_type":      winner["crop_type"],
            "model_used":     winner["model_used"],
            "all_candidates": candidates,
        }

    @staticmethod
    def _unclear(candidates=None) -> dict:
        return {
            "label":          "Unclear_Image",
            "confidence":     0.0,
            "crop_type":      "Unknown",
            "model_used":     "none",
            "all_candidates": candidates or [],
        }

    def predict_wheat(self, image):
        r = self._infer(self._wheat_model, WHEAT_CLASSES, image)
        return {**r, "crop_type": "Wheat", "model_used": "wheat_specialist"}

    def predict_chilli(self, image):
        r = self._infer(self._chilli_model, CHILLI_CLASSES, image)
        return {**r, "crop_type": "Chilli", "model_used": "chilli_specialist"}
