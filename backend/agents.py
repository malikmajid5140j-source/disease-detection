"""
AgriScan AI v3 — Multi-Agent System
─────────────────────────────────────
5 Agents:
  1. GatekeeperAgent   → CLIP: is this a plant leaf?
  2. CropRouterAgent   → CLIP: which crop?
  3. Specialists       → EfficientNetV2-S disease detection
  4. ConsensusValidator→ Cross-check all predictions
  5. ResponseBuilder   → Build final response (in main.py)
"""

from __future__ import annotations
import os, time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

MODELS_DIR = Path(__file__).parent / "specialist_models"
MODELS_DIR.mkdir(exist_ok=True)

HF_BASE = os.getenv(
    "HF_MODEL_URL",
    "https://huggingface.co/Mlaikmajid1063/agriscan-models/resolve/main"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

WHEAT_CLASSES = [
    "Aphid", "Black_Rust", "Brown_Rust", "Common_Root_Rot",
    "Fusarium_Head_Blight", "Healthy", "Leaf_Blight", "Mite",
    "Powdery_Mildew", "Septoria", "Smut", "Stem_Fly",
    "Tan_Spot", "Wheat_Blast", "Yellow_Rust"
]

CHILLI_CLASSES = [
    "Anthracnose", "Damping_Off", "Healthy", "Leaf_Curl_Virus",
    "Leaf_Spot", "Veinal_Mottle_Virus", "Whitefly", "Yellowish"
]


# ═══════════════════════════════════════════════════════════════
# AGENT 1 — GATEKEEPER
# ═══════════════════════════════════════════════════════════════
class GatekeeperAgent:
    PLANT_PROMPTS = [
        "a photo of a plant leaf",
        "a photo of a crop leaf with disease",
        "a close up photo of a leaf",
        "a diseased plant leaf",
    ]
    NOT_PLANT_PROMPTS = [
        "a photo of a person or human face",
        "a photo of food on a plate",
        "a photo of an object",
        "a photo of a building or street",
        "a photo of an animal",
        "a random photo or screenshot",
        "a raw potato or vegetable",
    ]

    def __init__(self):
        self.model = None
        self.preprocess = None
        self.ready = False

    def load(self):
        try:
            import clip
            self.model, self.preprocess = clip.load("ViT-B/32", device=DEVICE)
            self.ready = True
            print("[gatekeeper] CLIP ready")
        except Exception as e:
            print(f"[gatekeeper] CLIP unavailable: {e}")
            self.ready = False

    def check(self, image: Image.Image) -> dict:
        if not self.ready:
            return {"is_plant": True, "plant_score": 0.5,
                    "not_plant_score": 0.5, "reason": "unavailable"}

        import clip
        img_tensor  = self.preprocess(image).unsqueeze(0).to(DEVICE)
        all_prompts = self.PLANT_PROMPTS + self.NOT_PLANT_PROMPTS
        text_tensor = clip.tokenize(all_prompts).to(DEVICE)

        with torch.no_grad():
            img_feat  = self.model.encode_image(img_tensor)
            txt_feat  = self.model.encode_text(text_tensor)
            img_feat  = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat  = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            probs     = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)[0].cpu().numpy()

        n_plant     = len(self.PLANT_PROMPTS)
        plant_score = float(probs[:n_plant].sum())
        not_score   = float(probs[n_plant:].sum())
        is_plant    = plant_score > not_score and plant_score >= 0.40

        top_not_idx = int(probs[n_plant:].argmax())
        top_not     = self.NOT_PLANT_PROMPTS[top_not_idx]

        return {
            "is_plant":        is_plant,
            "plant_score":     plant_score,
            "not_plant_score": not_score,
            "reason": f"plant={plant_score:.2f} not={not_score:.2f} top_reject={top_not}",
        }


# ═══════════════════════════════════════════════════════════════
# AGENT 2 — CROP ROUTER
# ═══════════════════════════════════════════════════════════════
class CropRouterAgent:
    CROP_PROMPTS = {
        "wheat":  "a photo of wheat leaves or wheat plant",
        "chilli": "a photo of chilli pepper leaf or mirch plant",
        "tomato": "a photo of tomato plant leaf",
        "potato": "a photo of potato plant leaf",
        "corn":   "a photo of corn or maize leaf",
        "other":  "a photo of some other plant leaf",
    }

    def __init__(self, gatekeeper: GatekeeperAgent):
        self.gk = gatekeeper

    def route(self, image: Image.Image) -> dict:
        if not self.gk.ready:
            return {"crop": "unknown", "confidence": 0.0, "scores": {}}

        import clip
        img_tensor  = self.gk.preprocess(image).unsqueeze(0).to(DEVICE)
        keys        = list(self.CROP_PROMPTS.keys())
        prompts     = list(self.CROP_PROMPTS.values())
        text_tensor = clip.tokenize(prompts).to(DEVICE)

        with torch.no_grad():
            img_feat  = self.gk.model.encode_image(img_tensor)
            txt_feat  = self.gk.model.encode_text(text_tensor)
            img_feat  = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat  = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            probs     = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)[0].cpu().numpy()

        scores   = {k: float(v) for k, v in zip(keys, probs)}
        top_idx  = int(probs.argmax())
        top_crop = keys[top_idx]
        top_conf = float(probs[top_idx])

        route_to = top_crop if (top_crop in ("wheat", "chilli") and top_conf > 0.22) else "other"

        return {"crop": route_to, "top": top_crop,
                "confidence": top_conf, "scores": scores}


# ═══════════════════════════════════════════════════════════════
# AGENT 3 — SPECIALISTS
# ═══════════════════════════════════════════════════════════════
def _download(url: str, dest: Path) -> bool:
    import urllib.request
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        return True
    try:
        print(f"[download] {dest.name}...")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"[download] failed: {e}")
        return False


def _load_model(path: Path, n_cls: int):
    m = models.efficientnet_v2_s(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_cls)
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ck["model_state_dict"])
    m.eval().to(DEVICE)
    return m, ck.get("val_acc", 0)


def _run_inference(model, classes: list, image: Image.Image) -> dict:
    img_t = EVAL_TF(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(img_t)[0], dim=0).cpu().numpy()
    idx      = int(probs.argmax())
    sorted_p = np.sort(probs)[::-1]
    entropy  = float(-np.sum(probs * np.log(probs + 1e-8)) / np.log(len(probs)))
    return {
        "label":      classes[idx],
        "confidence": float(probs[idx]),
        "top2_gap":   float(sorted_p[0] - sorted_p[1]),
        "entropy":    entropy,
    }


class WheatAgent:
    def __init__(self):
        self.model = None
        self.ready = False

    def load(self):
        p = MODELS_DIR / "wheat_efficientnetv2s_best.pth"
        if _download(f"{HF_BASE}/wheat_efficientnetv2s_best.pth", p):
            try:
                self.model, acc = _load_model(p, len(WHEAT_CLASSES))
                self.ready = True
                print(f"[wheat] ready {acc:.1f}%")
            except Exception as e:
                print(f"[wheat] failed: {e}")

    def predict(self, image: Image.Image) -> dict:
        r = _run_inference(self.model, WHEAT_CLASSES, image)
        return {**r, "crop_type": "Wheat", "model_used": "wheat_specialist"}


class ChilliAgent:
    def __init__(self):
        self.model = None
        self.ready = False

    def load(self):
        p = MODELS_DIR / "chilli_efficientnetv2s_best.pth"
        if _download(f"{HF_BASE}/chilli_efficientnetv2s_best.pth", p):
            try:
                self.model, acc = _load_model(p, len(CHILLI_CLASSES))
                self.ready = True
                print(f"[chilli] ready {acc:.1f}%")
            except Exception as e:
                print(f"[chilli] failed: {e}")

    def predict(self, image: Image.Image) -> dict:
        r = _run_inference(self.model, CHILLI_CLASSES, image)
        return {**r, "crop_type": "Chilli", "model_used": "chilli_specialist"}


# ═══════════════════════════════════════════════════════════════
# AGENT 4 — CONSENSUS VALIDATOR
# ═══════════════════════════════════════════════════════════════
class ConsensusValidator:
    MIN_CONF = 0.68
    MAX_ENT  = 0.42
    MIN_GAP  = 0.42

    def validate(self, pred: dict, route: dict, gk: dict) -> dict:
        if not gk.get("is_plant", True):
            return self._reject("not_a_plant",
                "This does not appear to be a plant leaf.",
                "یہ پودے کا پتہ نہیں لگتی۔")

        router_crop = route.get("crop", "unknown")
        spec_crop   = pred.get("crop_type", "").lower()
        if (router_crop not in ("unknown", "other") and
                router_crop != spec_crop and
                route.get("confidence", 0) > 0.35):
            return self._reject(
                f"wrong_crop",
                f"This looks like {router_crop.title()}, not {spec_crop.title()}.",
                f"یہ {router_crop} لگتا ہے، {spec_crop} نہیں۔")

        if pred["confidence"] < self.MIN_CONF:
            return self._reject("low_confidence",
                "Image unclear. Please take a closer photo.",
                "تصویر واضح نہیں۔ قریب سے دوبارہ لیں۔")

        if pred["entropy"] > self.MAX_ENT:
            return self._reject("high_entropy",
                "Multiple items in image. Focus on one leaf.",
                "تصویر میں زیادہ چیزیں ہیں۔ ایک پتے پر فوکس کریں۔")

        if pred["top2_gap"] < self.MIN_GAP:
            return self._reject("ambiguous",
                "Two diseases look similar. Upload a clearer photo.",
                "دو بیماریاں ملتی جلتی ہیں۔ واضح تصویر لیں۔")

        score = pred["confidence"] * pred["top2_gap"] * (1 - pred["entropy"])
        return {"valid": True, "score": score}

    @staticmethod
    def _reject(reason, msg_en, msg_ur):
        return {"valid": False, "reason": reason,
                "msg_en": msg_en, "msg_ur": msg_ur}


# ═══════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════
class MultiAgentSystem:

    def __init__(self, general_model=None):
        self.gatekeeper = GatekeeperAgent()
        self.router     = None
        self.wheat      = WheatAgent()
        self.chilli     = ChilliAgent()
        self.validator  = ConsensusValidator()
        self.general    = general_model

    def load(self):
        print("[system] Loading agents...")
        self.gatekeeper.load()
        self.router = CropRouterAgent(self.gatekeeper)
        self.wheat.load()
        self.chilli.load()
        print("[system] All agents ready!")

    def status(self) -> dict:
        return {
            "gatekeeper": "ready" if self.gatekeeper.ready else "fallback",
            "wheat":      "ready" if self.wheat.ready else "not_loaded",
            "chilli":     "ready" if self.chilli.ready else "not_loaded",
        }

    def analyze(self, image: Image.Image) -> dict:
        t0  = time.perf_counter()
        log = []

        # Agent 1
        gk = self.gatekeeper.check(image)
        log.append(f"[gatekeeper] {gk['reason']}")

        if not gk["is_plant"] and gk["not_plant_score"] > 0.70:
            return self._unclear(
                "not_a_plant",
                "This does not appear to be a plant leaf. Please upload a clear close-up photo of an affected crop leaf.",
                "یہ تصویر پودے کا پتہ نہیں لگتی۔ براہ کرم متاثرہ فصل کے پتے کی واضح تصویر اپلوڈ کریں۔",
                log, t0)

        # Agent 2
        route = self.router.route(image)
        log.append(f"[router] {route['crop']} conf={route['confidence']:.2f}")

        # Agent 3 + 4
        if route["crop"] == "wheat" and self.wheat.ready:
            pred = self.wheat.predict(image)
            log.append(f"[wheat] {pred['label']} {pred['confidence']:.2f}")
            v = self.validator.validate(pred, route, gk)
            if v["valid"]:
                return self._success(pred, v["score"], log, t0)
            log.append(f"[validator] rejected: {v['reason']}")
            return self._unclear(v["reason"], v["msg_en"], v["msg_ur"], log, t0)

        if route["crop"] == "chilli" and self.chilli.ready:
            pred = self.chilli.predict(image)
            log.append(f"[chilli] {pred['label']} {pred['confidence']:.2f}")
            v = self.validator.validate(pred, route, gk)
            if v["valid"]:
                return self._success(pred, v["score"], log, t0)
            log.append(f"[validator] rejected: {v['reason']}")
            return self._unclear(v["reason"], v["msg_en"], v["msg_ur"], log, t0)

        # General fallback
        if self.general is not None:
            try:
                preds = self.general.predict(image, top_k=1)
                if preds and preds[0]["confidence"] > 0.62:
                    top   = preds[0]
                    label = top["label"]
                    crop  = label.split("___")[0].replace("_", " ") if "___" in label else "Plant"
                    log.append(f"[general] {label} {top['confidence']:.2f}")
                    return self._success(
                        {"label": label, "confidence": top["confidence"],
                         "crop_type": crop, "model_used": "general_plantvillage",
                         "top2_gap": 0.5, "entropy": 0.3},
                        top["confidence"], log, t0)
                log.append("[general] low confidence")
            except Exception as e:
                log.append(f"[general] error: {e}")

        return self._unclear(
            "no_prediction",
            "Could not identify the disease. Upload a clear, well-lit close-up of the affected leaf.",
            "بیماری کی شناخت نہیں ہو سکی۔ واضح اور روشن تصویر لیں۔",
            log, t0)

    def _success(self, pred, score, log, t0) -> dict:
        return {
            "success":      True,
            "label":        pred["label"],
            "confidence":   pred["confidence"],
            "crop_type":    pred["crop_type"],
            "model_used":   pred["model_used"],
            "agent_log":    log,
            "inference_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    @staticmethod
    def _unclear(reason, msg_en, msg_ur, log, t0) -> dict:
        return {
            "success":      False,
            "label":        "Unclear_Image",
            "confidence":   0.0,
            "crop_type":    "Unknown",
            "model_used":   "none",
            "reason":       reason,
            "msg_en":       msg_en,
            "msg_ur":       msg_ur,
            "agent_log":    log,
            "inference_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
