"""
AgriScan AI v3 — Multi-Agent System
─────────────────────────────────────
Key insight: Universal Object Identifier using CLIP.
Identifies exactly what the object is, rejecting non-supported objects
with clear messages, and routing supported crops to specialists.
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
# AGENT 1 — GATEKEEPER (Universal Object Identifier)
# ═══════════════════════════════════════════════════════════════
class GatekeeperAgent:
    """
    CLIP-based Universal Object Identifier.
    Classifies the incoming image into specific categories, identifying whether
    it is a supported crop or a clearly unsupported object.
    """
    CATEGORIES = {
        "human": {
            "prompts": ["a photo of a human face", "a selfie of a person", "a portrait of a person"],
            "is_supported": False,
            "name_en": "Human Face / Person",
            "name_ur": "انسان کا چہرہ / شخص"
        },
        "potato_vegetable": {
            "prompts": ["a photo of raw potato vegetable", "a potato tuber", "potatoes in a pile"],
            "is_supported": False,
            "name_en": "Potato Tuber / Vegetable",
            "name_ur": "آلو کی سبزی"
        },
        "food": {
            "prompts": ["a photo of cooked food on a plate", "a bowl of soup or meal", "a dish of food"],
            "is_supported": False,
            "name_en": "Cooked Food / Meal",
            "name_ur": "پکا ہوا کھانا / خوراک"
        },
        "car": {
            "prompts": ["a photo of a car", "a photo of a vehicle", "a truck or automobile"],
            "is_supported": False,
            "name_en": "Car / Vehicle",
            "name_ur": "گاڑی / وہیکل"
        },
        "animal": {
            "prompts": ["a photo of a dog or cat", "a photo of a cow or sheep", "a photo of an animal"],
            "is_supported": False,
            "name_en": "Animal / Pet",
            "name_ur": "جانور / پالتو جانور"
        },
        "building": {
            "prompts": ["a photo of a house or building", "an indoor room", "a street scene"],
            "is_supported": False,
            "name_en": "Building / Indoor Room",
            "name_ur": "عمارت / اندورنِ کمرہ"
        },
        "device": {
            "prompts": ["a photo of a smartphone", "a laptop or computer", "an electronic gadget"],
            "is_supported": False,
            "name_en": "Electronic Device / Gadget",
            "name_ur": "الیکٹرانک ڈیوائس"
        },
        "text": {
            "prompts": ["a screenshot of a website", "a page of a document or book", "written text on paper"],
            "is_supported": False,
            "name_en": "Screenshot / Document Text",
            "name_ur": "اسکرین شاٹ / تحریری دستاویز"
        },
        "wheat": {
            "prompts": ["a photo of wheat leaves", "a photo of wheat plant", "a photo of a wheat crop field"],
            "is_supported": True,
            "crop_name": "wheat",
            "name_en": "Wheat Plant / Leaf",
            "name_ur": "گندم کا پودا / پتہ"
        },
        "chilli": {
            "prompts": ["a photo of chilli plant", "a photo of pepper crop", "a photo of green or red chilli pepper"],
            "is_supported": True,
            "crop_name": "chilli",
            "name_en": "Chilli Plant / Leaf",
            "name_ur": "مرچ کا پودا / پتہ"
        },
        "tomato": {
            "prompts": ["a photo of tomato plant", "a photo of tomato leaf"],
            "is_supported": True,
            "crop_name": "tomato",
            "name_en": "Tomato Plant / Leaf",
            "name_ur": "ٹماٹر کا پودا / پتہ"
        },
        "potato_leaf": {
            "prompts": ["a photo of potato plant leaf", "a photo of potato leaves"],
            "is_supported": True,
            "crop_name": "potato",
            "name_en": "Potato Plant Leaf",
            "name_ur": "آلو کے پودے کا پتہ"
        },
        "corn": {
            "prompts": ["a photo of corn plant", "a photo of maize leaf", "a photo of corn leaf"],
            "is_supported": True,
            "crop_name": "corn",
            "name_en": "Corn / Maize Plant",
            "name_ur": "مکئی کا پودا"
        },
        "apple_leaf": {
            "prompts": ["a photo of apple leaf", "a photo of apple plant leaf"],
            "is_supported": True,
            "crop_name": "apple",
            "name_en": "Apple Plant Leaf",
            "name_ur": "سیب کے پودے کا پتہ"
        },
        "other_plant": {
            "prompts": ["a photo of some other plant leaf", "a photo of green leaves of a tree", "a photo of house plant"],
            "is_supported": True,
            "crop_name": "other",
            "name_en": "Other Plant Leaf",
            "name_ur": "کسی دوسرے پودے کا پتہ"
        }
    }

    def __init__(self):
        self.model      = None
        self.preprocess = None
        self.ready      = False

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
            return {
                "is_supported": True,
                "crop_name": "other",
                "name_en": "Unknown Plant",
                "name_ur": "نامعلوم پودا",
                "reason": "clip_unavailable",
                "best_cat": "other_plant"
            }

        import clip
        img_tensor  = self.preprocess(image).unsqueeze(0).to(DEVICE)
        
        all_prompts = []
        prompt_to_category = []
        for cat_key, cat_val in self.CATEGORIES.items():
            for p in cat_val["prompts"]:
                all_prompts.append(p)
                prompt_to_category.append(cat_key)
                
        text_tensor = clip.tokenize(all_prompts).to(DEVICE)

        with torch.no_grad():
            img_feat  = self.model.encode_image(img_tensor)
            txt_feat  = self.model.encode_text(text_tensor)
            img_feat  = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat  = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            probs     = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)[0].cpu().numpy()

        cat_scores = {}
        for idx, prob in enumerate(probs):
            cat_key = prompt_to_category[idx]
            cat_scores[cat_key] = cat_scores.get(cat_key, 0.0) + float(prob)

        best_cat = max(cat_scores, key=cat_scores.get)
        best_score = cat_scores[best_cat]
        cat_info = self.CATEGORIES[best_cat]

        return {
            "is_supported": cat_info["is_supported"],
            "crop_name": cat_info.get("crop_name", "other"),
            "name_en": cat_info["name_en"],
            "name_ur": cat_info["name_ur"],
            "best_cat": best_cat,
            "reason": f"Detected: {cat_info['name_en']} (conf={best_score:.2f})",
        }


# ═══════════════════════════════════════════════════════════════
# AGENT 2 — CROP ROUTER
# ═══════════════════════════════════════════════════════════════
class CropRouterAgent:
    def __init__(self, gatekeeper: GatekeeperAgent):
        self.gk = gatekeeper

    def route(self, image: Image.Image, gk_res: dict = None) -> dict:
        if gk_res is not None:
            return {"crop": gk_res["crop_name"], "confidence": 1.0, "scores": {}}
        
        # Fallback if called without gk_res
        if not self.gk.ready:
            return {"crop": "unknown", "confidence": 0.0, "scores": {}}
        res = self.gk.check(image)
        return {"crop": res["crop_name"], "confidence": 1.0, "scores": {}}


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
        print(f"[download] done {dest.stat().st_size/1e6:.1f}MB")
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
    """
    Specialist prediction cross-check.
    Realistic thresholds — real field images thodi noisy hoti hain.
    """
    MIN_CONF = 0.55
    MAX_ENT  = 0.65
    MIN_GAP  = 0.30

    def validate(self, pred: dict, route: dict, gk: dict) -> dict:
        if not gk.get("is_supported", True):
            return self._reject("unsupported_object",
                "This image is not supported.",
                "یہ تصویر سپورٹڈ نہیں ہے۔")

        # Confidence check only — entropy aur gap hata do
        # Field photos mein entropy naturally high hoti hai
        if pred["confidence"] < self.MIN_CONF:
            return self._reject("low_confidence",
                "Cannot identify disease clearly. Please take a closer photo of the affected area.",
                "بیماری واضح نہیں۔ متاثرہ حصے کے قریب سے تصویر لیں۔")

        score = pred["confidence"]
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

        # ── Agent 1: Gatekeeper / Object Recognizer ──────────
        gk = self.gatekeeper.check(image)
        log.append(f"[gatekeeper] {gk['reason']}")

        if not gk["is_supported"]:
            return self._unclear(
                "unsupported_object",
                f"You uploaded a photo of a {gk['name_en']}. This has no relation to the crops we support (Wheat, Chilli, Tomato, Potato, Corn, Apple, etc.). Please upload an affected leaf.",
                f"آپ نے {gk['name_ur']} کی تصویر اپلوڈ کی ہے۔ یہ ہمارے سپورٹڈ پودوں (گندم، مرچ، ٹماٹر، آلو، مکئی، سیب) سے مطابقت نہیں رکھتی۔ براہ کرم متاثرہ پتے کی تصویر اپلوڈ کریں۔",
                log, t0)

        # ── Agent 2: Router ──────────────────────────────────
        route = self.router.route(image, gk_res=gk)
        log.append(f"[router] → {route['crop']}")

        # ── Agent 3: Specialists ─────────────────────────────
        # Try wheat specialist
        if route["crop"] == "wheat" and self.wheat.ready:
            pred = self.wheat.predict(image)
            log.append(f"[wheat] {pred['label']} conf={pred['confidence']:.2f} "
                       f"gap={pred['top2_gap']:.2f} ent={pred['entropy']:.2f}")
            v = self.validator.validate(pred, route, gk)
            if v["valid"]:
                return self._success(pred, v["score"], log, t0)
            log.append(f"[validator] rejected wheat: {v['reason']}")
            return self._unclear(v["reason"], v["msg_en"], v["msg_ur"], log, t0)

        # Try chilli specialist
        if route["crop"] == "chilli" and self.chilli.ready:
            pred = self.chilli.predict(image)
            log.append(f"[chilli] {pred['label']} conf={pred['confidence']:.2f} "
                       f"gap={pred['top2_gap']:.2f} ent={pred['entropy']:.2f}")
            v = self.validator.validate(pred, route, gk)
            if v["valid"]:
                return self._success(pred, v["score"], log, t0)
            log.append(f"[validator] rejected chilli: {v['reason']}")
            return self._unclear(v["reason"], v["msg_en"], v["msg_ur"], log, t0)

        # General PlantVillage fallback (other crops)
        if self.general is not None:
            try:
                preds = self.general.predict(image, top_k=1)
                if preds and preds[0]["confidence"] > 0.58:
                    top   = preds[0]
                    label = top["label"]
                    crop  = (label.split("___")[0].replace("_", " ")
                             if "___" in label else "Plant")
                    log.append(f"[general] {label} conf={top['confidence']:.2f}")
                    return self._success(
                        {"label": label, "confidence": top["confidence"],
                         "crop_type": crop, "model_used": "general_plantvillage",
                         "top2_gap": 0.5, "entropy": 0.3},
                        top["confidence"], log, t0)
                log.append(f"[general] low conf={preds[0]['confidence']:.2f if preds else 0:.2f}")
            except Exception as e:
                log.append(f"[general] error: {e}")

        return self._unclear(
            "no_prediction",
            "Could not identify the crop or disease. Please upload a clear, well-lit photo of the affected plant.",
            "فصل یا بیماری کی شناخت نہیں ہو سکی۔ براہ کرم متاثرہ پودے کی واضح اور روشن تصویر لیں۔",
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