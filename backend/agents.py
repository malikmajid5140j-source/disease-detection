"""
AgriScan AI v3 — Multi-Agent System
─────────────────────────────────────
Key insight: CLIP dekhta hai PURI image — sirf leaf nahi.
Wheat field, diseased crop, plant stem — sab accept karo.
Sirf clearly non-plant reject karo (human face, car, food on plate).
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
# Puri image dekhta hai — sirf clearly non-agricultural reject
# ═══════════════════════════════════════════════════════════════
class GatekeeperAgent:
    """
    CLIP se check karta hai: Kya yeh agricultural image hai?
    
    ACCEPT:
      - Wheat leaves, stalks, field, crop
      - Chilli plant, pepper, diseased crop
      - Any plant, vegetation, farming scene
      - Blurry/dark crop photo — specialists handle it
      - Whole plant, stem, roots — not just leaf
    
    REJECT (ONLY clearly non-agricultural):
      - Human face / selfie
      - Car, building, furniture
      - Cooked food on plate / restaurant
      - Screenshot / text / document
      - Random objects (phone, bottle, etc.)
    """

    # Broad agricultural prompts — field/crop/plant sab accept
    AGRI_PROMPTS = [
        "a photo of agricultural crops or plants",
        "a photo of wheat or cereal crops in a field",
        "a photo of chilli or pepper plant",
        "a photo of diseased or unhealthy plant",
        "a photo of green or yellow plant vegetation",
        "a photo of a crop field or farm",
        "a photo of plant leaves stems or roots",
        "a photo of a sick or infected crop",
        "a close up photo of plant texture or surface",
        "a photo of farming or agriculture",
    ]

    # Only CLEARLY non-agricultural things
    NOT_AGRI_PROMPTS = [
        "a photo of a human face or person",
        "a selfie or portrait photo",
        "a photo of a car or vehicle",
        "a photo of a building or indoor room",
        "a photo of cooked food on a plate or in a bowl",
        "a photo of a smartphone or electronic device",
        "a screenshot of text or a website",
        "a photo of an animal like a dog or cat",
        "a photo of furniture or household objects",
        "a photo of clothing or fashion",
    ]

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
        """
        Returns:
          is_agri: True = agricultural image, proceed
          agri_score: 0-1
          not_agri_score: 0-1
          reason: debug string
        """
        if not self.ready:
            # CLIP nahi hai — specialists par trust karo
            return {
                "is_agri":        True,
                "agri_score":     0.6,
                "not_agri_score": 0.4,
                "reason":         "clip_unavailable_trusting_specialists",
            }

        import clip
        img_t     = self.preprocess(image).unsqueeze(0).to(DEVICE)
        all_txt   = self.AGRI_PROMPTS + self.NOT_AGRI_PROMPTS
        text_t    = clip.tokenize(all_txt).to(DEVICE)

        with torch.no_grad():
            img_feat  = self.model.encode_image(img_t)
            txt_feat  = self.model.encode_text(text_t)
            img_feat  = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat  = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            probs     = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)[0].cpu().numpy()

        n_agri      = len(self.AGRI_PROMPTS)
        agri_score  = float(probs[:n_agri].sum())
        not_score   = float(probs[n_agri:].sum())

        top_agri_idx = int(probs[:n_agri].argmax())
        top_not_idx  = int(probs[n_agri:].argmax())
        top_agri     = self.AGRI_PROMPTS[top_agri_idx]
        top_not      = self.NOT_AGRI_PROMPTS[top_not_idx]

        # STRICT rejection — sirf tab reject karo jab:
        # 1. not_agri clearly wins (>0.65) — ya
        # 2. agri score bahut kam hai (<0.30)
        # Dubious cases mein specialists par chhoddo
        is_agri = not (not_score > 0.65 or agri_score < 0.30)

        reason = (f"agri={agri_score:.2f} not={not_score:.2f} "
                  f"top_agri='{top_agri[:30]}' "
                  f"top_not='{top_not[:30]}'")

        return {
            "is_agri":        is_agri,
            "agri_score":     agri_score,
            "not_agri_score": not_score,
            "reason":         reason,
        }


# ═══════════════════════════════════════════════════════════════
# AGENT 2 — CROP ROUTER
# Puri image se crop identify karta hai
# ═══════════════════════════════════════════════════════════════
class CropRouterAgent:
    """
    Puri image context se crop identify karta hai.
    Wheat field, yellowing crop, diseased stalks — sab wheat.
    """

    CROP_PROMPTS = {
        "wheat":  "a photo of wheat crop, wheat field, or wheat plant with disease",
        "chilli": "a photo of chilli plant, pepper crop, or mirch with disease",
        "tomato": "a photo of tomato plant or tomato leaves with disease",
        "potato": "a photo of potato plant leaves or potato crop disease",
        "corn":   "a photo of corn or maize crop or maize plant",
        "rice":   "a photo of rice crop or paddy field or rice plant",
        "other":  "a photo of some other agricultural plant or crop",
    }

    def __init__(self, gatekeeper: GatekeeperAgent):
        self.gk = gatekeeper

    def route(self, image: Image.Image) -> dict:
        if not self.gk.ready:
            return {"crop": "unknown", "confidence": 0.0, "scores": {}}

        import clip
        img_t   = self.gk.preprocess(image).unsqueeze(0).to(DEVICE)
        keys    = list(self.CROP_PROMPTS.keys())
        prompts = list(self.CROP_PROMPTS.values())
        text_t  = clip.tokenize(prompts).to(DEVICE)

        with torch.no_grad():
            img_feat  = self.gk.model.encode_image(img_t)
            txt_feat  = self.gk.model.encode_text(text_t)
            img_feat  = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat  = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            probs     = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)[0].cpu().numpy()

        scores   = {k: float(v) for k, v in zip(keys, probs)}
        top_idx  = int(probs.argmax())
        top_crop = keys[top_idx]
        top_conf = float(probs[top_idx])

        # Lower threshold — specialist models zyada accurate hain
        if top_crop in ("wheat", "chilli") and top_conf > 0.18:
            route_to = top_crop
        elif top_crop in ("tomato", "potato", "corn", "rice") and top_conf > 0.20:
            route_to = "other"
        else:
            route_to = "other"

        print(f"[router] top={top_crop} conf={top_conf:.2f} → routing to {route_to}")
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
    MIN_CONF = 0.65   # 65% minimum
    MAX_ENT  = 0.45   # entropy limit
    MIN_GAP  = 0.40   # top-2 gap

    def validate(self, pred: dict, route: dict, gk: dict) -> dict:

        # Not agricultural at all
        if not gk.get("is_agri", True):
            return self._reject("not_agricultural",
                "This does not appear to be an agricultural image.",
                "یہ زرعی تصویر نہیں لگتی۔")

        # Router strongly says different crop
        router_crop = route.get("crop", "unknown")
        spec_crop   = pred.get("crop_type", "").lower()
        if (router_crop not in ("unknown", "other") and
                router_crop != spec_crop and
                route.get("confidence", 0) > 0.40):
            return self._reject("wrong_crop",
                f"This looks like {router_crop.title()}, not {spec_crop.title()}.",
                f"یہ {router_crop} لگتا ہے، {spec_crop} نہیں۔")

        # Confidence check
        if pred["confidence"] < self.MIN_CONF:
            return self._reject("low_confidence",
                "Image unclear or too far. Please take a closer photo.",
                "تصویر واضح نہیں یا دور سے لی ہے۔ قریب سے دوبارہ لیں۔")

        # Entropy check
        if pred["entropy"] > self.MAX_ENT:
            return self._reject("high_entropy",
                "Too much variation in image. Focus on the affected area.",
                "تصویر میں بہت زیادہ variation ہے۔ متاثرہ حصے پر فوکس کریں۔")

        # Top-2 gap check
        if pred["top2_gap"] < self.MIN_GAP:
            return self._reject("ambiguous",
                "Two possibilities detected. Please upload a clearer photo.",
                "دو ممکنہ بیماریاں ہیں۔ واضح تصویر لیں۔")

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

        # ── Agent 1: Gatekeeper ──────────────────────────────
        gk = self.gatekeeper.check(image)
        log.append(f"[gatekeeper] {gk['reason']}")

        # Sirf clearly non-agricultural reject karo
        if not gk["is_agri"]:
            return self._unclear(
                "not_agricultural",
                "This does not appear to be a crop or plant image. Please upload a photo of an affected plant.",
                "یہ فصل یا پودے کی تصویر نہیں لگتی۔ براہ کرم متاثرہ پودے کی تصویر اپلوڈ کریں۔",
                log, t0)

        # ── Agent 2: Router ──────────────────────────────────
        route = self.router.route(image)
        log.append(f"[router] → {route['crop']} (conf={route['confidence']:.2f})")

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