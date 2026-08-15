"""
AgriScan AI v3 — Multi-Agent System
Smart Gatekeeper: tries CLIP first, falls back gracefully
GradCAM: highlights disease area on leaf image
"""

from __future__ import annotations
import os, time, io, base64
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2

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
# GRADCAM — Disease Area Highlighter
# Uses existing EfficientNetV2-S model — no extra training needed
# ═══════════════════════════════════════════════════════════════
class GradCAM:
    """
    Generates heatmap showing WHERE the model found disease on the leaf.
    Works with EfficientNetV2-S — hooks into last conv layer.
    """
    def __init__(self, model: nn.Module):
        self.model    = model
        self.gradients = None
        self.activations = None
        self._hook_layers()

    def _hook_layers(self):
        # Hook into last conv block of EfficientNetV2-S
        target_layer = self.model.features[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image: Image.Image, class_idx: int) -> str:
        """
        Returns base64 encoded PNG of original image with disease heatmap overlay.
        """
        self.model.eval()
        img_tensor = EVAL_TF(image).unsqueeze(0).to(DEVICE)
        img_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(img_tensor)

        # Backward pass for target class
        self.model.zero_grad()
        output[0, class_idx].backward()

        if self.gradients is None or self.activations is None:
            return ""

        # Compute GradCAM weights
        weights   = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam       = (weights * self.activations).sum(dim=1).squeeze()
        cam       = F.relu(cam)
        cam       = cam.cpu().numpy()

        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to original image size
        orig_w, orig_h = image.size
        cam_resized = cv2.resize(cam, (orig_w, orig_h))

        # Apply colormap (JET: blue=healthy, red=disease)
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay on original image
        orig_np  = np.array(image.resize((orig_w, orig_h)))
        overlay  = (0.55 * orig_np + 0.45 * heatmap).astype(np.uint8)

        # Draw label on image
        result_img = Image.fromarray(overlay)

        # Convert to base64
        buffer = io.BytesIO()
        result_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ═══════════════════════════════════════════════════════════════
# AGENT 1 — SMART GATEKEEPER
# Tries CLIP first. If CLIP unavailable, uses EfficientNet confidence.
# Self-healing: never crashes, always makes a decision.
# ═══════════════════════════════════════════════════════════════
class GatekeeperAgent:
    """
    Smart self-healing gatekeeper.
    Strategy 1: CLIP zero-shot (best accuracy)
    Strategy 2: EfficientNet confidence fallback (if CLIP unavailable)
    """

    CLIP_PROMPTS = {
        "wheat": [
            "a photo of wheat plant leaves in a field",
            "wheat crop leaf showing yellow rust orange stripes",
            "wheat leaf with brown rust disease spots",
            "green wheat plant with fungal disease",
            "wheat field close up with leaf blight",
            "cereal grain crop with diseased leaf",
            "wheat stem and leaf with rust disease",
            "wheat plant leaf showing septoria",
            "wheat crop with powdery mildew on leaf",
            "diseased wheat leaf in agricultural field",
        ],
        "chilli": [
            "a photo of chilli plant leaf",
            "pepper plant leaf with disease",
            "chilli leaf showing leaf curl virus symptoms",
            "green chilli plant with diseased leaves",
            "red chilli pepper plant leaf",
            "capsicum plant leaf with spots or disease",
            "hot pepper plant leaf showing damage",
            "chilli crop leaf with whitefly or anthracnose",
            "chilli plant with yellowing leaves",
            "pepper plant leaf close-up in garden",
        ],
        "other": [
            "a photo of a human face or person",
            "cooked food like pizza burger or meal on plate",
            "a car truck or motor vehicle",
            "a dog cat or other animal",
            "raw potato onion carrot or vegetable",
            "a mobile phone laptop or electronic device",
            "a building house or indoor room",
            "furniture like chair table or sofa",
            "fruit like apple banana or orange",
            "a random non-agricultural object",
        ]
    }

    def __init__(self):
        self.clip_model    = None
        self.clip_preproc  = None
        self.text_features = None
        self.prompt_map    = []
        self.clip_ready    = False

    def load(self):
        try:
            import clip
            self.clip_model, self.clip_preproc = clip.load("RN50", device=DEVICE)

            all_prompts = []
            for category, prompts in self.CLIP_PROMPTS.items():
                for p in prompts:
                    all_prompts.append(p)
                    self.prompt_map.append(category)

            tokens = clip.tokenize(all_prompts).to(DEVICE)
            with torch.no_grad():
                self.text_features = self.clip_model.encode_text(tokens)
                self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

            self.clip_ready = True
            print("[gatekeeper] CLIP RN50 ready OK")
        except Exception as e:
            print(f"[gatekeeper] CLIP RN50 unavailable: {e} — will use EfficientNet fallback")
            self.clip_ready = False

    def check(self, image: Image.Image, wheat_model=None, chilli_model=None) -> dict:
        """
        Try CLIP first. If unavailable, use EfficientNet confidence scores.
        """
        if self.clip_ready:
            return self._clip_check(image)
        else:
            return self._efficientnet_check(image, wheat_model, chilli_model)

    def _clip_check(self, image: Image.Image) -> dict:
        import clip
        img_t = self.clip_preproc(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            img_feat = self.clip_model.encode_image(img_t)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sims = (100.0 * img_feat @ self.text_features.T).softmax(dim=-1)[0].cpu().numpy()

        # Average scores per category
        cat_scores = {cat: 0.0 for cat in self.CLIP_PROMPTS}
        counts     = {cat: 0   for cat in self.CLIP_PROMPTS}
        for idx, score in enumerate(sims):
            cat = self.prompt_map[idx]
            cat_scores[cat] += float(score)
            counts[cat]     += 1
        for cat in cat_scores:
            cat_scores[cat] /= counts[cat]

        best_cat   = max(cat_scores, key=cat_scores.get)
        best_score = cat_scores[best_cat]
        plant_best = max(cat_scores["wheat"], cat_scores["chilli"])

        print(f"[gatekeeper] CLIP wheat:{cat_scores['wheat']:.3f} "
              f"chilli:{cat_scores['chilli']:.3f} "
              f"other:{cat_scores['other']:.3f} -> {best_cat}")

        # Reject only if other clearly dominates plant scores
        if best_cat == "other" and (cat_scores["other"] - plant_best) > 0.02:
            return {
                "is_supported": False,
                "crop_name":    "unsupported",
                "name_en":      "Unsupported Image",
                "name_ur":      "غیر سپورٹڈ تصویر — صرف گندم یا مرچ کا پتہ اپلوڈ کریں",
                "reason":       f"CLIP: other={cat_scores['other']:.3f} > plant={plant_best:.3f}",
            }

        # If ambiguous between other and plant, trust the plant
        if best_cat == "other":
            best_cat = "wheat" if cat_scores["wheat"] >= cat_scores["chilli"] else "chilli"

        return {
            "is_supported": True,
            "crop_name":    best_cat,
            "name_en":      "Wheat Plant Leaf" if best_cat == "wheat" else "Chilli Plant Leaf",
            "name_ur":      "گندم کا پتہ" if best_cat == "wheat" else "مرچ کا پتہ",
            "reason":       f"CLIP: {best_cat} (score={best_score:.3f})",
        }

    def _efficientnet_check(self, image, wheat_model, chilli_model) -> dict:
        """
        Fallback: run both models, pick higher confidence crop.
        Reject if both are very low confidence (random image).
        """
        scores = {}
        if wheat_model is not None:
            r = _run_inference(wheat_model, WHEAT_CLASSES, image)
            scores["wheat"] = r["confidence"]
        if chilli_model is not None:
            r = _run_inference(chilli_model, CHILLI_CLASSES, image)
            scores["chilli"] = r["confidence"]

        if not scores:
            return {"is_supported": False, "crop_name": "unsupported",
                    "name_en": "Models not loaded", "name_ur": "ماڈل لوڈ نہیں",
                    "reason": "no models available"}

        best_crop = max(scores, key=scores.get)
        best_conf = scores[best_crop]

        print(f"[gatekeeper] EfficientNet fallback — "
              f"wheat:{scores.get('wheat',0):.2f} chilli:{scores.get('chilli',0):.2f}")

        # Reject if confidence too low — clearly not wheat or chilli
        if best_conf < 0.40:
            return {
                "is_supported": False,
                "crop_name":    "unsupported",
                "name_en":      "Image not recognized as wheat or chilli leaf",
                "name_ur":      "تصویر گندم یا مرچ کا پتہ نہیں لگ رہی",
                "reason":       f"EfficientNet fallback: low conf={best_conf:.2f}",
            }

        return {
            "is_supported": True,
            "crop_name":    best_crop,
            "name_en":      "Wheat Plant Leaf" if best_crop == "wheat" else "Chilli Plant Leaf",
            "name_ur":      "گندم کا پتہ" if best_crop == "wheat" else "مرچ کا پتہ",
            "reason":       f"EfficientNet fallback: {best_crop} conf={best_conf:.2f}",
        }


# ═══════════════════════════════════════════════════════════════
# HELPERS
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
        "class_idx":  idx,
    }


# ═══════════════════════════════════════════════════════════════
# AGENT 3 — SPECIALISTS
# ═══════════════════════════════════════════════════════════════
class WheatAgent:
    def __init__(self):
        self.model   = None
        self.gradcam = None
        self.ready   = False

    def load(self):
        p = MODELS_DIR / "wheat_efficientnetv2s_best.pth"
        if _download(f"{HF_BASE}/wheat_efficientnetv2s_best.pth", p):
            try:
                self.model, acc = _load_model(p, len(WHEAT_CLASSES))
                self.gradcam    = GradCAM(self.model)
                self.ready      = True
                print(f"[wheat] ready — val_acc={acc:.1f}%")
            except Exception as e:
                print(f"[wheat] failed: {e}")

    def predict(self, image: Image.Image) -> dict:
        r        = _run_inference(self.model, WHEAT_CLASSES, image)
        heatmap  = self.gradcam.generate(image, r["class_idx"])
        return {**r, "crop_type": "Wheat", "model_used": "wheat_specialist",
                "heatmap_base64": heatmap}


class ChilliAgent:
    def __init__(self):
        self.model   = None
        self.gradcam = None
        self.ready   = False

    def load(self):
        p = MODELS_DIR / "chilli_efficientnetv2s_best.pth"
        if _download(f"{HF_BASE}/chilli_efficientnetv2s_best.pth", p):
            try:
                self.model, acc = _load_model(p, len(CHILLI_CLASSES))
                self.gradcam    = GradCAM(self.model)
                self.ready      = True
                print(f"[chilli] ready — val_acc={acc:.1f}%")
            except Exception as e:
                print(f"[chilli] failed: {e}")

    def predict(self, image: Image.Image) -> dict:
        r        = _run_inference(self.model, CHILLI_CLASSES, image)
        heatmap  = self.gradcam.generate(image, r["class_idx"])
        return {**r, "crop_type": "Chilli", "model_used": "chilli_specialist",
                "heatmap_base64": heatmap}


# ═══════════════════════════════════════════════════════════════
# VALIDATOR
# ═══════════════════════════════════════════════════════════════
class ConsensusValidator:
    MIN_CONF = 0.30

    def validate(self, pred: dict, gk: dict) -> dict:
        if not gk.get("is_supported", True):
            return self._reject("unsupported_object",
                "This image is not supported.",
                "یہ تصویر سپورٹڈ نہیں ہے۔")
        if pred["confidence"] < self.MIN_CONF:
            return self._reject("low_confidence",
                "Cannot identify disease clearly. Please take a closer photo.",
                "بیماری واضح نہیں۔ قریب سے تصویر لیں۔")
        return {"valid": True, "score": pred["confidence"]}

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
        self.wheat      = WheatAgent()
        self.chilli     = ChilliAgent()
        self.validator  = ConsensusValidator()

    def load(self):
        print("[system] Loading agents...")
        self.gatekeeper.load()
        self.wheat.load()
        self.chilli.load()
        print("[system] All agents ready!")

    def status(self) -> dict:
        return {
            "gatekeeper": "clip"  if self.gatekeeper.clip_ready else "efficientnet_fallback",
            "wheat":      "ready" if self.wheat.ready  else "not_loaded",
            "chilli":     "ready" if self.chilli.ready else "not_loaded",
        }

    def analyze(self, image: Image.Image) -> dict:
        t0  = time.perf_counter()
        log = []

        # Agent 1: Gatekeeper (CLIP or EfficientNet fallback)
        gk = self.gatekeeper.check(
            image,
            wheat_model  = self.wheat.model  if self.wheat.ready  else None,
            chilli_model = self.chilli.model if self.chilli.ready else None,
        )
        log.append(f"[gatekeeper] {gk['reason']}")

        if not gk["is_supported"]:
            return self._unclear(
                "unsupported_object",
                f"Unsupported image. Please upload a wheat or chilli leaf photo.",
                "غیر سپورٹڈ تصویر۔ براہ کرم گندم یا مرچ کے پتے کی تصویر اپلوڈ کریں۔",
                log, t0)

        crop = gk["crop_name"]
        log.append(f"[router] → {crop}")

        # Agent 3: Specialist + GradCAM
        if crop == "wheat" and self.wheat.ready:
            pred = self.wheat.predict(image)
            log.append(f"[wheat] {pred['label']} conf={pred['confidence']:.2f}")
            v = self.validator.validate(pred, gk)
            if v["valid"]:
                return self._success(pred, v["score"], log, t0)
            return self._unclear(v["reason"], v["msg_en"], v["msg_ur"], log, t0)

        if crop == "chilli" and self.chilli.ready:
            pred = self.chilli.predict(image)
            log.append(f"[chilli] {pred['label']} conf={pred['confidence']:.2f}")
            v = self.validator.validate(pred, gk)
            if v["valid"]:
                return self._success(pred, v["score"], log, t0)
            return self._unclear(v["reason"], v["msg_en"], v["msg_ur"], log, t0)

        return self._unclear("no_specialist",
            "Specialist model not ready.",
            "اسپیشلسٹ ماڈل تیار نہیں۔", log, t0)

    def _success(self, pred, score, log, t0) -> dict:
        return {
            "success":          True,
            "label":            pred["label"],
            "confidence":       pred["confidence"],
            "crop_type":        pred["crop_type"],
            "model_used":       pred["model_used"],
            "heatmap_base64":   pred.get("heatmap_base64", ""),
            "agent_log":        log,
            "inference_ms":     round((time.perf_counter() - t0) * 1000, 1),
        }

    @staticmethod
    def _unclear(reason, msg_en, msg_ur, log, t0) -> dict:
        return {
            "success":        False,
            "label":          "Unclear_Image",
            "confidence":     0.0,
            "crop_type":      "Unknown",
            "model_used":     "none",
            "heatmap_base64": "",
            "reason":         reason,
            "msg_en":         msg_en,
            "msg_ur":         msg_ur,
            "agent_log":      log,
            "inference_ms":   round((time.perf_counter() - t0) * 1000, 1),
        }