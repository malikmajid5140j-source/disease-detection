"""
AgriScan Multi-Agent System v3
────────────────────────────────
5 Agents working together:

1. Gatekeeper Agent    → Is this a plant leaf at all?
2. Crop Router Agent   → Which crop? Wheat/Chilli/Other
3. Specialist Agents   → Disease detection (Wheat/Chilli/General)
4. Consensus Validator → Cross-check predictions
5. Response Builder    → Generate final response with treatment
"""

from __future__ import annotations
import os, io, time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models


# ═══════════════════════════════════════════════════════════════
# Base configuration
# ═══════════════════════════════════════════════════════════════

MODELS_DIR = Path(__file__).parent / "specialist_models"
MODELS_DIR.mkdir(exist_ok=True)

HF_BASE = os.getenv(
    "HF_MODEL_URL",
    "https://huggingface.co/Mlaikmajid1063/agriscan-models/resolve/main"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Standard transform for our specialist models
EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
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


# ═══════════════════════════════════════════════════════════════
# AGENT 1: GATEKEEPER AGENT
# ═══════════════════════════════════════════════════════════════
class GatekeeperAgent:
    """
    Uses CLIP to check: Is this even a plant leaf?
    Rejects humans, food, objects, random photos.
    """

    def __init__(self):
        self.model = None
        self.preprocess = None
        self.ready = False

        # Text prompts for zero-shot classification
        self.plant_prompts = [
            "a photo of a plant leaf",
            "a photo of a crop leaf with disease",
            "a photo of green leaves",
            "a close up photo of a leaf",
        ]

        self.not_plant_prompts = [
            "a photo of a person",
            "a photo of a human face",
            "a photo of food or vegetable",
            "a photo of an object",
            "a photo of a building",
            "a photo of an animal",
            "a random photo",
            "a screenshot or drawing",
        ]

    def load(self):
        try:
            import clip
            print("[gatekeeper] Loading CLIP model...")
            self.model, self.preprocess = clip.load("ViT-B/32", device=DEVICE)
            self.ready = True
            print("[gatekeeper] [OK] CLIP ready")
        except ImportError:
            print("[gatekeeper] [WARN] CLIP not installed - using fallback")
            self.ready = False
        except Exception as e:
            print(f"[gatekeeper] [FAIL] Load failed: {e}")
            self.ready = False

    def is_plant(self, image: Image.Image) -> dict:
        """
        Returns:
            {
                'is_plant': bool,
                'plant_score': float (0-1),
                'not_plant_score': float (0-1),
                'reasoning': str
            }
        """
        if not self.ready:
            # Fallback — always assume plant (specialists will filter)
            return {
                'is_plant': True,
                'plant_score': 0.5,
                'not_plant_score': 0.5,
                'reasoning': 'gatekeeper_unavailable'
            }

        import clip
        img_t = self.preprocess(image).unsqueeze(0).to(DEVICE)
        all_prompts = self.plant_prompts + self.not_plant_prompts
        text_t = clip.tokenize(all_prompts).to(DEVICE)

        with torch.no_grad():
            img_feat = self.model.encode_image(img_t)
            txt_feat = self.model.encode_text(text_t)
            # Normalize
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            probs = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)
            probs = probs[0].cpu().numpy()

        n_plant = len(self.plant_prompts)
        plant_score     = float(probs[:n_plant].sum())
        not_plant_score = float(probs[n_plant:].sum())

        is_plant = plant_score > not_plant_score

        # Extra safety — need strong plant signal
        if plant_score < 0.55:
            is_plant = False

        top_not_plant_idx = int(probs[n_plant:].argmax())
        top_not_plant     = self.not_plant_prompts[top_not_plant_idx]

        reasoning = (f"plant={plant_score:.2f} not_plant={not_plant_score:.2f} "
                    f"top_reject={top_not_plant}")

        return {
            'is_plant': is_plant,
            'plant_score': plant_score,
            'not_plant_score': not_plant_score,
            'reasoning': reasoning,
        }


# ═══════════════════════════════════════════════════════════════
# AGENT 2: CROP ROUTER AGENT
# ═══════════════════════════════════════════════════════════════
class CropRouterAgent:
    """
    Uses CLIP to identify which crop.
    Routes to appropriate specialist.
    """

    def __init__(self, gatekeeper: GatekeeperAgent):
        # Reuse gatekeeper's CLIP model
        self.gk = gatekeeper

        self.crop_prompts = {
            'wheat':   "a photo of a wheat leaf or wheat plant",
            'chilli':  "a photo of a chilli pepper leaf or chilli plant",
            'tomato':  "a photo of a tomato leaf or tomato plant",
            'potato':  "a photo of a potato leaf or potato plant",
            'corn':    "a photo of a corn or maize leaf",
            'apple':   "a photo of an apple leaf",
            'grape':   "a photo of a grape leaf",
            'other':   "a photo of some other plant leaf",
        }

    def route(self, image: Image.Image) -> dict:
        """
        Returns:
            {
                'crop': 'wheat' | 'chilli' | 'other',
                'confidence': float,
                'all_scores': {crop_name: score}
            }
        """
        if not self.gk.ready:
            # Fallback — try all specialists
            return {
                'crop': 'unknown',
                'confidence': 0.0,
                'all_scores': {},
                'reasoning': 'router_unavailable'
            }

        import clip
        img_t = self.gk.preprocess(image).unsqueeze(0).to(DEVICE)
        prompts = list(self.crop_prompts.values())
        text_t = clip.tokenize(prompts).to(DEVICE)

        with torch.no_grad():
            img_feat = self.gk.model.encode_image(img_t)
            txt_feat = self.gk.model.encode_text(text_t)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            probs = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)
            probs = probs[0].cpu().numpy()

        crop_keys = list(self.crop_prompts.keys())
        scores = {k: float(v) for k, v in zip(crop_keys, probs)}

        # Best match
        top_idx = int(probs.argmax())
        top_crop = crop_keys[top_idx]
        top_conf = float(probs[top_idx])

        # Decide routing
        if top_crop == 'wheat' and top_conf > 0.25:
            route_to = 'wheat'
        elif top_crop == 'chilli' and top_conf > 0.25:
            route_to = 'chilli'
        else:
            route_to = 'other'

        return {
            'crop': route_to,
            'top_match': top_crop,
            'confidence': top_conf,
            'all_scores': scores,
        }


# ═══════════════════════════════════════════════════════════════
# AGENT 3: SPECIALIST AGENTS (Wheat / Chilli / General)
# ═══════════════════════════════════════════════════════════════

def _download_model(url: str, dest: Path) -> bool:
    import urllib.request
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        return True
    try:
        print(f"[download] {dest.name}...")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"[download] Failed: {e}")
        return False


def _load_specialist(pth_path: Path, num_classes: int):
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    ckpt = torch.load(pth_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(DEVICE)
    return model, ckpt.get('val_acc', 0)


class WheatSpecialistAgent:
    def __init__(self):
        self.model = None
        self.acc = 0
        self.ready = False
        self.classes = WHEAT_CLASSES

    def load(self):
        path = MODELS_DIR / "wheat_efficientnetv2s_best.pth"
        url  = f"{HF_BASE}/wheat_efficientnetv2s_best.pth"
        if _download_model(url, path):
            try:
                self.model, self.acc = _load_specialist(path, len(self.classes))
                self.ready = True
                print(f"[wheat] [OK] Ready - {self.acc:.1f}%")
            except Exception as e:
                print(f"[wheat] [FAIL] {e}")

    def predict(self, image: Image.Image) -> dict:
        if not self.ready:
            return {'ready': False}

        img_t = EVAL_TF(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(self.model(img_t)[0], dim=0).cpu().numpy()

        idx = int(probs.argmax())
        sorted_p = np.sort(probs)[::-1]
        entropy = float(-np.sum(probs * np.log(probs + 1e-8)) / np.log(len(probs)))

        return {
            'ready': True,
            'label':      self.classes[idx],
            'confidence': float(probs[idx]),
            'top2_gap':   float(sorted_p[0] - sorted_p[1]),
            'entropy':    entropy,
            'crop_type':  'Wheat',
            'model_used': 'wheat_specialist',
        }


class ChilliSpecialistAgent:
    def __init__(self):
        self.model = None
        self.acc = 0
        self.ready = False
        self.classes = CHILLI_CLASSES

    def load(self):
        path = MODELS_DIR / "chilli_efficientnetv2s_best.pth"
        url  = f"{HF_BASE}/chilli_efficientnetv2s_best.pth"
        if _download_model(url, path):
            try:
                self.model, self.acc = _load_specialist(path, len(self.classes))
                self.ready = True
                print(f"[chilli] [OK] Ready - {self.acc:.1f}%")
            except Exception as e:
                print(f"[chilli] [FAIL] {e}")

    def predict(self, image: Image.Image) -> dict:
        if not self.ready:
            return {'ready': False}

        img_t = EVAL_TF(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(self.model(img_t)[0], dim=0).cpu().numpy()

        idx = int(probs.argmax())
        sorted_p = np.sort(probs)[::-1]
        entropy = float(-np.sum(probs * np.log(probs + 1e-8)) / np.log(len(probs)))

        return {
            'ready': True,
            'label':      self.classes[idx],
            'confidence': float(probs[idx]),
            'top2_gap':   float(sorted_p[0] - sorted_p[1]),
            'entropy':    entropy,
            'crop_type':  'Chilli',
            'model_used': 'chilli_specialist',
        }


# ═══════════════════════════════════════════════════════════════
# AGENT 4: CONSENSUS VALIDATOR
# ═══════════════════════════════════════════════════════════════
class ConsensusValidator:
    """
    Cross-checks specialist prediction with router.
    Rejects if:
    - Router said not wheat but wheat model gave high confidence (suspicious)
    - Very low confidence
    - High entropy
    - Small top-2 gap
    """

    # Strict thresholds
    MIN_CONFIDENCE = 0.75
    MAX_ENTROPY    = 0.35
    MIN_TOP2_GAP   = 0.50

    def validate(self, specialist_pred: dict,
                 router_result: dict,
                 gatekeeper_result: dict) -> dict:
        """
        Returns:
            {
                'valid': bool,
                'reason': str,
                'confidence_boost': float (adjusted score)
            }
        """
        # 1. Gatekeeper says not plant → reject
        if not gatekeeper_result.get('is_plant', True):
            return {
                'valid': False,
                'reason': 'not_a_plant',
                'user_message_en': 'This does not appear to be a plant leaf.',
                'user_message_ur': 'یہ پودے کا پتہ نہیں لگتی۔',
            }

        # 2. Router disagrees with specialist crop → suspicious
        router_crop     = router_result.get('crop', 'unknown')
        specialist_crop = specialist_pred.get('crop_type', '').lower()

        if router_crop != 'unknown':
            if router_crop != specialist_crop and router_crop != 'other':
                # Router is confident about a different crop
                if router_result.get('confidence', 0) > 0.35:
                    return {
                        'valid': False,
                        'reason': f'router_says_{router_crop}_not_{specialist_crop}',
                        'user_message_en': f'This appears to be {router_crop}, not {specialist_crop}.',
                        'user_message_ur': f'یہ {router_crop} لگتا ہے، {specialist_crop} نہیں۔',
                    }

        # 3. Specialist metrics check
        conf     = specialist_pred.get('confidence', 0)
        entropy  = specialist_pred.get('entropy', 1.0)
        top2_gap = specialist_pred.get('top2_gap', 0)

        if conf < self.MIN_CONFIDENCE:
            return {
                'valid': False,
                'reason': f'low_confidence_{conf:.2f}',
                'user_message_en': 'Not confident enough. Please upload a clearer image.',
                'user_message_ur': 'کافی یقین نہیں ہے۔ واضح تصویر اپلوڈ کریں۔',
            }

        if entropy > self.MAX_ENTROPY:
            return {
                'valid': False,
                'reason': f'high_entropy_{entropy:.2f}',
                'user_message_en': 'Image is unclear or contains multiple items.',
                'user_message_ur': 'تصویر واضح نہیں ہے۔',
            }

        if top2_gap < self.MIN_TOP2_GAP:
            return {
                'valid': False,
                'reason': f'small_gap_{top2_gap:.2f}',
                'user_message_en': 'Multiple diseases show similar features. Please retake photo.',
                'user_message_ur': 'کئی بیماریاں ملتی جلتی ہیں۔ دوبارہ تصویر لیں۔',
            }

        return {
            'valid': True,
            'reason': 'all_checks_passed',
            'confidence_boost': conf * (1 - entropy) * top2_gap,
        }


# ═══════════════════════════════════════════════════════════════
# ORCHESTRATOR — Runs all agents in sequence
# ═══════════════════════════════════════════════════════════════
class MultiAgentSystem:
    def __init__(self, general_model=None):
        self.gatekeeper = GatekeeperAgent()
        self.router     = None  # created after gatekeeper loads
        self.wheat      = WheatSpecialistAgent()
        self.chilli     = ChilliSpecialistAgent()
        self.validator  = ConsensusValidator()
        self.general    = general_model  # existing PlantVillage CNN

    def load(self):
        print("[system] Loading multi-agent system...")
        self.gatekeeper.load()
        self.router = CropRouterAgent(self.gatekeeper)
        self.wheat.load()
        self.chilli.load()
        print("[system] [OK] All agents loaded")

    def analyze(self, image: Image.Image) -> dict:
        t0 = time.perf_counter()
        log = []

        # ── AGENT 1: Gatekeeper ──────────────────────────────
        gk = self.gatekeeper.is_plant(image)
        log.append(f"[Gatekeeper] {gk['reasoning']}")

        if not gk['is_plant']:
            return self._unclear_response(
                reason='not_a_plant',
                message_en='This image does not appear to be a plant leaf.',
                message_ur='یہ تصویر پودے کا پتہ نہیں لگتی۔',
                log=log,
                elapsed=time.perf_counter() - t0,
            )

        # ── AGENT 2: Router ──────────────────────────────────
        route = self.router.route(image)
        log.append(f"[Router] → {route['crop']} (top={route.get('top_match')} conf={route['confidence']:.2f})")

        # ── AGENT 3: Specialists ─────────────────────────────
        candidates = []

        if route['crop'] == 'wheat' and self.wheat.ready:
            pred = self.wheat.predict(image)
            log.append(f"[Wheat] {pred['label']} ({pred['confidence']:.2f})")
            v = self.validator.validate(pred, route, gk)
            if v['valid']:
                candidates.append({**pred, 'combined': v['confidence_boost']})
            else:
                log.append(f"[Validator] ❌ Wheat rejected: {v['reason']}")

        elif route['crop'] == 'chilli' and self.chilli.ready:
            pred = self.chilli.predict(image)
            log.append(f"[Chilli] {pred['label']} ({pred['confidence']:.2f})")
            v = self.validator.validate(pred, route, gk)
            if v['valid']:
                candidates.append({**pred, 'combined': v['confidence_boost']})
            else:
                log.append(f"[Validator] ❌ Chilli rejected: {v['reason']}")

        elif route['crop'] == 'other' and self.general is not None:
            try:
                preds = self.general.predict(image, top_k=1)
                if preds and preds[0]['confidence'] > 0.65:
                    pred = {
                        'label':      preds[0]['label'],
                        'confidence': preds[0]['confidence'],
                        'crop_type':  preds[0]['label'].split('___')[0].replace('_', ' ')
                                        if '___' in preds[0]['label'] else 'Plant',
                        'model_used': 'general_plantvillage',
                        'combined':   preds[0]['confidence'],
                    }
                    candidates.append(pred)
                    log.append(f"[General] {pred['label']} ({pred['confidence']:.2f})")
                else:
                    log.append(f"[General] Low confidence, rejecting")
            except Exception as e:
                log.append(f"[General] Error: {e}")

        # ── AGENT 4: Final validator ─────────────────────────
        if not candidates:
            return self._unclear_response(
                reason='no_valid_prediction',
                message_en='Could not confidently identify the plant or disease. Please upload a clear close-up of an affected leaf.',
                message_ur='پودے یا بیماری کی شناخت نہیں ہو سکی۔ متاثرہ پتے کی واضح تصویر لیں۔',
                log=log,
                elapsed=time.perf_counter() - t0,
            )

        # Best candidate
        best = max(candidates, key=lambda x: x.get('combined', x['confidence']))

        return {
            'success':      True,
            'label':        best['label'],
            'confidence':   best['confidence'],
            'crop_type':    best['crop_type'],
            'model_used':   best['model_used'],
            'agent_log':    log,
            'inference_ms': round((time.perf_counter() - t0) * 1000, 1),
        }

    @staticmethod
    def _unclear_response(reason, message_en, message_ur, log, elapsed):
        return {
            'success':      False,
            'label':        'Unclear_Image',
            'confidence':   0.0,
            'crop_type':    'Unknown',
            'model_used':   'none',
            'reason':       reason,
            'message_en':   message_en,
            'message_ur':   message_ur,
            'agent_log':    log,
            'inference_ms': round(elapsed * 1000, 1),
        }
