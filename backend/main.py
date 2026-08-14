"""
AgriScan AI v3 — Multi-Agent Backend
─────────────────────────────────────
5 agents work together:
  1. Gatekeeper (CLIP) → is this a plant?
  2. Router (CLIP)     → which crop?
  3. Specialists       → disease detection
  4. Validator         → cross-check
  5. Response Builder  → treatments
"""

import io
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from model import PlantDiseaseModel
from agents import MultiAgentSystem
from disease_info import DISEASE_INFO


GENERAL_MODEL: PlantDiseaseModel | None = None
AGENTS: MultiAgentSystem | None = None
PT_PATH = Path(__file__).parent / "plant_disease_model_1_latest.pt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global GENERAL_MODEL, AGENTS

    print("[startup] Loading general PlantVillage model...")
    GENERAL_MODEL = PlantDiseaseModel(pt_path=PT_PATH)
    GENERAL_MODEL.load()
    print(f"[startup] General model ready — {GENERAL_MODEL.num_classes} classes")

    print("[startup] Initializing multi-agent system...")
    AGENTS = MultiAgentSystem(general_model=GENERAL_MODEL)
    AGENTS.load()
    print("[startup] [OK] Multi-agent system ready!")

    yield
    print("[shutdown] Releasing resources")


app = FastAPI(
    title="AgriScan AI v3 — Multi-Agent",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_response(agent_result: dict) -> dict:
    """Convert agent output to full frontend response."""

    # Unclear case
    if not agent_result.get('success'):
        return {
            'crop_type':             'Unknown',
            'plant_type':            'Unknown',
            'disease_name_en':       'Image Not Recognized',
            'disease_name_ur':       'تصویر کی شناخت نہیں ہوئی',
            'severity':              'Unknown',
            'confidence':            0,
            'description_en':        agent_result.get('message_en', 'Could not analyze this image.'),
            'description_ur':        agent_result.get('message_ur', 'اس تصویر کا تجزیہ نہیں ہو سکا۔'),
            'symptoms_en':           [
                'Upload a clear close-up of a plant leaf',
                'Use good daylight',
                'Focus on the affected area',
                'Distance: 15-25cm from leaf',
            ],
            'symptoms_ur':           [
                'پودے کے پتے کی واضح تصویر لیں',
                'اچھی روشنی میں',
                'متاثرہ حصے پر فوکس کریں',
                '15-25 سینٹی میٹر کے فاصلے سے',
            ],
            'organic_treatment_en':  'N/A - Retake photo',
            'organic_treatment_ur':  'قابل اطلاق نہیں',
            'chemical_treatment_en': 'N/A',
            'chemical_treatment_ur': 'قابل اطلاق نہیں',
            'prevention_en':         'N/A',
            'prevention_ur':         'قابل اطلاق نہیں',
            'model_used':            'none',
            'inference_ms':          agent_result.get('inference_ms', 0),
            'agent_log':             agent_result.get('agent_log', []),
            'reason':                agent_result.get('reason', ''),
        }

    # Success case
    label      = agent_result['label']
    info       = DISEASE_INFO.get(label, DISEASE_INFO.get('__default__'))
    is_healthy = 'healthy' in label.lower()

    if '___' in label:
        plant, disease = label.split('___')
        plant   = plant.replace('_', ' ')
        disease = disease.replace('_', ' ')
    else:
        plant   = agent_result.get('crop_type', 'Plant')
        disease = label.replace('_', ' ')

    return {
        'crop_type':             agent_result['crop_type'],
        'plant_type':            plant,
        'disease_name_en':       info.get('name_en', disease),
        'disease_name_ur':       info.get('name_ur', disease),
        'severity':              'Healthy' if is_healthy else info.get('severity', 'Moderate'),
        'confidence':            round(agent_result['confidence'] * 100, 2),
        'description_en':        info.get('desc_en', ''),
        'description_ur':        info.get('desc_ur', ''),
        'symptoms_en':           info.get('symptoms_en', []),
        'symptoms_ur':           info.get('symptoms_ur', []),
        'organic_treatment_en':  info.get('organic_en', ''),
        'organic_treatment_ur':  info.get('organic_ur', ''),
        'chemical_treatment_en': info.get('chemical_en', ''),
        'chemical_treatment_ur': info.get('chemical_ur', ''),
        'prevention_en':         info.get('prevention_en', ''),
        'prevention_ur':         info.get('prevention_ur', ''),
        'model_used':            agent_result['model_used'],
        'inference_ms':          agent_result['inference_ms'],
        'agent_log':             agent_result.get('agent_log', []),
    }


@app.get("/health")
def health():
    return {
        'status':          'ok',
        'model_loaded':    GENERAL_MODEL is not None,
        'agents_ready':    AGENTS is not None,
        'gatekeeper':      AGENTS.gatekeeper.ready if AGENTS else False,
        'wheat':           AGENTS.wheat.ready if AGENTS else False,
        'chilli':          AGENTS.chilli.ready if AGENTS else False,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if AGENTS is None:
        raise HTTPException(503, "Agents not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Image file required")

    try:
        raw   = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Image decode failed: {e}")

    result = AGENTS.analyze(image)
    return JSONResponse(build_response(result))


# ── Serve frontend ────────────────────────────────────────────
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(frontend_path / "index.html")
