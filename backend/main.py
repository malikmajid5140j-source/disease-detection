"""
AgriScan AI v3 — FastAPI Backend
──────────────────────────────────
Multi-agent disease detection.
Kisan sirf photo upload kare — sab automatic.
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


GENERAL: PlantDiseaseModel | None = None
SYSTEM:  MultiAgentSystem  | None = None
PT_PATH = Path(__file__).parent / "plant_disease_model_1_latest.pt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global GENERAL, SYSTEM

    if not PT_PATH.exists():
        raise FileNotFoundError(f"Model not found: {PT_PATH}")

    print("[startup] Loading general PlantVillage model...")
    GENERAL = PlantDiseaseModel(pt_path=PT_PATH)
    GENERAL.load()
    print(f"[startup] General model ready — {GENERAL.num_classes} classes")

    print("[startup] Loading multi-agent system...")
    SYSTEM = MultiAgentSystem(general_model=GENERAL)
    SYSTEM.load()
    print(f"[startup] System ready — {SYSTEM.status()}")

    yield

    print("[shutdown] Done")


app = FastAPI(
    title="AgriScan AI v3",
    description="Multi-agent plant disease detection",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_response(result: dict) -> dict:
    """Convert agent result → full API response with Urdu treatments."""

    # Unclear / rejected image
    if not result.get("success"):
        return {
            "crop_type":             "Unknown",
            "plant_type":            "Unknown",
            "disease_name_en":       "Image Not Recognized",
            "disease_name_ur":       "تصویر کی شناخت نہیں ہوئی",
            "severity":              "Unknown",
            "confidence":            0,
            "description_en":        result.get("msg_en", "Could not analyze this image."),
            "description_ur":        result.get("msg_ur", "اس تصویر کا تجزیہ نہیں ہو سکا۔"),
            "symptoms_en": [
                "Upload a clear close-up of a plant leaf",
                "Use good daylight — avoid flash",
                "Focus on the affected area only",
                "Hold camera 15-25cm from leaf",
            ],
            "symptoms_ur": [
                "پودے کے پتے کی واضح تصویر لیں",
                "اچھی روشنی میں — فلیش سے گریز کریں",
                "صرف متاثرہ حصے پر فوکس کریں",
                "کیمرہ پتے سے 15-25 سینٹی میٹر دور رکھیں",
            ],
            "organic_treatment_en":  "N/A — please retake the photo.",
            "organic_treatment_ur":  "قابل اطلاق نہیں — دوبارہ تصویر لیں۔",
            "chemical_treatment_en": "N/A",
            "chemical_treatment_ur": "قابل اطلاق نہیں",
            "prevention_en":         "N/A",
            "prevention_ur":         "قابل اطلاق نہیں",
            "model_used":            "none",
            "inference_ms":          result.get("inference_ms", 0),
            "agent_log":             result.get("agent_log", []),
            "reason":                result.get("reason", ""),
        }

    # Successful detection
    label      = result["label"]
    info       = DISEASE_INFO.get(label, DISEASE_INFO["__default__"])
    is_healthy = "healthy" in label.lower()

    if "___" in label:
        plant   = label.split("___")[0].replace("_", " ")
        disease = label.split("___")[1].replace("_", " ")
    else:
        plant   = result.get("crop_type", "Plant")
        disease = label.replace("_", " ")

    return {
        "crop_type":             result["crop_type"],
        "plant_type":            plant,
        "disease_name_en":       info.get("name_en", disease),
        "disease_name_ur":       info.get("name_ur", disease),
        "severity":              "Healthy" if is_healthy else info.get("severity", "Moderate"),
        "confidence":            round(result["confidence"] * 100, 2),
        "description_en":        info.get("desc_en", ""),
        "description_ur":        info.get("desc_ur", ""),
        "symptoms_en":           info.get("symptoms_en", []),
        "symptoms_ur":           info.get("symptoms_ur", []),
        "organic_treatment_en":  info.get("organic_en", ""),
        "organic_treatment_ur":  info.get("organic_ur", ""),
        "chemical_treatment_en": info.get("chemical_en", ""),
        "chemical_treatment_ur": info.get("chemical_ur", ""),
        "prevention_en":         info.get("prevention_en", ""),
        "prevention_ur":         info.get("prevention_ur", ""),
        "model_used":            result["model_used"],
        "inference_ms":          result["inference_ms"],
        "agent_log":             result.get("agent_log", []),
    }


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": GENERAL is not None,
        "agents":       SYSTEM.status() if SYSTEM else {},
    }


@app.get("/classes")
def get_classes():
    if not GENERAL:
        raise HTTPException(503, "Not ready")
    return {
        "general_classes": GENERAL.class_names,
        "wheat_classes":   list(SYSTEM.wheat.ready and
                           ["Aphid","Black_Rust","Brown_Rust","Common_Root_Rot",
                            "Fusarium_Head_Blight","Healthy","Leaf_Blight","Mite",
                            "Powdery_Mildew","Septoria","Smut","Stem_Fly",
                            "Tan_Spot","Wheat_Blast","Yellow_Rust"] or []),
        "chilli_classes":  list(SYSTEM.chilli.ready and
                           ["Anthracnose","Damping_Off","Healthy","Leaf_Curl_Virus",
                            "Leaf_Spot","Veinal_Mottle_Virus","Whitefly","Yellowish"] or []),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Kisan sirf photo upload kare.
    AI automatically:
      1. Checks if it's a plant leaf
      2. Identifies crop type
      3. Detects disease
      4. Returns Urdu treatment
    """
    if SYSTEM is None:
        raise HTTPException(503, "System not ready")

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Image required, got {file.content_type}")

    try:
        raw   = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Cannot read image: {e}")

    result   = SYSTEM.analyze(image)
    response = _build_response(result)
    return JSONResponse(response)


# Serve frontend if present
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(frontend_path / "index.html")
