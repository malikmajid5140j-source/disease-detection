"""
AgriScan AI v3 — FastAPI Backend
──────────────────────────────────
Wheat & Chilli specialist models only.
General PlantVillage model removed.
"""

import io
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from agents import MultiAgentSystem
from disease_info import DISEASE_INFO


SYSTEM: MultiAgentSystem | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global SYSTEM

    print("[startup] Loading multi-agent system (Wheat + Chilli specialists)...")
    SYSTEM = MultiAgentSystem(general_model=None)   # No general model
    SYSTEM.load()
    print(f"[startup] System ready — {SYSTEM.status()}")

    yield

    print("[shutdown] Done")


app = FastAPI(
    title="AgriScan AI v3",
    description="Wheat & Chilli specialist disease detection",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    traceback.print_exc()
    return JSONResponse({"error": str(exc), "trace": traceback.format_exc()}, status_code=500)


def _build_response(result: dict) -> dict:
    """Convert agent result → full API response with Urdu treatments."""

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
                "Upload a clear close-up of a wheat or chilli leaf",
                "Use good daylight — avoid flash",
                "Focus on the affected area only",
                "Hold camera 15-25cm from leaf",
            ],
            "symptoms_ur": [
                "گندم یا مرچ کے پتے کی واضح تصویر لیں",
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
            "heatmap_base64":        "",
            "reason":                result.get("reason", ""),
        }

    label      = result["label"]
    info       = DISEASE_INFO.get(label, DISEASE_INFO["__default__"])
    is_healthy = "healthy" in label.lower()

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
        "heatmap_base64":        result.get("heatmap_base64", ""),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "agents": SYSTEM.status() if SYSTEM else {},
    }


@app.get("/classes")
def get_classes():
    if not SYSTEM:
        raise HTTPException(503, "Not ready")
    return {
        "wheat_classes":  ["Aphid","Black_Rust","Brown_Rust","Common_Root_Rot",
                           "Fusarium_Head_Blight","Healthy","Leaf_Blight","Mite",
                           "Powdery_Mildew","Septoria","Smut","Stem_Fly",
                           "Tan_Spot","Wheat_Blast","Yellow_Rust"],
        "chilli_classes": ["Anthracnose","Damping_Off","Healthy","Leaf_Curl_Virus",
                           "Leaf_Spot","Veinal_Mottle_Virus","Whitefly","Yellowish"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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


# Serve frontend
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(frontend_path / "index.html")