"""
AgriScan AI v2 — FastAPI Backend
──────────────────────────────────
FULLY AUTOMATIC crop detection:
  - Kisan sirf photo upload kare
  - AI khud detect kare: Wheat / Chilli / Other
  - No user input needed
"""

import time, io, os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from model import PlantDiseaseModel, ModelNotLoadedError
from specialist import SpecialistRouter
from disease_info import DISEASE_INFO

GENERAL_MODEL: PlantDiseaseModel | None = None
ROUTER: SpecialistRouter | None = None
PT_PATH = Path(__file__).parent / "plant_disease_model_1_latest.pt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global GENERAL_MODEL, ROUTER

    if not PT_PATH.exists():
        raise FileNotFoundError(f"Model not found: {PT_PATH}")

    print("[startup] Loading general PlantVillage model...")
    GENERAL_MODEL = PlantDiseaseModel(pt_path=PT_PATH)
    GENERAL_MODEL.load()
    print(f"[startup] General model ready — {GENERAL_MODEL.num_classes} classes")

    print("[startup] Loading specialist models (Wheat + Chilli)...")
    ROUTER = SpecialistRouter()
    ROUTER.load()
    print(f"[startup] Specialists: {ROUTER.status()}")

    yield
    GENERAL_MODEL = None
    ROUTER = None


app = FastAPI(
    title="AgriScan AI v2",
    description="Smart auto-detection — no user selection needed",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_response(label: str, confidence: float, crop_type: str,
                   model_used: str, inference_ms: float) -> dict:
    """Build complete response with Urdu treatments."""

    info = DISEASE_INFO.get(label, DISEASE_INFO.get("__default__"))

    # Parse label
    if "___" in label:
        parts   = label.split("___")
        plant   = parts[0].replace("_", " ")
        disease = parts[1].replace("_", " ")
    else:
        plant   = crop_type
        disease = label.replace("_", " ")

    is_healthy = "healthy" in label.lower()

    # Unclear image response
    if label == "Unclear_Image":
        return {
            "crop_type":            "Unknown",
            "plant_type":           "Unknown",
            "disease_name_en":      "Image Not Clear",
            "disease_name_ur":      "تصویر واضح نہیں",
            "severity":             "Unknown",
            "confidence":           0,
            "description_en":       "Could not detect crop type. Please upload a clear, close-up photo of the plant leaf in good lighting.",
            "description_ur":       "فصل کی قسم معلوم نہ ہو سکی۔ براہ کرم پتے کی واضح اور قریبی تصویر اچھی روشنی میں لیں۔",
            "symptoms_en":          ["Ensure leaf is clearly visible", "Use good lighting", "Take photo from 20-30cm distance"],
            "symptoms_ur":          ["پتہ واضح نظر آنا چاہیے", "اچھی روشنی استعمال کریں", "20-30 سینٹی میٹر کے فاصلے سے تصویر لیں"],
            "organic_treatment_en": "Please retake the photo with better lighting and focus on the leaf.",
            "organic_treatment_ur": "براہ کرم بہتر روشنی میں دوبارہ تصویر لیں اور پتے پر توجہ رکھیں۔",
            "chemical_treatment_en": "N/A",
            "chemical_treatment_ur": "قابل اطلاق نہیں",
            "prevention_en":        "N/A",
            "prevention_ur":        "قابل اطلاق نہیں",
            "model_used":           "none",
            "inference_ms":         inference_ms,
        }

    return {
        "crop_type":             crop_type,
        "plant_type":            plant,
        "disease_name_en":       info.get("name_en", disease),
        "disease_name_ur":       info.get("name_ur", disease),
        "severity":              "Healthy" if is_healthy else info.get("severity", "Moderate"),
        "confidence":            round(confidence * 100, 2),
        "description_en":        info.get("desc_en", f"{plant} {'is healthy' if is_healthy else 'shows signs of ' + disease}."),
        "description_ur":        info.get("desc_ur", f"پودا {'صحت مند ہے' if is_healthy else disease + ' سے متاثر ہے'}"),
        "symptoms_en":           info.get("symptoms_en", ["No visible symptoms"] if is_healthy else [f"Signs of {disease}"]),
        "symptoms_ur":           info.get("symptoms_ur", ["کوئی علامت نہیں"] if is_healthy else ["علامات نظر آ رہی ہیں"]),
        "organic_treatment_en":  info.get("organic_en", "Maintain good plant hygiene." if is_healthy else "Remove affected parts, use neem oil."),
        "organic_treatment_ur":  info.get("organic_ur", "پودوں کی صفائی کا خیال رکھیں۔" if is_healthy else "متاثرہ حصے ہٹائیں، نیم کا تیل استعمال کریں۔"),
        "chemical_treatment_en": info.get("chemical_en", "None needed." if is_healthy else "Consult local agriculture extension."),
        "chemical_treatment_ur": info.get("chemical_ur", "ضرورت نہیں" if is_healthy else "مقامی زرعی ماہر سے مشورہ کریں۔"),
        "prevention_en":         info.get("prevention_en", "Regular monitoring and crop rotation."),
        "prevention_ur":         info.get("prevention_ur", "باقاعدہ نگرانی اور فصل بدلیں۔"),
        "model_used":            model_used,
        "inference_ms":          inference_ms,
    }


@app.get("/health")
def health():
    return {
        "status":      "ok",
        "model_loaded": GENERAL_MODEL is not None,
        "specialist":  ROUTER.status() if ROUTER else {},
        "mode":        "fully_automatic",
    }


@app.get("/classes")
def get_classes():
    if GENERAL_MODEL is None:
        raise HTTPException(503, "Model not loaded")
    return {
        "wheat_classes":   ROUTER.wheat_classes() if ROUTER else [],
        "chilli_classes":  ROUTER.chilli_classes() if ROUTER else [],
        "general_classes": GENERAL_MODEL.class_names,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    FULLY AUTOMATIC prediction.
    Kisan sirf photo upload kare — crop type khud detect hoga.
    No crop_hint needed from frontend.
    """
    if GENERAL_MODEL is None or ROUTER is None:
        raise HTTPException(503, "Models not loaded yet")

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Image file chahiye, mila: '{file.content_type}'")

    try:
        raw   = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Image decode failed: {e}")

    try:
        t0     = time.perf_counter()

        # ── SMART AUTO DETECT ────────────────────────────────────────────────
        result = ROUTER.smart_predict(image, general_model=GENERAL_MODEL)
        # ────────────────────────────────────────────────────────────────────

        elapsed = round((time.perf_counter() - t0) * 1000, 1)

        response = build_response(
            label        = result["label"],
            confidence   = result["confidence"],
            crop_type    = result["crop_type"],
            model_used   = result["model_used"],
            inference_ms = elapsed,
        )
        return JSONResponse(response)

    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")


# ── Serve frontend ────────────────────────────────────────────────────────────
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(frontend_path / "index.html")
