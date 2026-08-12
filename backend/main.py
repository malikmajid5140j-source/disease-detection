"""
PlantGuard AI — FastAPI Backend
Serves YOLOv8 plant disease predictions via REST API.

Run:
    uvicorn main:app --reload --port 8000
"""

import time
import io
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from model import PlantDiseaseModel, ModelNotLoadedError


# ── Model singleton ──────────────────────────────────────────────────────────
MODEL: PlantDiseaseModel | None = None

PT_PATH = Path(__file__).parent / "plant_disease_model_1_latest.pt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global MODEL
    if not PT_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {PT_PATH}\n"
            "Place plant_disease_model_1_latest.pt in the same folder as main.py"
        )
    print(f"[startup] Loading model from {PT_PATH} ...")
    MODEL = PlantDiseaseModel(pt_path=PT_PATH)
    MODEL.load()
    print(f"[startup] Model ready — {MODEL.num_classes} classes loaded")
    yield
    print("[shutdown] Releasing model")
    MODEL = None


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PlantGuard AI",
    description="YOLOv8-based plant disease detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Quick liveness + model status check."""
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "num_classes": MODEL.num_classes if MODEL else 0,
    }


@app.get("/classes")
def get_classes():
    """Return all class names the model knows."""
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")
    return {"classes": MODEL.class_names, "count": MODEL.num_classes}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an image, run YOLOv8 classification, return top-5 predictions.

    Request:  multipart/form-data  { file: <image> }
    Response: JSON with predictions sorted by confidence descending
    """
    if MODEL is None:
        raise HTTPException(503, "Model not loaded yet — try again in a moment")

    # ── Validate file type ────────────────────────────────────────────────────
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            400,
            f"Expected an image file, got '{file.content_type}'.\n"
            "Please upload PNG, JPG, or WEBP."
        )

    # ── Read & decode image ───────────────────────────────────────────────────
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")

    # ── Run inference ─────────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        predictions = MODEL.predict(image, top_k=5)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    except ModelNotLoadedError:
        raise HTTPException(503, "Model not available")
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    if not predictions:
        raise HTTPException(500, "Model returned no predictions")

    top = predictions[0]

    label = top["label"]
    confidence = top["confidence"]
    
    # Check if prediction is unreliable (low confidence) or if it's explicitly classified as background
    if confidence < 0.5 or label == "Background_without_leaves":
        response_data = {
            "plant_type": "Unknown",
            "disease_name_en": "Unrecognized / Not a Plant",
            "disease_name_ur": "غیر متعلقہ / پودا نہیں",
            "severity": "Unknown",
            "confidence": confidence * 100,
            "description_en": "The uploaded image does not appear to be a recognized plant leaf or the model is unsure. Please upload a clear image of a plant leaf.",
            "description_ur": "یہ تصویر کسی پودے کے پتے کی نہیں لگتی۔ براہ کرم پتے کی واضح تصویر اپ لوڈ کریں۔",
            "symptoms_en": ["None"],
            "symptoms_ur": ["کوئی نہیں"],
            "organic_treatment_en": "N/A",
            "organic_treatment_ur": "قابل اطلاق نہیں",
            "chemical_treatment_en": "N/A",
            "chemical_treatment_ur": "قابل اطلاق نہیں",
            "prevention_en": "N/A",
            "prevention_ur": "قابل اطلاق نہیں"
        }
        return JSONResponse(response_data)
    
    # Generic parsing of label assuming format like "Tomato___Early_blight"
    parts = label.split("___")
    plant = parts[0].replace("_", " ") if len(parts) > 0 else "Unknown Plant"
    disease = parts[1].replace("_", " ") if len(parts) > 1 else label.replace("_", " ")
    
    is_healthy = "healthy" in disease.lower()
    severity = "Healthy" if is_healthy else "Moderate"

    response_data = {
        "plant_type": f"{plant}",
        "disease_name_en": disease,
        "disease_name_ur": f"{disease} (اردو)",
        "severity": severity,
        "confidence": confidence * 100,
        "description_en": f"The plant {plant} appears to be {'healthy' if is_healthy else 'affected by ' + disease}.",
        "description_ur": f"پودا {plant} {'صحت مند ہے' if is_healthy else disease + ' سے متاثر ہے'}",
        "symptoms_en": ["No visible symptoms"] if is_healthy else [f"Signs of {disease} on leaves", "Discoloration", "Spots or lesions"],
        "symptoms_ur": ["کوئی علامت نہیں"] if is_healthy else ["پتوں پر دھبے", "رنگ میں تبدیلی", "خرابی کے آثار"],
        "organic_treatment_en": "Keep optimal watering and sunlight." if is_healthy else "Remove affected leaves and use neem oil.",
        "organic_treatment_ur": "مناسب پانی اور دھوپ کا خیال رکھیں۔" if is_healthy else "متاثرہ پتوں کو ہٹا دیں اور نیم کا تیل استعمال کریں۔",
        "chemical_treatment_en": "None needed." if is_healthy else "Apply appropriate fungicide/bactericide.",
        "chemical_treatment_ur": "ضرورت نہیں" if is_healthy else "مناسب پھپھوندی کش دوا استعمال کریں۔",
        "prevention_en": "Maintain good plant hygiene.",
        "prevention_ur": "پودوں کی صفائی کا خیال رکھیں۔"
    }

    return JSONResponse(response_data)

# Serve frontend statically
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(frontend_path / "index.html")

