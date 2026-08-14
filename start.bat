@echo off
echo Starting AgriScan AI Backend...
start cmd /k "cd backend && ..\.venv\Scripts\activate && set HF_MODEL_URL=https://huggingface.co/Mlaikmajid1063/agriscan-models/resolve/main && uvicorn main:app --reload --port 8000"

echo Opening Frontend...
start "" "http://localhost:8000"

echo Project is running!
