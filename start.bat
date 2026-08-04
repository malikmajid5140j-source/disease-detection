@echo off
echo Starting AgriScan AI Backend...
start cmd /k "cd backend && uvicorn main:app --reload --port 8000"

echo Opening Frontend...
start "" "frontend\index.html"

echo Project is running!
