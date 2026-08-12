# AgriScan AI: Full Project Architecture & Design Details

## 1. High-Level Architecture Overview
AgriScan AI is a full-stack, AI-powered web application designed to identify plant diseases from leaf images and provide actionable organic and chemical treatments. The system follows a classic **Client-Server Architecture** decoupled into a lightweight frontend UI and a heavy AI-processing backend.

- **Frontend (Client):** A bilingual (English & Urdu) Single Page Application (SPA) that captures user input (images) and visualizes the AI pipeline and results.
- **Backend (Server):** A high-performance REST API that receives images, processes them through a deep learning model, and returns a JSON payload with the diagnosis and treatments.
- **AI Engine (Model):** A Convolutional Neural Network (CNN) trained on the standard PlantVillage dataset to classify 39 distinct crop conditions (38 diseases + 1 background).

---

## 2. Technology Stack

### Frontend Stack (UI/UX)
*   **HTML5:** Semantic structure of the web app.
*   **Tailwind CSS:** Used via CDN for rapid, modern, and responsive styling. It handles the glassmorphism effects, grid layouts, and custom animations (like the floating elements and SVG neural network paths).
*   **Vanilla JavaScript:** Handles file dragging/dropping, Base64 image encoding, API `fetch()` requests, dynamic DOM updates, and the English/Urdu language toggle.
*   **FontAwesome & Google Fonts:** Provides scalable iconography and modern typography (`Inter` for English, `Noto Nastaliq Urdu` for Urdu).

### Backend Stack (API & Processing)
*   **Python (3.11+):** The core programming language.
*   **FastAPI:** A modern, fast web framework for building the REST API. It handles CORS, routing, and serves the static frontend files.
*   **Uvicorn:** The ASGI web server used to run the FastAPI application.
*   **Pillow (PIL):** Used to read, decode, and convert the uploaded images into the correct RGB format.

### Machine Learning Stack
*   **PyTorch (`torch`, `torchvision`):** The deep learning framework powering the custom CNN architecture. It loads the `state_dict` (weights) from the `.pt` file and performs matrix multiplications (inference).

---

## 3. Detailed File Structure

```text
disease-detection/          # Root Repository Directory
│
├── frontend/               # The User Interface
│   └── index.html          # The single-page app containing all UI, Tailwind styling, and JS logic.
│
├── backend/                # The API & AI Engine
│   ├── main.py             # FastAPI application, routing (/predict), and logic to format the JSON response.
│   ├── model.py            # PyTorch CustomCNN class definition and model loading logic.
│   ├── plant_disease_model_1_latest.pt # The trained PyTorch neural network weights (The "Brain").
│   ├── requirements.txt    # Python dependencies (fastapi, torch, uvicorn, etc.)
│   ├── Procfile            # Deployment instruction for PaaS (like Railway/Heroku).
│   └── .python-version     # Specifies Python version (3.11) for cloud builders.
│
├── Dockerfile              # Containerization instructions for deploying to cloud environments.
├── deploy_to_hf.py         # Automated script to push the repository to Hugging Face Spaces.
├── start.bat               # Windows batch script to easily run the backend locally.
├── README.md               # Project documentation.
│
├── images/                 # Static assets or dataset samples.
├── screenshots/            # Images of the UI for the README.
└── test_images/            # Sample leaf images used for testing the model.
```

---

## 4. How the Data Flows (The Pipeline)

1.  **Image Upload:** The user drags and drops a leaf image onto the dropzone in `index.html`.
2.  **Request Generation:** The JavaScript frontend converts the image into a `FormData` object and sends a `POST` request to `http://<server-ip>:8000/predict`.
3.  **API Reception:** `main.py` (FastAPI) receives the file, validates that it is an image (PNG/JPG/WEBP), and reads it into memory using Pillow.
4.  **AI Inference:** 
    *   The image is passed to `model.py`.
    *   It gets resized to `224x224` pixels and normalized.
    *   It passes through 4 blocks of Convolutional layers (extracting shapes/patterns) and Dense layers.
    *   A `Softmax` function calculates the probability for all 39 classes.
5.  **Result Formatting:** `main.py` checks the top prediction. If confidence is > 50%, it looks up the specific disease, generates treatment advice in both English and Urdu, and wraps it in a JSON object.
6.  **UI Update:** The frontend receives the JSON response, updates the UI cards with the severity, confidence score, symptoms, and treatments, and stops the visual loading animation.

---

## 5. Design & UI Details
*   **Color Palette:** Heavily utilizes "Emerald" and "Brand (Green)" tones to signify agriculture and nature.
*   **Bilingual Support:** Designed specifically for regional farmers, allowing seamless switching between English and right-to-left (RTL) Urdu text without reloading the page.
*   **Interactive Visuals:** Includes an animated SVG "Neural Network Pipeline" that visually explains to the user what is happening behind the scenes while the AI processes the image.
*   **Sample Presets:** Pre-loaded images (Tomato Blight, Corn Rust, etc.) allow users to test the app without needing their own photos.
