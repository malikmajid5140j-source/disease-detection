// src/api/predict.js
// Drop this file in your React project at src/api/predict.js

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

/**
 * Send image to FastAPI and get predictions back.
 * @param {File} file  - image file from input/drop
 * @returns {Promise<{
 *   top_prediction: string,
 *   confidence: number,
 *   predictions: {label: string, confidence: number}[],
 *   model: string,
 *   inference_time_ms: number,
 *   image_size: number[]
 * }>}
 */
export async function predictDisease(file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }

  return res.json();
}

/** Check if the backend is alive and model is loaded */
export async function checkHealth() {
  const res = await fetch(`${API_URL}/health`);
  return res.json();
}

/** Get all class names the model supports */
export async function getClasses() {
  const res = await fetch(`${API_URL}/classes`);
  return res.json();
}
