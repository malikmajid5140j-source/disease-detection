// src/api/predict.js — AgriScan AI v3
const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function predictDisease(file) {
  const form = new FormData();
  form.append("file", file);  // ONLY file — AI decides everything automatically

  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }

  const data = await res.json();

  // Handle unclear image
  if (data.model_used === "none" || data.crop_type === "Unknown") {
    return { ...data, isUnclear: true };
  }

  return { ...data, isUnclear: false };
}

export async function checkHealth() {
  const res = await fetch(`${API_URL}/health`);
  return res.json();
}

export async function getClasses() {
  const res = await fetch(`${API_URL}/classes`);
  return res.json();
}

// Model badge helper
export function getModelBadge(modelUsed) {
  const badges = {
    wheat_specialist:     { emoji: "🌾", label: "Wheat AI",   acc: "92.3%", color: "#fbbf24" },
    chilli_specialist:    { emoji: "🌶️", label: "Chilli AI",  acc: "99.4%", color: "#f87171" },
    general_plantvillage: { emoji: "🌿", label: "General AI", acc: "39 crops", color: "#818cf8" },
    none:                 { emoji: "⚠️", label: "Unclear",     acc: "",       color: "#9ca3af" },
  };
  return badges[modelUsed] ?? badges.none;
}
