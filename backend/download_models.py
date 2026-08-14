import urllib.request
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BACKEND_DIR / "specialist_models"
MODELS_DIR.mkdir(exist_ok=True)

# 1. Download General Model (if LFS or missing)
general_path = BACKEND_DIR / "plant_disease_model_1_latest.pt"
general_url = "https://github.com/malikmajid5140j-source/disease-detection/raw/main/backend/plant_disease_model_1_latest.pt"

if not general_path.exists() or general_path.stat().st_size < 1024 * 1024:
    print("[build] Downloading general model...")
    try:
        urllib.request.urlretrieve(general_url, general_path)
        print(f"[build] ✅ General model downloaded ({general_path.stat().st_size/1e6:.1f}MB)")
    except Exception as e:
        print(f"[build] ❌ General model download failed: {e}")
else:
    print(f"[build] General model already exists ({general_path.stat().st_size/1e6:.1f}MB)")

# 2. Download Specialist Models from HuggingFace
HF_BASE = "https://huggingface.co/Mlaikmajid1063/agriscan-models/resolve/main"
specialists = {
    "wheat_efficientnetv2s_best.pth": MODELS_DIR / "wheat_efficientnetv2s_best.pth",
    "chilli_efficientnetv2s_best.pth": MODELS_DIR / "chilli_efficientnetv2s_best.pth"
}

for fname, dst in specialists.items():
    if not dst.exists() or dst.stat().st_size < 1024 * 1024:
        print(f"[build] Downloading {fname}...")
        try:
            urllib.request.urlretrieve(f"{HF_BASE}/{fname}", dst)
            print(f"[build] ✅ {fname} downloaded ({dst.stat().st_size/1e6:.1f}MB)")
        except Exception as e:
            print(f"[build] ❌ {fname} download failed: {e}")
    else:
        print(f"[build] {fname} already exists ({dst.stat().st_size/1e6:.1f}MB)")

print("[build] Pre-download of models completed successfully!")
