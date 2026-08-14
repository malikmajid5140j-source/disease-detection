"""
PlantDiseaseModel
─────────────────
Custom CNN architecture matching the provided plant_disease_model_1_latest.pt state_dict.
"""

from __future__ import annotations
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

class ModelNotLoadedError(RuntimeError):
    pass

class CustomCNN(nn.Module):
    def __init__(self, num_classes=39):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)
        )
        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dense_layers(x)
        return x

class PlantDiseaseModel:
    def __init__(self, pt_path: Path):
        self.pt_path = pt_path
        self._model = None
        # Standard PlantVillage 39 Classes
        self.class_names = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot',
            'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight', 'Grape___healthy', 'Orange___Haunglongbing',
            'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper_bell___Bacterial_spot', 'Pepper_bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
            'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus', 'Tomato___healthy', 'Background_without_leaves'
        ]
        self.num_classes = len(self.class_names)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Basic normalization (can be adjusted if model was trained with specific norm)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load(self) -> None:
        """Load the PyTorch state dictionary."""
        self._model = CustomCNN(num_classes=self.num_classes).to(self.device)
        
        # Railway Git LFS Fix: Download real model if file is just an LFS pointer (<1MB)
        import os
        import urllib.request
        if os.path.getsize(self.pt_path) < 1024 * 1024:
            print(f"[model] LFS pointer detected! Downloading real model from GitHub...")
            url = "https://github.com/malikmajid5140j-source/disease-detection/raw/main/backend/plant_disease_model_1_latest.pt"
            urllib.request.urlretrieve(url, self.pt_path)
            print(f"[model] Download complete.")

        # Load weights
        try:
            state_dict = torch.load(self.pt_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"[model] ❌ Load failed ({e}). File might be corrupted.")
            print(f"[model] Deleting corrupted file and redownloading...")
            if self.pt_path.exists():
                self.pt_path.unlink()
            
            url = "https://github.com/malikmajid5140j-source/disease-detection/raw/main/backend/plant_disease_model_1_latest.pt"
            urllib.request.urlretrieve(url, self.pt_path)
            print(f"[model] ✅ Redownload complete.")
            state_dict = torch.load(self.pt_path, map_location=self.device, weights_only=False)

        self._model.load_state_dict(state_dict)
        self._model.eval()

        # Warm-up pass
        dummy = Image.new("RGB", (224, 224), color=(34, 139, 34))
        self.predict(dummy, top_k=1)
        print(f"[model] Custom CNN loaded successfully on {self.device}.")

    def predict(self, image: Image.Image, top_k: int = 5) -> list[dict]:
        if self._model is None:
            raise ModelNotLoadedError("Call .load() before .predict()")

        img_t = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self._model(img_t)
            probs = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()

        all_preds = [
            {"label": self.class_names[i], "confidence": round(float(probs[i]), 6)}
            for i in range(self.num_classes)
        ]
        all_preds.sort(key=lambda x: x["confidence"], reverse=True)

        return all_preds[:top_k]

