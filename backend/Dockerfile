FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (git for CLIP, libgl1/libglib2.0-0 for OpenCV GUI required by YOLO, wget for models)
RUN apt-get update && apt-get install -y git libgl1 libglib2.0-0 wget && rm -rf /var/lib/apt/lists/*

# Copy backend requirements first (for better Docker caching)
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Pre-download models to bake them into the Docker image (reduces startup time from 40s to 5s)
RUN mkdir -p specialist_models && \
    wget -q https://huggingface.co/Mlaikmajid1063/agriscan-models/resolve/main/wheat_efficientnetv2s_best.pth -O specialist_models/wheat_efficientnetv2s_best.pth && \
    wget -q https://huggingface.co/Mlaikmajid1063/agriscan-models/resolve/main/chilli_efficientnetv2s_best.pth -O specialist_models/chilli_efficientnetv2s_best.pth

# Expose port
EXPOSE $PORT

# Run the FastAPI server
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
