FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (git for CLIP, libgl1/libglib2.0-0 for OpenCV GUI required by YOLO)
RUN apt-get update && apt-get install -y git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy backend requirements first (for better Docker caching)
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE $PORT

# Run the FastAPI server
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
