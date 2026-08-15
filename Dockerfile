FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for OpenCV and PyTorch
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements first (for better Docker caching)
COPY backend/requirements.txt ./backend/

# Install python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE $PORT

# Run the FastAPI server
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}
