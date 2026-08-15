FROM python:3.10-slim

# Set working directory
WORKDIR /app

# No system dependencies needed because we use opencv-python-headless

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
