# Use official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies (Tesseract OCR and required libraries for OpenCV/Pillow)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY backend/ ./backend/

# Copy the public folder (specifically for demo_assets/model.onnx)
# Since the backend code expects the model at ../public/demo_assets/model.onnx
COPY public/ ./public/

# Expose port 8000
EXPOSE 8000

# Set the command to run the FastAPI app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
