# Use official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies
# - tesseract-ocr   → OCR engine
# - libgl1          → graphics libraries for OpenCV/Pillow
# - execstack       → utility to clear the executable‑stack flag on the onnxruntime .so file
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    execstack \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Clear the executable‑stack flag on any onnxruntime shared objects
# This fixes the RuntimeError: cannot enable executable stack as shared object requires
RUN python - <<'PY'
import os, glob, onnxruntime
capi_dir = os.path.join(os.path.dirname(onnxruntime.__file__), 'capi')
for so in glob.glob(os.path.join(capi_dir, '*.so')):
    os.system(f'execstack -c {so}')
PY

# Copy backend code and public assets (including model)
COPY backend/ ./backend/
COPY public/ ./public/

# Render provides the PORT environment variable; expose a placeholder port
EXPOSE 10000

# Start FastAPI using Render's PORT variable
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
