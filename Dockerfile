# Use Debian Bullseye base (has patchelf)
FROM python:3.10-slim-bullseye

# Install system dependencies
# - tesseract-ocr : OCR engine
# - libgl1      : graphics libs for Pillow/OpenCV
# - patchelf   : clears the executable‑stack flag on onnxruntime .so
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Clear the executable‑stack flag on any onnxruntime shared objects using patchelf
RUN python - <<'PY'
import os, glob, onnxruntime
capi_dir = os.path.join(os.path.dirname(onnxruntime.__file__), 'capi')
for so in glob.glob(os.path.join(capi_dir, '*.so')):
    os.system(f'patchelf --clear-execstack {so}')
PY

# Copy application code and public assets (including the ONNX model)
COPY backend/ ./backend/
COPY public/ ./public/

# Expose placeholder port (Render will provide $PORT at runtime)
EXPOSE 10000

# Start FastAPI using the Render‑provided PORT variable
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
