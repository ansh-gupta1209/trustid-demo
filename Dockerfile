# Use Debian Bullseye (has execstack package) as base
FROM python:3.10-slim-bullseye

# Install system dependencies
# - tesseract-ocr : OCR engine
# - libgl1      : graphics libs for Pillow/OpenCV
# - execstack   : utility to clear the executable‑stack flag on onnxruntime .so
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    execstack \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python deps
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Clear the executable‑stack flag on any onnxruntime shared objects
RUN python - <<'PY'
import os, glob, onnxruntime
capi_dir = os.path.join(os.path.dirname(onnxruntime.__file__), 'capi')
for so in glob.glob(os.path.join(capi_dir, '*.so')):
    os.system(f'execstack -c {so}')
PY

# Copy backend code and public assets (including model)
COPY backend/ ./backend/
COPY public/ ./public/

# Expose placeholder port (Render will set $PORT at runtime)
EXPOSE 10000

# Start FastAPI using Render's PORT variable
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
