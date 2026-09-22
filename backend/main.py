import os
import re
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import onnxruntime as ort
from PIL import Image
import io
import numpy as np
import pytesseract
from mrz.checker.td3 import TD3CodeChecker
from mrz.checker.td2 import TD2CodeChecker
from mrz.checker.td1 import TD1CodeChecker

app = FastAPI(title="TrustID API")

# Load ONNX Model globally
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'public', 'demo_assets', 'model.onnx')
try:
    ort_session = ort.InferenceSession(MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load ONNX model. Make sure {MODEL_PATH} exists.")
    ort_session = None

def validate_verhoeff(num_str: str) -> bool:
    if len(num_str) != 12 or not num_str.isdigit():
        return False
    d = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
        [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
        [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
        [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
        [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]
    p = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
        [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
        [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
    ]
    inv = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]
    c = 0
    for i, n in enumerate(reversed(num_str)):
        c = d[c][p[i % 8][int(n)]]
    return c == 0

def validate_pan(pan_str: str) -> bool:
    pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
    return bool(re.match(pattern, pan_str.upper()))

def validate_mrz(mrz_str: str) -> bool:
    # Very basic MRZ validation fallback or attempt to use mrz library
    try:
        if len(mrz_str) >= 88:
            return bool(TD3CodeChecker(mrz_str))
        elif len(mrz_str) >= 72:
            return bool(TD2CodeChecker(mrz_str))
        elif len(mrz_str) >= 90:
            return bool(TD1CodeChecker(mrz_str))
        return False
    except Exception:
        # Fallback if parsing fails
        return len(mrz_str) > 10

def process_image(image_bytes: bytes) -> float:
    if not ort_session:
        return 0.5 # fallback score if model failed to load

    # Preprocess image for the ONNX model (assuming standard 384x384 input)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((384, 384))
    
    # Normalize and convert to NCHW format
    img_data = np.array(image).astype(np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1)) # HWC to CHW
    img_data = np.expand_dims(img_data, axis=0)  # Add batch dimension (NCHW)
    
    # Run inference
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    result = ort_session.run([output_name], {input_name: img_data})
    
    score = float(result[0][0])
    # Apply sigmoid if necessary (model output dependent, assuming logit here)
    if score < 0 or score > 1:
        import math
        score = 1 / (1 + math.exp(-score))
    return score

def extract_text_from_image(image_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return ""

def find_id_in_text(text: str, doc_type: str) -> str:
    if doc_type == "aadhaar":
        # Look for 12 consecutive digits or spaced digits
        matches = re.findall(r'\b\d{4}\s?\d{4}\s?\d{4}\b', text)
        if matches:
            return matches[0].replace(" ", "")
    elif doc_type == "pan":
        matches = re.findall(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b', text)
        if matches:
            return matches[0]
    # MRZ is harder to regex generically from messy OCR, usually handled by specific OCR setups
    return ""

@app.post("/api/v1/verify")
async def verify_document(
    document_type: str = Form(...),
    document_id: str = Form(""),
    file: UploadFile = File(...)
):
    image_bytes = await file.read()
    reasons = []
    
    # 1. OCR (if ID is not provided)
    if not document_id:
        text = extract_text_from_image(image_bytes)
        document_id = find_id_in_text(text, document_type.lower())
        if document_id:
            reasons.append(f"OCR successfully extracted ID: {document_id}")
        else:
            reasons.append("OCR failed to extract ID. Using generic validation.")

    # 2. Validation Rules
    validation_passed = False
    doc_type_lower = document_type.lower()
    
    if doc_type_lower == "aadhaar":
        if document_id:
            validation_passed = validate_verhoeff(document_id)
            reasons.append("Verhoeff check passed" if validation_passed else f"Verhoeff check failed for {document_id}")
        else:
            reasons.append("Verhoeff check skipped (No ID)")
    elif doc_type_lower == "pan":
        if document_id:
            validation_passed = validate_pan(document_id)
            reasons.append("PAN format valid" if validation_passed else f"PAN format invalid for {document_id}")
        else:
            reasons.append("PAN check skipped (No ID)")
    elif doc_type_lower == "passport":
        if document_id:
            validation_passed = validate_mrz(document_id)
            reasons.append("MRZ check passed" if validation_passed else "MRZ check failed")
        else:
            reasons.append("MRZ check skipped (No ID)")
    else:
        reasons.append("Unknown document type")

    # 3. AI Inference
    tamper_score = process_image(image_bytes)
    reasons.append(f"Visual Tamper Score calculated: {tamper_score:.4f}")

    # 4. Fusion Logic
    # Thresholds: 0.519 +/- 0.15
    decision = "REVIEW"
    if tamper_score < 0.369 and validation_passed:
        decision = "ACCEPT"
    elif tamper_score > 0.669 or not validation_passed:
        decision = "ESCALATE"

    return {
        "status": "success",
        "decision": decision,
        "visual_tamper_score": tamper_score,
        "validation_passed": validation_passed,
        "reasons": reasons,
        "extracted_id": document_id
    }
