import os
import re
import math
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageEnhance, ImageFilter
import io
import numpy as np
import pytesseract
from mrz.checker.td3 import TD3CodeChecker
from mrz.checker.td2 import TD2CodeChecker
from mrz.checker.td1 import TD1CodeChecker

app = FastAPI(title="TrustID API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["*"],
)

VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]

VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]

def validate_verhoeff(num_str: str) -> bool:
    clean_str = re.sub(r'\D', '', str(num_str))
    if len(clean_str) != 12:
        return False
    c = 0
    for i, n in enumerate(reversed(clean_str)):
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][int(n)]]
    return c == 0

def clean_ocr_text(raw_text: str) -> str:
    char_map = {'O': '0', 'o': '0', 'D': '0', 'I': '1', 'l': '1', '|': '1', 'B': '8', 'S': '5', 'Z': '2'}
    res = ""
    for char in raw_text:
        res += char_map.get(char, char)
    return res

def preprocess_image_for_ocr(pil_img: Image.Image) -> list[Image.Image]:
    images = []
    gray = pil_img.convert('L')
    sharp = ImageEnhance.Sharpness(gray).enhance(2.0)
    images.append(sharp)
    
    contrast = ImageEnhance.Contrast(gray).enhance(3.0)
    fn = lambda x : 255 if x > 140 else 0
    thresh = contrast.point(fn, mode='1')
    images.append(thresh)
    
    w, h = pil_img.size
    rescaled = gray.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
    images.append(rescaled)
    return images

def extract_text_from_image(image_bytes: bytes) -> str:
    combined_text = ""
    try:
        base_image = Image.open(io.BytesIO(image_bytes))
        variants = [base_image] + preprocess_image_for_ocr(base_image)
        configs = ['--oem 3 --psm 6', '--oem 3 --psm 11', '--oem 3 --psm 3']
        for img in variants:
            for cfg in configs:
                try:
                    text = pytesseract.image_to_string(img, config=cfg)
                    combined_text += "\n" + text
                except Exception:
                    pass
    except Exception:
        pass
    return combined_text

def find_id_in_text(text: str, doc_type: str) -> str:
    cleaned = clean_ocr_text(text)
    if doc_type == "aadhaar":
        matches = re.findall(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', cleaned)
        for match in matches:
            cand = re.sub(r'\D', '', match)
            if len(cand) == 12 and validate_verhoeff(cand):
                return cand
        
        all_digits = re.sub(r'\D', '', cleaned)
        for i in range(len(all_digits) - 11):
            cand = all_digits[i:i+12]
            if validate_verhoeff(cand):
                return cand
        if len(matches) > 0:
            return re.sub(r'\D', '', matches[0])

    elif doc_type == "pan":
        matches = re.findall(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b', text.upper())
        if matches:
            return matches[0]
            
    return ""

def validate_pan(pan_str: str) -> bool:
    pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
    return bool(re.match(pattern, pan_str.upper()))

def validate_mrz(mrz_str: str) -> bool:
    try:
        if len(mrz_str) >= 88:
            return bool(TD3CodeChecker(mrz_str))
        elif len(mrz_str) >= 72:
            return bool(TD2CodeChecker(mrz_str))
        elif len(mrz_str) >= 90:
            return bool(TD1CodeChecker(mrz_str))
        return False
    except Exception:
        return len(mrz_str) > 10

def process_image(image_bytes: bytes) -> float:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((384, 384))
        img_array = np.array(image).astype(np.float32)
        gray = np.mean(img_array, axis=2)
        noise = np.std(gray) / 255.0

        def laplacian_variance(channel):
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
            from numpy.lib.stride_tricks import sliding_window_view
            padded = np.pad(channel, 1, mode='reflect')
            windows = sliding_window_view(padded, (3, 3))
            conv = np.sum(windows * kernel, axis=(-2, -1))
            return np.var(conv)

        lap_var = laplacian_variance(gray)
        sharpness = min(lap_var / 5000.0, 1.0)
        channel_stds = [np.std(img_array[:, :, c]) / 255.0 for c in range(3)]
        imbalance = np.std(channel_stds)

        raw = 0.4 * (1.0 - noise) + 0.4 * sharpness + 0.2 * (imbalance * 10)
        score = max(0.0, min(1.0, raw))
        return round(score, 4)
    except Exception:
        return 0.5

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/v1/verify")
async def verify_document(
    document_type: str = Form(...),
    document_id: str = Form(""),
    file: UploadFile = File(...)
):
    try:
        image_bytes = await file.read()
        reasons = []
        
        if not document_id:
            text = extract_text_from_image(image_bytes)
            document_id = find_id_in_text(text, document_type.lower())
            if document_id:
                reasons.append(f"OCR successfully extracted ID: {document_id}")
            else:
                reasons.append("OCR failed to extract ID. Using generic validation.")

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

        tamper_score = process_image(image_bytes)
        reasons.append(f"Visual Tamper Score calculated: {tamper_score:.4f}")

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
    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
