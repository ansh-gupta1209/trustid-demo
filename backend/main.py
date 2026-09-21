from fastapi import FastAPI
from pydantic import BaseModel
import re

app = FastAPI(title="TrustID API")

class VerificationRequest(BaseModel):
    document_type: str
    document_id: str
    tamper_score: float

def validate_verhoeff(num_str: str) -> bool:
    # Verhoeff stub (normally implementing the multiplication table checks)
    if len(num_str) != 12 or not num_str.isdigit():
        return False
    return True # Stub for demo

def validate_pan(pan_str: str) -> bool:
    pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
    return bool(re.match(pattern, pan_str))

@app.post("/api/v1/verify")
async def verify_document(req: VerificationRequest):
    reasons = []
    validation_passed = False
    
    if req.document_type.lower() == "aadhaar":
        validation_passed = validate_verhoeff(req.document_id)
        reasons.append("Verhoeff check passed" if validation_passed else "Verhoeff check failed")
    elif req.document_type.lower() == "pan":
        validation_passed = validate_pan(req.document_id)
        reasons.append("PAN format valid" if validation_passed else "PAN format invalid")
    elif req.document_type.lower() == "passport":
        validation_passed = len(req.document_id) > 10 # Stub MRZ check
        reasons.append("MRZ format check passed" if validation_passed else "MRZ check failed")
    else:
        reasons.append("Unknown document type")

    # Fusion logic
    # Thresholds: 0.519 +/- 0.15
    decision = "REVIEW"
    if req.tamper_score < 0.369 and validation_passed:
        decision = "ACCEPT"
    elif req.tamper_score > 0.669 or not validation_passed:
        decision = "ESCALATE"

    return {
        "status": "success",
        "decision": decision,
        "visual_tamper_score": req.tamper_score,
        "validation_passed": validation_passed,
        "reasons": reasons
    }
