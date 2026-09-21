# TrustID API Documentation

## `POST /api/v1/verify`
Evaluates a document based on type and visual heuristics.

**Request:**
```json
{
  "document_type": "aadhaar",
  "document_text": "extracted text from frontend OCR",
  "document_id": "123456789012"
}
```

**Response:**
```json
{
  "status": "success",
  "decision": "MANUAL_REVIEW",
  "visual_tamper_score": 0.55,
  "validation_passed": true,
  "reasons": [
    "Verhoeff validation passed",
    "Visual tamper score in uncertainty band (0.369 - 0.669)"
  ]
}
```
