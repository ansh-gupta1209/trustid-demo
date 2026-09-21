# TrustID - Intelligent Document Verification & Fraud Detection

TrustID is a comprehensive product designed to verify identity documents (Aadhaar, PAN, Passports) and detect potential tampering or fraud using a fusion of deterministic rules, OCR consistency checks, and visual forensic AI models.

**Note for SIH / Evaluators:** 
The visual forensics model included in this repository provides an *advisory risk signal* based on general document tampering artifacts. It has *not* been trained or evaluated on Indian datasets (Aadhaar/PAN) due to data privacy laws. Explicit Indian-document support (Aadhaar, PAN, Passport) is handled via deterministic rules, layout checks, OCR entity extraction, Verhoeff validation, and MRZ check-digit validation.

## Features & Architecture

* **Document Router:** Automatically identifies or allows user-selection of the document type (Aadhaar, PAN, Passport).
* **Deterministic Validation:**
  * **Aadhaar:** 12-digit format checks, Verhoeff algorithm validation.
  * **PAN:** Format validation (e.g., `[A-Z]{5}[0-9]{4}[A-Z]{1}`), entity checks.
  * **Passport:** MRZ extraction and ICAO check-digit validation.
* **Visual Forensics AI:** An ONNX-based deep learning model to detect general visual tampering (splicing, copy-move).
  * **Operating Policy:** 
    * `Accept`: Low tamper probability (< 0.369)
    * `Manual Review`: Uncertainty band (0.369 - 0.669)
    * `Escalate`: High tamper probability (> 0.669)
* **Evidence Fusion Layer:** Combines OCR confidence, layout rules, and visual model scores.
* **Human-in-the-Loop Review Flow:** Interface for manual reviewers to accept, review, or escalate cases based on AI confidence.

## Repository Structure

- `frontend/`: Web UI for the reviewer flow and document upload.
- `backend/`: FastAPI backend for OCR, validation rules, and evidence fusion.
- `ml/`: Scripts for model training, evaluation, and ONNX export.
- `docs/`: Architecture diagrams and API documentation.
- `tests/`: Automated test suites for validation logic.
- `sample_data/`: Placeholder for test samples (No PII or real documents).

## Setup & Deployment

1. **Frontend:** Deployed via Vercel (static site in `public/` or `frontend/`). 
2. **Backend:** Python FastAPI backend.
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```

## License
MIT License
