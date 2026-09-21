# Model card - trustid-sidtd-mnv3s-v1

**Task:** binary classification of a document image: genuine (0) vs tampered (1).
**Architecture:** MobileNetV3-Small (ImageNet init, full fine-tune); input [1, 3, 384, 384]; output = P(tampered).
**Training data:** SIDTD (synthetic European ID/travel documents; not Indian documents). Split by base document (all forged variants of a document stay in one
partition), stratified by document type. Counts: train 1548, val 324,
test 350.
**Threshold:** 0.519 (chosen on validation only). Manual-review band: +/-0.15.
**Model file SHA-256:** `15f7e42c1e10c07e97460a3732012deec1e647f5e7d4e089f8673a8accc54138`

## Locked test-set results (measured on the exported ONNX file)
Test: 150 genuine / 200 tampered. A model that always says "tampered" scores
57.1% accuracy, so accuracy alone is not meaningful here.

| Metric (no review band) | Value | 95% CI |
|---|---|---|
| Precision | 88.4% | |
| Recall (tampered caught) | 65.0% | 58.2-71.3% |
| F1 | 74.9% | |
| False-positive rate (genuine flagged) | 11.3% | 7.2-17.4% |
| ROC AUC | 0.870 | |

Confusion matrix (rows = true genuine/tampered; cols = predicted genuine/tampered): [[133, 17], [70, 130]]

With the manual-review band: 58.0% of test images sent to manual review;
on the remainder precision 98.2%, recall 80.6%, FPR 1.2%.

## Per document type (no review band)
| Type | Genuine | Tampered | Recall | FPR |
|---|---|---|---|---|
| alb_id | 15 | 20 | 80.0% | 13.3% |
| aze_passport | 15 | 20 | 45.0% | 13.3% |
| esp_id | 15 | 23 | 100.0% | 0.0% |
| est_id | 15 | 20 | 80.0% | 0.0% |
| fin_id | 15 | 18 | 100.0% | 0.0% |
| grc_passport | 15 | 14 | 85.7% | 6.7% |
| lva_passport | 15 | 20 | 50.0% | 26.7% |
| rus_internalpassport | 15 | 26 | 26.9% | 20.0% |
| srb_passport | 15 | 24 | 29.2% | 6.7% |
| svk_id | 15 | 15 | 80.0% | 26.7% |

## Limitations (state these on the slide)
- Trained on **synthetic European** ID/travel documents. It has **not** been trained or evaluated on Aadhaar, PAN or Indian passports.
- Held-out split is by base document, **not** by template: every document type also appears in training.
- Small test set, especially genuine images; see the confidence intervals.
- Forgeries are digital crop-and-replace / inpainting; results do not transfer automatically to real-world fraud.
- JPEG normalisation used during training/eval: no.
- Browser image scaling differs slightly from the Python preprocessing used here.
