# Fake News Detection — Evaluation Report

## 1. Project Overview
The system classifies news text as REAL or FAKE using machine learning and natural language processing.

## 2. Test Dataset
- Held-out test samples: 1,284
- REAL class: 928 samples
- FAKE class: 356 samples

## 3. Recorded Model Results

| Metric | Result |
|---|---:|
| Accuracy | 65.97% |
| FAKE-class Precision | 38.84% |
| FAKE-class Recall | 39.61% |
| FAKE-class F1-score | 39.22% |

## 4. Confusion Matrix

| Actual / Predicted | REAL | FAKE |
|---|---:|---:|
| REAL | 706 | 222 |
| FAKE | 215 | 141 |

## 5. Interpretation
The model achieved 65.97% accuracy on the recorded held-out test set. Its FAKE-class F1-score was 39.22%, indicating limited ability to identify fake news reliably.

## 6. Limitations
- The model can misclassify real and fake news.
- The confidence-like score is not a calibrated probability.
- The application does not independently verify claims using authoritative sources.
- Predictions should not be considered definitive fact-checks.

## 7. Conclusion
The project demonstrates a machine-learning-based news classification workflow. Further work is required to improve generalization, evaluate alternative models, and validate performance on representative datasets.
