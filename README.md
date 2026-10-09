# Fake News Detection System

## About the Project
This project uses machine learning and natural language processing (NLP) to classify news text as REAL or FAKE. It provides predictions through a Flask web application.

## Features
- Real and fake news classification
- TF-IDF text features
- Word2Vec and auxiliary text features
- Flask web interface and prediction API
- Input validation and error handling
- Automated testing using pytest

## Technologies Used
- Python
- Flask
- Machine Learning
- Natural Language Processing (NLP)
- Scikit-learn
- TF-IDF and Word2Vec
- HTML and CSS
- Pytest

## Project Structure
- `app.py` — Main Flask application
- `templates/` — Web interface
- `models/` — Trained model files
- `tests/` — Automated tests
- `explainability/` — Model explanation outputs
- `data/` — Dataset documentation

## How to Run
1. Install dependencies:
   `python -m pip install -r requirements.txt`
2. Start the application:
   `python app.py`
3. Open `http://127.0.0.1:5000` in your browser.

## API Endpoints
- `/` — Home page
- `/health` — Application health check
- `/predict` — News prediction endpoint

## Testing
Run the automated tests using:

`python -m pytest -v`

The test suite covers the home page, health endpoint, invalid requests, missing or empty text, text length limits, and valid predictions.

## Model Evaluation
Recorded held-out test results:
- Accuracy: 65.97%
- FAKE-class precision: 38.84%
- FAKE-class recall: 39.61%
- FAKE-class F1-score: 39.22%

These results indicate that the model has limitations and may misclassify news.

## Limitations
The confidence-like score is not a calibrated probability. The model does not independently verify news against authoritative sources. Users should verify important claims through reliable sources.

## Disclaimer
This project is intended for educational purposes. Its predictions should not be treated as definitive proof that a news article is true or false.
