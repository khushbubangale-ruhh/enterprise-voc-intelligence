"""Optional REST inference service: uvicorn src.api:app --host 0.0.0.0 --port 8000"""
from functools import lru_cache
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'sentiment_model.joblib'
app = FastAPI(title='Voice-of-Customer Sentiment API', version='1.0.0')

class ReviewRequest(BaseModel):
    review_text: str = Field(min_length=3, max_length=5000)

@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError('Train a model before starting inference')
    # Only load model artifacts created by this trusted project.
    return joblib.load(MODEL_PATH)

@app.get('/health')
def health():
    return {'status': 'ok' if MODEL_PATH.exists() else 'model_missing'}

@app.post('/predict')
def predict(payload: ReviewRequest):
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    label = str(model.predict([payload.review_text])[0])
    scores = model.predict_proba([payload.review_text])[0]
    return {'sentiment': label, 'scores': {str(k): round(float(v), 5) for k, v in zip(model.classes_, scores)}}
