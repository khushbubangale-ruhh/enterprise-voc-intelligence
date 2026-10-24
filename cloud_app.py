"""Streamlit Community Cloud entrypoint: trains a trusted local artifact if missing."""
from pathlib import Path
import runpy
from src.pipeline import train, ROOT

if not (ROOT / 'models/sentiment_model.joblib').exists() or not (ROOT / 'reports/metrics.json').exists():
    public = ROOT / 'data/raw/amazon_reviews.csv'
    source = public if public.exists() else ROOT / 'data/raw/demo_reviews.csv'
    train(source)
runpy.run_path(str(ROOT / 'app.py'), run_name='__main__')
