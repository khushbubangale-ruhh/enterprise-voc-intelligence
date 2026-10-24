
"""Streamlit Cloud entrypoint: prepare public data and train model."""

from pathlib import Path
import runpy

from prepare_real_data import prepare
from src.pipeline import train, ROOT

model_path = ROOT / "models" / "sentiment_model.joblib"
metrics_path = ROOT / "reports" / "metrics.json"
data_path = ROOT / "data" / "raw" / "amazon_reviews.csv"

# Download the public Amazon review dataset if missing.
if not data_path.exists():
    prepare()

# Train when model or metrics are missing.
if not model_path.exists() or not metrics_path.exists():
    train(data_path)

# Launch the Streamlit dashboard.
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
