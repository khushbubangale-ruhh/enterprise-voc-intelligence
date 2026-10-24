from pathlib import Path
import pandas as pd
import pytest
from src.pipeline import load_reviews, candidate, train

ROOT = Path(__file__).resolve().parents[1]

def test_demo_data_valid():
    df = load_reviews(ROOT / 'data/raw/demo_reviews.csv')
    assert len(df) >= 20
    assert set(df.sentiment) == {'positive', 'negative', 'neutral'}

def test_prediction_pipeline():
    df = load_reviews(ROOT / 'data/raw/demo_reviews.csv')
    model = candidate('Logistic Regression').fit(df.review_text, df.sentiment)
    assert model.predict(['Customer service was helpful']).shape == (1,)

def test_conflicting_labels_rejected(tmp_path):
    data = pd.DataFrame({'review_text':['same review']*2 + ['great work']*4 + ['poor service']*4,
                         'sentiment':['positive','negative']+['positive']*4+['negative']*4})
    p = tmp_path / 'conflict.csv'; data.to_csv(p,index=False)
    with pytest.raises(ValueError, match='Conflicting'):
        load_reviews(p)

def test_train_writes_artifacts(tmp_path):
    result = train(ROOT / 'data/raw/demo_reviews.csv', output=tmp_path)
    assert 0 <= result['macro_f1'] <= 1
    assert (tmp_path / 'models/sentiment_model.joblib').exists()
    assert (tmp_path / 'reports/metrics.json').exists()
