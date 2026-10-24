"""Train the sentiment classifier using an explicit CSV input."""
import argparse
from .pipeline import train, ROOT

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate Voice-of-Customer sentiment models')
    parser.add_argument('--data', default=str(ROOT / 'data/raw/demo_reviews.csv'))
    args = parser.parse_args()
    metrics = train(args.data)
    print(f"Dataset: {metrics['dataset']}")
    print(f"Rows: {metrics['rows_after_cleaning']} | train: {metrics['train_count']} | test: {metrics['test_count']}")
    print(f"Selected: {metrics['selected_model']} | test macro F1: {metrics['macro_f1']:.3f}")
    print('Saved model to models/sentiment_model.joblib and reports to reports/')
