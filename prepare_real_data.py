"""Prepare the UCI Sentiment Labelled Sentences Amazon subset.

Run: python prepare_real_data.py
Offline: python prepare_real_data.py --zip path/to/sentiment_labelled_sentences.zip
"""
import argparse
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen
import pandas as pd

ROOT = Path(__file__).resolve().parent
URL = 'https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip'


def prepare(zip_path=None):
    if zip_path:
        payload = Path(zip_path).read_bytes()
    else:
        print('Downloading the UCI Sentiment Labelled Sentences dataset...')
        with urlopen(URL, timeout=45) as response:
            payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        matches = [name for name in archive.namelist() if name.endswith('amazon_cells_labelled.txt')]
        if len(matches) != 1:
            raise ValueError('Expected one amazon_cells_labelled.txt file inside the UCI archive')
        with archive.open(matches[0]) as source:
            df = pd.read_csv(source, sep='\t', header=None, names=['review_text', 'label'])
    df['sentiment'] = df['label'].map({0: 'negative', 1: 'positive'})
    if df['sentiment'].isna().any():
        raise ValueError('Unexpected sentiment label in the dataset')
    df = df[['review_text', 'sentiment']].dropna()
    df['review_text'] = df['review_text'].astype(str).str.strip()
    df = df[df['review_text'].str.len() >= 3]
    df = df.drop_duplicates(subset=['review_text']).reset_index(drop=True)
    output = ROOT / 'data' / 'raw' / 'amazon_reviews.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f'Saved {len(df)} reviews to {output}')
    print(df['sentiment'].value_counts().to_string())
    print('Next: python -m src.train --data data/raw/amazon_reviews.csv')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', help='Optional manually downloaded UCI ZIP path')
    args = parser.parse_args()
    prepare(args.zip)
