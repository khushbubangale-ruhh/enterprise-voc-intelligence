$ErrorActionPreference = 'Stop'
if (-not (Test-Path '.venv')) { py -3.11 -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m src.train --data data/raw/demo_reviews.csv
& .\.venv\Scripts\python.exe -m streamlit run app.py
