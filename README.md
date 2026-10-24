# Voice-of-Customer Intelligence — End-to-End NLP Portfolio

**Scope:** Industry-style project structure and engineering practices; not a production-validated sentiment model. All bundled reviews are synthetic. Public real reviews can be downloaded separately from UCI. This is a **binary** (positive/negative) task on UCI Amazon reviews, and a **three-class** task on the bundled demo data. No neutral labels are fabricated.

## Architecture

`CSV -> schema validation/deduplication -> stratified train/test split -> train-only TF-IDF + classifier -> train-only CV model selection -> holdout metrics and error analysis -> trusted joblib model -> Streamlit UI / FastAPI inference`

The pipeline prevents vocabulary leakage by fitting TF-IDF only within training folds. It detects duplicate texts with conflicting labels, produces a confusion matrix and classification report, and keeps the holdout separate from model selection. A single 75/25 split on a tiny dataset is unstable; use a larger labelled dataset for meaningful assessment.

## Files

- `data/raw/demo_reviews.csv`: 30 **synthetic** labelled reviews, included for an immediate offline run
- `prepare_real_data.py`: downloads and prepares UCI Amazon 1,000 labelled review sentences
- `src/pipeline.py`: validation, model training, comparison, evaluation, artifact creation
- `src/train.py`: training command
- `src/api.py`: FastAPI prediction service
- `app.py`: Streamlit analytics and prediction dashboard
- `tests/`: validation, prediction, artifact tests
- `.github/workflows/ci.yml`: GitHub Actions continuous integration
- `Dockerfile`: reproducible dashboard container
- `models/`, `reports/`, `data/processed/`: created by training

## 1. Quick start on Windows PowerShell (Python 3.11 recommended)

Open VS Code terminal **inside the extracted project folder**, then run each command:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.train --data data/raw/demo_reviews.csv
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If `py -3.11` is unavailable but `python --version` shows Python 3.10–3.12, use `python -m venv .venv` instead. Commands use the venv's Python directly, so PowerShell activation is not required. Open http://localhost:8501.

## 2. Train on real public labelled Amazon reviews

Requires internet access for first download. The UCI dataset is **Sentiment Labelled Sentences**, source: https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences. The Amazon subset contains 1,000 positive/negative examples. Check licensing and attribution before redistribution.

```powershell
.\.venv\Scripts\python.exe prepare_real_data.py
.\.venv\Scripts\python.exe -m src.train --data data/raw/amazon_reviews.csv
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If download is blocked, download `sentiment+labelled+sentences.zip` manually from UCI and run:

```powershell
.\.venv\Scripts\python.exe prepare_real_data.py --zip "C:\path\to\sentiment+labelled+sentences.zip"
```

After retraining, **stop and restart Streamlit** to ensure its cached model refreshes. Dataset schema for your own CSV: required columns `review_text,sentiment`; labels must be `negative`, `neutral`, or `positive`; at least two classes and four distinct texts per class. Optional `category` and `review_date` columns enable extra charts. Do not invent category/date metadata for the Amazon data.

## 3. Evaluation outputs

- `reports/metrics.json`: CV scores, selected algorithm, held-out test accuracy, macro F1, classification report and confusion matrix
- `reports/test_predictions.csv`: test examples with actual/predicted labels, usable for error analysis
- `data/processed/clean_reviews.csv`: cleaned dataset
- `models/sentiment_model.joblib`: trusted locally trained pipeline

Scores are dataset-specific and do not imply enterprise production accuracy. Inspect mixed reviews, negation, sarcasm, class balance, duplicates, data provenance and domain shift.

## 4. Optional prediction API

After training:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

API docs: http://127.0.0.1:8000/docs

PowerShell example in another terminal:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body '{"review_text":"The delivery was quick and customer service was excellent"}'
```

Do not expose this API publicly without authentication, rate limits, logging, and privacy controls.

## 5. Deploy free-tier-style to Streamlit Community Cloud

1. Create a GitHub repository and upload this project's **source** files (including `.streamlit`, `requirements.txt`, `data/raw/demo_reviews.csv`, `src/`, `tests/`, and `app.py`). Do not upload `.venv`, secrets, private customer data, or arbitrary downloaded joblib models.
2. To deploy **with the real UCI dataset**, run the preparation and training commands locally, then commit the public `data/raw/amazon_reviews.csv` if licensing permits and the file size is acceptable. The model is ignored by `.gitignore`, so deployment needs to train at startup or use a trusted published model artifact.
3. The easiest repeatable approach is to add a **bootstrap launcher** as the Streamlit entry point: `cloud_app.py`. It trains from `data/raw/amazon_reviews.csv` when present, otherwise from the bundled synthetic demo. Training may take time at first boot and on restarts.
4. Go to https://share.streamlit.io/, sign in with GitHub, select your repository, branch, and entrypoint `cloud_app.py`, then deploy. Use the app's logs to diagnose dependency or training errors.
5. The cloud app URL can be added to your resume, explicitly describing whether it runs on synthetic or real public data. Free hosting has resource/sleep limits; consult current provider terms.

## 6. Deploy with Docker locally

Install Docker Desktop, then from the project folder:

```powershell
docker build -t voc-intelligence:demo .
docker run --rm -p 8501:8501 voc-intelligence:demo
```

Open http://localhost:8501. The supplied Dockerfile trains on **synthetic demo** data at build time. For real-data Docker deployment, edit the `RUN python -m src.train ...` line to use the prepared `amazon_reviews.csv`, make sure that CSV is included in the build context, and rebuild.

## 7. Suggested production hardening (not implemented)

Real production would also need authorized customer-data ingestion, PII minimization/redaction, role-based access, encryption, monitoring, model/version registry, drift alerts, scheduled retraining, labeling QA, a larger representative dataset, robust out-of-domain tests, and a deployment approval process. The included API is for local demonstrations only.

## Interview explanation

"I built a modular NLP pipeline that validates customer-review data, compares TF-IDF + Logistic Regression with Multinomial Naive Bayes using stratified cross-validation, evaluates the selected model on held-out reviews, and exposes results through Streamlit and FastAPI. I used synthetic data for offline testing and the public UCI Amazon dataset for a realistic binary sentiment benchmark. I would validate generalization and data governance before production deployment."


## Attractive dashboard
The default `app.py` now provides a dark executive dashboard with four workspaces. The previous dashboard is preserved in `app_classic.py`. Run `python -m streamlit run app.py` after training.
