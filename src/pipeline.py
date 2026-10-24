from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"review_text", "sentiment"}
LABELS = {"positive", "neutral", "negative"}

def load_reviews(path):
    df = pd.read_csv(path)
    if not REQUIRED.issubset(df.columns):
        raise ValueError("CSV needs review_text and sentiment columns")
    df = df.dropna(subset=["review_text", "sentiment"]).copy()
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df = df[(df.review_text.str.len() >= 3) & df.sentiment.isin(LABELS)]
    df["dedup_key"] = df.review_text.str.casefold().str.replace(r"\s+", " ", regex=True)
    conflicts = df.groupby("dedup_key").sentiment.nunique()
    if (conflicts > 1).any():
        raise ValueError("Conflicting sentiment labels for identical review text; resolve them first")
    df = df.drop_duplicates("dedup_key").drop(columns="dedup_key")
    if df.sentiment.nunique() < 2 or df.sentiment.value_counts().min() < 4:
        raise ValueError("Need at least two classes with four distinct reviews per class")
    return df

def candidate(name):
    classifier = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42) if name == "Logistic Regression" else MultinomialNB()
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=20000)), ("classifier", classifier)])

def train(csv_path, output=ROOT):
    output = Path(output)
    df = load_reviews(csv_path)
    # Split before TF-IDF fitting to prevent vocabulary/IDF leakage.
    x_train, x_test, y_train, y_test = train_test_split(df.review_text, df.sentiment, test_size=0.25, stratify=df.sentiment, random_state=42)
    nfolds = min(3, int(y_train.value_counts().min()))
    if nfolds < 2:
        raise ValueError("Insufficient examples per class in training split")
    comparison=[]
    for name in ["Logistic Regression", "Multinomial Naive Bayes"]:
        scores = cross_validate(candidate(name), x_train, y_train, cv=StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42), scoring="f1_macro")
        comparison.append({"model":name,"cv_macro_f1_mean":float(scores['test_score'].mean()),"cv_macro_f1_std":float(scores['test_score'].std())})
    comparison.sort(key=lambda r:r['cv_macro_f1_mean'], reverse=True)
    best = candidate(comparison[0]['model']).fit(x_train,y_train)
    preds=best.predict(x_test)
    labels=sorted(y_test.unique().tolist())
    report={"dataset":"PUBLIC OR USER-SUPPLIED DATA (verify provenance)" if Path(csv_path).resolve()!= (ROOT/'data/raw/demo_reviews.csv').resolve() else "SMALL HAND-AUTHORED DEMONSTRATION", "source_file":str(csv_path), "rows_after_cleaning":int(len(df)),"train_count":len(x_train),"test_count":len(x_test),"selected_model":comparison[0]['model'],"cv_comparison":comparison,"accuracy":float(accuracy_score(y_test,preds)),"macro_f1":float(f1_score(y_test,preds,average='macro')),"labels":labels,"confusion_matrix":confusion_matrix(y_test,preds,labels=labels).tolist(),"classification_report":classification_report(y_test,preds,output_dict=True,zero_division=0),"limitations":"Small demonstration data is not evidence of production accuracy. Check representativeness, labeling, data leakage, and out-of-domain examples."}
    (output/'models').mkdir(parents=True,exist_ok=True); (output/'reports').mkdir(parents=True,exist_ok=True);(output/'data/processed').mkdir(parents=True,exist_ok=True)
    joblib.dump(best,output/'models/sentiment_model.joblib')
    (output/'reports/metrics.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    pd.DataFrame({"review_text":x_test,"actual":y_test,"predicted":preds}).to_csv(output/'reports/test_predictions.csv',index=False)
    df.to_csv(output/'data/processed/clean_reviews.csv',index=False)
    return report
