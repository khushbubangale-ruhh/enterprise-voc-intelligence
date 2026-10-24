from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title="Voice of Customer | Portfolio",layout="wide")
st.title("💬 Voice-of-Customer Intelligence")
st.caption("Educational NLP portfolio • Evaluation results reflect the selected dataset, not a company production system")
model_path=ROOT/'models/sentiment_model.joblib'
metrics_path=ROOT/'reports/metrics.json'
if not model_path.exists() or not metrics_path.exists():
    st.error("First train the model: python -m src.train")
    st.stop()
@st.cache_resource
def get_model():
    return joblib.load(model_path)
model=get_model()
metrics=json.loads(metrics_path.read_text(encoding='utf8'))
st.info("Dataset: " + metrics['dataset'] + ". These are benchmark/demo results, not production performance. Available labels: " + ", ".join(metrics["labels"]))
a,b,c,d=st.columns(4)
a.metric("Clean reviews",metrics['rows_after_cleaning']);b.metric("Train",metrics['train_count']);c.metric("Test",metrics['test_count']);d.metric("Test macro F1",f"{metrics['macro_f1']:.2f}")
st.subheader("Model comparison (training-only cross-validation)")
st.dataframe(pd.DataFrame(metrics['cv_comparison']),hide_index=True,use_container_width=True)
st.subheader("Sentiment prediction")
review=st.text_area("Enter a customer review",value="The app is useful but the payment fails often")
if st.button("Predict") and review.strip():
    prediction=model.predict([review])[0]
    probabilities=model.predict_proba([review])[0]
    st.write("Predicted sentiment:",prediction)
    st.dataframe(pd.DataFrame({'sentiment':model.classes_,'model_probability':probabilities}),hide_index=True)
    st.caption("Probabilities are model estimates, not guaranteed confidence or correctness. Mixed reviews may be misclassified.")
st.subheader("Evaluation")
st.write("Selected model:",metrics['selected_model'])
st.write("Confusion matrix (rows = actual, columns = predicted)")
st.dataframe(pd.DataFrame(metrics['confusion_matrix'],index=metrics['labels'],columns=metrics['labels']))
st.write("Classification report")
st.dataframe(pd.DataFrame(metrics['classification_report']).T)
st.subheader("Error analysis")
errors=pd.read_csv(ROOT/'reports/test_predictions.csv')
st.dataframe(errors[errors.actual != errors.predicted],use_container_width=True,hide_index=True)
st.subheader("Dataset exploration")
df=pd.read_csv(ROOT/'data/processed/clean_reviews.csv')
st.bar_chart(df.sentiment.value_counts())
if 'category' in df.columns:
    st.bar_chart(pd.crosstab(df.category,df.sentiment))
if 'review_date' in df.columns:
    dates=pd.to_datetime(df.review_date,errors='coerce')
    if dates.notna().any():
        chart=df.assign(month=dates.dt.to_period('M').astype(str)).groupby(['month','sentiment']).size().unstack(fill_value=0)
        st.line_chart(chart)
st.caption("Do not upload private customer data without permission. Load joblib files only from trusted sources.")
