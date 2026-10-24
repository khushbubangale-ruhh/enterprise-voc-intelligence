from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="VOC Intelligence | Insights", page_icon="💠", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stApp"] {font-family:'Inter',sans-serif;}
[data-testid="stApp"] {background:linear-gradient(145deg,#090e22 0%,#111a35 55%,#0d1630 100%);color:#e8eeff;}
[data-testid="stSidebar"] {background:#0b132b;border-right:1px solid #293858;}
[data-testid="stHeader"] {background:transparent;}
h1,h2,h3 {color:#f4f6ff !important;letter-spacing:-.025em;}
p, label {color:#cbd5e1;}
.hero {background:linear-gradient(110deg,#4f46e5,#7c3aed 58%,#0e7490);padding:32px;border-radius:22px;margin-bottom:22px;box-shadow:0 18px 42px #0004;}
.hero h1 {margin:0;color:white!important;font-size:2.1rem;}
.hero p {color:#e0e7ff;margin:8px 0 0;}
.kpi {border:1px solid #35446a;background:linear-gradient(135deg,#1a2747,#141d36);border-radius:16px;padding:20px 20px 16px;margin-bottom:12px;}
.kpi-label {font-size:.83rem;color:#b5c2dd;}
.kpi-value {font-size:2rem;font-weight:800;color:white;margin-top:5px;}
.kpi-foot {font-size:.74rem;color:#a5b4fc;margin-top:4px;}
.section-note {color:#aebbd6;font-size:.92rem;margin-bottom:12px;}
[data-testid="stDataFrame"], [data-testid="stChart"] {border-radius:14px;overflow:hidden;}
div.stButton > button[kind="primary"] {background:linear-gradient(90deg,#6366f1,#8b5cf6);border:none;color:white;font-weight:700;border-radius:10px;}
[data-testid="stAlert"] {border-radius:12px;}
</style>""", unsafe_allow_html=True)

model_path = ROOT / 'models/sentiment_model.joblib'
metrics_path = ROOT / 'reports/metrics.json'
if not model_path.exists() or not metrics_path.exists():
    st.error('Model not trained yet. Run: python -m src.train --data data/raw/demo_reviews.csv')
    st.stop()

@st.cache_resource
def load_model(path, mtime):
    return joblib.load(path)

model = load_model(str(model_path), model_path.stat().st_mtime)
metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
labels = metrics.get('labels', [])
dataset = metrics.get('dataset', 'Unknown dataset')

st.sidebar.markdown('## 💠 VOC INTELLIGENCE')
st.sidebar.caption('Customer Experience Analytics')
page = st.sidebar.radio('Workspace', ['📊 Executive overview', '🧠 Sentiment studio', '📈 Model performance', '🗂️ Review explorer'])
st.sidebar.divider()
st.sidebar.caption('MODEL STATUS')
st.sidebar.success('Model ready')
st.sidebar.caption(f'Dataset: {dataset}')
st.sidebar.caption('Educational / portfolio deployment')

st.markdown('<div class="hero"><h1>Customer Voice, Clear Insights.</h1><p>Explore customer sentiment, discover friction points and inspect model quality in one workspace.</p></div>', unsafe_allow_html=True)

def kpi(label, value, foot):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-foot">{foot}</div></div>', unsafe_allow_html=True)

if page == '📊 Executive overview':
    st.subheader('Executive overview')
    cols = st.columns(4)
    entries = [('Reviews', metrics.get('rows_after_cleaning', '—'), 'After cleaning'), ('Training rows', metrics.get('train_count', '—'), 'Model training'), ('Test rows', metrics.get('test_count', '—'), 'Held-out evaluation'), ('Macro F1', f"{metrics.get('macro_f1', 0):.2f}", 'Held-out score')]
    for col, (label, val, foot) in zip(cols, entries):
        with col: kpi(label, val, foot)
    st.info(f'Current dataset: {dataset}. Scores are specific to this dataset and are not evidence of production accuracy.')
    data_path = ROOT / 'data/processed/clean_reviews.csv'
    if data_path.exists():
        df = pd.read_csv(data_path)
        left, right = st.columns(2)
        with left:
            st.markdown('#### Sentiment distribution')
            st.bar_chart(df['sentiment'].value_counts().rename_axis('sentiment').rename('reviews'))
        with right:
            st.markdown('#### Category breakdown')
            if 'category' in df.columns:
                st.bar_chart(pd.crosstab(df['category'].fillna('Unknown'), df['sentiment']))
            else:
                st.caption('No category field in the selected dataset. This is expected for Amazon labelled sentences.')
        if 'review_date' in df.columns:
            dates = pd.to_datetime(df['review_date'], errors='coerce')
            if dates.notna().any():
                st.markdown('#### Sentiment over time')
                monthly = df.loc[dates.notna()].assign(month=dates[dates.notna()].dt.to_period('M').astype(str)).groupby(['month','sentiment']).size().unstack(fill_value=0)
                st.line_chart(monthly)
            else: st.caption('No valid dates available for a trend chart.')
        else: st.caption('Review dates are not available in this dataset; no time trend is inferred.')

elif page == '🧠 Sentiment studio':
    st.subheader('Sentiment prediction studio')
    st.markdown('<div class="section-note">Try realistic feedback. Predictions are model estimates and can be wrong, especially for mixed sentiment.</div>', unsafe_allow_html=True)
    with st.form('predict_form'):
        review = st.text_area('Customer review', value='The app is useful but the payment fails often', height=130, max_chars=4000)
        submitted = st.form_submit_button('✨ Analyze sentiment', type='primary', use_container_width=True)
    if submitted:
        if not review.strip(): st.warning('Please enter a customer review.')
        else:
            prediction = model.predict([review])[0]
            probabilities = model.predict_proba([review])[0]
            st.success(f'Predicted sentiment: **{str(prediction).upper()}**')
            proba_df = pd.DataFrame({'Sentiment':model.classes_, 'Estimated probability':probabilities}).set_index('Sentiment')
            st.bar_chart(proba_df)
            st.caption('Estimated probabilities are not calibrated confidence guarantees. Neutral may not be available for a binary dataset.')

elif page == '📈 Model performance':
    st.subheader('Model quality & evaluation')
    st.caption(f"Selected model: {metrics.get('selected_model', 'Unknown')}")
    st.markdown('#### Training-only cross-validation comparison')
    st.dataframe(pd.DataFrame(metrics.get('cv_comparison', [])), use_container_width=True, hide_index=True)
    st.markdown('#### Confusion matrix')
    cm = metrics.get('confusion_matrix', [])
    if cm and len(cm) == len(labels):
        st.dataframe(pd.DataFrame(cm, index=labels, columns=labels), use_container_width=True)
        st.caption('Rows are actual labels; columns are predicted labels.')
    st.markdown('#### Classification report')
    st.dataframe(pd.DataFrame(metrics.get('classification_report', {})).T, use_container_width=True)
    st.markdown('#### Misclassified test reviews')
    errors_path = ROOT / 'reports/test_predictions.csv'
    if errors_path.exists():
        errors = pd.read_csv(errors_path)
        if {'actual','predicted'}.issubset(errors.columns):
            wrong = errors[errors['actual'] != errors['predicted']]
            if wrong.empty: st.success('No errors in this test split. This does not guarantee generalization.')
            else: st.dataframe(wrong, use_container_width=True, hide_index=True)

else:
    st.subheader('Review explorer')
    data_path = ROOT / 'data/processed/clean_reviews.csv'
    if not data_path.exists(): st.warning('Processed review data not found.')
    else:
        df = pd.read_csv(data_path)
        selection = st.selectbox('Filter by sentiment', ['All'] + sorted(df['sentiment'].dropna().astype(str).unique().tolist()))
        if selection != 'All': df = df[df['sentiment'] == selection]
        search = st.text_input('Search review text', placeholder='e.g. delivery, refund, support')
        if search and 'review' in df.columns: df = df[df['review'].astype(str).str.contains(search, case=False, regex=False, na=False)]
        st.caption(f'{len(df)} reviews matched')
        st.dataframe(df.head(500), use_container_width=True, hide_index=True)
        st.download_button('Download filtered CSV', df.to_csv(index=False).encode('utf-8'), 'filtered_reviews.csv', 'text/csv')

st.divider()
st.caption('Portfolio demonstration • Do not upload private customer data without permission • Load only trusted model files')
