from pathlib import Path
import html
import json
import joblib
import pandas as pd
import altair as alt
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="VOC | Customer Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {color-scheme:light;}
html, body, [data-testid="stApp"] {font-family:'DM Sans',sans-serif;}
[data-testid="stApp"] {background:#F5F7FA;color:#172B4D;}
[data-testid="stHeader"] {background:#F5F7FA;}
[data-testid="stSidebar"] {background:#FFFFFF;border-right:1px solid #E5EAF1;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#53657D;}
.block-container {max-width:1450px;padding-top:1.9rem;padding-bottom:3rem;}
h1,h2,h3 {font-family:'Manrope',sans-serif!important;color:#16263D!important;letter-spacing:-.035em!important;}
h1 {font-size:2.1rem!important;} h2 {font-size:1.4rem!important;} h3 {font-size:1.1rem!important;}
[data-testid="stSidebar"] .stRadio label {border-radius:9px;padding:7px 8px;}
[data-testid="stSidebar"] .stRadio label:hover {background:#F0F5FA;}
[data-testid="stMetric"] {background:white;border:1px solid #E5EAF1;border-radius:14px;padding:18px;}
[data-testid="stDataFrame"] {border:1px solid #E5EAF1;border-radius:12px;overflow:hidden;}
[data-testid="stForm"] {background:#FFFFFF;border:1px solid #E5EAF1;border-radius:16px;padding:22px;}
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"] {background:#145C76!important;border-color:#145C76!important;color:white!important;border-radius:9px!important;font-weight:700!important;}
.stButton>button,.stDownloadButton>button {border-radius:9px!important;}
[data-testid="stAlert"] {border-radius:11px;}
.voc-eyebrow {font-size:.74rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;color:#44738A;margin-bottom:6px;}
.voc-hero {background:#FFFFFF;border:1px solid #E5EAF1;border-radius:18px;padding:28px 30px 23px;margin-bottom:22px;box-shadow:0 5px 24px rgba(24,46,76,.035);}
.voc-hero h1 {font-family:'Manrope',sans-serif;font-size:2rem;margin:0 0 8px;color:#172B4D;line-height:1.25;}
.voc-hero p {margin:0;color:#66788D;font-size:.96rem;}
.voc-kpi {background:white;border:1px solid #E5EAF1;border-radius:14px;padding:21px 22px 18px;min-height:130px;box-shadow:0 3px 15px rgba(24,46,76,.025);}
.voc-kpi .label {color:#63768C;font-size:.83rem;font-weight:600;}
.voc-kpi .value {color:#172B4D;font:800 2.05rem 'Manrope',sans-serif;margin:7px 0 3px;}
.voc-kpi .foot {font-size:.77rem;color:#7A8CA0;}
.voc-section {font:800 1.05rem 'Manrope',sans-serif;color:#243951;margin:25px 0 12px;}
.voc-pill {display:inline-block;background:#E8F3F3;color:#156A69;border:1px solid #CFE8E5;padding:5px 11px;border-radius:20px;font-size:.77rem;font-weight:700;}
.voc-note {background:#EDF4F8;border-left:3px solid #38748B;border-radius:6px;padding:12px 16px;color:#466278;font-size:.88rem;margin:10px 0 20px;}
.voc-brand {font:800 1.28rem 'Manrope',sans-serif;color:#16374A;margin:9px 0 3px;}
.voc-brand-sub {font-size:.76rem;color:#8190A1;letter-spacing:.055em;text-transform:uppercase;}
</style>""", unsafe_allow_html=True)

MODEL = ROOT / 'models/sentiment_model.joblib'
METRICS = ROOT / 'reports/metrics.json'
DATA = ROOT / 'data/processed/clean_reviews.csv'
if not MODEL.exists() or not METRICS.exists():
    st.error('Model files are missing. Run `python -m src.train --data data/raw/demo_reviews.csv` in the project folder, then refresh.')
    st.stop()

@st.cache_resource
def get_model(path, mtime):
    return joblib.load(path)  # Only load trusted local model artifacts.

@st.cache_data
def get_reviews(path, mtime):
    return pd.read_csv(path)

model = get_model(str(MODEL), MODEL.stat().st_mtime)
metrics = json.loads(METRICS.read_text(encoding='utf-8'))
dataset = str(metrics.get('dataset', 'Unknown'))
labels = metrics.get('labels', [])
df = get_reviews(str(DATA), DATA.stat().st_mtime) if DATA.exists() else pd.DataFrame()

with st.sidebar:
    st.markdown('<div class="voc-brand">◈ &nbsp; VOC Intelligence</div><div class="voc-brand-sub">Customer experience analytics</div>', unsafe_allow_html=True)
    st.divider()
    page = st.radio('WORKSPACE', ['Executive overview','Sentiment analysis','Model evaluation','Review explorer','Business insights'], label_visibility='collapsed')
    st.divider()
    st.markdown('<span class="voc-pill">● &nbsp; Model available</span>', unsafe_allow_html=True)
    st.caption('PORTFOLIO ENVIRONMENT')
    st.caption('Evaluation depends on the selected dataset. Not a production performance claim.')

st.markdown('<div class="voc-hero"><div class="voc-eyebrow">CUSTOMER EXPERIENCE / ANALYTICS</div><h1>Voice of Customer Intelligence</h1><p>Turn customer feedback into measurable insights. Explore sentiment, evaluate model quality and investigate recurring issues.</p></div>', unsafe_allow_html=True)

def kpi(label, value, foot):
    st.markdown(f'<div class="voc-kpi"><div class="label">{html.escape(str(label))}</div><div class="value">{html.escape(str(value))}</div><div class="foot">{html.escape(str(foot))}</div></div>', unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="voc-section">{html.escape(title)}</div>', unsafe_allow_html=True)

def bar(data, x, y, color=None, horizontal=False):
    enc = {'x':alt.X(x, axis=alt.Axis(title=None, labelColor='#64748B', gridColor='#ECF0F4')),
           'y':alt.Y(y, axis=alt.Axis(title=None, labelColor='#64748B', gridColor='#ECF0F4'))}
    if color: enc['color'] = alt.Color(color, scale=alt.Scale(domain=['negative','neutral','positive'],range=['#C97373','#A9A6A0','#328A80']),legend=alt.Legend(orient='bottom',title=None))
    chart=alt.Chart(data).mark_bar(cornerRadiusEnd=4).encode(**enc).properties(height=300)
    st.altair_chart(chart, use_container_width=True)

if page == 'Executive overview':
    st.markdown('<div class="voc-eyebrow">EXECUTIVE SUMMARY</div>', unsafe_allow_html=True)
    c=st.columns(4,gap='medium')
    items=[('Total reviews',metrics.get('rows_after_cleaning','—'),'Validated records'),('Training set',metrics.get('train_count','—'),'Reviews used for fitting'),('Test set',metrics.get('test_count','—'),'Held-out reviews'),('Macro F1',f"{metrics.get('macro_f1',0):.2f}",'Held-out evaluation')]
    for col, item in zip(c,items):
        with col:kpi(*item)
    st.markdown(f'<div class="voc-note"><strong>Data context:</strong> {html.escape(dataset)}. Metrics describe this dataset only; do not interpret them as production performance.</div>',unsafe_allow_html=True)
    if not df.empty:
        left,right=st.columns([1,1],gap='large')
        with left:
            section('Sentiment mix')
            counts=df['sentiment'].value_counts().rename_axis('sentiment').reset_index(name='reviews')
            bar(counts,'sentiment:N','reviews:Q','sentiment:N')
        with right:
            section('Feedback by category')
            if 'category' in df.columns and df['category'].notna().any():
                category=df.groupby(['category','sentiment']).size().reset_index(name='reviews')
                bar(category,'category:N','reviews:Q','sentiment:N')
            else:st.info('This dataset has no category labels. No categories have been inferred.')
        section('Review volume over time')
        if 'review_date' in df.columns:
            dates=pd.to_datetime(df['review_date'],errors='coerce')
            if dates.notna().any():
                trend=df.loc[dates.notna(),['sentiment']].copy()
                trend['month']=dates[dates.notna()].dt.to_period('M').astype(str).values
                trend=trend.groupby(['month','sentiment']).size().reset_index(name='reviews')
                st.altair_chart(alt.Chart(trend).mark_line(point=True,strokeWidth=2).encode(x=alt.X('month:N',title=None),y=alt.Y('reviews:Q',title='Reviews'),color=alt.Color('sentiment:N',scale=alt.Scale(domain=['negative','neutral','positive'],range=['#C97373','#A9A6A0','#328A80']))).properties(height=250),use_container_width=True)
            else:st.caption('No valid review dates in this dataset.')
        else:st.caption('Review dates are not available in this dataset. No timeline is fabricated.')

elif page == 'Sentiment analysis':
    st.markdown('<div class="voc-eyebrow">INFERENCE WORKSPACE</div>',unsafe_allow_html=True)
    st.subheader('Analyze a customer review')
    st.caption('Enter customer feedback to see the model’s estimated sentiment and class probabilities.')
    with st.form('prediction'):
        review=st.text_area('Customer feedback',value='The app is useful but the payment fails often',height=150,max_chars=4000)
        submitted=st.form_submit_button('Analyze feedback',type='primary',use_container_width=True)
    if submitted:
        if not review.strip():st.warning('Please enter a review before analyzing.')
        else:
            predicted=str(model.predict([review])[0])
            probs=model.predict_proba([review])[0]
            section('Prediction result')
            a,b=st.columns([1,2],gap='large')
            with a:kpi('Predicted sentiment',predicted.title(),'Model estimate, not a verified label')
            with b:
                prob_df=pd.DataFrame({'sentiment':model.classes_,'probability':probs})
                st.altair_chart(alt.Chart(prob_df).mark_bar(cornerRadiusEnd=5).encode(x=alt.X('probability:Q',axis=alt.Axis(format='%'),scale=alt.Scale(domain=[0,1]),title=None),y=alt.Y('sentiment:N',sort='-x',title=None),color=alt.Color('sentiment:N',scale=alt.Scale(domain=['negative','neutral','positive'],range=['#C97373','#A9A6A0','#328A80']),legend=None)).properties(height=160),use_container_width=True)
            st.caption('Class probabilities are uncalibrated estimates. A binary dataset cannot predict neutral.')

elif page == 'Model evaluation':
    st.markdown('<div class="voc-eyebrow">MODEL GOVERNANCE / EVALUATION</div>',unsafe_allow_html=True)
    st.subheader('Model performance')
    st.caption(f"Selected algorithm: {metrics.get('selected_model','Unknown')} · Held-out macro F1: {metrics.get('macro_f1',0):.3f}")
    section('Training-only cross-validation')
    st.dataframe(pd.DataFrame(metrics.get('cv_comparison',[])),use_container_width=True,hide_index=True)
    section('Confusion matrix')
    cm=metrics.get('confusion_matrix',[])
    if cm and len(cm)==len(labels):
        cells=[{'Actual':a,'Predicted':b,'Count':cm[i][j]} for i,a in enumerate(labels) for j,b in enumerate(labels)]
        heat=alt.Chart(pd.DataFrame(cells)).mark_rect(cornerRadius=3).encode(x=alt.X('Predicted:N',title='Predicted'),y=alt.Y('Actual:N',title='Actual'),color=alt.Color('Count:Q',scale=alt.Scale(scheme='blues'),legend=None),tooltip=['Actual','Predicted','Count'])
        numbers=alt.Chart(pd.DataFrame(cells)).mark_text(fontSize=15,fontWeight='bold').encode(x='Predicted:N',y='Actual:N',text='Count:Q',color=alt.condition(alt.datum.Count>max(1,max(max(r) for r in cm)*.55),alt.value('white'),alt.value('#243951')))
        st.altair_chart((heat+numbers).properties(height=260),use_container_width=True)
    section('Classification report')
    st.dataframe(pd.DataFrame(metrics.get('classification_report',{})).T,use_container_width=True)
    section('Error analysis')
    err=ROOT/'reports/test_predictions.csv'
    if err.exists():
        predictions=pd.read_csv(err)
        wrong=predictions[predictions['actual']!=predictions['predicted']]
        if wrong.empty:st.info('No errors on this test split; this does not guarantee generalization.')
        else:st.dataframe(wrong,use_container_width=True,hide_index=True)


elif page == 'Business insights':
    st.markdown('<div class="voc-eyebrow">ACTIONABLE CUSTOMER INTELLIGENCE</div>', unsafe_allow_html=True)
    st.subheader('Complaint themes & recommended actions')
    st.caption('Exploratory keyword-based analysis of negative reviews. Themes may overlap, and a mention does not prove a complaint about that topic.')
    if df.empty or not {'review_text', 'sentiment'}.issubset(df.columns):
        st.warning('Review text and sentiment data are required for this analysis.')
    else:
        negative = df.loc[df['sentiment'].astype(str).str.lower().eq('negative')].copy()
        themes = {
            'Delivery & shipping': (r'\b(?:delivery|delivered|shipping|shipment|courier|late|delay|delayed)\b', 'Investigate delivery delays and shipping handoffs.'),
            'Product quality': (r'\b(?:quality|broken|damaged|defect|defective|poorly made|faulty)\b', 'Review quality checks and recurring product defects.'),
            'Customer support': (r'\b(?:support|customer service|helpdesk|representative|agent|response)\b', 'Audit response times and unresolved support cases.'),
            'Returns & refunds': (r'\b(?:refund|return|replacement|exchange|money back)\b', 'Review refund and return friction points.'),
            'Pricing & value': (r'\b(?:price|pricing|expensive|overpriced|cost|value for money)\b', 'Review price expectations and perceived value.'),
            'Payment & billing': (r'\b(?:payment|billing|charged|charge|transaction|checkout)\b', 'Check payment failures and billing disputes.'),
        }
        if negative.empty:
            st.info('There are no negative-labeled reviews in the current dataset.')
        else:
            summary = []
            for name, (pattern, action) in themes.items():
                mask = negative['review_text'].fillna('').astype(str).str.contains(pattern, case=False, regex=True, na=False)
                summary.append({'Theme': name, 'Reviews mentioning theme': int(mask.sum()), 'Suggested investigation': action})
            result = pd.DataFrame(summary).sort_values('Reviews mentioning theme', ascending=False)
            c1, c2, c3 = st.columns(3)
            with c1: st.metric('Negative reviews', f'{len(negative):,}')
            with c2: st.metric('Themes checked', len(themes))
            with c3: st.metric('Reviews with theme mentions', f"{sum(negative['review_text'].fillna('').astype(str).str.contains(p, case=False, regex=True, na=False) for p, _ in themes.values()).gt(0).sum():,}")
            section('Most-mentioned themes')
            bar(result, 'Reviews mentioning theme:Q', 'Theme:N')
            st.dataframe(result, use_container_width=True, hide_index=True)
            section('Explore supporting reviews')
            chosen = st.selectbox('Choose a theme', list(themes))
            pattern, action = themes[chosen]
            matches = negative.loc[negative['review_text'].fillna('').astype(str).str.contains(pattern, case=False, regex=True, na=False)]
            st.info(f'Suggested next step: {action}')
            st.caption(f'{len(matches):,} negative reviews mention keywords related to {chosen.lower()}.')
            if matches.empty:
                st.info('No matching reviews for this theme.')
            else:
                columns = [c for c in ['review_text', 'sentiment', 'category', 'review_date'] if c in matches.columns]
                st.dataframe(matches[columns].head(100), use_container_width=True, hide_index=True)
            st.caption('Interpretation note: keyword matches are not validated issue classifications; overlapping themes can count the same review more than once. Recommendations are investigation prompts, not confirmed root causes.')

else:
    st.markdown('<div class="voc-eyebrow">DATA EXPLORATION</div>',unsafe_allow_html=True)
    st.subheader('Customer review explorer')
    if df.empty:st.warning('Processed dataset unavailable. Train a model first.')
    else:
        a,b=st.columns([1,2])
        with a:sentiment=st.selectbox('Sentiment',['All']+sorted(df['sentiment'].dropna().unique().tolist()))
        with b:search=st.text_input('Search reviews',placeholder='Search for product, support, refund...')
        filtered=df.copy()
        if sentiment!='All':filtered=filtered[filtered['sentiment']==sentiment]
        if search:filtered=filtered[filtered['review_text'].astype(str).str.contains(search,case=False,regex=False,na=False)]
        st.caption(f'{len(filtered):,} matching reviews · Showing first 500')
        st.dataframe(filtered.head(500),use_container_width=True,hide_index=True)
        st.download_button('Export filtered reviews (CSV)',filtered.to_csv(index=False).encode('utf-8'),'voc_filtered_reviews.csv','text/csv')

st.divider()
st.caption('VOC Intelligence · Portfolio demonstration · Do not upload private customer data without authorization · Load only trusted model files')
