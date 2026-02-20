import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnLens · Prediction Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b26 60%, #0d1117 100%);
    color: #e6edf3;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b26 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #58a6ff;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.4rem;
    margin-top: 1.4rem;
}
[data-testid="stSidebar"] label {
    color: #8b949e !important;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.04em;
}
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, #1a2332 0%, #0f1923 50%, #1a1a2e 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(88,166,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-header::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(63,185,80,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.02em;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}
.hero-title span {
    color: #58a6ff;
}
.hero-sub {
    color: #8b949e;
    font-size: 1rem;
    font-weight: 400;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(88,166,255,0.15);
    border: 1px solid rgba(88,166,255,0.3);
    color: #58a6ff;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    margin-bottom: 1rem;
    letter-spacing: 0.1em;
}

/* ── KPI cards ── */
.kpi-card {
    background: #161b26;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: border-color 0.2s;
    height: 100%;
}
.kpi-card:hover { border-color: #58a6ff; }
.kpi-label {
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1;
}
.kpi-delta {
    font-size: 0.78rem;
    color: #8b949e;
    margin-top: 0.4rem;
}

/* ── Risk badge ── */
.risk-badge-high {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(248,81,73,0.15);
    border: 1px solid rgba(248,81,73,0.4);
    color: #f85149;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    padding: 0.5rem 1.2rem;
    border-radius: 8px;
    letter-spacing: 0.08em;
}
.risk-badge-medium {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(210,153,34,0.15);
    border: 1px solid rgba(210,153,34,0.4);
    color: #d29922;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    padding: 0.5rem 1.2rem;
    border-radius: 8px;
    letter-spacing: 0.08em;
}
.risk-badge-low {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(63,185,80,0.15);
    border: 1px solid rgba(63,185,80,0.4);
    color: #3fb950;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    padding: 0.5rem 1.2rem;
    border-radius: 8px;
    letter-spacing: 0.08em;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.6rem;
    margin: 2rem 0 1.2rem 0;
}

/* ── Chart containers ── */
.chart-card {
    background: #161b26;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem;
}

/* ── Feature table ── */
.feature-table {
    background: #161b26;
    border: 1px solid #30363d;
    border-radius: 12px;
    overflow: hidden;
}
.feature-row {
    display: flex;
    justify-content: space-between;
    padding: 0.7rem 1.2rem;
    border-bottom: 1px solid #21262d;
    font-size: 0.88rem;
}
.feature-row:last-child { border-bottom: none; }
.feature-key { color: #8b949e; font-weight: 500; }
.feature-val { color: #e6edf3; font-family: 'Space Mono', monospace; font-size: 0.82rem; }

/* ── Interpretation box ── */
.interp-card {
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-top: 0.5rem;
}
.interp-card.high {
    background: rgba(248,81,73,0.08);
    border: 1px solid rgba(248,81,73,0.25);
}
.interp-card.medium {
    background: rgba(210,153,34,0.08);
    border: 1px solid rgba(210,153,34,0.25);
}
.interp-card.low {
    background: rgba(63,185,80,0.08);
    border: 1px solid rgba(63,185,80,0.25);
}
.interp-title { font-weight: 600; font-size: 1rem; margin-bottom: 0.5rem; }
.interp-title.high { color: #f85149; }
.interp-title.medium { color: #d29922; }
.interp-title.low { color: #3fb950; }
.interp-body { color: #8b949e; font-size: 0.88rem; line-height: 1.6; }

/* ── Threshold info ── */
.threshold-row {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-top: 0.8rem;
}
.threshold-pill {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: 700;
    letter-spacing: 0.06em;
}
.pill-low  { background: rgba(63,185,80,0.15); color: #3fb950; border: 1px solid rgba(63,185,80,0.3); }
.pill-mid  { background: rgba(210,153,34,0.15); color: #d29922; border: 1px solid rgba(210,153,34,0.3); }
.pill-high { background: rgba(248,81,73,0.15); color: #f85149; border: 1px solid rgba(248,81,73,0.3); }

/* ── Streamlit overrides ── */
div[data-testid="metric-container"] {
    background: #161b26;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
div[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #e6edf3 !important;
    font-family: 'Space Mono', monospace !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 1.5rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.04em;
    cursor: pointer;
    transition: opacity 0.2s;
    margin-top: 1rem;
}
.stButton > button:hover { opacity: 0.85; }

/* Plotly chart transparent bg */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

div[data-testid="stExpander"] {
    background: #161b26;
    border: 1px solid #30363d !important;
    border-radius: 12px;
}
div[data-testid="stExpander"] summary {
    color: #8b949e;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FEATURE ORDER  (must match training pipeline)
# ─────────────────────────────────────────────
FEATURE_ORDER = [
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_France', 'Geography_Germany', 'Geography_Spain'
]

# ─────────────────────────────────────────────
# LOAD MODEL & PREPROCESSORS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = tf.keras.models.load_model('trained_model/churn_model.h5')
    with open('preprocessing_models/one_hot_encoder_geography.pkl', 'rb') as f:
        ohe = pickle.load(f)
    with open('preprocessing_models/label_encoder_gender.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('preprocessing_models/scaler_features.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, ohe, le, scaler

try:
    model, one_hot_encoder, label_encoder, scaler = load_artifacts()
except Exception as e:
    st.error(f"❌ Failed to load model or preprocessors: {e}")
    st.stop()

# ─────────────────────────────────────────────
# SIDEBAR — USER INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
        <span style='font-family:Space Mono,monospace; font-size:1.1rem;
                     font-weight:700; color:#58a6ff; letter-spacing:0.05em;'>
            🔮 ChurnLens
        </span><br>
        <span style='font-size:0.75rem; color:#8b949e;'>Customer Risk Analyzer</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Personal Info")
    geography = st.selectbox("Geography", options=['France', 'Spain', 'Germany'])
    gender    = st.selectbox("Gender",    options=['Male', 'Female'])
    age       = st.number_input("Age",    min_value=18, max_value=100, value=38)

    st.markdown("## Financial Profile")
    credit_score     = st.number_input("Credit Score",     min_value=300, max_value=850, value=650)
    balance          = st.number_input("Account Balance",  min_value=0.0, value=75000.0, step=1000.0)
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=80000.0, step=1000.0)

    st.markdown("## Banking Activity")
    tenure          = st.slider("Tenure (years)",        min_value=0,  max_value=10, value=5)
    num_of_products = st.slider("Number of Products",    min_value=1,  max_value=4,  value=2)
    has_cr_card     = st.selectbox("Has Credit Card",     options=[1, 0],
                                   format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.selectbox("Active Member",      options=[1, 0],
                                    format_func=lambda x: "Yes" if x == 1 else "No")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  Run Prediction")

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">AI-POWERED ANALYTICS · REAL-TIME PREDICTION</div>
    <h1 class="hero-title">Customer Churn<br><span>Intelligence Dashboard</span></h1>
    <p class="hero-sub">
        Deep-learning model · TensorFlow · Predict churn risk from customer behaviour signals
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLACEHOLDER  (before first prediction)
# ─────────────────────────────────────────────
if "churn_prob" not in st.session_state:
    st.markdown("""
    <div style='text-align:center; padding:5rem 2rem; color:#8b949e;'>
        <div style='font-size:3rem; margin-bottom:1rem;'>🔮</div>
        <p style='font-size:1rem; font-weight:500; color:#58a6ff;'>
            Configure customer details in the sidebar and click <strong>Run Prediction</strong>
        </p>
        <p style='font-size:0.85rem; max-width:420px; margin:0 auto; line-height:1.6;'>
            The model will analyse 12 features and return a real-time churn probability
            with full risk breakdown and visual confidence indicators.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RUN PREDICTION
# ─────────────────────────────────────────────
if predict_btn:
    try:
        input_df = pd.DataFrame({
            'CreditScore':     [credit_score],
            'Gender':          [gender],
            'Age':             [age],
            'Tenure':          [tenure],
            'Balance':         [balance],
            'NumOfProducts':   [num_of_products],
            'HasCrCard':       [has_cr_card],
            'IsActiveMember':  [is_active_member],
            'EstimatedSalary': [estimated_salary],
        })

        # Encode Gender
        input_df['Gender'] = label_encoder.transform(input_df['Gender'])

        # OHE Geography
        geography_encoded = one_hot_encoder.transform([[geography]]).toarray()
        geography_df = pd.DataFrame(
            geography_encoded,
            columns=one_hot_encoder.get_feature_names_out(['Geography'])
        )

        # Combine & reorder
        input_df = pd.concat([input_df, geography_df], axis=1)
        input_df = input_df[FEATURE_ORDER]

        # Scale & predict
        scaled_input = scaler.transform(input_df)
        prob = float(model.predict(scaled_input, verbose=0)[0][0])

        # Persist result
        st.session_state.churn_prob  = prob
        st.session_state.input_df    = input_df
        st.session_state.input_raw   = {
            "Geography": geography, "Gender": gender, "Age": age,
            "Credit Score": credit_score, "Balance": f"${balance:,.2f}",
            "Est. Salary": f"${estimated_salary:,.2f}",
            "Tenure": f"{tenure} yr", "Products": num_of_products,
            "Credit Card": "Yes" if has_cr_card else "No",
            "Active Member": "Yes" if is_active_member else "No",
        }

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ─────────────────────────────────────────────
# RESULTS DASHBOARD
# ─────────────────────────────────────────────
if "churn_prob" in st.session_state:
    prob    = st.session_state.churn_prob
    no_prob = 1.0 - prob
    pct     = prob * 100

    # Derive risk tier
    if prob >= 0.65:
        risk_tier   = "HIGH"
        risk_color  = "#f85149"
        risk_class  = "high"
        badge_html  = f'<span class="risk-badge-high">⚠ HIGH RISK &nbsp;·&nbsp; {pct:.1f}%</span>'
        conf_label  = "Strong Churn Signal"
    elif prob >= 0.40:
        risk_tier   = "MEDIUM"
        risk_color  = "#d29922"
        risk_class  = "medium"
        badge_html  = f'<span class="risk-badge-medium">◈ MODERATE RISK &nbsp;·&nbsp; {pct:.1f}%</span>'
        conf_label  = "Moderate Churn Signal"
    else:
        risk_tier   = "LOW"
        risk_color  = "#3fb950"
        risk_class  = "low"
        badge_html  = f'<span class="risk-badge-low">✔ LOW RISK &nbsp;·&nbsp; {pct:.1f}%</span>'
        conf_label  = "Retention Signal"

    # ── KPI Row ──────────────────────────────
    st.markdown('<p class="section-header">▸ Overview Metrics</p>', unsafe_allow_html=True)
    kc1, kc2, kc3, kc4 = st.columns(4)

    with kc1:
        st.metric("Churn Probability", f"{pct:.2f}%")
    with kc2:
        st.metric("Retention Probability", f"{no_prob*100:.2f}%")
    with kc3:
        st.metric("Risk Level", risk_tier)
    with kc4:
        confidence = abs(prob - 0.5) / 0.5 * 100
        st.metric("Model Confidence", f"{confidence:.1f}%")

    # ── Risk badge + progress bar ─────────────
    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns([1, 3])
    with b1:
        st.markdown(badge_html, unsafe_allow_html=True)
    with b2:
        bar_filled = int(round(pct))
        bar_empty  = 100 - bar_filled
        st.markdown(f"""
        <div style='margin-top:0.55rem;'>
            <div style='display:flex; align-items:center; gap:0.6rem;'>
                <div style='
                    flex:1; height:10px; border-radius:10px;
                    background:linear-gradient(90deg, {risk_color} {bar_filled}%, #21262d {bar_filled}%);
                    border: 1px solid #30363d;
                '></div>
                <span style='font-family:Space Mono,monospace; font-size:0.78rem;
                             color:{risk_color}; min-width:3.5rem;'>
                    {pct:.1f} %
                </span>
            </div>
            <div style='font-size:0.72rem; color:#8b949e; margin-top:0.3rem;
                        letter-spacing:0.05em;'>
                CHURN PROBABILITY · {conf_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ───────────────────────────
    st.markdown('<p class="section-header">▸ Visual Breakdown</p>', unsafe_allow_html=True)
    ch1, ch2, ch3 = st.columns([1.4, 1, 1])

    # Gauge
    with ch1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            number={"suffix": "%", "font": {"size": 34, "color": "#e6edf3",
                                             "family": "Space Mono"}},
            delta={"reference": 50, "increasing": {"color": "#f85149"},
                   "decreasing": {"color": "#3fb950"},
                   "font": {"size": 14}},
            title={"text": "Churn Risk Gauge", "font": {"size": 13,
                   "color": "#8b949e", "family": "DM Sans"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1,
                          "tickcolor": "#30363d", "tickfont": {"color": "#8b949e",
                                                                "size": 10}},
                "bar":  {"color": risk_color, "thickness": 0.28},
                "bgcolor": "#161b26",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  40], "color": "rgba(63,185,80,0.12)"},
                    {"range": [40, 65], "color": "rgba(210,153,34,0.12)"},
                    {"range": [65, 100],"color": "rgba(248,81,73,0.12)"},
                ],
                "threshold": {
                    "line": {"color": "#e6edf3", "width": 2},
                    "thickness": 0.8, "value": 50
                }
            }
        ))
        gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=10), height=280,
            font=dict(family="DM Sans")
        )
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

    # Bar chart
    with ch2:
        bar = go.Figure(go.Bar(
            x=["Churn", "Stay"],
            y=[prob, no_prob],
            marker_color=[risk_color, "#3fb950"],
            marker_line_width=0,
            text=[f"{prob*100:.1f}%", f"{no_prob*100:.1f}%"],
            textposition="outside",
            textfont=dict(family="Space Mono", size=12, color="#e6edf3"),
            width=0.45,
        ))
        bar.update_layout(
            title={"text": "Churn vs Retention", "font": {"size": 13,
                   "color": "#8b949e", "family": "DM Sans"}},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=10), height=280,
            yaxis=dict(range=[0, 1.25], showgrid=True, gridcolor="#21262d",
                       tickformat=".0%", tickfont=dict(color="#8b949e", size=10),
                       zeroline=False),
            xaxis=dict(tickfont=dict(color="#8b949e", size=11),
                       linecolor="#30363d"),
            font=dict(family="DM Sans"),
            bargap=0.3,
        )
        st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})

    # Pie chart
    with ch3:
        pie = go.Figure(go.Pie(
            labels=["Churn Risk", "Retention"],
            values=[prob, no_prob],
            hole=0.55,
            marker=dict(colors=[risk_color, "#3fb950"],
                        line=dict(color="#0d1117", width=2)),
            textinfo="percent",
            textfont=dict(family="Space Mono", size=11, color="#e6edf3"),
            hovertemplate="%{label}: %{percent}<extra></extra>",
        ))
        pie.add_annotation(
            text=f"<b>{pct:.0f}%</b>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=22, family="Space Mono", color=risk_color)
        )
        pie.update_layout(
            title={"text": "Risk Distribution", "font": {"size": 13,
                   "color": "#8b949e", "family": "DM Sans"}},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=10), height=280,
            legend=dict(font=dict(color="#8b949e", size=10),
                        bgcolor="rgba(0,0,0,0)"),
            font=dict(family="DM Sans"),
            showlegend=True,
        )
        st.plotly_chart(pie, use_container_width=True, config={"displayModeBar": False})

    # ── Feature Summary + Interpretation ─────
    st.markdown('<p class="section-header">▸ Customer Profile & Risk Interpretation</p>',
                unsafe_allow_html=True)
    ft_col, interp_col = st.columns([1, 1.3])

    with ft_col:
        raw = st.session_state.input_raw
        rows_html = "".join([
            f'<div class="feature-row">'
            f'  <span class="feature-key">{k}</span>'
            f'  <span class="feature-val">{v}</span>'
            f'</div>'
            for k, v in raw.items()
        ])
        st.markdown(f'<div class="feature-table">{rows_html}</div>', unsafe_allow_html=True)

    with interp_col:
        if risk_tier == "HIGH":
            interp_title = "⚠ High Churn Risk Detected"
            interp_body  = (
                f"With a churn probability of <strong style='color:{risk_color}'>{pct:.1f}%</strong>, "
                "this customer shows strong disengagement signals. "
                "Immediate retention action is recommended — consider personalised offers, "
                "account review calls, or loyalty rewards. "
                "Key risk contributors may include low activity, limited product adoption, "
                "or dissatisfaction indicators."
            )
        elif risk_tier == "MEDIUM":
            interp_title = "◈ Moderate Churn Risk"
            interp_body  = (
                f"Churn probability of <strong style='color:{risk_color}'>{pct:.1f}%</strong> "
                "places this customer in an undecided zone. "
                "Proactive but light-touch engagement is advised — email campaigns, "
                "feature highlights, or satisfaction surveys. "
                "Monitor account activity over the next 30 days."
            )
        else:
            interp_title = "✔ Low Churn Risk — Retention Likely"
            interp_body  = (
                f"Churn probability of <strong style='color:{risk_color}'>{pct:.1f}%</strong> "
                "indicates this customer is likely to remain with the bank. "
                "Focus on cross-sell opportunities and deepening the relationship "
                "through additional product offerings. This profile represents a "
                "strong candidate for premium tier upgrades."
            )

        st.markdown(f"""
        <div class="interp-card {risk_class}">
            <div class="interp-title {risk_class}">{interp_title}</div>
            <div class="interp-body">{interp_body}</div>
        </div>
        """, unsafe_allow_html=True)

        # Threshold explanation
        st.markdown("""
        <div style='margin-top:1.2rem;'>
            <div style='font-size:0.72rem; color:#8b949e; text-transform:uppercase;
                        letter-spacing:0.1em; margin-bottom:0.5rem;'>
                Decision Thresholds
            </div>
            <div class="threshold-row">
                <span class="threshold-pill pill-low">0–40% · Low Risk</span>
                <span class="threshold-pill pill-mid">40–65% · Moderate</span>
                <span class="threshold-pill pill-high">65–100% · High Risk</span>
            </div>
            <div style='font-size:0.75rem; color:#8b949e; margin-top:0.6rem; line-height:1.5;'>
                Model threshold set at <strong style='color:#e6edf3;'>0.50</strong>.
                Predictions above this value classify as churn-positive.
                Confidence reflects distance from the decision boundary.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Advanced Details (Expander) ───────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔬  Advanced Details — Scaled Feature Vector & Model Info"):
        adv1, adv2 = st.columns(2)
        with adv1:
            st.markdown("**Scaled Feature Vector (model input)**")
            scaled_df = pd.DataFrame(
                scaler.transform(st.session_state.input_df),
                columns=FEATURE_ORDER
            ).T.rename(columns={0: "Scaled Value"})
            scaled_df["Scaled Value"] = scaled_df["Scaled Value"].round(4)
            st.dataframe(scaled_df, use_container_width=True, height=340)

        with adv2:
            st.markdown("**Raw Feature Values**")
            raw_df = st.session_state.input_df.T.rename(columns={0: "Raw Value"})
            st.dataframe(raw_df, use_container_width=True, height=340)

            st.markdown(f"""
            <div style='margin-top:1rem; font-size:0.8rem; color:#8b949e; line-height:1.7;'>
                <strong style='color:#e6edf3;'>Model Architecture:</strong> TensorFlow ANN (.h5)<br>
                <strong style='color:#e6edf3;'>Encoder:</strong> LabelEncoder + OneHotEncoder<br>
                <strong style='color:#e6edf3;'>Scaler:</strong> StandardScaler (z-score)<br>
                <strong style='color:#e6edf3;'>Output:</strong> Sigmoid · Binary classification<br>
                <strong style='color:#e6edf3;'>Features:</strong> {len(FEATURE_ORDER)} engineered inputs
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────
    st.markdown("""
    <div style='margin-top:3rem; padding: 1.5rem; text-align:center;
                border-top: 1px solid #21262d;'>
        <span style='font-size:0.75rem; color:#484f58; font-family:Space Mono,monospace;
                    letter-spacing:0.08em;'>
            ChurnLens · Powered by TensorFlow + Streamlit
            &nbsp;·&nbsp; For internal analytical use only
        </span>
    </div>
    """, unsafe_allow_html=True)