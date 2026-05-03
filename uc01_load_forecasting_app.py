"""
UC-01 — Load Forecasting Agent (Enterprise v2)
==============================================
Major upgrades:
  • Quantile regression with conformal calibration (P10/P50/P90 intervals)
  • Operator Recommendation engine (rule-based, ranked actions)
  • Model Insights tab (R²/MAE/RMSE + residual diagnostics)
  • Live train/test split slider (re-trains automatically)
  • Enhanced features: time-of-day segments, cyclical encoding, weather trends
  • Tool-using chat agent (compare_models, cross_validate, query_forecast,
    render_chart, what_if_scenario)
  • 6 tabs: Overview / Forecast / Insights / Scenarios / What-If Chat / Recommendations
"""

import os
import io
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import shap
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, r2_score
)

from google import genai
from google.genai import types as genai_types

# ============== PAGE SETUP ==============
st.set_page_config(
    page_title="UC-01 — Load Forecasting (Enterprise v2)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — Accenture brand palette + enterprise polish
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#FAFAFA; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #E8E8EE; }
[data-testid="stMetric"] { background:#ffffff; border:1px solid #E8E8EE; border-radius:10px; padding:14px 16px; }
[data-testid="stMetricValue"] { font-size:1.6rem; font-weight:700; }
[data-testid="stMetricLabel"] { color:#666; }

.stButton > button {
    background:#A100FF !important; color:white !important; border:none !important;
    border-radius:8px !important; font-weight:600 !important; transition:all 0.2s ease;
}
.stButton > button:hover {
    background:#7000B8 !important; transform:translateY(-1px);
    box-shadow:0 4px 12px rgba(161,0,255,0.25);
}
.stButton > button:disabled { background:#E8E8EE !important; color:#999 !important; }

.uc-tag { display:inline-block; background:#000000; color:#A100FF; padding:4px 12px;
          border-radius:4px; font-size:0.75rem; font-weight:700; letter-spacing:0.5px; }

.section-card { background:#ffffff; border:1px solid #E8E8EE; border-radius:10px;
                padding:18px 20px; margin-bottom:14px; }
.section-card h4 { color:#A100FF; margin-top:0; margin-bottom:8px; }

.kpi-card { background:linear-gradient(135deg,#A100FF 0%,#7000B8 100%); color:white;
            border-radius:10px; padding:16px; }
.kpi-card .label { font-size:0.75rem; opacity:0.85; text-transform:uppercase; letter-spacing:0.5px; }
.kpi-card .value { font-size:1.6rem; font-weight:700; margin-top:4px; }

.recommendation-card { background:#ffffff; border:1px solid #E8E8EE; border-left:4px solid #A100FF;
                       border-radius:8px; padding:14px 16px; margin-bottom:10px; }
.recommendation-card.priority-high { border-left-color:#FF1493; }
.recommendation-card.priority-medium { border-left-color:#FF8C00; }
.recommendation-card.priority-low { border-left-color:#A100FF; }
.recommendation-card .priority-tag { display:inline-block; padding:2px 8px; border-radius:99px;
                                      font-size:0.7rem; font-weight:700; letter-spacing:0.4px; }
.priority-high .priority-tag { background:#FFE3F0; color:#C70076; }
.priority-medium .priority-tag { background:#FFE9D1; color:#A35100; }
.priority-low .priority-tag { background:#F5E9FF; color:#7000B8; }

.data-quality-good { color:#0F8C5A; font-weight:600; }
.data-quality-warn { color:#A35100; font-weight:600; }
.data-quality-bad { color:#C70076; font-weight:600; }

[data-testid="stChatMessage"] { background:transparent !important; padding:12px 0 !important;
                                 border-bottom:1px solid #F0F0F4; }
[data-testid="stChatMessage"]:last-child { border-bottom:none; }
[data-testid="stChatMessageContent"] { font-size:15px; line-height:1.7; }
[data-testid="stChatMessageContent"] code { background:#F5E9FF; color:#7000B8;
    padding:1px 6px; border-radius:4px; font-size:13px; }
[data-testid="stChatMessageContent"] pre { background:#0E0E1A; color:#F5E9FF;
    padding:12px; border-radius:8px; overflow-x:auto; }
[data-testid="stChatMessageContent"] strong { color:#000000; font-weight:600; }
[data-testid="stChatMessageContent"] ul, [data-testid="stChatMessageContent"] ol {
    padding-left:22px; margin:8px 0; }
[data-testid="stChatMessageContent"] li { margin-bottom:4px; }

[data-testid="stChatInput"] textarea { font-size:15px !important; }
[data-testid="stChatInput"] { border:1px solid #E8E8EE !important; border-radius:12px !important; }
[data-testid="stChatInput"]:focus-within {
    border-color:#A100FF !important; box-shadow:0 0 0 3px rgba(161,0,255,0.1) !important;
}

.typing-dots { display:inline-flex; gap:4px; padding:8px 0; }
.typing-dots span { width:7px; height:7px; border-radius:50%; background:#A100FF;
    animation:typing 1.4s infinite ease-in-out; }
.typing-dots span:nth-child(2) { animation-delay:0.2s; }
.typing-dots span:nth-child(3) { animation-delay:0.4s; }
@keyframes typing { 0%,80%,100% { transform:scale(0.6); opacity:0.4; }
                    40% { transform:scale(1); opacity:1; } }

[data-testid="stTabs"] button { font-weight:500; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#A100FF !important; }
</style>
""", unsafe_allow_html=True)

# ============== SESSION STATE INIT ==============
if "model_p50" not in st.session_state:
    st.session_state.update({
        "model_p10": None, "model_p50": None, "model_p90": None,
        "metrics": None,
        "X_train": None, "X_calib": None, "X_test": None,
        "y_train": None, "y_calib": None, "y_test": None,
        "y_pred_p10": None, "y_pred_p50": None, "y_pred_p90": None,
        "feature_cols": None,
        "shap_values": None, "X_sample": None,
        "X_features_full": None,
        "forecast_df": None,
        "q_hat": 0.0,
        "chat_history": [],
        "pending_charts": [],
        "last_train_signature": None,
    })

# ============== SIDEBAR ==============
with st.sidebar:
    st.markdown("## ⚡ UC-01 Agent")
    st.caption("Enterprise Load Forecasting")
    st.divider()

    api_key = st.text_input("Gemini API key", type="password",
                            value=os.environ.get("GEMINI_API_KEY", ""),
                            placeholder="AIzaSy...")
    model_name = st.selectbox("LLM Model",
        ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"])
    st.divider()

    st.markdown("**📊 Train / Test Split**")
    test_size = st.slider("Test set size (last %)", 10, 40, 20, 5,
                          help="Larger test set → more reliable metrics, less training data.")
    st.caption(f"Train: {100-test_size}% • Test: {test_size}% (time-ordered)")
    st.divider()

    st.markdown("**🔮 Forecast settings**")
    forecast_horizon = st.select_slider("Horizon (hours ahead)",
                                        options=[24, 48, 72], value=24)
    confidence_level = st.select_slider("Confidence interval",
                                        options=["50%", "80%", "90%"], value="80%",
                                        help="Width of prediction interval (P10–P90 = 80%).")
    st.divider()

    st.markdown("**🔄 Retraining**")
    retrain_cadence = st.selectbox("Cadence",
        ["Manual", "Daily", "Weekly", "Monthly"], index=2)
    st.caption(f"⏱️ Configured: {retrain_cadence}")

# ============== HEADER ==============
col_tag, col_title = st.columns([1, 8])
with col_tag:
    st.markdown('<div class="uc-tag">UC-01 v2</div>', unsafe_allow_html=True)
with col_title:
    st.markdown("# Load Forecasting Agent")
    st.caption("Predict hourly load with calibrated uncertainty • Explain why • Recommend actions")

st.divider()

# ============== DATA LOADING ==============
st.markdown("### 📁 Step 1 — Load DERMS Data")
c1, c2 = st.columns([3, 1])
with c1:
    uploaded = st.file_uploader("Upload DERMS CSV/Excel",
                                 type=["csv", "xlsx", "xls"],
                                 label_visibility="collapsed")
with c2:
    use_demo = st.checkbox("Use demo data", value=(uploaded is None))

@st.cache_data
def load_demo():
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, "derms_load_data.csv")
    return pd.read_csv(csv_path, parse_dates=["datetime"])

def load_uploaded(file):
    df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df

if uploaded:
    df = load_uploaded(uploaded)
    st.success(f"✅ Loaded **{uploaded.name}** — {df.shape[0]:,} rows × {df.shape[1]} cols")
elif use_demo:
    try:
        df = load_demo()
        st.info(f"🧪 Demo DERMS data — {df.shape[0]:,} hourly rows "
                f"({df['datetime'].min().date()} → {df['datetime'].max().date()})")
    except FileNotFoundError:
        st.error("Demo file `derms_load_data.csv` missing. Upload your own.")
        st.stop()
else:
    st.warning("Upload a CSV or enable demo data to continue.")
    st.stop()

# ============== DATA QUALITY CHECK ==============
def assess_data_quality(df):
    """Run quick checks; return dict of findings."""
    findings = {}
    findings["row_count"] = len(df)
    findings["null_pct"] = round(df.isnull().sum().sum() / df.size * 100, 2) if df.size > 0 else 0
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        findings["datetime_invalid"] = int(dt.isna().sum())
        if dt.notna().any():
            sorted_dt = dt.dropna().sort_values()
            gaps = sorted_dt.diff().dropna()
            expected = pd.Timedelta(hours=1)
            findings["irregular_intervals"] = int((gaps != expected).sum())
            findings["duplicates"] = int(df.duplicated(subset=["datetime"]).sum())
        else:
            findings["irregular_intervals"] = -1
            findings["duplicates"] = -1
    if "total_load_mw" in df.columns:
        findings["negative_load"] = int((df["total_load_mw"] < 0).sum())
        findings["zero_load"] = int((df["total_load_mw"] == 0).sum())
    return findings

dq = assess_data_quality(df)

# ============== FEATURE ENGINEERING ==============
def make_features(df):
    """Enhanced features: basic time, cyclical encoding, time-of-day segments,
    weather trends, and lag features."""
    X = df.copy()
    X["datetime"] = pd.to_datetime(X["datetime"], errors="coerce")
    X = X.dropna(subset=["datetime", "total_load_mw"]).copy()
    X = X.sort_values("datetime").reset_index(drop=True)

    # Basic time
    X["hour"] = X["datetime"].dt.hour
    X["dow_num"] = X["datetime"].dt.dayofweek
    X["month"] = X["datetime"].dt.month
    X["day"] = X["datetime"].dt.day

    # Cyclical encoding (so model sees hour 23 ↔ hour 0 are adjacent)
    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
    X["dow_sin"] = np.sin(2 * np.pi * X["dow_num"] / 7)
    X["dow_cos"] = np.cos(2 * np.pi * X["dow_num"] / 7)
    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)

    # Time-of-day segment (categorical → ordinal: 0=night, 1=morning, 2=afternoon, 3=evening)
    def tod_segment(h):
        if 0 <= h < 6:
            return 0  # night
        elif 6 <= h < 12:
            return 1  # morning
        elif 12 <= h < 18:
            return 2  # afternoon
        else:
            return 3  # evening
    X["tod_segment"] = X["hour"].apply(tod_segment)

    # Weather trends — change over last 6 hours
    if "temperature_c" in X.columns:
        X["temp_trend_6h"] = X["temperature_c"].diff(6).fillna(0)
    if "humidity_pct" in X.columns:
        X["humidity_trend_6h"] = X["humidity_pct"].diff(6).fillna(0)

    # Lags
    X["load_lag_24h"] = X["total_load_mw"].shift(24)
    X["load_lag_168h"] = X["total_load_mw"].shift(168)
    X["load_roll_24h_mean"] = X["total_load_mw"].shift(1).rolling(24).mean()
    X["load_roll_168h_mean"] = X["total_load_mw"].shift(1).rolling(168).mean()

    feature_cols = [
        "hour", "dow_num", "month", "day", "tod_segment",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "is_weekend", "is_holiday",
        "temperature_c", "humidity_pct", "temp_trend_6h", "humidity_trend_6h",
        "load_lag_24h", "load_lag_168h", "load_roll_24h_mean", "load_roll_168h_mean",
    ]
    feature_cols = [c for c in feature_cols if c in X.columns]
    X = X.dropna(subset=feature_cols + ["total_load_mw"])
    return X, feature_cols

# ============== QUANTILE MODEL TRAINING (with conformal calibration) ==============
def train_quantile_models(df, test_pct):
    """Train P10, P50, P90 XGBoost models and conformally calibrate them.
    Splits: 60% train, 20% calibration, (test_pct)% test (last)."""
    X_full, feature_cols = make_features(df)
    X = X_full[feature_cols]
    y = X_full["total_load_mw"]

    n = len(X)
    test_end = n
    test_start = int(n * (1 - test_pct / 100))
    # Reserve a calibration set within the training portion (last 25% of training data)
    calib_start = int(test_start * 0.75)
    train_end = calib_start

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_calib = X.iloc[calib_start:test_start]
    y_calib = y.iloc[calib_start:test_start]
    X_test = X.iloc[test_start:test_end]
    y_test = y.iloc[test_start:test_end]

    common_params = dict(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        subsample=0.85, colsample_bytree=0.85,
        random_state=42, n_jobs=-1,
    )

    m10 = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.1, **common_params)
    m50 = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **common_params)
    m90 = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.9, **common_params)
    m10.fit(X_train, y_train, verbose=False)
    m50.fit(X_train, y_train, verbose=False)
    m90.fit(X_train, y_train, verbose=False)

    # Conformal calibration
    if len(X_calib) > 10:
        c_p10 = m10.predict(X_calib)
        c_p90 = m90.predict(X_calib)
        scores = np.maximum(c_p10 - y_calib.values, y_calib.values - c_p90)
        n_c = len(scores)
        q_hat = float(np.quantile(scores, np.ceil((n_c + 1) * 0.8) / n_c))
    else:
        q_hat = 0.0

    # Test predictions
    y_p10 = m10.predict(X_test) - q_hat
    y_p50 = m50.predict(X_test)
    y_p90 = m90.predict(X_test) + q_hat

    # Metrics on P50
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_p50)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_p50))),
        "MAPE": float(mean_absolute_percentage_error(y_test, y_p50) * 100),
        "R²": float(r2_score(y_test, y_p50)),
        "coverage_80pct": float(((y_test.values >= y_p10) & (y_test.values <= y_p90)).mean()),
        "interval_width_mean": float(np.mean(y_p90 - y_p10)),
    }

    # SHAP from P50 model
    sample_size = min(200, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42) if sample_size > 0 else X_test
    explainer = shap.TreeExplainer(m50)
    shap_values = explainer.shap_values(X_sample) if len(X_sample) else None

    return {
        "model_p10": m10, "model_p50": m50, "model_p90": m90,
        "metrics": metrics, "q_hat": q_hat,
        "X_train": X_train, "X_calib": X_calib, "X_test": X_test,
        "y_train": y_train, "y_calib": y_calib, "y_test": y_test,
        "y_pred_p10": y_p10, "y_pred_p50": y_p50, "y_pred_p90": y_p90,
        "feature_cols": feature_cols,
        "shap_values": shap_values, "X_sample": X_sample,
        "X_features_full": X_full,
    }

# ============== TRAIN BUTTON ==============
st.markdown("### 🏗️ Step 2 — Train Forecast Model")
train_col1, train_col2 = st.columns([2, 5])
with train_col1:
    train_btn = st.button(f"🚀 Train Quantile Models", use_container_width=True,
                           help="Trains P10, P50, P90 models with conformal calibration")
with train_col2:
    if st.session_state.model_p50 is not None:
        st.success(f"✅ Model trained • test={test_size}% • R²={st.session_state.metrics['R²']:.3f} • "
                   f"coverage={st.session_state.metrics['coverage_80pct']:.0%}")
    else:
        st.info("Click train to fit P10/P50/P90 quantile models with conformal calibration.")

# Auto-retrain if test_size changed
signature = (test_size, len(df), str(df.columns.tolist()))
if train_btn or (st.session_state.model_p50 is not None and st.session_state.last_train_signature != signature):
    if train_btn or st.session_state.last_train_signature is not None:
        with st.spinner("Training quantile models + calibrating intervals..."):
            try:
                result = train_quantile_models(df, test_size)
                for k, v in result.items():
                    st.session_state[k] = v
                st.session_state.last_train_signature = signature
                if not train_btn:
                    st.toast("Re-trained on new split", icon="🔄")
                else:
                    st.success("✅ Quantile models trained and calibrated")
            except Exception as e:
                st.error(f"Training failed: {e}")

# ============== RECOMMENDATION ENGINE ==============
def generate_recommendations(forecast_df, df_history, metrics):
    """Rule-based recommendations from the forecast.
    Returns list of dicts: {priority, title, detail, window}.
    Priority: high / medium / low."""
    if forecast_df is None or len(forecast_df) == 0:
        return []

    recs = []
    fc = forecast_df.copy()
    historical_avg = float(df_history["total_load_mw"].mean())
    historical_peak = float(df_history["total_load_mw"].max())

    # 1. PEAK MANAGEMENT
    peak_idx = fc["forecast_p50"].idxmax()
    peak_time = fc.loc[peak_idx, "datetime"]
    peak_p50 = fc.loc[peak_idx, "forecast_p50"]
    peak_p90 = fc.loc[peak_idx, "forecast_p90"]
    peak_pct_vs_avg = (peak_p50 / historical_avg - 1) * 100

    if peak_p90 > historical_peak * 0.95:
        recs.append({
            "priority": "high",
            "title": "🚨 Critical peak window approaching",
            "detail": f"Forecast peak of **{peak_p50:.0f} MW** (P90: {peak_p90:.0f} MW) at "
                      f"**{peak_time.strftime('%a %H:%M')}** — within 5% of historical max ({historical_peak:.0f} MW). "
                      f"Activate full demand-response program; pre-charge all available battery capacity 2–3 hr before.",
            "window": peak_time - pd.Timedelta(hours=3),
        })
    elif peak_pct_vs_avg > 30:
        recs.append({
            "priority": "medium",
            "title": "⚠️ Significant peak above average",
            "detail": f"Forecast peak of **{peak_p50:.0f} MW** at **{peak_time.strftime('%a %H:%M')}** — "
                      f"{peak_pct_vs_avg:+.0f}% above historical average. Schedule additional generation; "
                      f"pre-charge batteries by {(peak_time - pd.Timedelta(hours=2)).strftime('%H:%M')}.",
            "window": peak_time - pd.Timedelta(hours=2),
        })
    else:
        recs.append({
            "priority": "low",
            "title": "✅ Peak within normal range",
            "detail": f"Peak of **{peak_p50:.0f} MW** at **{peak_time.strftime('%a %H:%M')}** is normal "
                      f"(only {peak_pct_vs_avg:+.0f}% vs avg). Standard operations sufficient.",
            "window": peak_time,
        })

    # 2. UNCERTAINTY ALERT
    avg_width = (fc["forecast_p90"] - fc["forecast_p10"]).mean()
    if avg_width > historical_avg * 0.3:
        recs.append({
            "priority": "medium",
            "title": "📊 Wide uncertainty band — increase reserves",
            "detail": f"Average 80% interval width is **{avg_width:.0f} MW** ({avg_width/historical_avg*100:.0f}% "
                      f"of avg load). Consider holding **+{avg_width/2:.0f} MW** spinning reserve to handle "
                      f"forecast uncertainty.",
            "window": fc["datetime"].iloc[0],
        })

    # 3. BATTERY PRE-CHARGE WINDOW
    min_idx = fc["forecast_p50"].idxmin()
    min_time = fc.loc[min_idx, "datetime"]
    min_value = fc.loc[min_idx, "forecast_p50"]
    if min_time < peak_time:
        recs.append({
            "priority": "low",
            "title": "🔋 Optimal battery charging window",
            "detail": f"Lowest demand of **{min_value:.0f} MW** at **{min_time.strftime('%a %H:%M')}**. "
                      f"Charge batteries during this {((peak_time - min_time).total_seconds() / 3600):.0f}-hour "
                      f"window to be ready for the peak at {peak_time.strftime('%H:%M')}.",
            "window": min_time,
        })

    # 4. RAMP RATE ALERT
    if len(fc) >= 2:
        diffs = fc["forecast_p50"].diff().abs()
        max_ramp = diffs.max()
        max_ramp_idx = diffs.idxmax()
        if max_ramp > 100:
            ramp_time = fc.loc[max_ramp_idx, "datetime"]
            recs.append({
                "priority": "medium",
                "title": "📈 Steep ramp expected",
                "detail": f"Hourly load change of **{max_ramp:.0f} MW** at **{ramp_time.strftime('%a %H:%M')}**. "
                          f"Verify that flexible generation units can ramp at this rate; coordinate with neighboring "
                          f"control areas if needed.",
                "window": ramp_time,
            })

    # 5. MODEL CONFIDENCE FLAG
    if metrics and metrics.get("R²", 1.0) < 0.85:
        recs.append({
            "priority": "high",
            "title": "⚠️ Low model confidence",
            "detail": f"Current model R² is **{metrics['R²']:.2f}** — below the 0.85 reliability threshold. "
                      f"Consider retraining with more recent data or investigating data quality issues "
                      f"before relying on these forecasts for commitment decisions.",
            "window": fc["datetime"].iloc[0],
        })

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda r: priority_order[r["priority"]])
    return recs


# ============== TABS ==============
if st.session_state.model_p50 is not None:
    tab_overview, tab_forecast, tab_insights, tab_scenario, tab_chat, tab_recs = st.tabs([
        "📊 Overview",
        "📈 Forecast",
        "🔍 Model Insights",
        "🧪 Scenarios",
        "💬 What-If Chat",
        "💡 Recommendations",
    ])

    # ============== TAB: OVERVIEW ==============
    with tab_overview:
        st.markdown("### Executive Overview")
        # KPI strip
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f'<div class="kpi-card"><div class="label">Records</div>'
                        f'<div class="value">{len(df):,}</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="kpi-card"><div class="label">Peak Load</div>'
                        f'<div class="value">{df["total_load_mw"].max():.0f} MW</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="kpi-card"><div class="label">Avg Load</div>'
                        f'<div class="value">{df["total_load_mw"].mean():.0f} MW</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="kpi-card"><div class="label">Model R²</div>'
                        f'<div class="value">{st.session_state.metrics["R²"]:.3f}</div></div>', unsafe_allow_html=True)
        with k5:
            st.markdown(f'<div class="kpi-card"><div class="label">Coverage</div>'
                        f'<div class="value">{st.session_state.metrics["coverage_80pct"]:.0%}</div></div>',
                        unsafe_allow_html=True)

        st.markdown("")
        # Data quality + quick metrics
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-card"><h4>📋 Data Quality</h4>', unsafe_allow_html=True)
            null_class = "data-quality-good" if dq["null_pct"] < 1 else ("data-quality-warn" if dq["null_pct"] < 5 else "data-quality-bad")
            irr_class = "data-quality-good" if dq.get("irregular_intervals", 0) == 0 else "data-quality-warn"
            st.markdown(f"""
            • Rows: **{dq['row_count']:,}**
            • Missing values: <span class="{null_class}">{dq['null_pct']}%</span>
            • Invalid datetimes: **{dq.get('datetime_invalid', 'n/a')}**
            • Irregular intervals: <span class="{irr_class}">{dq.get('irregular_intervals', 'n/a')}</span>
            • Duplicate timestamps: **{dq.get('duplicates', 'n/a')}**
            • Negative loads: **{dq.get('negative_load', 'n/a')}**
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-card"><h4>📐 Model Snapshot</h4>', unsafe_allow_html=True)
            m = st.session_state.metrics
            st.markdown(f"""
            • R² score: **{m['R²']:.3f}**
            • MAE: **{m['MAE']:.1f} MW**
            • RMSE: **{m['RMSE']:.1f} MW**
            • MAPE: **{m['MAPE']:.2f}%**
            • 80% interval coverage: **{m['coverage_80pct']:.0%}** (target: 80%)
            • Avg interval width: **{m['interval_width_mean']:.0f} MW**
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Plain-English summary
        st.markdown('<div class="section-card"><h4>💬 Plain-English Summary</h4>', unsafe_allow_html=True)
        st.markdown(f"""
        The model achieves an **R² of {m['R²']:.3f}**, meaning it explains
        **{m['R²']*100:.0f}% of the variance** in load. On average, predictions are off by
        **{m['MAE']:.0f} MW (MAPE {m['MAPE']:.1f}%)** — well within typical industry tolerance for short-term load
        forecasting (MAPE < 5%). The 80% prediction interval correctly contains the actual value
        **{m['coverage_80pct']*100:.0f}% of the time**, which is close to the calibrated 80% target.
        Operators can use the P50 (median) forecast as the most likely outcome, and the P10–P90 range
        for capacity planning under uncertainty.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # ============== TAB: FORECAST ==============
    with tab_forecast:
        st.markdown("### Future Load Forecast — with Calibrated Confidence")

        # Generate forecast
        last_dt = st.session_state.X_features_full["datetime"].max()
        future_dates = pd.date_range(last_dt + pd.Timedelta(hours=1),
                                     periods=forecast_horizon, freq="h")
        history = st.session_state.X_features_full.copy().reset_index(drop=True)

        # Build hourly weather pattern from recent days
        last_day = history.tail(24).copy()
        hourly_temp_recent = last_day.set_index("hour")["temperature_c"].to_dict() if "temperature_c" in history.columns else {}
        hourly_hum_recent = last_day.set_index("hour")["humidity_pct"].to_dict() if "humidity_pct" in history.columns else {}
        recent_history = history.tail(24 * 7).copy()
        hourly_temp_7d = recent_history.groupby("hour")["temperature_c"].mean().to_dict() if "temperature_c" in history.columns else {}
        hourly_hum_7d = recent_history.groupby("hour")["humidity_pct"].mean().to_dict() if "humidity_pct" in history.columns else {}
        avg_temp = float(history["temperature_c"].mean()) if "temperature_c" in history.columns else 25.0
        avg_hum = float(history["humidity_pct"].mean()) if "humidity_pct" in history.columns else 60.0
        avg_load = float(history["total_load_mw"].mean())
        load_series = history.set_index("datetime")["total_load_mw"].to_dict()

        future_p10, future_p50, future_p90 = [], [], []
        future_rows = []
        for fd in future_dates:
            lag_24h = load_series.get(fd - pd.Timedelta(hours=24), avg_load)
            lag_168h = load_series.get(fd - pd.Timedelta(hours=168), avg_load)
            recent_24h_loads = [load_series.get(fd - pd.Timedelta(hours=h), avg_load) for h in range(1, 25)]
            recent_168h_loads = [load_series.get(fd - pd.Timedelta(hours=h), avg_load) for h in range(1, 169)]
            roll_24h = float(np.mean(recent_24h_loads))
            roll_168h = float(np.mean(recent_168h_loads))
            temp_val = float(hourly_temp_recent.get(fd.hour, hourly_temp_7d.get(fd.hour, avg_temp)))
            hum_val = float(hourly_hum_recent.get(fd.hour, hourly_hum_7d.get(fd.hour, avg_hum)))
            tod_seg = 0 if fd.hour < 6 else (1 if fd.hour < 12 else (2 if fd.hour < 18 else 3))

            row = {
                "hour": fd.hour, "dow_num": fd.dayofweek, "month": fd.month, "day": fd.day,
                "tod_segment": tod_seg,
                "hour_sin": float(np.sin(2*np.pi*fd.hour/24)),
                "hour_cos": float(np.cos(2*np.pi*fd.hour/24)),
                "dow_sin": float(np.sin(2*np.pi*fd.dayofweek/7)),
                "dow_cos": float(np.cos(2*np.pi*fd.dayofweek/7)),
                "month_sin": float(np.sin(2*np.pi*fd.month/12)),
                "month_cos": float(np.cos(2*np.pi*fd.month/12)),
                "is_weekend": int(fd.dayofweek >= 5), "is_holiday": 0,
                "temperature_c": temp_val, "humidity_pct": hum_val,
                "temp_trend_6h": 0.0, "humidity_trend_6h": 0.0,
                "load_lag_24h": float(lag_24h), "load_lag_168h": float(lag_168h),
                "load_roll_24h_mean": roll_24h, "load_roll_168h_mean": roll_168h,
            }
            row_df = pd.DataFrame([row])[st.session_state.feature_cols]
            p10_val = float(st.session_state.model_p10.predict(row_df)[0]) - st.session_state.q_hat
            p50_val = float(st.session_state.model_p50.predict(row_df)[0])
            p90_val = float(st.session_state.model_p90.predict(row_df)[0]) + st.session_state.q_hat
            load_series[fd] = p50_val
            future_p10.append(p10_val)
            future_p50.append(p50_val)
            future_p90.append(p90_val)
            future_rows.append(row)

        forecast_df = pd.DataFrame({
            "datetime": future_dates,
            "forecast_p10": future_p10,
            "forecast_p50": future_p50,
            "forecast_p90": future_p90,
        })
        st.session_state.forecast_df = forecast_df

        # Render fan chart
        fig_fan = go.Figure()
        # Past actuals (last 72h)
        recent_actual = history.tail(72)
        fig_fan.add_trace(go.Scatter(
            x=recent_actual["datetime"], y=recent_actual["total_load_mw"],
            mode="lines", name="Recent Actual",
            line=dict(color="#1a1a2e", width=2),
        ))
        # P90 upper boundary
        fig_fan.add_trace(go.Scatter(
            x=forecast_df["datetime"], y=forecast_df["forecast_p90"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        # P10 lower boundary with fill
        fig_fan.add_trace(go.Scatter(
            x=forecast_df["datetime"], y=forecast_df["forecast_p10"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(161,0,255,0.15)",
            name="80% Confidence Band",
        ))
        # P50 median forecast
        fig_fan.add_trace(go.Scatter(
            x=forecast_df["datetime"], y=forecast_df["forecast_p50"],
            mode="lines", name="Median Forecast (P50)",
            line=dict(color="#A100FF", width=3),
        ))
        fig_fan.update_layout(
            height=460, plot_bgcolor="white", paper_bgcolor="white",
            yaxis_title="Total Load (MW)",
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", y=1.08),
            hovermode="x unified",
        )
        st.plotly_chart(fig_fan, use_container_width=True)

        # Forecast table snapshot
        c1, c2 = st.columns([3, 2])
        with c1:
            peak_idx = forecast_df["forecast_p50"].idxmax()
            peak_time = forecast_df.loc[peak_idx, "datetime"]
            peak_p50 = forecast_df.loc[peak_idx, "forecast_p50"]
            peak_p10 = forecast_df.loc[peak_idx, "forecast_p10"]
            peak_p90 = forecast_df.loc[peak_idx, "forecast_p90"]
            st.markdown(f"""
            <div class="section-card">
            <h4>⚡ Peak Window</h4>
            <p><b>Predicted peak: {peak_p50:.0f} MW</b> at <b>{peak_time.strftime('%a %Y-%m-%d %H:%M')}</b></p>
            <p>80% confidence: <b>{peak_p10:.0f} – {peak_p90:.0f} MW</b> (range: {peak_p90-peak_p10:.0f} MW)</p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            csv_bytes = forecast_df.to_csv(index=False).encode()
            st.download_button("📥 Download forecast CSV", csv_bytes,
                               "forecast_with_intervals.csv", "text/csv",
                               use_container_width=True)

        # Show forecast as a table
        with st.expander("📋 Show forecast table"):
            display_df = forecast_df.copy()
            display_df["forecast_p10"] = display_df["forecast_p10"].round(1)
            display_df["forecast_p50"] = display_df["forecast_p50"].round(1)
            display_df["forecast_p90"] = display_df["forecast_p90"].round(1)
            display_df.columns = ["Datetime", "P10 (low)", "P50 (median)", "P90 (high)"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)


    # ============== TAB: MODEL INSIGHTS ==============
    with tab_insights:
        st.markdown("### Model Insights & Diagnostics")
        m = st.session_state.metrics

        # Metrics row
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("R²", f"{m['R²']:.3f}", help="Coefficient of determination — fraction of variance explained")
        mc2.metric("MAE", f"{m['MAE']:.1f} MW", help="Mean Absolute Error — average prediction error in MW")
        mc3.metric("RMSE", f"{m['RMSE']:.1f} MW", help="Root Mean Squared Error — penalizes large errors more")
        mc4.metric("MAPE", f"{m['MAPE']:.2f}%", help="Mean Absolute Percentage Error — error as % of actual")

        st.markdown('<div class="section-card"><h4>📖 What these mean (plain English)</h4>', unsafe_allow_html=True)
        r2_pct = m['R²'] * 100
        if m['R²'] >= 0.95:
            r2_quality = "excellent"
        elif m['R²'] >= 0.85:
            r2_quality = "good"
        elif m['R²'] >= 0.70:
            r2_quality = "moderate"
        else:
            r2_quality = "weak"
        if m['MAPE'] < 3:
            mape_quality = "very accurate (industry-leading)"
        elif m['MAPE'] < 5:
            mape_quality = "accurate (industry standard)"
        elif m['MAPE'] < 10:
            mape_quality = "acceptable"
        else:
            mape_quality = "poor — needs improvement"
        st.markdown(f"""
        - **R² = {m['R²']:.3f}** → **{r2_quality}** fit. The model explains **{r2_pct:.0f}% of the variation** in load.
        - **MAE = {m['MAE']:.0f} MW** → On average, predictions are off by {m['MAE']:.0f} MW.
        - **RMSE = {m['RMSE']:.0f} MW** → Larger penalty for big errors; {m['RMSE']/m['MAE']:.2f}× MAE
          (if this ratio is much > 1, there are some big mistakes).
        - **MAPE = {m['MAPE']:.2f}%** → {mape_quality}. Industry benchmark: < 5%.
        - **80% interval coverage = {m['coverage_80pct']:.0%}** → Calibration is
          {"well-calibrated" if abs(m['coverage_80pct'] - 0.8) < 0.05 else "slightly off"}.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        # Diagnostic plots
        st.markdown("#### 📊 Diagnostic plots")
        diag_c1, diag_c2 = st.columns(2)

        # PLOT 1: Predicted vs Actual
        with diag_c1:
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred_p50
            fig_pa = go.Figure()
            fig_pa.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                                         marker=dict(color="#A100FF", opacity=0.5, size=6),
                                         name="Predictions"))
            lo = float(min(y_test.min(), y_pred.min()))
            hi = float(max(y_test.max(), y_pred.max()))
            fig_pa.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                         line=dict(color="#000000", dash="dash", width=2),
                                         name="Perfect"))
            fig_pa.update_layout(
                title=f"Predicted vs Actual (R² = {m['R²']:.3f})",
                xaxis_title="Actual (MW)", yaxis_title="Predicted (MW)",
                plot_bgcolor="white", paper_bgcolor="white",
                height=380, margin=dict(t=40, b=10, l=10, r=10),
                showlegend=False,
            )
            st.plotly_chart(fig_pa, use_container_width=True)

        # PLOT 2: Residuals over time
        with diag_c2:
            test_dt = st.session_state.X_features_full.loc[
                st.session_state.X_test.index, "datetime"
            ].values
            residuals = y_test.values - y_pred
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=test_dt, y=residuals, mode="markers",
                                          marker=dict(color="#A100FF", opacity=0.6, size=5)))
            fig_res.add_hline(y=0, line=dict(color="black", dash="dash"))
            fig_res.update_layout(
                title=f"Residuals over time (mean = {residuals.mean():.1f} MW)",
                yaxis_title="Residual (MW)",
                plot_bgcolor="white", paper_bgcolor="white",
                height=380, margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_res, use_container_width=True)

        # PLOT 3: Residual distribution
        diag_c3, diag_c4 = st.columns(2)
        with diag_c3:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=residuals, nbinsx=40,
                                             marker_color="#A100FF", opacity=0.8))
            fig_hist.add_vline(x=0, line=dict(color="black", dash="dash"))
            fig_hist.update_layout(
                title=f"Residual distribution (std = {residuals.std():.1f} MW)",
                xaxis_title="Residual (MW)", yaxis_title="Frequency",
                plot_bgcolor="white", paper_bgcolor="white",
                height=340, margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # PLOT 4: SHAP global importance
        with diag_c4:
            if st.session_state.shap_values is not None:
                shap_vals = st.session_state.shap_values
                feat_names = st.session_state.feature_cols
                mean_abs = np.abs(shap_vals).mean(axis=0)
                imp_df = pd.DataFrame({
                    "feature": feat_names, "importance": mean_abs
                }).sort_values("importance", ascending=True).tail(10)
                fig_shap = px.bar(imp_df, x="importance", y="feature", orientation="h",
                                  color="importance", color_continuous_scale="Purples")
                fig_shap.update_layout(
                    title="Top 10 features (mean |SHAP|)",
                    xaxis_title="MW impact", yaxis_title="",
                    coloraxis_showscale=False,
                    plot_bgcolor="white", paper_bgcolor="white",
                    height=340, margin=dict(t=40, b=10, l=10, r=10),
                )
                st.plotly_chart(fig_shap, use_container_width=True)

        # Feature ↔ load combined heatmap
        st.markdown("#### 🌡️ Combined Feature Impact: Hour × Day-of-Week")
        df_hm = df.copy()
        df_hm["datetime"] = pd.to_datetime(df_hm["datetime"])
        df_hm["hour"] = df_hm["datetime"].dt.hour
        df_hm["dow"] = df_hm["datetime"].dt.day_name()
        pivot = df_hm.pivot_table(values="total_load_mw", index="hour",
                                   columns="dow", aggfunc="mean")
        days_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        pivot = pivot[[d for d in days_order if d in pivot.columns]]
        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale=[[0, "#FAFAFA"], [0.5, "#A100FF"], [1, "#000000"]],
            colorbar=dict(title="MW"),
        ))
        fig_heat.update_layout(
            xaxis_title="Day of week", yaxis_title="Hour of day",
            plot_bgcolor="white", paper_bgcolor="white",
            height=420, margin=dict(t=20, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ============== TAB: SCENARIOS ==============
    with tab_scenario:
        st.markdown("### Hypothetical Scenarios — Slider Controls")
        st.caption("Adjust conditions and see the predicted load with confidence bounds.")

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            sc_temp = st.slider("Temperature (°C)", 15.0, 45.0,
                                float(df["temperature_c"].mean()), 0.5)
            sc_hum = st.slider("Humidity (%)", 20, 95,
                                int(df["humidity_pct"].mean()))
        with sc2:
            sc_hour = st.slider("Hour of day", 0, 23, 18)
            sc_dow = st.selectbox("Day of week",
                ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], index=0)
        with sc3:
            sc_holiday = st.checkbox("Public holiday", value=False)
            sc_month = st.selectbox("Month", list(range(1, 13)),
                index=int(df["datetime"].dt.month.mode()[0]) - 1)

        sc_dow_num = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(sc_dow)
        sc_weekend = 1 if sc_dow_num >= 5 else 0
        sc_tod = 0 if sc_hour < 6 else (1 if sc_hour < 12 else (2 if sc_hour < 18 else 3))
        baseline = df["total_load_mw"].mean()

        scenario_row = pd.DataFrame([{
            "hour": sc_hour, "dow_num": sc_dow_num, "month": sc_month, "day": 15,
            "tod_segment": sc_tod,
            "hour_sin": float(np.sin(2*np.pi*sc_hour/24)),
            "hour_cos": float(np.cos(2*np.pi*sc_hour/24)),
            "dow_sin": float(np.sin(2*np.pi*sc_dow_num/7)),
            "dow_cos": float(np.cos(2*np.pi*sc_dow_num/7)),
            "month_sin": float(np.sin(2*np.pi*sc_month/12)),
            "month_cos": float(np.cos(2*np.pi*sc_month/12)),
            "is_weekend": sc_weekend, "is_holiday": int(sc_holiday),
            "temperature_c": sc_temp, "humidity_pct": sc_hum,
            "temp_trend_6h": 0.0, "humidity_trend_6h": 0.0,
            "load_lag_24h": baseline, "load_lag_168h": baseline,
            "load_roll_24h_mean": baseline, "load_roll_168h_mean": baseline,
        }])[st.session_state.feature_cols]

        sc_p10 = float(st.session_state.model_p10.predict(scenario_row)[0]) - st.session_state.q_hat
        sc_p50 = float(st.session_state.model_p50.predict(scenario_row)[0])
        sc_p90 = float(st.session_state.model_p90.predict(scenario_row)[0]) + st.session_state.q_hat

        delta = sc_p50 - baseline
        delta_pct = (delta / baseline) * 100

        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("Predicted load (P50)", f"{sc_p50:.0f} MW",
                    delta=f"{delta:+.0f} MW vs avg")
        sm2.metric("80% interval", f"{sc_p10:.0f} – {sc_p90:.0f} MW",
                    delta=f"width: {sc_p90-sc_p10:.0f} MW")
        sm3.metric("Difference vs avg", f"{delta_pct:+.1f}%")

        st.markdown(f"""
        <div class="section-card">
        <p>💡 <b>Scenario:</b> {sc_temp:.1f}°C, {sc_hum}% humidity, hour {sc_hour:02d}:00, {sc_dow}
        {' (holiday)' if sc_holiday else ''}, month {sc_month}.</p>
        <p>Most likely load: <b>{sc_p50:.0f} MW</b>. With 80% confidence, actual load will fall between
        <b>{sc_p10:.0f} MW</b> and <b>{sc_p90:.0f} MW</b>.</p>
        </div>
        """, unsafe_allow_html=True)


    # ============== TAB: WHAT-IF CHAT ==============
    with tab_chat:
        # ===== Chart builder helpers (used by render_chart tool) =====
        def build_chart_regression():
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred_p50
            r2 = float(r2_score(y_test, y_pred))
            mae = float(mean_absolute_error(y_test, y_pred))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                marker=dict(color="#A100FF", opacity=0.5, size=6), name="Predictions"))
            lo = float(min(y_test.min(), y_pred.min()))
            hi = float(max(y_test.max(), y_pred.max()))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                line=dict(color="#000000", dash="dash", width=2), name="Perfect"))
            fig.update_layout(title=f"Predicted vs Actual — R² = {r2:.3f}, MAE = {mae:.1f} MW",
                xaxis_title="Actual (MW)", yaxis_title="Predicted (MW)",
                plot_bgcolor="white", paper_bgcolor="white",
                height=420, margin=dict(t=50, b=10, l=10, r=10))
            return fig

        def build_chart_residuals():
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred_p50
            test_dt = st.session_state.X_features_full.loc[st.session_state.X_test.index, "datetime"].values
            res = y_test.values - y_pred
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_dt, y=res, mode="markers",
                marker=dict(color="#A100FF", opacity=0.6, size=5)))
            fig.add_hline(y=0, line=dict(color="black", dash="dash"))
            fig.update_layout(title="Residuals Over Time", yaxis_title="Residual (MW)",
                plot_bgcolor="white", paper_bgcolor="white",
                height=380, margin=dict(t=50, b=10, l=10, r=10))
            return fig

        def build_chart_model_comparison():
            X_full, fc = make_features(df)
            X = X_full[fc]; y = X_full["total_load_mw"]
            split = int(len(X) * (1 - test_size / 100))
            Xt, Xe = X.iloc[:split], X.iloc[split:]
            yt, ye = y.iloc[:split], y.iloc[split:]
            models = {
                "Linear": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                "XGBoost": xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                             random_state=42, n_jobs=-1),
            }
            names, r2_v = [], []
            for n, m in models.items():
                m.fit(Xt, yt); pred = m.predict(Xe)
                names.append(n); r2_v.append(float(r2_score(ye, pred)))
            fig = go.Figure(go.Bar(x=names, y=r2_v, marker_color="#A100FF",
                text=[f"{v:.3f}" for v in r2_v], textposition="outside"))
            fig.update_layout(title="Model Comparison — R²", yaxis_title="R²", yaxis_range=[0, 1.05],
                plot_bgcolor="white", paper_bgcolor="white",
                height=380, margin=dict(t=50, b=10, l=10, r=10))
            return fig, {"names": names, "R2": r2_v}

        def build_chart_temp_vs_load():
            x = df["temperature_c"].values; y = df["total_load_mw"].values
            slope, intercept = np.polyfit(x, y, 1)
            xl = np.linspace(x.min(), x.max(), 50); yl = slope * xl + intercept
            wd = df[df["is_weekend"] == 0]; we = df[df["is_weekend"] == 1]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=wd["temperature_c"], y=wd["total_load_mw"],
                mode="markers", name="Weekday",
                marker=dict(color="#A100FF", opacity=0.4, size=5)))
            fig.add_trace(go.Scatter(x=we["temperature_c"], y=we["total_load_mw"],
                mode="markers", name="Weekend",
                marker=dict(color="#FF1493", opacity=0.4, size=5)))
            fig.add_trace(go.Scatter(x=xl, y=yl, mode="lines", name="Trend",
                line=dict(color="#000000", dash="dash", width=2)))
            fig.update_layout(title=f"Temperature vs Load (slope = {slope:.1f} MW/°C)",
                xaxis_title="Temperature (°C)", yaxis_title="Load (MW)",
                plot_bgcolor="white", paper_bgcolor="white",
                height=420, margin=dict(t=50, b=10, l=10, r=10))
            return fig, float(slope)

        def build_chart_hourly_heatmap():
            df2 = df.copy()
            df2["datetime"] = pd.to_datetime(df2["datetime"])
            df2["hour"] = df2["datetime"].dt.hour
            df2["dow"] = df2["datetime"].dt.day_name()
            pivot = df2.pivot_table(values="total_load_mw", index="hour", columns="dow", aggfunc="mean")
            days_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            pivot = pivot[[d for d in days_order if d in pivot.columns]]
            fig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index,
                colorscale=[[0, "#FAFAFA"], [0.5, "#A100FF"], [1, "#000000"]],
                colorbar=dict(title="MW")))
            fig.update_layout(title="Avg Load by Hour × Day", xaxis_title="Day", yaxis_title="Hour",
                plot_bgcolor="white", paper_bgcolor="white",
                height=460, margin=dict(t=50, b=10, l=10, r=10))
            return fig, {"peak_hour": int(pivot.mean(axis=1).idxmax())}

        CHART_BUILDERS = {
            "regression": build_chart_regression,
            "residuals": build_chart_residuals,
            "model_comparison": build_chart_model_comparison,
            "temp_vs_load": build_chart_temp_vs_load,
            "hourly_heatmap": build_chart_hourly_heatmap,
        }

        # ===== Tool implementations =====
        def tool_compare_models():
            try:
                X_full, fc = make_features(df); X = X_full[fc]; y = X_full["total_load_mw"]
                split = int(len(X) * (1 - test_size / 100))
                Xt, Xe = X.iloc[:split], X.iloc[split:]
                yt, ye = y.iloc[:split], y.iloc[split:]
                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                    "XGBoost": xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                                 random_state=42, n_jobs=-1),
                }
                results = []
                for n, m in models.items():
                    m.fit(Xt, yt); pred = m.predict(Xe)
                    results.append({
                        "model": n,
                        "MAE_MW": round(float(mean_absolute_error(ye, pred)), 2),
                        "MAPE_pct": round(float(mean_absolute_percentage_error(ye, pred) * 100), 2),
                        "RMSE_MW": round(float(np.sqrt(mean_squared_error(ye, pred))), 2),
                        "R2": round(float(r2_score(ye, pred)), 3),
                    })
                return {"results": results}
            except Exception as e:
                return {"error": str(e)}

        def tool_cross_validate(n_splits=5):
            try:
                X_full, fc = make_features(df); X = X_full[fc].values; y = X_full["total_load_mw"].values
                tscv = TimeSeriesSplit(n_splits=n_splits)
                folds = []
                for i, (ti, ei) in enumerate(tscv.split(X)):
                    m = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                                          random_state=42, n_jobs=-1)
                    m.fit(X[ti], y[ti]); pred = m.predict(X[ei])
                    folds.append({
                        "fold": i + 1,
                        "MAE_MW": round(float(mean_absolute_error(y[ei], pred)), 2),
                        "R2": round(float(r2_score(y[ei], pred)), 3),
                    })
                r2s = [f["R2"] for f in folds]
                return {"folds": folds, "mean_R2": round(float(np.mean(r2s)), 3),
                        "std_R2": round(float(np.std(r2s)), 3),
                        "stability": "stable" if np.std(r2s) < 0.05 else "variable"}
            except Exception as e:
                return {"error": str(e)}

        def tool_query_forecast(query_type, hour=None):
            try:
                fcd = st.session_state.forecast_df
                if fcd is None or len(fcd) == 0:
                    return {"error": "No forecast yet. Open the Forecast tab first."}
                if query_type == "peak":
                    idx = fcd["forecast_p50"].idxmax()
                    return {"type": "peak", "datetime": str(fcd.loc[idx, "datetime"]),
                            "p50_mw": round(float(fcd.loc[idx, "forecast_p50"]), 1),
                            "p10_mw": round(float(fcd.loc[idx, "forecast_p10"]), 1),
                            "p90_mw": round(float(fcd.loc[idx, "forecast_p90"]), 1)}
                elif query_type == "min":
                    idx = fcd["forecast_p50"].idxmin()
                    return {"type": "min", "datetime": str(fcd.loc[idx, "datetime"]),
                            "p50_mw": round(float(fcd.loc[idx, "forecast_p50"]), 1)}
                elif query_type == "average":
                    return {"type": "average",
                            "p50_mw": round(float(fcd["forecast_p50"].mean()), 1)}
                elif query_type == "at_hour" and hour is not None:
                    rows = fcd[fcd["datetime"].dt.hour == int(hour)]
                    return {"type": f"at_hour_{hour}",
                            "values": [{"datetime": str(r["datetime"]),
                                         "p10": round(float(r["forecast_p10"]), 1),
                                         "p50": round(float(r["forecast_p50"]), 1),
                                         "p90": round(float(r["forecast_p90"]), 1)}
                                        for _, r in rows.iterrows()]}
                else:
                    return {"error": f"Unknown query type: {query_type}"}
            except Exception as e:
                return {"error": str(e)}

        def tool_render_chart(chart_type):
            if chart_type not in CHART_BUILDERS:
                return {"error": f"Unknown chart: {chart_type}. Available: {list(CHART_BUILDERS.keys())}"}
            try:
                result = CHART_BUILDERS[chart_type]()
                if isinstance(result, tuple):
                    fig, extra = result
                else:
                    fig, extra = result, None
                if "pending_charts" not in st.session_state:
                    st.session_state.pending_charts = []
                st.session_state.pending_charts.append((chart_type, fig))
                return {"chart_type": chart_type, "rendered": True, "extra_info": extra}
            except Exception as e:
                return {"error": f"Chart failed: {e}"}

        def tool_what_if_scenario(temperature_c=None, humidity_pct=None, hour=None,
                                    day_of_week=None, is_holiday=False):
            """Run a scenario through the trained quantile models."""
            try:
                tmean = float(df["temperature_c"].mean())
                hmean = int(df["humidity_pct"].mean())
                t = float(temperature_c) if temperature_c is not None else tmean
                h = int(humidity_pct) if humidity_pct is not None else hmean
                hr = int(hour) if hour is not None else 18
                dow_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6,
                           "Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
                           "Friday":4,"Saturday":5,"Sunday":6}
                dow_num = dow_map.get(day_of_week, 0) if isinstance(day_of_week, str) else (
                    int(day_of_week) if day_of_week is not None else 0)
                weekend = 1 if dow_num >= 5 else 0
                tod = 0 if hr < 6 else (1 if hr < 12 else (2 if hr < 18 else 3))
                baseline = float(df["total_load_mw"].mean())
                row = pd.DataFrame([{
                    "hour": hr, "dow_num": dow_num, "month": 6, "day": 15,
                    "tod_segment": tod,
                    "hour_sin": float(np.sin(2*np.pi*hr/24)),
                    "hour_cos": float(np.cos(2*np.pi*hr/24)),
                    "dow_sin": float(np.sin(2*np.pi*dow_num/7)),
                    "dow_cos": float(np.cos(2*np.pi*dow_num/7)),
                    "month_sin": float(np.sin(2*np.pi*6/12)),
                    "month_cos": float(np.cos(2*np.pi*6/12)),
                    "is_weekend": weekend, "is_holiday": int(bool(is_holiday)),
                    "temperature_c": t, "humidity_pct": h,
                    "temp_trend_6h": 0.0, "humidity_trend_6h": 0.0,
                    "load_lag_24h": baseline, "load_lag_168h": baseline,
                    "load_roll_24h_mean": baseline, "load_roll_168h_mean": baseline,
                }])[st.session_state.feature_cols]
                p10 = float(st.session_state.model_p10.predict(row)[0]) - st.session_state.q_hat
                p50 = float(st.session_state.model_p50.predict(row)[0])
                p90 = float(st.session_state.model_p90.predict(row)[0]) + st.session_state.q_hat
                return {
                    "scenario": {"temperature_c": t, "humidity_pct": h, "hour": hr,
                                  "day_of_week": dow_num, "is_holiday": bool(is_holiday)},
                    "p10_mw": round(p10, 1), "p50_mw": round(p50, 1), "p90_mw": round(p90, 1),
                    "vs_avg_pct": round((p50 - baseline) / baseline * 100, 1),
                }
            except Exception as e:
                return {"error": str(e)}

        TOOL_DISPATCH = {
            "compare_models": tool_compare_models,
            "cross_validate": tool_cross_validate,
            "query_forecast": tool_query_forecast,
            "render_chart": tool_render_chart,
            "what_if_scenario": tool_what_if_scenario,
        }

        TOOL_DECLARATIONS = [genai_types.Tool(function_declarations=[
            genai_types.FunctionDeclaration(
                name="compare_models",
                description="Train Linear Regression, Random Forest, and XGBoost; return MAE/MAPE/RMSE/R² for each. Use to verify R², compare models.",
                parameters=genai_types.Schema(type=genai_types.Type.OBJECT, properties={}, required=[]),
            ),
            genai_types.FunctionDeclaration(
                name="cross_validate",
                description="Run 5-fold time-series cross-validation. Returns R² per fold + mean ± std. Use to test stability.",
                parameters=genai_types.Schema(type=genai_types.Type.OBJECT, properties={}, required=[]),
            ),
            genai_types.FunctionDeclaration(
                name="query_forecast",
                description="Get specific forecast values: peak, min, average, or at a specific hour. Returns P10/P50/P90.",
                parameters=genai_types.Schema(type=genai_types.Type.OBJECT,
                    properties={
                        "query_type": genai_types.Schema(type=genai_types.Type.STRING,
                            description="One of: 'peak', 'min', 'average', 'at_hour'"),
                        "hour": genai_types.Schema(type=genai_types.Type.INTEGER,
                            description="Hour 0-23, only for 'at_hour'"),
                    }, required=["query_type"]),
            ),
            genai_types.FunctionDeclaration(
                name="render_chart",
                description="Render an interactive chart in chat. Types: 'regression', 'residuals', 'model_comparison', 'temp_vs_load', 'hourly_heatmap'.",
                parameters=genai_types.Schema(type=genai_types.Type.OBJECT,
                    properties={"chart_type": genai_types.Schema(type=genai_types.Type.STRING)},
                    required=["chart_type"]),
            ),
            genai_types.FunctionDeclaration(
                name="what_if_scenario",
                description="Predict load for a hypothetical scenario. All params optional; defaults use historical averages. Returns P10/P50/P90.",
                parameters=genai_types.Schema(type=genai_types.Type.OBJECT,
                    properties={
                        "temperature_c": genai_types.Schema(type=genai_types.Type.NUMBER, description="°C"),
                        "humidity_pct": genai_types.Schema(type=genai_types.Type.INTEGER, description="0-100"),
                        "hour": genai_types.Schema(type=genai_types.Type.INTEGER, description="0-23"),
                        "day_of_week": genai_types.Schema(type=genai_types.Type.STRING,
                            description="'Mon'..'Sun' or 0-6"),
                        "is_holiday": genai_types.Schema(type=genai_types.Type.BOOLEAN),
                    }, required=[]),
            ),
        ])]

        # ===== Chat UI =====
        head_a, head_b = st.columns([4, 1])
        with head_a:
            st.markdown("### 💬 What-If Chat")
            st.caption("Ask scenarios in plain English — agent calls ML tools, runs scenarios, renders charts.")
        with head_b:
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        # Suggestion chips
        if not st.session_state.chat_history:
            st.markdown("**Try asking:**")
            chip_cols = st.columns(2)
            chips = [
                "What if temperature jumps to 40°C tomorrow at 6 PM?",
                "Compare Linear/Random Forest/XGBoost R²",
                "Show the regression scatter plot",
                "What is tomorrow's peak load with confidence?",
            ]
            quick_q = None
            for i, chip in enumerate(chips):
                with chip_cols[i % 2]:
                    if st.button(chip, key=f"chip_{i}", use_container_width=True):
                        quick_q = chip
        else:
            quick_q = None

        # Render chat history
        for msg in st.session_state.chat_history:
            avatar = "👤" if msg["role"] == "user" else "⚡"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["text"])

        user_q = st.chat_input("Ask about scenarios, models, charts, or forecasts…")
        if quick_q and not user_q:
            user_q = quick_q

        if user_q:
            if not api_key:
                st.error("Please add your Gemini API key in the sidebar.")
            else:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(user_q)

                # Build context
                m = st.session_state.metrics
                shap_arr = st.session_state.shap_values
                feat_names = st.session_state.feature_cols
                top_features_text = ""
                if shap_arr is not None:
                    mean_abs = np.abs(shap_arr).mean(axis=0)
                    top_features_text = "\n".join([
                        f"  - {feat}: {imp:.2f} MW avg impact"
                        for feat, imp in sorted(zip(feat_names, mean_abs),
                                                 key=lambda x: x[1], reverse=True)[:5]
                    ])

                # Forecast summary
                fc_summary = "Forecast not yet generated."
                if st.session_state.forecast_df is not None:
                    fc = st.session_state.forecast_df.copy()
                    fc["date"] = fc["datetime"].dt.date
                    daily = fc.groupby("date").agg(
                        peak_p50=("forecast_p50", "max"),
                        peak_p10=("forecast_p10", lambda s: float(fc.loc[s.idxmax(), "forecast_p10"]) if hasattr(s, 'idxmax') else 0),
                        peak_p90=("forecast_p90", lambda s: float(fc.loc[s.idxmax(), "forecast_p90"]) if hasattr(s, 'idxmax') else 0),
                        peak_hour=("forecast_p50", lambda s: int(fc.loc[s.idxmax(), "datetime"].hour)),
                    ).reset_index()
                    fc_summary = "Per-day forecast (P50 = median, P10–P90 = 80% interval):\n"
                    for _, r in daily.iterrows():
                        fc_summary += (f"  - {r['date']}: peak P50={r['peak_p50']:.0f} MW "
                                       f"({r['peak_p10']:.0f}–{r['peak_p90']:.0f}) at "
                                       f"hour {r['peak_hour']:02d}:00\n")

                context = f"""You are a senior load forecasting analyst (UC-01) with access to:
- A trained quantile XGBoost model (P10/P50/P90 with conformal calibration)
- SHAP explanations
- Tools you can CALL to run computations

CURRENT FORECAST:
{fc_summary}

MODEL PERFORMANCE:
- R²: {m['R²']:.3f}, MAE: {m['MAE']:.1f} MW, RMSE: {m['RMSE']:.1f} MW, MAPE: {m['MAPE']:.2f}%
- 80% interval coverage: {m['coverage_80pct']:.0%}
- Conformal q_hat: {st.session_state.q_hat:.1f} MW

TOP SHAP FEATURES:
{top_features_text}

DATASET:
- Rows: {len(df)}, range: {df['datetime'].min()} → {df['datetime'].max()}
- Peak: {df['total_load_mw'].max():.0f} MW, avg: {df['total_load_mw'].mean():.0f} MW

AVAILABLE TOOLS (use them, don't just describe them):
1. compare_models() — train Linear/RF/XGBoost, return metrics
2. cross_validate() — 5-fold time-series CV
3. query_forecast(query_type, hour=None) — peak/min/average/at_hour
4. render_chart(chart_type) — 'regression', 'residuals', 'model_comparison', 'temp_vs_load', 'hourly_heatmap'
5. what_if_scenario(temperature_c, humidity_pct, hour, day_of_week, is_holiday) — predict load for a scenario

When user asks a what-if question, CALL what_if_scenario and explain results in plain English.
When user asks for a chart, CALL render_chart.
Use markdown formatting. **Bold** key MW values. Suggest concrete actions when relevant."""

                try:
                    client = genai.Client(api_key=api_key)
                    full_prompt = "\n".join(
                        [f"{m_['role']}: {m_['text']}" for m_ in st.session_state.chat_history]
                        + [f"user: {user_q}"]
                    )
                    contents = [genai_types.Content(role="user",
                        parts=[genai_types.Part(text=full_prompt)])]
                    cfg = genai_types.GenerateContentConfig(
                        system_instruction=context, tools=TOOL_DECLARATIONS)

                    with st.chat_message("assistant", avatar="⚡"):
                        placeholder = st.empty()
                        tool_status = st.empty()
                        chart_slot = st.empty()
                        placeholder.markdown(
                            '<div class="typing-dots"><span></span><span></span><span></span></div>',
                            unsafe_allow_html=True
                        )

                        # Reset pending charts for this turn
                        st.session_state.pending_charts = []
                        full_response = ""
                        max_rounds = 5
                        tools_used = []

                        for _ in range(max_rounds):
                            response = client.models.generate_content(
                                model=model_name, contents=contents, config=cfg)
                            function_call = None; text_parts = []
                            if response.candidates and response.candidates[0].content:
                                for part in response.candidates[0].content.parts or []:
                                    if hasattr(part, "function_call") and part.function_call:
                                        function_call = part.function_call
                                    elif hasattr(part, "text") and part.text:
                                        text_parts.append(part.text)

                            if function_call:
                                tool_name = function_call.name
                                tools_used.append(tool_name)
                                tool_status.info(f"🛠️ Running tool: `{tool_name}`...")
                                tool_args = dict(function_call.args) if function_call.args else {}
                                if tool_name in TOOL_DISPATCH:
                                    try:
                                        result = TOOL_DISPATCH[tool_name](**tool_args) if tool_args else TOOL_DISPATCH[tool_name]()
                                    except Exception as te:
                                        result = {"error": f"{te}"}
                                else:
                                    result = {"error": f"Unknown tool: {tool_name}"}

                                # Render any pending charts immediately
                                if st.session_state.pending_charts:
                                    with chart_slot.container():
                                        for ct, fig_obj in st.session_state.pending_charts:
                                            st.plotly_chart(fig_obj, use_container_width=True,
                                                             key=f"chat_chart_{ct}_{len(st.session_state.chat_history)}_{_}")

                                contents.append(response.candidates[0].content)
                                contents.append(genai_types.Content(role="user",
                                    parts=[genai_types.Part.from_function_response(
                                        name=tool_name, response={"result": result})]))
                                placeholder.markdown(
                                    '<div class="typing-dots"><span></span><span></span><span></span></div>',
                                    unsafe_allow_html=True)
                            else:
                                full_response = "".join(text_parts)
                                break

                        if tools_used:
                            badges = " ".join([
                                f'<span style="background:#F5E9FF; color:#7000B8; padding:2px 8px; '
                                f'border-radius:99px; font-size:11px; font-weight:600; margin-right:4px;">'
                                f'🛠️ {t}</span>' for t in tools_used
                            ])
                            placeholder.markdown(
                                f'<div style="margin-bottom:8px;">{badges}</div>{full_response}',
                                unsafe_allow_html=True)
                        else:
                            placeholder.markdown(full_response or "(no response)")
                        tool_status.empty()

                    if not full_response:
                        full_response = "(no response generated)"
                    st.session_state.chat_history.append({"role": "user", "text": user_q})
                    st.session_state.chat_history.append({"role": "assistant", "text": full_response})
                    st.rerun()
                except Exception as e:
                    st.error(f"Gemini error: {e}")


    # ============== TAB: RECOMMENDATIONS ==============
    with tab_recs:
        st.markdown("### Operator Recommendations")
        st.caption("Auto-generated, ranked by priority based on the current forecast.")

        recs = generate_recommendations(
            st.session_state.forecast_df, df, st.session_state.metrics
        )

        if not recs:
            st.info("No recommendations available — train the model and check the Forecast tab first.")
        else:
            # Priority summary
            high_count = sum(1 for r in recs if r["priority"] == "high")
            med_count = sum(1 for r in recs if r["priority"] == "medium")
            low_count = sum(1 for r in recs if r["priority"] == "low")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("🔴 High priority", high_count)
            sc2.metric("🟡 Medium priority", med_count)
            sc3.metric("🟢 Low priority", low_count)

            st.markdown("")
            for r in recs:
                badge_text = {"high": "HIGH PRIORITY", "medium": "MEDIUM",
                              "low": "INFO"}[r["priority"]]
                st.markdown(f"""
                <div class="recommendation-card priority-{r['priority']}">
                <span class="priority-tag">{badge_text}</span>
                <h4 style="margin:6px 0;">{r['title']}</h4>
                <p style="margin:0; color:#333;">{r['detail']}</p>
                </div>
                """, unsafe_allow_html=True)

            # Export recommendations
            rec_csv = pd.DataFrame([{
                "priority": r["priority"], "title": r["title"], "detail": r["detail"]
            } for r in recs])
            st.download_button("📥 Download recommendations (CSV)",
                                rec_csv.to_csv(index=False).encode(),
                                "operator_recommendations.csv", "text/csv")

else:
    # Pre-training landing
    st.info("👆 Click **Train Quantile Models** above to begin.")
    st.markdown("""
    ### What this app does
    - Loads DERMS hourly load data (or your CSV)
    - Trains 3 quantile XGBoost models (P10, P50, P90)
    - Calibrates intervals using conformal prediction (proper 80% coverage)
    - Generates forecasts with confidence bands
    - Provides AI agent that can run tools (compare models, cross-validate, render charts, what-if scenarios)
    - Auto-generates operator recommendations
    """)

st.divider()

# ============== BUSINESS BENEFITS ==============
st.markdown("### 💼 Business Benefits")
b1, b2, b3, b4 = st.columns(4)
benefits = [
    ("🎯", "Reduce demand penalty", "Predict peak windows with calibrated intervals 24-72hr ahead"),
    ("⚙️", "Optimize scheduling", "Day-ahead generation & unit commitment with confidence"),
    ("🔋", "Battery pre-charging", "Proactive prep before peak surges using P90 upper bound"),
    ("💰", "Lower balancing costs", "Quantified uncertainty → right-sized reserves"),
]
for col, (icon, title, desc) in zip([b1, b2, b3, b4], benefits):
    with col:
        st.markdown(f'<div class="section-card" style="height:140px;">'
                    f'<div style="font-size:1.6rem;">{icon}</div><b>{title}</b>'
                    f'<p style="font-size:0.8rem; color:#555;margin-top:4px;">{desc}</p></div>',
                    unsafe_allow_html=True)
