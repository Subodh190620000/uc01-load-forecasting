"""
UC-01 — Load Forecasting Agent
==============================
DERMS-style hourly load forecasting with XGBoost + SHAP + Gemini chat.

Sections mirror the UC-01 enterprise spec:
  - Input data from DERMS
  - AI/ML help (XGBoost, SHAP)
  - Business benefits (24-72hr forecast, peak windows, scheduling)
  - Improve accuracy (hypothetical scenarios)
  - Forecast horizon (24/48/72 hr)
  - Retraining cadence
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

from google import genai
from google.genai import types as genai_types

# ============== PAGE SETUP ==============
st.set_page_config(
    page_title="UC-01 — Load Forecasting Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — purple theme matching the UC-01 slide
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#fafbfc; }
[data-testid="stSidebar"] { background:#ffffff; border-right: 1px solid #e8eaed; }
[data-testid="stMetric"] { background:#ffffff; border:1px solid #e8eaed; border-radius:10px; padding:12px 16px; }
.stButton > button { background:#8B14E0 !important; color:white !important; border:none !important; border-radius:8px !important; font-weight:600 !important; }
.stButton > button:hover { background:#6B0FAB !important; }
.uc-tag { display:inline-block; background:#8B14E0; color:white; padding:4px 12px; border-radius:4px; font-size:0.75rem; font-weight:700; letter-spacing:0.5px; margin-bottom:8px; }
.section-card { background:#ffffff; border:1px solid #e8eaed; border-radius:10px; padding:18px 20px; margin-bottom:14px; }
.section-card h4 { color:#8B14E0; margin-top:0; }
.kpi-card { background:linear-gradient(135deg,#8B14E0 0%,#6B0FAB 100%); color:white; border-radius:10px; padding:16px; }
.kpi-card .label { font-size:0.78rem; opacity:0.85; text-transform:uppercase; letter-spacing:0.5px; }
.kpi-card .value { font-size:1.6rem; font-weight:700; margin-top:4px; }
.chat-user { background:#F3E8FF; padding:10px 14px; border-radius:10px; margin:6px 0; }
.chat-agent { background:#F7F8FA; border:1px solid #e8eaed; padding:10px 14px; border-radius:10px; margin:6px 0; }
</style>
""", unsafe_allow_html=True)

# ============== SIDEBAR ==============
with st.sidebar:
    st.markdown("## ⚡ UC-01 Agent")
    st.caption("Load Forecasting — DERMS")
    st.divider()

    api_key = st.text_input(
        "Gemini API key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        placeholder="AIzaSy..."
    )
    model_name = st.selectbox(
        "LLM Model",
        ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
    )
    st.divider()

    st.markdown("**Forecast settings**")
    forecast_horizon = st.select_slider(
        "Horizon (hours ahead)",
        options=[24, 48, 72],
        value=24,
    )
    test_size = st.slider("Test set size (last %)", 10, 30, 20, 5)
    st.divider()

    st.markdown("**Retraining**")
    retrain_cadence = st.selectbox(
        "Cadence",
        ["Manual", "Daily", "Weekly", "Monthly"],
        index=2,
    )
    st.caption(f"⏱️ Configured: {retrain_cadence}")

# ============== HEADER (mirrors slide) ==============
col_tag, col_title = st.columns([1, 8])
with col_tag:
    st.markdown('<div class="uc-tag">UC-01</div>', unsafe_allow_html=True)
with col_title:
    st.markdown("# Load Forecasting")
    st.caption("Predict hourly and daily electricity consumption to enable proactive grid management and cost optimization.")

st.divider()

# ============== SECTION 1 — DATA INPUT ==============
st.markdown("### 📁 Input Data from DERMS")

c1, c2 = st.columns([2, 1])
with c1:
    uploaded = st.file_uploader(
        "Upload DERMS CSV (hourly load + temperature + holiday flags)",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed"
    )
with c2:
    use_demo = st.checkbox("Use demo data (3 months)", value=True if uploaded is None else False)

@st.cache_data
def load_demo():
    return pd.read_csv("derms_load_data.csv", parse_dates=["datetime"])

def load_uploaded(file):
    df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df

# Load data
if uploaded:
    df = load_uploaded(uploaded)
    st.success(f"✅ Loaded **{uploaded.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")
elif use_demo:
    try:
        df = load_demo()
        st.info(f"🧪 Using demo DERMS data — {df.shape[0]:,} hourly rows ({df['datetime'].min().date()} → {df['datetime'].max().date()})")
    except FileNotFoundError:
        st.error("Demo data file `derms_load_data.csv` not found in the repo. Upload your own CSV.")
        st.stop()
else:
    st.warning("Upload a CSV or enable demo data to continue.")
    st.stop()

# KPI strip
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="kpi-card"><div class="label">Records</div><div class="value">{len(df):,}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card"><div class="label">Peak Load</div><div class="value">{df["total_load_mw"].max():.0f} MW</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi-card"><div class="label">Avg Load</div><div class="value">{df["total_load_mw"].mean():.0f} MW</div></div>', unsafe_allow_html=True)
with k4:
    holidays_count = int(df["is_holiday"].sum() // 24) if "is_holiday" in df.columns else 0
    st.markdown(f'<div class="kpi-card"><div class="label">Holidays</div><div class="value">{holidays_count}</div></div>', unsafe_allow_html=True)

with st.expander("Preview data"):
    st.dataframe(df.head(20), use_container_width=True)

st.divider()

# ============== FEATURE ENGINEERING ==============
def make_features(df):
    """Build features for XGBoost from the raw DERMS data."""
    X = df.copy()
    X["datetime"] = pd.to_datetime(X["datetime"])
    X["hour"] = X["datetime"].dt.hour
    X["dow_num"] = X["datetime"].dt.dayofweek
    X["month"] = X["datetime"].dt.month
    X["day"] = X["datetime"].dt.day

    # Lag features (yesterday same hour, last week same hour)
    X = X.sort_values("datetime").reset_index(drop=True)
    X["load_lag_24h"] = X["total_load_mw"].shift(24)
    X["load_lag_168h"] = X["total_load_mw"].shift(168)  # 1 week
    X["load_roll_24h_mean"] = X["total_load_mw"].shift(1).rolling(24).mean()

    feature_cols = [
        "hour", "dow_num", "month", "day",
        "is_weekend", "is_holiday",
        "temperature_c", "humidity_pct",
        "load_lag_24h", "load_lag_168h", "load_roll_24h_mean",
    ]
    X = X.dropna(subset=feature_cols + ["total_load_mw"])
    return X, feature_cols

# ============== SECTION 2 — AI/ML ==============
st.markdown("### 🧠 How AI/ML Helps")

ai_c1, ai_c2, ai_c3 = st.columns(3)
with ai_c1:
    st.markdown("""
    <div class="section-card">
    <h4>🌲 XGBoost</h4>
    <p style="font-size:0.88rem;">Gradient-boosted trees capture trends, seasonality, weather effects, and calendar patterns from historical data.</p>
    </div>
    """, unsafe_allow_html=True)
with ai_c2:
    st.markdown("""
    <div class="section-card">
    <h4>🔍 SHAP</h4>
    <p style="font-size:0.88rem;">Explains why each prediction was made — feature contributions for transparency and trust.</p>
    </div>
    """, unsafe_allow_html=True)
with ai_c3:
    st.markdown("""
    <div class="section-card">
    <h4>🤖 Gemini Agent</h4>
    <p style="font-size:0.88rem;">Conversational interface for asking questions, interpreting SHAP, and exploring scenarios.</p>
    </div>
    """, unsafe_allow_html=True)

# ============== TRAIN MODEL ==============
st.markdown("### 🏗️ Model Training")

if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.metrics = None
    st.session_state.X_test = None
    st.session_state.y_test = None
    st.session_state.y_pred = None
    st.session_state.feature_cols = None
    st.session_state.shap_values = None
    st.session_state.X_features_full = None

train_btn = st.button(f"🚀 Train XGBoost Model ({retrain_cadence})", use_container_width=False)

if train_btn:
    with st.spinner("Engineering features and training XGBoost..."):
        X_full, feature_cols = make_features(df)
        X = X_full[feature_cols]
        y = X_full["total_load_mw"]

        # Time-based split (last N% as test)
        split_idx = int(len(X) * (1 - test_size / 100))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)

        metrics = {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "MAPE": float(mean_absolute_percentage_error(y_test, y_pred) * 100),
            "R²": float(r2_score(y_test, y_pred)),
        }

        # SHAP — sample 200 rows to keep it fast
        sample_size = min(200, len(X_test))
        X_sample = X_test.sample(n=sample_size, random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Save to session
        st.session_state.model = model
        st.session_state.metrics = metrics
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.feature_cols = feature_cols
        st.session_state.shap_values = shap_values
        st.session_state.X_sample = X_sample
        st.session_state.X_features_full = X_full

    st.success("✅ Model trained")

# Show metrics if model exists
if st.session_state.model is not None:
    m = st.session_state.metrics
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("MAE (MW)", f"{m['MAE']:.2f}")
    mc2.metric("MAPE (%)", f"{m['MAPE']:.2f}")
    mc3.metric("R²", f"{m['R²']:.3f}")

    # ========== TABS ==========
    tab_forecast, tab_shap, tab_scenario, tab_chat = st.tabs([
        "📈 Forecast",
        "🔍 SHAP Explainability",
        "🧪 Hypothetical Scenarios",
        "💬 Chat with Agent",
    ])

    # ----- FORECAST TAB -----
    with tab_forecast:
        st.markdown(f"#### Actual vs Predicted (last {test_size}% of data)")
        # Use .loc with the actual index labels (not positional .iloc) to avoid out-of-bounds
        test_datetimes = st.session_state.X_features_full.loc[st.session_state.X_test.index, "datetime"].values
        plot_df = pd.DataFrame({
            "datetime": test_datetimes,
            "Actual": st.session_state.y_test.values,
            "Predicted": st.session_state.y_pred,
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df["datetime"], y=plot_df["Actual"],
                                 name="Actual", line=dict(color="#1a1a2e", width=2)))
        fig.add_trace(go.Scatter(x=plot_df["datetime"], y=plot_df["Predicted"],
                                 name="Predicted", line=dict(color="#8B14E0", width=2, dash="dot")))
        fig.update_layout(
            height=420, plot_bgcolor="white", paper_bgcolor="white",
            yaxis_title="Total Load (MW)", xaxis_title="",
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Forecast next N hours
        st.markdown(f"#### Forecast — next {forecast_horizon} hours")
        last_datetime = st.session_state.X_features_full["datetime"].max()
        future_dates = pd.date_range(last_datetime + pd.Timedelta(hours=1),
                                     periods=forecast_horizon, freq="h")

        # Build features for future timestamps using last known weather + lags from history
        history = st.session_state.X_features_full.copy()
        recent_temp = history["temperature_c"].tail(48).mean()
        recent_humidity = history["humidity_pct"].tail(48).mean()

        future_rows = []
        for fd in future_dates:
            row = {
                "hour": fd.hour,
                "dow_num": fd.dayofweek,
                "month": fd.month,
                "day": fd.day,
                "is_weekend": int(fd.dayofweek >= 5),
                "is_holiday": 0,
                "temperature_c": recent_temp,
                "humidity_pct": recent_humidity,
                "load_lag_24h": float(history["total_load_mw"].iloc[-24]) if len(history) >= 24 else float(history["total_load_mw"].mean()),
                "load_lag_168h": history["total_load_mw"].iloc[-168] if len(history) >= 168 else history["total_load_mw"].mean(),
                "load_roll_24h_mean": history["total_load_mw"].tail(24).mean(),
            }
            future_rows.append(row)

        future_df = pd.DataFrame(future_rows)
        future_pred = st.session_state.model.predict(future_df[st.session_state.feature_cols])

        forecast_df = pd.DataFrame({"datetime": future_dates, "forecast_mw": future_pred})

        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(
            x=history["datetime"].tail(72), y=history["total_load_mw"].tail(72),
            name="Recent Actual", line=dict(color="#1a1a2e")
        ))
        fig_f.add_trace(go.Scatter(
            x=forecast_df["datetime"], y=forecast_df["forecast_mw"],
            name=f"Forecast (+{forecast_horizon}h)", line=dict(color="#8B14E0", width=3)
        ))
        fig_f.update_layout(
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            yaxis_title="Load (MW)", margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_f, use_container_width=True)

        # Peak window detection
        peak_idx = forecast_df["forecast_mw"].idxmax()
        peak_time = forecast_df.loc[peak_idx, "datetime"]
        peak_value = forecast_df.loc[peak_idx, "forecast_mw"]

        st.markdown(f"""
        <div class="section-card">
        <h4>⚠️ Peak Window Detected</h4>
        <p><b>Predicted peak:</b> {peak_value:.1f} MW at <b>{peak_time.strftime('%Y-%m-%d %H:%M')}</b></p>
        <p style="font-size:0.85rem; color:#555;">
        💡 Business benefit: Pre-charge batteries 2-3 hrs before peak, schedule generation, avoid demand penalty charges.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.download_button(
            "📥 Download forecast CSV",
            forecast_df.to_csv(index=False).encode(),
            "load_forecast.csv", "text/csv"
        )

    # ----- SHAP TAB -----
    with tab_shap:
        st.markdown("#### Global feature importance (SHAP values)")
        st.caption("Which features drive load predictions most across all data?")

        shap_vals = st.session_state.shap_values
        feature_names = st.session_state.feature_cols
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        imp_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=True)

        fig_shap = px.bar(imp_df, x="mean_abs_shap", y="feature", orientation="h",
                         color="mean_abs_shap", color_continuous_scale="Purples")
        fig_shap.update_layout(
            height=400, plot_bgcolor="white", paper_bgcolor="white",
            yaxis_title="", xaxis_title="Mean |SHAP value| (MW impact)",
            coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # Single-prediction explanation
        st.markdown("#### Why was a specific prediction made?")
        idx_choice = st.slider("Sample index", 0, len(st.session_state.X_sample) - 1, 0)
        sample_row = st.session_state.X_sample.iloc[idx_choice]
        sample_shap = shap_vals[idx_choice]

        contrib_df = pd.DataFrame({
            "feature": feature_names,
            "value": sample_row.values,
            "shap_contribution_mw": sample_shap,
        }).sort_values("shap_contribution_mw", key=abs, ascending=True)

        fig_w = go.Figure(go.Bar(
            x=contrib_df["shap_contribution_mw"],
            y=contrib_df["feature"],
            orientation="h",
            marker_color=["#8B14E0" if v > 0 else "#E0148B" for v in contrib_df["shap_contribution_mw"]],
            text=[f"{v:+.1f} MW" for v in contrib_df["shap_contribution_mw"]],
            textposition="outside",
        ))
        fig_w.update_layout(
            height=400, plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Contribution to prediction (MW)",
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_w, use_container_width=True)

        st.caption("🟣 Purple bars push prediction up • 🩷 Pink bars push it down")

    # ----- SCENARIO TAB -----
    with tab_scenario:
        st.markdown("#### What-if hypothetical scenarios")
        st.caption("Adjust conditions and see how predicted load would change.")

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            sc_temp = st.slider("Temperature (°C)", 15.0, 45.0,
                                float(df["temperature_c"].mean()), 0.5)
            sc_hum = st.slider("Humidity (%)", 20, 95, int(df["humidity_pct"].mean()))
        with sc2:
            sc_hour = st.slider("Hour", 0, 23, 18)
            sc_dow = st.selectbox("Day of week",
                                  ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], index=0)
        with sc3:
            sc_holiday = st.checkbox("Public holiday", value=False)
            sc_month = st.selectbox("Month",
                                    list(range(1, 13)), index=int(df["datetime"].dt.month.mode()[0]) - 1)

        sc_dow_num = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(sc_dow)
        sc_weekend = 1 if sc_dow_num >= 5 else 0

        scenario_row = pd.DataFrame([{
            "hour": sc_hour,
            "dow_num": sc_dow_num,
            "month": sc_month,
            "day": 15,
            "is_weekend": sc_weekend,
            "is_holiday": int(sc_holiday),
            "temperature_c": sc_temp,
            "humidity_pct": sc_hum,
            "load_lag_24h": df["total_load_mw"].mean(),
            "load_lag_168h": df["total_load_mw"].mean(),
            "load_roll_24h_mean": df["total_load_mw"].mean(),
        }])

        sc_pred = st.session_state.model.predict(scenario_row[st.session_state.feature_cols])[0]

        baseline = df["total_load_mw"].mean()
        delta = sc_pred - baseline
        delta_pct = (delta / baseline) * 100

        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("Predicted load", f"{sc_pred:.1f} MW")
        sm2.metric("vs avg", f"{delta:+.1f} MW", delta_color="inverse")
        sm3.metric("Difference", f"{delta_pct:+.1f}%", delta_color="inverse")

        st.markdown(f"""
        <div class="section-card">
        <p>💡 If conditions are <b>{sc_temp:.1f}°C, {sc_hum}% humidity, hour {sc_hour}, {sc_dow}</b>{' (holiday)' if sc_holiday else ''},
        the model predicts <b>{sc_pred:.1f} MW</b>. Use this for capacity planning, demand response triggers, and battery scheduling.</p>
        </div>
        """, unsafe_allow_html=True)

    # ----- CHAT TAB -----
    with tab_chat:
        st.markdown("#### Ask the agent about your forecast")
        st.caption("Powered by Gemini — interprets results, explains SHAP, suggests actions.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            css = "chat-user" if msg["role"] == "user" else "chat-agent"
            label = "You" if msg["role"] == "user" else "🤖 Agent"
            st.markdown(f'<div class="{css}"><b>{label}:</b> {msg["text"]}</div>',
                        unsafe_allow_html=True)

        user_q = st.chat_input("e.g. Why is the peak so high tomorrow evening?")
        if user_q:
            if not api_key:
                st.error("Please add your Gemini API key in the sidebar.")
            else:
                genai.configure(api_key=api_key)
                # Build rich context for the agent
                context = f"""
You are a senior load forecasting analyst for an electric utility (UC-01 use case).
You have access to a trained XGBoost model and SHAP explanations.

Model performance:
- MAE: {st.session_state.metrics['MAE']:.2f} MW
- MAPE: {st.session_state.metrics['MAPE']:.2f}%
- R²: {st.session_state.metrics['R²']:.3f}

Top SHAP features (most important):
{imp_df.tail(5).to_string(index=False)}

Recent dataset stats:
- Rows: {len(df)}
- Date range: {df['datetime'].min()} to {df['datetime'].max()}
- Avg total load: {df['total_load_mw'].mean():.1f} MW
- Peak load: {df['total_load_mw'].max():.1f} MW
- Avg temperature: {df['temperature_c'].mean():.1f}°C

Forecast horizon: {forecast_horizon} hours
Retraining cadence: {retrain_cadence}

Answer questions clearly and concisely. Suggest concrete utility actions
(battery scheduling, demand response, generation commitment) where relevant.
"""
                try:
                    client = genai.Client(api_key=api_key)
                    full_prompt = "\n".join(
                        [f"{m['role']}: {m['text']}" for m in st.session_state.chat_history]
                        + [f"user: {user_q}"]
                    )
                    response = client.models.generate_content(
                        model=model_name,
                        contents=full_prompt,
                        config=genai_types.GenerateContentConfig(
                            system_instruction=context,
                        ),
                    )
                    answer = response.text
                    st.session_state.chat_history.append({"role": "user", "text": user_q})
                    st.session_state.chat_history.append({"role": "assistant", "text": answer})
                    st.rerun()
                except Exception as e:
                    st.error(f"Gemini error: {e}")

        st.caption("💡 Try: *Why is load high in evenings?* • *What if temp rises 5°C tomorrow?* • *Explain the top SHAP feature*")

else:
    st.info("👆 Click **Train XGBoost Model** to start.")

st.divider()

# ============== BUSINESS BENEFITS ==============
st.markdown("### 💼 Business Benefits")
b1, b2, b3, b4 = st.columns(4)
benefits = [
    ("🎯", "Reduce demand penalty", "Predict peak windows 24-72 hr ahead"),
    ("⚙️", "Optimize scheduling", "Day-ahead generation & unit commitment"),
    ("🔋", "Battery pre-charging", "Proactive prep before demand surges"),
    ("💰", "Lower balancing costs", "Reduce utility balancing mechanism costs"),
]
for col, (icon, title, desc) in zip([b1, b2, b3, b4], benefits):
    with col:
        st.markdown(f"""
        <div class="section-card" style="height:130px;">
        <div style="font-size:1.6rem;">{icon}</div>
        <b>{title}</b>
        <p style="font-size:0.8rem; color:#555; margin-top:4px;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)
