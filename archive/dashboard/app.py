"""
Streamlit Dashboard — Digital Twin Microgrid Monitor.

Run with:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys

# Add parent dir for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(
    page_title="Digital Twin — Microgrid Monitor",
    layout="wide",
)

st.title("⚡ Digital Twin: Microgrid Monitoring & Blackout Prediction")


# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'simulation_results.csv')
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


df = load_data()

if df is None:
    st.error("⚠️ No simulation data found. Run `python main.py` first to generate results.")
    st.stop()


# ── Add engineered features (for ML prediction) ──────────────────────────────

def add_features(df):
    df = df.copy()
    df['solar_load_ratio'] = df['p_solar_mw'] / (df['p_load_mw'] + 1e-6)
    df['soc_deficit'] = 1.0 - df['soc']
    df['v_violation_margin'] = df['v_min'] - 0.92
    df['net_power_balance'] = df['p_solar_mw'] + df['p_battery_mw'] - df['p_load_mw']
    df['is_night'] = ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 20)).astype(int)
    return df

df_feat = add_features(df)


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("📊 View Options")
max_day = int(df['day'].max())
day_range = st.sidebar.slider(
    "Select Day Range",
    0, max_day,
    (0, min(7, max_day)),
)

filtered = df[(df['day'] >= day_range[0]) & (df['day'] <= day_range[1])]


# ── Row 1: KPIs ──────────────────────────────────────────────────────────────

st.subheader("📈 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Timesteps", len(filtered))
col2.metric("Blackout Hours", int(filtered['blackout'].sum()))
lolp = filtered['blackout'].mean() * 100 if len(filtered) > 0 else 0
col3.metric("LOLP", f"{lolp:.1f}%")
col4.metric("Min Voltage (pu)", f"{filtered['v_min'].min():.4f}" if len(filtered) > 0 else "N/A")


# ── Row 2: Power Profiles ────────────────────────────────────────────────────

st.subheader("⚡ Power Profiles")
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(filtered['timestep'], filtered['p_load_mw'], label='Load (MW)', linewidth=1)
ax.plot(filtered['timestep'], filtered['p_solar_mw'], label='Solar (MW)', linewidth=1)
ax.plot(filtered['timestep'], filtered['p_battery_mw'], label='Battery (MW)', linewidth=1)
ax.fill_between(
    filtered['timestep'],
    filtered['blackout'] * filtered['p_load_mw'].max(),
    alpha=0.2, color='red', label='Blackout Event'
)
ax.legend()
ax.set_xlabel("Timestep")
ax.set_ylabel("MW")
ax.grid(True, alpha=0.3)
st.pyplot(fig)


# ── Row 3: Voltage + SOC ─────────────────────────────────────────────────────

col_v, col_s = st.columns(2)

with col_v:
    st.subheader("🔌 Voltage Profile")
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.plot(filtered['timestep'], filtered['v_min'], label='V_min (pu)', color='orange')
    ax2.axhline(0.92, color='red', linestyle='--', label='0.92 pu limit')
    ax2.legend()
    ax2.set_ylabel("Voltage (pu)")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

with col_s:
    st.subheader("🔋 Battery SOC")
    fig3, ax3 = plt.subplots(figsize=(7, 3))
    ax3.plot(filtered['timestep'], filtered['soc'], color='green', label='SOC')
    ax3.axhline(0.1, color='red', linestyle='--', label='SOC_min')
    ax3.axhline(0.9, color='blue', linestyle='--', label='SOC_max')
    ax3.legend()
    ax3.set_ylabel("State of Charge")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)


# ── Row 4: Blackout Probability (if model available) ─────────────────────────

model_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'model', 'best_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'model', 'scaler.pkl')
feat_path  = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'model', 'feature_cols.pkl')

if os.path.exists(model_path) and os.path.exists(scaler_path):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Load feature list from the saved model training — avoids hardcoded mismatch
        if os.path.exists(feat_path):
            feat_cols = joblib.load(feat_path)
        else:
            feat_cols = [c for c in df_feat.columns if c not in ['timestep','day','blackout','severity','converged']]

        available_cols = [c for c in feat_cols if c in df_feat.columns]
        X = df_feat[available_cols].values
        probs = model.predict_proba(X)[:, 1]
        df['blackout_prob'] = probs

        filtered2 = df[(df['day'] >= day_range[0]) & (df['day'] <= day_range[1])]

        st.subheader("🎯 Probabilistic Blackout Risk")
        fig4, ax4 = plt.subplots(figsize=(14, 3))
        ax4.plot(filtered2['timestep'], filtered2['blackout_prob'],
                 color='crimson', label='Blackout Probability')
        ax4.axhline(0.5, color='grey', linestyle='--', label='0.5 threshold')
        ax4.set_ylabel("P(Blackout)")
        ax4.set_xlabel("Timestep")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)

    except Exception as e:
        st.warning(f"Could not load ML model: {e}")
else:
    st.info("💡 Train the ML model first (run main.py) to see probabilistic predictions.")


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption("Digital Twin Microgrid — Mini Project | IEEE 33-Bus + PV + Battery | PHYSIC-DT-RISK Algorithm")
