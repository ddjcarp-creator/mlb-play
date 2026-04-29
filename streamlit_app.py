import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from pybaseball import statcast
from sklearn.ensemble import RandomForestClassifier

import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Savant Replica Lite", layout="wide")

st.title("⚾ Baseball Savant Replica (Streamlit Lite)")
st.caption("Stable Statcast-based pitcher vs batter analytics dashboard")

# ----------------------------
# SAFE DATA LOAD (CRITICAL FIX)
# ----------------------------
@st.cache_data(show_spinner=True)
def load_data():
    df = statcast("2024-04-01", "2024-04-07")

    # Defensive cleaning (IMPORTANT FIX)
    needed_cols = ["launch_speed", "launch_angle", "events", "batter", "pitcher"]
    df = df.dropna(subset=[c for c in needed_cols if c in df.columns])

    return df


df = load_data()

# ----------------------------
# MODEL (SAFE + NO RE-TRAIN LOOP)
# ----------------------------
MODEL_PATH = "hr_model.pkl"

@st.cache_resource
def train_model(data):
    data = data.copy()

    # Safety check
    if "events" not in data.columns:
        return None

    data["home_run"] = (data["events"] == "home_run").astype(int)

    X = data[["launch_speed", "launch_angle"]]
    y = data["home_run"]

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    return model


if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = train_model(df)

# ----------------------------
# SIDEBAR INPUTS
# ----------------------------
st.sidebar.header("Matchup Controls")

batter_id = st.sidebar.number_input("Batter ID", value=592450)
pitcher_id = st.sidebar.number_input("Pitcher ID", value=425844)

# ----------------------------
# SAFE FILTER FUNCTIONS
# ----------------------------
def safe_filter_batter(data, batter_id):
    if "batter" not in data.columns:
        return pd.DataFrame()
    return data[data["batter"] == batter_id]


def safe_filter_pitcher(data, pitcher_id):
    if "pitcher" not in data.columns:
        return pd.DataFrame()
    return data[data["pitcher"] == pitcher_id]


batter_df = safe_filter_batter(df, batter_id)
pitcher_df = safe_filter_pitcher(df, pitcher_id)

# ----------------------------
# MAIN ANALYSIS
# ----------------------------
if st.sidebar.button("Run Savant Analysis"):

    st.header("📊 Player Profiles")

    col1, col2, col3 = st.columns(3)

    # ---------------- BATTER ----------------
    with col1:
        st.subheader("Batter")

        if not batter_df.empty:
            st.metric("Avg Exit Velocity", round(batter_df["launch_speed"].mean(), 2))
            st.metric("Avg Launch Angle", round(batter_df["launch_angle"].mean(), 2))

            barrel = (
                (batter_df["launch_speed"] >= 98) &
                (batter_df["launch_angle"].between(25, 35))
            ).mean()

            st.metric("Barrel Rate", round(barrel, 3))
        else:
            st.warning("No batter data in sample range")

    # ---------------- PITCHER ----------------
    with col2:
        st.subheader("Pitcher")

        if not pitcher_df.empty:
            st.metric("Avg Velocity", round(pitcher_df["release_speed"].mean(), 2))

            k_rate = (pitcher_df["events"] == "strikeout").mean()
            st.metric("Strikeout Rate", round(k_rate, 3))
        else:
            st.warning("No pitcher data in sample range")

    # ---------------- EDGE ----------------
    with col3:
        st.subheader("Matchup Edge")

        if not batter_df.empty and not pitcher_df.empty:
            edge = batter_df["launch_speed"].mean() - pitcher_df["release_speed"].mean()

            st.metric("Power Edge", round(edge, 2))

            if edge > 5:
                st.success("Strong hitter advantage")
            elif edge > 0:
                st.info("Slight hitter advantage")
            else:
                st.warning("Pitcher advantage")

# ----------------------------
# HR PROBABILITY
# ----------------------------
st.divider()
st.header("🔥 Home Run Probability Model")

if model is not None:

    ls = st.slider("Exit Velocity", 50, 120, 95)
    la = st.slider("Launch Angle", -10, 60, 25)

    prob = model.predict_proba([[ls, la]])[0][1]

    st.metric("HR Probability", f"{prob:.3f}")

else:
    st.error("Model not available")

# ----------------------------
# STRIKE ZONE HEATMAP
# ----------------------------
st.divider()
st.header("🎯 Pitch Location Heatmap")

if not pitcher_df.empty and "plate_x" in pitcher_df.columns and "plate_z" in pitcher_df.columns:

    heat = pitcher_df.dropna(subset=["plate_x", "plate_z"])

    fig = go.Figure()

    fig.add_trace(go.Histogram2dContour(
        x=heat["plate_x"],
        y=heat["plate_z"],
        colorscale="Blues",
        contours=dict(showlabels=False)
    ))

    # Strike zone box (approx MLB zone)
    fig.add_trace(go.Scatter(
        x=[-0.83, 0.83, 0.83, -0.83, -0.83],
        y=[1.5, 1.5, 3.5, 3.5, 1.5],
        mode="lines",
        line=dict(color="red", width=2),
        name="Strike Zone"
    ))

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# SPRAY CHART
# ----------------------------
st.header("⚾ Spray Chart")

if not batter_df.empty and "hc_x" in batter_df.columns and "hc_y" in batter_df.columns:

    spray = batter_df.dropna(subset=["hc_x", "hc_y"]).sample(
        min(len(batter_df), 300)
    )

    fig = px.scatter(
        spray,
        x="hc_x",
        y="hc_y",
        color="launch_speed",
        color_continuous_scale="reds",
        title="Batted Ball Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# RAW DATA
# ----------------------------
st.divider()
st.header("📦 Raw Statcast Data")

tab1, tab2 = st.tabs(["Batter", "Pitcher"])

with tab1:
    st.dataframe(batter_df.head(50))

with tab2:
    st.dataframe(pitcher_df.head(50))
