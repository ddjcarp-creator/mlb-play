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
# APP CONFIG
# ----------------------------
st.set_page_config(page_title="MLB Savant-Lite 2.0", layout="wide")

st.title("⚾ MLB Savant-Lite 2.0 (Stable Version)")
st.caption("Statcast-powered MLB analytics + ML home run model")

# ----------------------------
# SAFE DATA LOADING (FIXED CORE)
# ----------------------------
@st.cache_data(show_spinner=True)
def load_statcast():
    # small range for stability (expand later if self-hosting)
    df = statcast("2024-04-01", "2024-04-07")
    
    # clean
    df = df.dropna(subset=["launch_speed", "launch_angle"])
    return df


data = load_statcast()

# ----------------------------
# MODEL (SAFE TRAINING)
# ----------------------------
MODEL_PATH = "hr_model.pkl"

@st.cache_resource
def train_model(df):
    df = df.copy()
    df["home_run"] = (df["events"] == "home_run").astype(int)

    X = df[["launch_speed", "launch_angle"]]
    y = df["home_run"]

    model = RandomForestClassifier(n_estimators=120, max_depth=6)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    return model


if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = train_model(data)

# ----------------------------
# SIDEBAR INPUTS
# ----------------------------
st.sidebar.header("Matchup Inputs")

batter_id = st.sidebar.number_input("Batter ID", value=592450)
pitcher_id = st.sidebar.number_input("Pitcher ID", value=425844)

# ----------------------------
# FILTER PLAYER DATA SAFELY
# ----------------------------
def get_batter(df, batter_id):
    return df[df["batter"] == batter_id]

def get_pitcher(df, pitcher_id):
    return df[df["pitcher"] == pitcher_id]


batter_df = get_batter(data, batter_id)
pitcher_df = get_pitcher(data, pitcher_id)

# ----------------------------
# MATCHUP ANALYSIS
# ----------------------------
if st.sidebar.button("Run Analysis"):

    st.header("📊 Player Breakdown")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Batter")

        if not batter_df.empty:
            st.metric("Avg Exit Velocity", round(batter_df["launch_speed"].mean(), 2))
            st.metric("Avg Launch Angle", round(batter_df["launch_angle"].mean(), 2))
            st.metric("Barrel Rate", round(((batter_df["launch_speed"] > 98) &
                                            (batter_df["launch_angle"].between(26, 30))).mean(), 3))
        else:
            st.warning("No batter data found in sample range")

    with col2:
        st.subheader("Pitcher")

        if not pitcher_df.empty:
            st.metric("Avg Velocity", round(pitcher_df["release_speed"].mean(), 2))
            st.metric("Strikeout Rate", round((pitcher_df["events"] == "strikeout").mean(), 3))
        else:
            st.warning("No pitcher data found in sample range")

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
# HR PROBABILITY ENGINE
# ----------------------------
st.divider()
st.header("⚾ Home Run Probability Model")

ls = st.slider("Exit Velocity", 50, 120, 95)
la = st.slider("Launch Angle", -10, 60, 25)

prob = model.predict_proba([[ls, la]])[0][1]

st.metric("HR Probability", f"{prob:.3f}")

# ----------------------------
# SPRAY CHART
# ----------------------------
st.divider()
st.header("📍 Spray Chart")

if not batter_df.empty and "hc_x" in batter_df.columns:

    sample = batter_df.dropna(subset=["hc_x", "hc_y"]).sample(min(300, len(batter_df)))

    fig = px.scatter(
        sample,
        x="hc_x",
        y="hc_y",
        color="launch_speed",
        color_continuous_scale="reds",
        title="Batted Ball Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# STRIKE ZONE HEATMAP
# ----------------------------
st.header("🎯 Pitch Location Heatmap")

if not pitcher_df.empty and "plate_x" in pitcher_df.columns:

    heat = pitcher_df.dropna(subset=["plate_x", "plate_z"])

    fig = go.Figure()

    fig.add_trace(go.Histogram2dContour(
        x=heat["plate_x"],
        y=heat["plate_z"],
        colorscale="Blues",
        contours=dict(showlabels=True)
    ))

    fig.update_layout(
        title="Pitch Location Density",
        xaxis_title="Plate X",
        yaxis_title="Plate Z"
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# RAW DATA VIEW
# ----------------------------
st.divider()
st.header("📦 Raw Statcast Data")

tab1, tab2 = st.tabs(["Batter", "Pitcher"])

with tab1:
    st.dataframe(batter_df.head(50))

with tab2:
    st.dataframe(pitcher_df.head(50))
