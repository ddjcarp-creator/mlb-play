import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from pybaseball import statcast_batter, statcast_pitcher
from sklearn.linear_model import LogisticRegression

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="MLB Matchup Analyzer", layout="wide")

st.title("⚾ MLB Pitcher vs Batter Analyzer")
st.write("Statcast-powered matchup insights + HR probability model")

# ----------------------------
# LOAD / TRAIN MODEL
# ----------------------------
MODEL_PATH = "hr_model.pkl"

@st.cache_resource
def train_model():
    st.write("Training HR model (first run only)...")

    data = statcast_batter("2024-04-01", "2024-10-01")
    data = data.dropna(subset=["launch_speed", "launch_angle", "home_run"])

    X = data[["launch_speed", "launch_angle"]]
    y = data["home_run"]

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    return model


if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = train_model()

# ----------------------------
# SIDEBAR INPUTS
# ----------------------------
st.sidebar.header("Player Inputs")

batter_id = st.sidebar.number_input("Batter ID", value=592450)
pitcher_id = st.sidebar.number_input("Pitcher ID", value=425844)

# ----------------------------
# MATCHUP ANALYSIS
# ----------------------------
if st.sidebar.button("Analyze Matchup"):

    with st.spinner("Pulling Statcast data..."):

        try:
            batter = statcast_batter("2024-04-01", "2024-10-01", player_id=batter_id)
            pitcher = statcast_pitcher("2024-04-01", "2024-10-01", player_id=pitcher_id)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Batter Profile")
                st.metric("Avg Exit Velocity", round(batter["launch_speed"].mean(), 2))
                st.metric("Avg Launch Angle", round(batter["launch_angle"].mean(), 2))
                st.metric("HR Rate", round(batter["home_run"].mean(), 3))

            with col2:
                st.subheader("Pitcher Profile")
                st.metric("Avg Pitch Velocity", round(pitcher["release_speed"].mean(), 2))
                st.metric("Strikeout Rate", round((pitcher["events"] == "strikeout").mean(), 3))

            with col3:
                st.subheader("Power Matchup Insight")

                power_diff = batter["launch_speed"].mean() - pitcher["release_speed"].mean()

                st.metric("Power Differential", round(power_diff, 2))

                if power_diff > 5:
                    st.success("High offensive matchup advantage")
                elif power_diff > 0:
                    st.info("Slight hitter advantage")
                else:
                    st.warning("Pitcher advantage")

        except Exception as e:
            st.error(f"Error loading data: {e}")

# ----------------------------
# HR PROBABILITY TOOL
# ----------------------------
st.divider()
st.header("⚾ Home Run Probability Model")

col1, col2 = st.columns(2)

with col1:
    launch_speed = st.slider("Exit Velocity (mph)", 50, 120, 95)

with col2:
    launch_angle = st.slider("Launch Angle (degrees)", -10, 60, 25)

if st.button("Calculate HR Probability"):

    prob = model.predict_proba([[launch_speed, launch_angle]])[0][1]

    st.subheader("Result")
    st.metric("Home Run Probability", f"{prob:.3f}")

    if prob > 0.5:
        st.success("High HR likelihood")
    elif prob > 0.2:
        st.info("Moderate HR chance")
    else:
        st.warning("Low HR probability")

# ----------------------------
# RAW DATA VIEWER (OPTIONAL)
# ----------------------------
st.divider()
st.header("📊 Raw Statcast Snapshot")

if st.checkbox("Show Batter Data Sample"):
    try:
        sample = statcast_batter("2024-04-01", "2024-04-05", player_id=batter_id)
        st.dataframe(sample.head(20))
    except:
        st.warning("No data available for this player/sample range")
