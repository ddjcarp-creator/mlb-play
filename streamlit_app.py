import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import plotly.express as px
import plotly.graph_objects as go

from pybaseball import statcast_batter, statcast_pitcher

from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="MLB Savant-Lite", layout="wide")

st.title("⚾ MLB Savant-Lite: Pitcher vs Batter Intelligence System")

st.caption("Statcast-powered analytics + ML-driven HR probability + visual dashboards")

# ----------------------------
# MODEL (UPGRADED FROM LOGISTIC REGRESSION)
# ----------------------------
MODEL_PATH = "hr_model.pkl"

@st.cache_resource
def train_model():
    data = statcast_batter("2024-04-01", "2024-10-01")
    data = data.dropna(subset=["launch_speed", "launch_angle", "home_run"])

    X = data[["launch_speed", "launch_angle"]]
    y = data["home_run"]

    model = RandomForestClassifier(n_estimators=150, max_depth=6)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    return model


if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = train_model()

# ----------------------------
# FEATURE ENGINEERING HELPERS
# ----------------------------
def estimate_xwoba(df):
    if df.empty:
        return 0
    weights = {
        "single": 0.9,
        "double": 1.25,
        "triple": 1.6,
        "home_run": 2.0
    }
    df["value"] = df["events"].map(weights).fillna(0)
    return df["value"].mean()


def barrel_rate(df):
    if df.empty:
        return 0
    barrels = df[(df["launch_speed"] >= 98) & (df["launch_angle"].between(26, 30))]
    return len(barrels) / len(df)


def pitch_type_split(df):
    if "pitch_type" not in df.columns:
        return {}
    return df["pitch_type"].value_counts(normalize=True).to_dict()

# ----------------------------
# INPUTS
# ----------------------------
st.sidebar.header("Player Inputs")

batter_id = st.sidebar.number_input("Batter ID", value=592450)
pitcher_id = st.sidebar.number_input("Pitcher ID", value=425844)

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data(show_spinner=False)
def load_data(batter_id, pitcher_id):
    batter = statcast_batter("2024-04-01", "2024-10-01", player_id=batter_id)
    pitcher = statcast_pitcher("2024-04-01", "2024-10-01", player_id=pitcher_id)
    return batter, pitcher


if st.sidebar.button("Run Full Analysis"):

    batter, pitcher = load_data(batter_id, pitcher_id)

    # ----------------------------
    # METRICS
    # ----------------------------
    st.header("📊 Player Profiles")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Batter Metrics")
        st.metric("Avg EV", round(batter["launch_speed"].mean(), 2))
        st.metric("Avg LA", round(batter["launch_angle"].mean(), 2))
        st.metric("Barrel Rate", round(barrel_rate(batter), 3))
        st.metric("xwOBA (est)", round(estimate_xwoba(batter), 3))

    with col2:
        st.subheader("Pitcher Metrics")
        st.metric("Avg Velocity", round(pitcher["release_speed"].mean(), 2))
        st.metric("Strikeout Rate", round((pitcher["events"] == "strikeout").mean(), 3))
        st.metric("xSLG (est)", round(estimate_xwoba(pitcher), 3))

    with col3:
        st.subheader("Matchup Edge Score")

        edge = batter["launch_speed"].mean() - pitcher["release_speed"].mean()

        st.metric("Power Differential", round(edge, 2))

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
    st.header("⚾ Home Run Probability Engine")

    ls = st.slider("Exit Velocity", 50, 120, 95)
    la = st.slider("Launch Angle", -10, 60, 25)

    prob = model.predict_proba([[ls, la]])[0][1]

    st.metric("HR Probability", f"{prob:.3f}")

    # ----------------------------
    # SPRAY CHART (SIMULATED)
    # ----------------------------
    st.divider()
    st.header("📍 Spray Chart (Approximation)")

    if not batter.empty:
        sample = batter.dropna(subset=["hc_x", "hc_y"]).sample(min(500, len(batter)))

        fig = px.scatter(
            sample,
            x="hc_x",
            y="hc_y",
            color="launch_speed",
            title="Batted Ball Distribution",
            color_continuous_scale="reds"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # STRIKE ZONE HEATMAP
    # ----------------------------
    st.header("🎯 Strike Zone Heatmap (Pitcher)")

    if not pitcher.empty and "plate_x" in pitcher.columns:

        heat = pitcher.dropna(subset=["plate_x", "plate_z"])

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
    # PITCH TYPE BREAKDOWN
    # ----------------------------
    st.header("⚾ Pitch Type Distribution")

    pitch_dist = pitch_type_split(pitcher)

    if pitch_dist:
        fig = px.pie(values=list(pitch_dist.values()), names=list(pitch_dist.keys()))
        st.plotly_chart(fig)

    # ----------------------------
    # RAW DATA
    # ----------------------------
    st.divider()
    st.header("📦 Raw Data Explorer")

    tab1, tab2 = st.tabs(["Batter Data", "Pitcher Data"])

    with tab1:
        st.dataframe(batter.head(50))

    with tab2:
        st.dataframe(pitcher.head(50))
