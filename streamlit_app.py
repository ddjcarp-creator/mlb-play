import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from pybaseball import statcast
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# PAGE SETUP
# ----------------------------
st.set_page_config(page_title="Savant Replica Lite", layout="wide")

st.title("⚾ Baseball Savant Replica (Lite)")
st.caption("Statcast-powered pitch + batted ball visualization system")

# ----------------------------
# LOAD DATA (SAFE + CACHE)
# ----------------------------
@st.cache_data
def load_data():
    df = statcast("2024-04-01", "2024-04-07")
    df = df.dropna(subset=["launch_speed", "launch_angle"])
    return df

df = load_data()

# ----------------------------
# MODEL (HR PROBABILITY)
# ----------------------------
@st.cache_resource
def train_model(data):
    data = data.copy()
    data["hr"] = (data["events"] == "home_run").astype(int)

    X = data[["launch_speed", "launch_angle"]]
    y = data["hr"]

    model = RandomForestClassifier(n_estimators=100, max_depth=6)
    model.fit(X, y)
    return model

model = train_model(df)

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
st.sidebar.header("Filters")

batter_id = st.sidebar.number_input("Batter ID", value=592450)
pitcher_id = st.sidebar.number_input("Pitcher ID", value=425844)

batter_df = df[df["batter"] == batter_id]
pitcher_df = df[df["pitcher"] == pitcher_id]

# ----------------------------
# MAIN DASHBOARD TABS
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🎯 Pitch Map",
    "⚾ Spray Chart",
    "🔥 HR Model"
])

# ----------------------------
# TAB 1: OVERVIEW
# ----------------------------
with tab1:
    st.header("Player Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Batter Metrics")

        if not batter_df.empty:
            st.metric("Avg Exit Velocity", round(batter_df["launch_speed"].mean(), 2))
            st.metric("Avg Launch Angle", round(batter_df["launch_angle"].mean(), 2))
            st.metric("Hard Hit %", round((batter_df["launch_speed"] > 95).mean(), 3))

    with col2:
        st.subheader("Pitcher Metrics")

        if not pitcher_df.empty:
            st.metric("Avg Velocity", round(pitcher_df["release_speed"].mean(), 2))
            st.metric("Strikeout %", round((pitcher_df["events"] == "strikeout").mean(), 3))

# ----------------------------
# TAB 2: PITCH MAP (SAVANT CORE)
# ----------------------------
with tab2:
    st.header("🎯 Pitch Location Heatmap (Savant Style)")

    pitch_data = pitcher_df.dropna(subset=["plate_x", "plate_z"])

    if not pitch_data.empty:

        fig = go.Figure()

        # Heatmap density
        fig.add_trace(go.Histogram2dContour(
            x=pitch_data["plate_x"],
            y=pitch_data["plate_z"],
            colorscale="Blues",
            contours=dict(showlabels=True),
            hoverinfo="skip"
        ))

        # Strike zone box (MLB standard approx)
        strike_zone = go.Scatter(
            x=[-0.83, 0.83, 0.83, -0.83, -0.83],
            y=[1.5, 1.5, 3.5, 3.5, 1.5],
            mode="lines",
            line=dict(color="red", width=2),
            name="Strike Zone"
        )

        fig.add_trace(strike_zone)

        fig.update_layout(
            title="Pitch Heatmap + Strike Zone Overlay",
            xaxis_title="Plate X",
            yaxis_title="Plate Z",
            plot_bgcolor="black"
        )

        st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# TAB 3: SPRAY CHART
# ----------------------------
with tab3:
    st.header("⚾ Batted Ball Spray Chart")

    spray = batter_df.dropna(subset=["hc_x", "hc_y"])

    if not spray.empty:

        fig = px.scatter(
            spray,
            x="hc_x",
            y="hc_y",
            color="launch_speed",
            size="launch_speed",
            color_continuous_scale="reds",
            title="Batted Ball Distribution"
        )

        # Field outline (simple approximation)
        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=-250, y0=-250,
            x1=250, y1=250,
            line=dict(color="green")
        )

        st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# TAB 4: HR PROBABILITY MODEL
# ----------------------------
with tab4:
    st.header("🔥 Home Run Probability Engine")

    ls = st.slider("Exit Velocity", 50, 120, 95)
    la = st.slider("Launch Angle", -10, 60, 25)

    prob = model.predict_proba([[ls, la]])[0][1]

    st.metric("HR Probability", f"{prob:.3f}")

    if prob > 0.5:
        st.success("Elite HR launch conditions")
    elif prob > 0.2:
        st.info("Moderate HR chance")
    else:
        st.warning("Low HR probability")
