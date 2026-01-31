# ============================================
# FULL STREAMLIT APP – ELECTRICAL LOAD FORECAST
# Built directly from Electrical_load.ipynb
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# --------------------------------------------
# PAGE CONFIG
# --------------------------------------------
st.set_page_config(
    page_title="Electrical Load Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Electrical Load Forecasting App")
st.markdown("Prediction-only deployment of the **XGBoost time-series model** trained in the Jupyter notebook.")

# --------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("xgb_tuned_load_forecast_model_.pkl")

model = load_model()
model_features = model.feature_names_in_

# --------------------------------------------
# FEATURE ENGINEERING (SAME AS NOTEBOOK)
# --------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['quarter'] = df.index.quarter
    df['is_weekend'] = (df.index.weekday >= 5).astype(int)

    df['demand_lag_24hr'] = df['demand'].shift(24)
    df['demand_lag_168hr'] = df['demand'].shift(168)
    df['demand_rolling_mean_24hr'] = df['demand'].rolling(24).mean()
    df['demand_rolling_std_24hr'] = df['demand'].rolling(24).std()

    return df.dropna()

# --------------------------------------------
# SIDEBAR MODE SELECTION
# --------------------------------------------
st.sidebar.header("Mode Selection")
mode = st.sidebar.radio(
    "Choose prediction type",
    ["Single Forecast", "Batch CSV Forecast"]
)

# ============================================
# SINGLE FORECAST MODE
# ============================================
if mode == "Single Forecast":
    st.subheader("🔮 Single-Time Forecast")

    st.markdown("Upload **recent historical demand data** (minimum 168 hours).")

    uploaded_file = st.file_uploader(
        "Upload CSV (columns: datetime, demand)",
        type="csv"
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=['datetime'])
        df = df.set_index('datetime')
        df = df.sort_index()

        if len(df) < 168:
            st.error("At least 168 hours of data is required for lag features.")
        else:
            features_df = engineer_features(df)
            X_latest = features_df[model_features].iloc[-1:]

            forecast_date = st.date_input("Forecast Date", X_latest.index[0].date())
            forecast_hour = st.slider("Forecast Hour", 0, 23, X_latest.index[0].hour)

            if st.button("Predict Load"):
                prediction = model.predict(X_latest)[0]
                st.success(f"Predicted Electrical Load: **{prediction:.2f} MW**")

                st.markdown("### Recent Load Trend")
                fig, ax = plt.subplots()
                ax.plot(df.tail(200).index, df.tail(200)['demand'], label='Actual Load')
                ax.legend()
                st.pyplot(fig)

# ============================================
# BATCH CSV FORECAST MODE
# ============================================
if mode == "Batch CSV Forecast":
    st.subheader("📂 Batch Load Forecast")

    st.markdown("Upload a CSV file containing continuous historical demand data.")

    uploaded_file = st.file_uploader(
        "Upload CSV (columns: datetime, demand)",
        type="csv",
        key="batch"
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=['datetime'])
        df = df.set_index('datetime')
        df = df.sort_index()

        if len(df) < 168:
            st.error("At least 168 hours of data is required for batch forecasting.")
        else:
            features_df = engineer_features(df)
            X = features_df[model_features]

            features_df['prediction'] = model.predict(X)

            st.success("Batch prediction completed successfully")

            st.markdown("### Forecast Results")
            st.dataframe(features_df[['demand', 'prediction']].tail(100))

            st.markdown("### Actual vs Predicted Load")
            fig, ax = plt.subplots()
            ax.plot(features_df.index, features_df['demand'], label='Actual')
            ax.plot(features_df.index, features_df['prediction'], label='Predicted')
            ax.legend()
            st.pyplot(fig)

            st.download_button(
                label="Download Predictions as CSV",
                data=features_df.to_csv().encode('utf-8'),
                file_name="load_forecast_predictions.csv",
                mime="text/csv"
            )

# --------------------------------------------
# FOOTER
# --------------------------------------------
st.markdown("---")
st.caption("Electrical Load Forecasting • XGBoost • Streamlit Deployment")
