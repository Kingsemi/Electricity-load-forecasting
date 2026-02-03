# ============================================
# FULL STREAMLIT APP – ELECTRICAL LOAD FORECAST
# Robust and Production-Ready
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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
# FEATURE ENGINEERING
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
# CSV LOADING & PREPROCESSING
# --------------------------------------------
def load_and_process_csv(uploaded_file):
    try:
        # Load CSV
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip().lower() for c in df.columns]

        # Detect datetime column
        datetime_col = None
        for col in df.columns:
            if 'date' in col or 'time' in col:
                datetime_col = col
                break
        if datetime_col is None:
            st.error("No datetime-like column found. Please ensure your CSV has a date/time column.")
            return None

        # Convert to datetime and set index
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        df = df.dropna(subset=[datetime_col])
        df = df.set_index(datetime_col).sort_index()

        # Check for demand column
        if 'demand' not in df.columns:
            st.error("CSV must contain a column named 'demand'.")
            return None

        # Interpolate missing demand
        df['demand'] = df['demand'].interpolate(method='linear')

        return df

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

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
        df = load_and_process_csv(uploaded_file)
        if df is not None:
            if len(df) < 168:
                st.error("At least 168 hours of data is required for lag features.")
            else:
                features_df = engineer_features(df)
                X_latest = features_df[model_features].iloc[-1:]

                # Forecast date & hour input
                forecast_date = st.date_input("Forecast Date", X_latest.index[-1].date())
                forecast_hour = st.slider("Forecast Hour", 0, 23, X_latest.index[-1].hour)
                forecast_dt = pd.Timestamp.combine(forecast_date, datetime.min.time()) + pd.Timedelta(hours=forecast_hour)
                X_latest.index = [forecast_dt]

                if st.button("Predict Load"):
                    try:
                        prediction = model.predict(X_latest)[0]
                        st.success(f"Predicted Electrical Load: **{prediction:.2f} MW**")

                        # Recent load trend
                        st.markdown("### Recent Load Trend")
                        fig, ax = plt.subplots()
                        ax.plot(df.tail(200).index, df.tail(200)['demand'], label='Actual Load')
                        ax.set_xlabel("Datetime")
                        ax.set_ylabel("Load (MW)")
                        ax.legend()
                        fig.autofmt_xdate()
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

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
        df = load_and_process_csv(uploaded_file)
        if df is not None:
            if len(df) < 168:
                st.error("At least 168 hours of data is required for batch forecasting.")
            else:
                features_df = engineer_features(df)
                X = features_df[model_features]

                try:
                    features_df['prediction'] = model.predict(X)
                    st.success("Batch prediction completed successfully")

                    # Forecast results table
                    st.markdown("### Forecast Results")
                    st.dataframe(features_df[['demand', 'prediction']].tail(100))

                    # Actual vs predicted plot
                    st.markdown("### Actual vs Predicted Load")
                    fig, ax = plt.subplots()
                    ax.plot(features_df.index, features_df['demand'], label='Actual')
                    ax.plot(features_df.index, features_df['prediction'], label='Predicted')
                    ax.set_xlabel("Datetime")
                    ax.set_ylabel("Load (MW)")
                    ax.legend()
                    fig.autofmt_xdate()
                    st.pyplot(fig)

                    # Download button
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=features_df.to_csv().encode('utf-8'),
                        file_name="load_forecast_predictions.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

# --------------------------------------------
# FOOTER
# --------------------------------------------
st.markdown("---")
st.caption("Electrical Load Forecasting • XGBoost • Streamlit Deployment")

