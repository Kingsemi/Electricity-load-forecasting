# load_forecast_model.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ===============================
# FUTURE FORECAST ENGINE
# ===============================
def forecast_future(df, model, feature_cols, horizon=24, temp_future=None):
    future_preds = []
    last = df.copy()

    for i in range(horizon):
        row = last.iloc[-1:].copy()

        # Inject future temperature
        if temp_future is not None:
            row["temperature"] = temp_future[i]

        # Lag features
        row["demand_lag_168hr"] = last["demand"].iloc[-168]
        row["demand_rolling_mean_24hr"] = last["demand"].iloc[-24:].mean()
        row["demand_rolling_std_24hr"] = last["demand"].iloc[-24:].std()

        X = row[feature_cols]
        y_hat = model.predict(X)[0]

        new_time = row.index[0] + pd.Timedelta(hours=1)
        new_row = row.copy()
        new_row.index = [new_time]
        new_row["demand"] = y_hat

        last = pd.concat([last, new_row])
        future_preds.append((new_time, y_hat))

    return pd.DataFrame(future_preds, columns=["date","forecast"]).set_index("date")

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Electricity Load Forecasting", layout="wide")
st.title("⚡ Future Electricity Load Forecast")

# -----------------------
# Upload historical data
# -----------------------
uploaded_file = st.file_uploader("Upload historical load CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required = {"date", "demand", "temperature"}
    if not required.issubset(df.columns):
        st.error("CSV must contain: date, demand, temperature")
        st.stop()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").asfreq("H")

    # Time features
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["weekofyear"] = df.index.isocalendar().week.astype(int)
    df["quarter"] = df.index.quarter
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # Lag & rolling
    df["demand_lag_168hr"] = df["demand"].shift(168)
    df["demand_rolling_mean_24hr"] = df["demand"].rolling(24).mean()
    df["demand_rolling_std_24hr"] = df["demand"].rolling(24).std()

    df = df.dropna()

    st.success("Data loaded successfully!")

    # -----------------------
    # Load trained pipeline
    # -----------------------
    model_file = st.file_uploader("Upload load_forecast_pipeline.pkl", type=["pkl"])
    features_file = st.file_uploader("Upload model_features.pkl", type=["pkl"])

    if model_file and features_file:
        pipe = pickle.load(model_file)
        feature_cols = pickle.load(features_file)

        st.success("Model & features loaded!")

        # -----------------------
        # Forecast controls
        # -----------------------
        horizon = st.slider("Forecast horizon (hours)", 1, 168, 24)
        temp_value = st.slider("Expected temperature (°C)", 10, 45, 30)

        temp_future = [temp_value] * horizon

        future_df = forecast_future(df, pipe, feature_cols, horizon, temp_future)

        # -----------------------
        # Plot
        # -----------------------
        fig, ax = plt.subplots(figsize=(15,5))
        ax.plot(df.tail(200).index, df.tail(200)["demand"], label="History")
        ax.plot(future_df.index, future_df["forecast"], label="Forecast", linestyle="--")
        ax.set_xlabel("Date")
        ax.set_ylabel("Demand")
        ax.legend()
        st.pyplot(fig)

        # -----------------------
        # Show table
        # -----------------------
        st.subheader("Forecasted Demand")
        st.dataframe(future_df)
