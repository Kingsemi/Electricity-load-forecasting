import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from collections import deque
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.inspection import permutation_importance

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(page_title="Electrical Load Forecasting", layout="wide")

MODEL_PATH = "xgb_tuned_load_forecast_model_.pkl"

FEATURE_COLS = [
    "hour", "month", "weekofyear", "quarter", "is_weekend",
    "demand_lag_24hr", "demand_lag_168hr",
    "demand_rolling_mean_24hr", "demand_rolling_std_24hr"
]

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# =========================================================
# FEATURE ENGINEERING
# =========================================================

def create_features(data):

    df = data.copy()

    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["weekofyear"] = df.index.isocalendar().week.astype(int)
    df["quarter"] = df.index.quarter
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    df["demand_lag_24hr"] = df["demand"].shift(24)
    df["demand_lag_168hr"] = df["demand"].shift(168)

    df["demand_rolling_mean_24hr"] = df["demand"].shift(1).rolling(24).mean()
    df["demand_rolling_std_24hr"] = df["demand"].shift(1).rolling(24).std()

    return df

# =========================================================
# FAST FORECAST O(N)
# =========================================================

def forecast_future_fast(model, history, horizon):

    hist = history.copy()

    lag24 = deque(hist["demand"].iloc[-24:], maxlen=24)
    lag168 = deque(hist["demand"].iloc[-168:], maxlen=168)
    roll24 = deque(hist["demand"].iloc[-24:], maxlen=24)

    preds = []
    last_time = hist.index[-1]

    for _ in range(horizon):

        next_time = last_time + pd.Timedelta(hours=1)

        row = pd.DataFrame([{
            "hour": next_time.hour,
            "month": next_time.month,
            "weekofyear": int(next_time.isocalendar().week),
            "quarter": next_time.quarter,
            "is_weekend": int(next_time.dayofweek >= 5),
            "demand_lag_24hr": lag24[0],
            "demand_lag_168hr": lag168[0],
            "demand_rolling_mean_24hr": np.mean(roll24),
            "demand_rolling_std_24hr": np.std(roll24)
        }])

        y = model.predict(row)[0]

        preds.append(y)
        lag24.append(y)
        lag168.append(y)
        roll24.append(y)

        last_time = next_time

    idx = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="H")

    return pd.DataFrame({"demand": preds}, index=idx)

# =========================================================
# CONFIDENCE INTERVALS
# =========================================================

def forecast_with_ci(model, history, horizon):

    sims = []

    for _ in range(25):
        noisy = history.copy()
        noisy["demand"] += np.random.normal(0, history["demand"].std()*0.05, len(history))
        f = forecast_future_fast(model, noisy, horizon)
        sims.append(f["demand"].values)

    sims = np.array(sims)

    idx = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="H")

    return pd.DataFrame({
        "P10": np.percentile(sims, 10, axis=0),
        "P50": np.percentile(sims, 50, axis=0),
        "P90": np.percentile(sims, 90, axis=0)
    }, index=idx)

# =========================================================
# UI
# =========================================================

st.title("⚡ Electrical Load Forecasting")

horizon = st.sidebar.slider("Forecast Horizon (hours)", 24, 168, 24)

uploaded = st.file_uploader("Upload CSV with columns: date, demand", type="csv")

# =========================================================
# MAIN
# =========================================================

if uploaded:

    df = pd.read_csv(uploaded)

    df["LoadDate"] = pd.to_datetime(df["date"])
    df = df.set_index("LoadDate").sort_index()

    df_feat = create_features(df).dropna()

    st.subheader("Recent Load")
    st.line_chart(df_feat["demand"].tail(168))

    # ---------------- MAPE ----------------
    split = int(len(df_feat)*0.8)

    X_test = df_feat[FEATURE_COLS].iloc[split:]
    y_test = df_feat["demand"].iloc[split:]

    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds)*100

    st.metric("MAPE (%)", f"{mape:.2f}")

    # ---------------- SINGLE PREDICTION ----------------
    st.subheader("🔮 Predict Next Hour")

    predict_time = st.datetime_input(
        "Prediction Time",
        value=df_feat.index[-1] + pd.Timedelta(hours=1)
    )

    if st.button("Predict Load"):

        lag24 = df_feat["demand"].iloc[-24]
        lag168 = df_feat["demand"].iloc[-168]

        row = pd.DataFrame([{
            "hour": predict_time.hour,
            "month": predict_time.month,
            "weekofyear": int(predict_time.isocalendar().week),
            "quarter": (predict_time.month - 1)//3 + 1,
            "is_weekend": int(predict_time.weekday() >= 5),
            "demand_lag_24hr": lag24,
            "demand_lag_168hr": lag168,
            "demand_rolling_mean_24hr": df_feat["demand"].iloc[-24:].mean(),
            "demand_rolling_std_24hr": df_feat["demand"].iloc[-24:].std()
        }])

        y = model.predict(row)[0]
        st.success(f"Predicted Load: {y:.2f}")

    # ---------------- FEATURE IMPORTANCE ----------------
    st.subheader("Feature Importance")

    xgb = model.named_steps["model"]
    fi = pd.Series(xgb.feature_importances_, index=FEATURE_COLS).sort_values()

    fig1, ax1 = plt.subplots(figsize=(4,3))
    fi.plot(kind="barh", ax=ax1)
    ax1.tick_params(labelsize=7)
    plt.tight_layout()
    st.pyplot(fig1)

    # ---------------- PERMUTATION IMPORTANCE ----------------
    st.subheader("Permutation Importance")

    perm = permutation_importance(model, X_test, y_test, n_repeats=3, random_state=42)
    pi = pd.Series(perm.importances_mean, index=FEATURE_COLS).sort_values()

    fig2, ax2 = plt.subplots(figsize=(4,3))
    pi.plot(kind="barh", ax=ax2)
    ax2.tick_params(labelsize=7)
    plt.tight_layout()
    st.pyplot(fig2)

    # ---------------- FORECAST ----------------
    if st.button("Run Forecast"):

        ci = forecast_with_ci(model, df_feat, horizon)

        st.subheader("Forecast")

        fig, ax = plt.subplots()
        ax.plot(df_feat["demand"].tail(168))
        ax.plot(ci["P50"])
        ax.fill_between(ci.index, ci["P10"], ci["P90"], alpha=0.3)
        ax.axvline(df_feat.index[-1], linestyle="--")
        st.pyplot(fig)

else:
    st.info("Upload CSV to begin.")
