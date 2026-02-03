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
    "hour","month","weekofyear","quarter","is_weekend",
    "demand_lag_24hr","demand_lag_168hr",
    "demand_rolling_mean_24hr","demand_rolling_std_24hr"
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
# FAST RECURSIVE FORECAST (O(N))
# =========================================================

def forecast_future_xgb_fast(model, history, horizon=24):

    hist = history.copy()

    lag_24 = deque(hist["demand"].iloc[-24:], maxlen=24)
    lag_168 = deque(hist["demand"].iloc[-168:], maxlen=168)
    roll_24 = deque(hist["demand"].iloc[-24:], maxlen=24)

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
            "demand_lag_24hr": lag_24[0],
            "demand_lag_168hr": lag_168[0],
            "demand_rolling_mean_24hr": np.mean(roll_24),
            "demand_rolling_std_24hr": np.std(roll_24)
        }])

        y_pred = model.predict(row)[0]
        preds.append(y_pred)

        lag_24.append(y_pred)
        lag_168.append(y_pred)
        roll_24.append(y_pred)

        last_time = next_time

    idx = pd.date_range(
        start=history.index[-1] + pd.Timedelta(hours=1),
        periods=horizon,
        freq="H"
    )

    return pd.DataFrame({"demand": preds}, index=idx)

# =========================================================
# CONFIDENCE INTERVALS (BOOTSTRAP)
# =========================================================

def forecast_with_ci(model, history, horizon=24, n_samples=30):

    sims = []

    for _ in range(n_samples):

        noisy = history.copy()
        noise = np.random.normal(0, history["demand"].std()*0.05, size=len(history))
        noisy["demand"] = noisy["demand"] + noise

        f = forecast_future_xgb_fast(model, noisy, horizon)
        sims.append(f["demand"].values)

    sims = np.array(sims)

    p10 = np.percentile(sims, 10, axis=0)
    p50 = np.percentile(sims, 50, axis=0)
    p90 = np.percentile(sims, 90, axis=0)

    idx = pd.date_range(
        start=history.index[-1] + pd.Timedelta(hours=1),
        periods=horizon,
        freq="H"
    )

    return pd.DataFrame({"P10": p10, "P50": p50, "P90": p90}, index=idx)

# =========================================================
# UI
# =========================================================

st.title("⚡ Electrical Load Forecasting (Production App)")

st.sidebar.header("Settings")

horizon = st.sidebar.slider("Forecast Horizon (hours)", 24, 168, 24)
enable_retrain = False


uploaded_file = st.file_uploader("Upload CSV (must contain 'date' and 'demand')", type=["csv"])

# =========================================================
# MAIN
# =========================================================

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    if not {"date","demand"}.issubset(df.columns):
        st.error("CSV must contain 'date' and 'demand'")
        st.stop()

    df["LoadDate"] = pd.to_datetime(df["date"])
    df = df.set_index("LoadDate")
    df = df.sort_index()

    df_feat = create_features(df).dropna()

    st.subheader("Recent Load")
    st.line_chart(df_feat["demand"].tail(168))

    # ---------------- AUTO RETRAIN ----------------


    # ---------------- MAPE ----------------
    holdout = int(len(df_feat)*0.8)
    X_test = df_feat[FEATURE_COLS].iloc[holdout:]
    y_test = df_feat["demand"].iloc[holdout:]
    preds = model.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, preds)*100
    st.metric("MAPE (%)", f"{mape:.2f}")

    # ---------------- SINGLE STEP PREDICTION ----------------
    st.subheader("🔮 Predict Load for One Hour")

    predict_time = st.datetime_input(
        "Select Date & Time",
        value=df_feat.index[-1] + pd.Timedelta(hours=1)
    )

    if st.button("Predict Single Load"):

        lag_24 = df_feat["demand"].iloc[-24]
        lag_168 = df_feat["demand"].iloc[-168]

        roll_mean_24 = df_feat["demand"].iloc[-24:].mean()
        roll_std_24 = df_feat["demand"].iloc[-24:].std()

        row = pd.DataFrame([{
            "hour": predict_time.hour,
            "month": predict_time.month,
            "weekofyear": int(predict_time.isocalendar().week),
            "quarter": (predict_time.month - 1) // 3 + 1,
            "is_weekend": int(predict_time.weekday() >= 5),
            "demand_lag_24hr": lag_24,
            "demand_lag_168hr": lag_168,
            "demand_rolling_mean_24hr": roll_mean_24,
            "demand_rolling_std_24hr": roll_std_24
        }])

        pred_load = model.predict(row)[0]

        st.success(f"Predicted Load: {pred_load:.2f}")

    # ---------------- FEATURE IMPORTANCE ----------------
   st.subheader("Feature Importance")

xgb = model.named_steps["model"]
fi = pd.Series(xgb.feature_importances_, index=FEATURE_COLS).sort_values()

fig_imp, ax_imp = plt.subplots(figsize=(4, 3))

fi.plot(kind="barh", ax=ax_imp)

ax_imp.set_title("XGBoost Importance", fontsize=9)
ax_imp.tick_params(axis='both', labelsize=7)

plt.tight_layout()
st.pyplot(fig_imp)

    # ---------------- PERMUTATION IMPORTANCE ----------------
   st.subheader("Permutation Importance")

perm = permutation_importance(model, X_test, y_test, n_repeats=3, random_state=42)
perm_imp = pd.Series(perm.importances_mean, index=FEATURE_COLS).sort_values()

fig_perm, ax_perm = plt.subplots(figsize=(4, 3))

perm_imp.plot(kind="barh", ax=ax_perm)

ax_perm.set_title("Permutation Importance", fontsize=9)
ax_perm.tick_params(axis='both', labelsize=7)

plt.tight_layout()
st.pyplot(fig_perm)


    # ---------------- FORECAST ----------------
    if st.button("Run Forecast"):

        ci = forecast_with_ci(model, df_feat, horizon)

        st.subheader("Forecast with Confidence Intervals")
        st.dataframe(ci)

        combined = pd.concat([df_feat[["demand"]].tail(168), ci["P50"]])

        fig, ax = plt.subplots()
        ax.plot(combined.index, combined.values)
        ax.fill_between(ci.index, ci["P10"], ci["P90"], alpha=0.3)
        ax.axvline(df_feat.index[-1], linestyle="--")
        st.pyplot(fig)

else:
    st.info("Upload CSV to begin.")



