# ⚡ Electric Load Forecast

A machine learning-based electricity demand forecasting system using XGBoost that predicts future power consumption patterns.

## 📋 Overview

This project provides an interactive web application for forecasting electrical load demand over specified time horizons (1-72 hours). It uses a pre-trained XGBoost model and historical demand data to generate accurate predictions with temporal and statistical features.

## 🎯 Features

- **Interactive Forecasting**: Web-based UI for real-time demand predictions
- **Historical Data Visualization**: Display historical demand patterns
- **Flexible Forecasting Horizon**: Predict 1 to 72 hours into the future
- **Advanced Feature Engineering**:
  - Temporal features (hour, month, week, quarter, weekend indicator)
  - Lag features (24-hour and 168-hour demand lags)
  - Rolling statistics (24-hour mean and standard deviation)
- **CSV Export**: Download forecasts for further analysis
- **XGBoost Model**: Pre-trained tuned model for accurate predictions

## 📁 Project Structure

```
Electric Load Forecast/
├── electric_load.py                    # Main Streamlit application
├── PDB_Load_History.csv               # Historical demand data
├── xgb_tuned_load_forecast_model_.pkl # Pre-trained XGBoost model
├── Untitled.ipynb                     # Jupyter notebook with analysis
└── README.md                          # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Required packages:
  - streamlit
  - pandas
  - matplotlib
  - joblib
  - xgboost (for model training)
  - scikit-learn (for preprocessing)

### Installation

1. Clone or download this repository
2. Install required packages:
   ```bash
   pip install streamlit pandas matplotlib joblib xgboost scikit-learn
   ```

### Running the Application

Start the Streamlit app:

```bash
streamlit run electric_load.py
```

The application will open in your default browser at `http://localhost:8501`

## 📊 How to Use

1. **View Historical Data**: The main page displays historical electricity demand over time
2. **Configure Forecast Options** (sidebar):
   - Select a start date for the forecast
   - Choose forecast horizon (1-72 hours)
3. **Run Forecast**: Click the "Run Forecast" button to generate predictions
4. **View Results**:
   - Forecast table with predicted demand values
   - Combined chart showing historical data + forecast
5. **Export Data**: Download forecast as CSV for further analysis

## 🔧 Model Details

### Features Used

| Feature                    | Description                                     |
| -------------------------- | ----------------------------------------------- |
| `hour`                     | Hour of day (0-23)                              |
| `month`                    | Month of year (1-12)                            |
| `weekofyear`               | Week number (1-52)                              |
| `quarter`                  | Quarter of year (1-4)                           |
| `is_weekend`               | Binary indicator for weekend days               |
| `demand_lag_24hr`          | Demand from 24 hours prior                      |
| `demand_lag_168hr`         | Demand from 168 hours (1 week) prior            |
| `demand_rolling_mean_24hr` | Average demand over last 24 hours               |
| `demand_rolling_std_24hr`  | Standard deviation of demand over last 24 hours |

### Forecasting Algorithm

The `forecast_future_xgb()` function:

1. Takes the most recent data as the starting point
2. Iteratively predicts the next hour's demand using the XGBoost model
3. Updates lag features and rolling statistics for the next prediction
4. Repeats for the specified forecast horizon

## 📈 Data Format

### Input Data (PDB_Load_History.csv)

Required columns:

- `date`: Timestamp of the observation
- `demand`: Electricity demand (in appropriate units)

Example:

```
date,demand
2023-01-01 00:00:00,5000
2023-01-01 01:00:00,4800
```

### Output Data

Forecast results include:

- `Hour`: Forecast step number (1 to horizon)
- `Predicted Demand`: Predicted electricity demand for that hour

## 📝 File Descriptions

- **electric_load.py**: Main Streamlit application containing:
  - Model and data loading
  - UI layout and controls
  - Forecasting logic
  - Visualization and export functionality

- **Untitled.ipynb**: Jupyter notebook with:
  - Data exploration and analysis
  - Model training and evaluation
  - Feature importance analysis
  - Performance metrics

- **PDB_Load_History.csv**: Historical electricity demand data used for:
  - Training the XGBoost model
  - Feature engineering
  - Historical trend visualization

- **xgb*tuned_load_forecast_model*.pkl**: Serialized XGBoost model with optimized hyperparameters

## 🔍 Key Functions

### `load_model()`

Loads the pre-trained XGBoost model from the pickle file with caching.

### `load_data()`

Loads and preprocesses historical demand data:

- Parses datetime
- Computes all engineered features
- Removes missing values

### `forecast_future_xgb(df, model, feature_cols, horizon)`

Generates predictions for the specified horizon:

- **Parameters**:
  - `df`: Historical dataframe
  - `model`: Trained XGBoost model
  - `feature_cols`: List of feature column names
  - `horizon`: Number of hours to forecast
- **Returns**: List of predicted demand values

## 💡 Tips for Best Results

- Start forecasts with recent historical data for better accuracy
- Consider external factors (weather, holidays, events) that may affect demand
- Validate forecast accuracy against actual values periodically
- Use multiple forecast horizons to identify pattern changes

## 📦 Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **matplotlib**: Plotting and visualization
- **joblib**: Model serialization
- **xgboost**: Gradient boosting framework
- **scikit-learn**: Machine learning utilities

## 🔄 Model Performance

The XGBoost model is pre-trained and tuned on historical load data. For updated performance metrics and model retraining, refer to `Untitled.ipynb`.

## 📞 Support

For issues or questions:

1. Check the Jupyter notebook for analysis and examples
2. Verify data format matches the expected structure
3. Ensure all required packages are installed

## 📄 License

[Specify your project license here]

## 🙏 Acknowledgments

- Data source: PDB Load History dataset
- Built with Streamlit and XGBoost
