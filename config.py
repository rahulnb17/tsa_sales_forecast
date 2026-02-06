"""
Configuration file for the forecasting system
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Data configuration
DATA_FILE = "Forecasting Case-Study.xlsx"
DATE_COLUMN = "Date"
STATE_COLUMN = "State"
TARGET_COLUMN = "Total"
CATEGORY_COLUMN = "Category"

# Forecasting configuration
FORECAST_HORIZON_WEEKS = 8
FORECAST_HORIZON_DAYS = FORECAST_HORIZON_WEEKS * 7

# Model configuration
TEST_SIZE = 0.2  # 20% for testing (last 20% of time series)
VALIDATION_SIZE = 0.2  # 20% for validation (before test)

# Feature engineering
LAG_FEATURES = [1, 7, 30]  # t-1, t-7, t-30
ROLLING_WINDOWS = [7, 30]  # 7-day and 30-day rolling windows

# Model hyperparameters
ARIMA_ORDER = (2, 1, 2)  # (p, d, q)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 7)  # (P, D, Q, s) - weekly seasonality

PROPHET_CONFIG = {
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "seasonality_mode": "multiplicative",
}

XGBOOST_CONFIG = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

LSTM_CONFIG = {
    "units": 64,  # Increased for better capacity
    "epochs": 100,  # More epochs with early stopping
    "batch_size": 32,
    "validation_split": 0.2,
    "lookback": 30,  # Number of previous time steps to use
}

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

