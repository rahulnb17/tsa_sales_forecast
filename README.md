# Sales Forecasting System

A production-ready end-to-end time series forecasting system that trains multiple forecasting algorithms, compares their performance, and serves predictions via a REST API.

## Features

- **Multiple Forecasting Models**: Implements ARIMA/SARIMA, Facebook Prophet, XGBoost, and LSTM
- **Automatic Model Selection**: Compares all models and selects the best performing one for each state
- **Comprehensive Feature Engineering**: 
  - Lag features (t-1, t-7, t-30)
  - Rolling statistics (mean, std, min, max)
  - Time-based features (day of week, month, holidays)
  - Cyclical encoding for seasonality
- **Missing Data Handling**: Automatically handles missing dates and values
- **REST API**: FastAPI-based API for serving predictions
- **Production-Ready**: Proper error handling, model persistence, and API documentation

## Project Structure

```
casestudy/
├── config.py                 # Configuration settings
├── data_preprocessing.py     # Data loading and feature engineering
├── model_comparison.py       # Model training and comparison
├── train.py                  # Main training script
├── api.py                    # REST API server
├── requirements.txt          # Python dependencies
├── models/                   # Model implementations
│   ├── __init__.py
│   ├── arima_model.py
│   ├── prophet_model.py
│   ├── xgboost_model.py
│   └── lstm_model.py
├── data/                     # Data directory (place Excel file here)
├── models/                   # Saved trained models
└── results/                  # Training results and forecasts
```

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies**:
   
   **For Python 3.12 (Windows):**
   ```bash
   install_requirements.bat
   ```
   
   **For Python 3.12 (Linux/Mac):**
   ```bash
   chmod +x install_requirements.sh
   ./install_requirements.sh
   ```
   
   **Or manually:**
   ```bash
   pip install --upgrade setuptools wheel
   pip install -r requirements.txt
   ```
   
   **Note:** If you encounter `distutils` errors with Python 3.12, make sure to install `setuptools` first as shown above.

3. **Place your data file**:
   - Copy `Forecasting Case-Study.xlsx` to the `data/` directory
   - The Excel file should contain columns: `State`, `Date`, `Total`, `Category`

## Usage

### Step 1: Train Models

Train all models and compare their performance:

```bash
python train.py
```

Train models for specific states:

```bash
python train.py --states Alabama Texas California
```

Specify custom data file:

```bash
python train.py --data-file path/to/your/data.xlsx
```

The training script will:
1. Load and preprocess the data
2. Train all 4 models (ARIMA/SARIMA, Prophet, XGBoost, LSTM) for each state
3. Evaluate models on test data
4. Select the best model for each state based on RMSE
5. Save trained models to `models/` directory
6. Generate 8-week forecasts and save to `results/forecasts.json`
7. Save comparison results to `results/model_comparison_results.json`

### Step 2: Start API Server

Start the REST API server:

```bash
python api.py
```

Or using uvicorn directly:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Step 3: Use the API

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Get Available States
```bash
curl http://localhost:8000/states
```

#### Generate Forecast (POST)
```bash
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{"state": "Alabama", "weeks": 8}'
```

#### Generate Forecast (GET)
```bash
curl "http://localhost:8000/forecast/Alabama?weeks=8"
```

#### API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI)

## API Endpoints

### `GET /`
Health check endpoint

### `GET /health`
Health check endpoint

### `GET /states`
Returns list of states with trained models

**Response:**
```json
{
  "states": ["Alabama", "Texas", ...],
  "count": 50
}
```

### `POST /forecast`
Generate forecasts for a specific state

**Request Body:**
```json
{
  "state": "Alabama",
  "weeks": 8
}
```

**Response:**
```json
{
  "state": "Alabama",
  "forecasts": [
    {
      "date": "2024-01-01T00:00:00",
      "predicted_sales": 1234.56
    },
    ...
  ],
  "model_used": "Prophet",
  "forecast_horizon_weeks": 8
}
```

### `GET /forecast/{state}`
Convenience GET endpoint for forecasts

**Parameters:**
- `state`: State name (path parameter)
- `weeks`: Number of weeks to forecast (query parameter, default: 8)

## Model Details

### ARIMA/SARIMA
- Statistical time series model
- Handles trend and seasonality
- Uses weekly seasonality (SARIMA)

### Facebook Prophet
- Additive/multiplicative seasonality
- Handles holidays and special events
- Robust to missing data

### XGBoost
- Gradient boosting with engineered features
- Uses lag features, rolling statistics, and time features
- Handles non-linear patterns

### LSTM
- Deep learning model for sequential data
- Uses 30-day lookback window
- Captures long-term dependencies

## Feature Engineering

The system automatically creates:

1. **Lag Features**: Sales values from 1, 7, and 30 days ago
2. **Rolling Statistics**: 
   - 7-day and 30-day rolling mean, std, min, max
3. **Time Features**:
   - Day of week, month, day of month, week of year
   - Is weekend, is holiday
   - Cyclical encoding (sin/cos) for day of week and month

## Configuration

Edit `config.py` to customize:

- Forecast horizon (default: 8 weeks)
- Train/test split ratio
- Model hyperparameters
- Feature engineering parameters
- API host and port

## Output Files

After training, you'll find:

- `models/{state}_{model}.joblib` or `.h5`: Trained models
- `results/model_comparison_results.json`: Model comparison metrics
- `results/forecasts.json`: Generated forecasts for all states

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

## Notes

- The system handles missing dates by creating a complete date range and interpolating values
- Models are evaluated using MAE, RMSE, and MAPE metrics
- Best model selection is based on RMSE (lower is better)
- All predictions are ensured to be non-negative
- The API caches loaded models for faster subsequent requests

## Troubleshooting

1. **Data file not found**: Ensure `Forecasting Case-Study.xlsx` is in the `data/` directory
2. **Model training fails**: Check that you have sufficient data (at least 50 data points per state)
3. **API errors**: Ensure models are trained before starting the API server
4. **Memory issues**: For large datasets, consider training models for specific states only

## License

This project is provided as-is for the case study assignment.

