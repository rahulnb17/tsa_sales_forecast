"""
REST API for serving forecasts
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import config
from data_preprocessing import DataPreprocessor
from models import XGBoostModel
import os

app = FastAPI(
    title="Sales Forecasting API",
    description="API for forecasting sales",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
preprocessor = DataPreprocessor()
loaded_models = {}  # Cache for loaded models


# Request/Response models
class ForecastRequest(BaseModel):
    state: str
    weeks: Optional[int] = 8


class ForecastResponse(BaseModel):
    state: str
    forecasts: List[Dict]
    model_used: str
    forecast_horizon_weeks: int


class HealthResponse(BaseModel):
    status: str
    message: str


def load_model_for_state(state: str):
    """Load the XGBoost model for a given state"""
    if state in loaded_models:
        return loaded_models[state]
    
    # Try to find saved XGBoost model
    xgb_file = config.MODELS_DIR / f"{state}_xgboost.joblib"
    
    if not xgb_file.exists():
        # Debugging: List actual files to understand why lookup failed
        try:
            available = os.listdir(config.MODELS_DIR)
        except Exception as e:
            available = [f"Error listing dir: {e}"]
        raise ValueError(f"No XGBoost model found for state: {state}. Looked for {xgb_file}. Available files: {available}")
    
    try:
        model = XGBoostModel(state)
        # Force load the xgboost joblib file
        model.load(str(xgb_file))
        model_name = 'XGBoost'
        print(f"Loaded XGBoost model for {state}")
            
    except Exception as e:
        print(f"Error loading model for {state}: {e}")
        raise ValueError(f"Failed to load XGBoost model for {state}: {e}")

    loaded_models[state] = {
        'model': model,
        'model_name': model_name,
        'file_path': str(xgb_file)
    }
    
    return loaded_models[state]

def _resolve_data_file() -> Path:
    """
    Resolve the dataset path.
    - If config.DATA_FILE exists in data/, use it.
    - Else, if exactly one .xlsx exists in data/, use it.
    """
    default_path = config.DATA_DIR / config.DATA_FILE
    if default_path.exists():
        return default_path

    candidates = sorted(config.DATA_DIR.glob("*.xlsx"))
    if len(candidates) == 1:
        return candidates[0]
    return default_path


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Sales Forecasting API is running (XGBoost Only)"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "API is operational"
    }


@app.get("/states")
async def get_available_states():
    """Get list of states with trained models"""
    model_files = list(config.MODELS_DIR.glob("*_xgboost.joblib"))
    
    states = set()
    for model_file in model_files:
        state = model_file.stem.split('_')[0]
        states.add(state)
    
    return {
        "states": sorted(list(states)),
        "count": len(states)
    }


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    """
    Generate forecasts for a specific state
    """
    try:
        # Load model for the state
        model_info = load_model_for_state(request.state)
        model = model_info['model']
        model_name = model_info['model_name']
        
        # Load historical data (if available)
        data_file = _resolve_data_file()
        if not data_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Data file not found. Please ensure an .xlsx file is present in {config.DATA_DIR}"
            )
        
        df = preprocessor.load_data(str(data_file))
        
        # Prepare data for the state
        state_df = df[df[config.STATE_COLUMN] == request.state].copy()
        if len(state_df) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for state: {request.state}"
            )
        
        state_df = preprocessor.handle_missing_dates(state_df, state=None)
        state_df = preprocessor.create_all_features(state_df)
        
        # Get last date
        last_date = pd.to_datetime(state_df[config.DATE_COLUMN].max())
        
        # Calculate forecast horizon
        forecast_days = request.weeks * 7
        
        # Generate forecast dates
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        # Create future dataframe with features for XGBoost
        future_df = pd.DataFrame({config.DATE_COLUMN: forecast_dates})
        future_df[config.STATE_COLUMN] = request.state
        if config.CATEGORY_COLUMN in state_df.columns:
            future_df[config.CATEGORY_COLUMN] = state_df[config.CATEGORY_COLUMN].iloc[0]
        
        # Create time features
        future_df = preprocessor.create_time_features(future_df)
        
        # Use last known values for lag features
        last_values = state_df[config.TARGET_COLUMN].tail(max(config.LAG_FEATURES)).values
        for lag in config.LAG_FEATURES:
            if lag <= len(last_values):
                future_df[f'lag_{lag}'] = last_values[-lag]
            else:
                future_df[f'lag_{lag}'] = state_df[config.TARGET_COLUMN].iloc[-1]
        
        # Rolling features (use last known values)
        for window in config.ROLLING_WINDOWS:
            col_mean = f'rolling_mean_{window}'
            col_std = f'rolling_std_{window}'
            col_min = f'rolling_min_{window}'
            col_max = f'rolling_max_{window}'
            
            if col_mean in state_df.columns:
                future_df[col_mean] = state_df[col_mean].iloc[-1]
                future_df[col_std] = state_df[col_std].iloc[-1]
                future_df[col_min] = state_df[col_min].iloc[-1]
                future_df[col_max] = state_df[col_max].iloc[-1]
            else:
                last_val = state_df[config.TARGET_COLUMN].iloc[-1]
                future_df[col_mean] = last_val
                future_df[col_std] = 0
                future_df[col_min] = last_val
                future_df[col_max] = last_val
        
        predictions = model.predict(future_df, preprocessor)
        
        # Format response
        forecasts = [
            {
                "date": date.isoformat(),
                "predicted_sales": float(pred)
            }
            for date, pred in zip(forecast_dates, predictions)
        ]
        
        return ForecastResponse(
            state=request.state,
            forecasts=forecasts,
            model_used=model_name,
            forecast_horizon_weeks=request.weeks
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")


@app.get("/forecast/{state}")
async def forecast_get(state: str, weeks: int = 8):
    """GET endpoint for forecasts (convenience endpoint)"""
    request = ForecastRequest(state=state, weeks=weeks)
    return await forecast(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)

