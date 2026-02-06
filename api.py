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
from models import ARIMAModel, XGBoostModel, LSTMModel, StackedEnsembleModel, PROPHET_AVAILABLE
from model_comparison import ModelComparator

# Try to import Prophet if available
if PROPHET_AVAILABLE:
    from models import ProphetModel

app = FastAPI(
    title="Sales Forecasting API",
    description="API for forecasting sales using multiple ML models",
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
model_comparator = ModelComparator()
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
    """Load the best model for a given state"""
    if state in loaded_models:
        return loaded_models[state]
    
    # Try to find saved model
    model_files = list(config.MODELS_DIR.glob(f"{state}_*.joblib")) + \
                  list(config.MODELS_DIR.glob(f"{state}_*.h5"))
    
    if not model_files:
        raise ValueError(f"No trained model found for state: {state}")
    
    # Load model comparison results to find the best model
    try:
        import json
        with open(config.RESULTS_DIR / "model_comparison_results.json", 'r') as f:
            comparison_results = json.load(f)
        
        if state in comparison_results:
            best_model_name = comparison_results[state]['best_model']
            
            # Construct expected filename
            if best_model_name == 'LSTM':
                expected_file = config.MODELS_DIR / f"{state}_lstm.h5"
            elif best_model_name == 'StackedEnsemble':
                expected_file = config.MODELS_DIR / f"{state}_stacked_ensemble.joblib"
            else:
                expected_file = config.MODELS_DIR / f"{state}_{best_model_name.lower()}.joblib"
                
            if expected_file.exists():
                model_file = expected_file
                print(f"Loading best model for {state}: {best_model_name}")
            else:
                print(f"Warning: Best model {best_model_name} for {state} not found at {expected_file}. Falling back to any available model.")
                model_file = model_files[0]
        else:
             print(f"Warning: No comparison results for {state}. Falling back to any available model.")
             model_file = model_files[0]
             
    except Exception as e:
        print(f"Warning: Error reading best model info: {e}. Falling back to any available model.")
        model_file = model_files[0]

    # Determine model type from filename
    model_name_lower = model_file.stem.split('_')[-1]
    model_name_lower = model_file.stem.split('_')[-1]
    
    if model_name_lower == 'lstm':
        model = LSTMModel(state)
        model.load(str(model_file))
        model_name = 'LSTM'
    elif model_name_lower == 'prophet':
        if not PROPHET_AVAILABLE:
            raise ValueError(f"Prophet model found for {state} but Prophet is not installed. Install it with: pip install prophet --no-build-isolation")
        model = ProphetModel(state)
        model.load(str(model_file))
        model_name = 'Prophet'
    elif model_name_lower == 'xgboost':
        model = XGBoostModel(state)
        model.load(str(model_file))
        model_name = 'XGBoost'
    elif model_name_lower == 'ensemble' or 'stacked' in model_name_lower:
        model = StackedEnsembleModel(state)
        model.load(str(model_file))
        model_name = 'StackedEnsemble'
        
        # We need to reload base models for the ensemble to work
        # This is expensive but necessary for the API
        base_models = {}
        target_base_models = ['ARIMA', 'XGBoost', 'Prophet', 'LSTM']
        
        for base_name in target_base_models:
            try:
                if base_name == 'ARIMA':
                    base_path = config.MODELS_DIR / f"{state}_arima.joblib"
                    if base_path.exists():
                        m = ARIMAModel(state)
                        m.load(str(base_path))
                        base_models['ARIMA'] = m
                elif base_name == 'XGBoost':
                    base_path = config.MODELS_DIR / f"{state}_xgboost.joblib"
                    if base_path.exists():
                        m = XGBoostModel(state)
                        m.load(str(base_path))
                        base_models['XGBoost'] = m
                elif base_name == 'Prophet' and PROPHET_AVAILABLE:
                    base_path = config.MODELS_DIR / f"{state}_prophet.joblib"
                    if base_path.exists():
                        m = ProphetModel(state)
                        m.load(str(base_path))
                        base_models['Prophet'] = m
                elif base_name == 'LSTM':
                    base_path = config.MODELS_DIR / f"{state}_lstm.h5"
                    if base_path.exists():
                        m = LSTMModel(state)
                        m.load(str(base_path))
                        base_models['LSTM'] = m
            except Exception as e:
                print(f"Warning: Failed to load base model {base_name} for ensemble: {e}")
        
        model.base_models = base_models
        
    else:  # arima
        model = ARIMAModel(state)
        model.load(str(model_file))
        model_name = 'ARIMA'
    
    loaded_models[state] = {
        'model': model,
        'model_name': model_name,
        'file_path': str(model_file)
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
        "message": "Sales Forecasting API is running"
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
    model_files = list(config.MODELS_DIR.glob("*_*.joblib")) + \
                  list(config.MODELS_DIR.glob("*_*.h5"))
    
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
    
    Args:
        request: ForecastRequest with state and optional weeks parameter
    
    Returns:
        ForecastResponse with predictions
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
        
        # Generate predictions based on model type
        if model_name == 'XGBoost':
            # Create future dataframe with features
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
            
        elif model_name == 'LSTM':
            predictions = model.predict_from_data(state_df, forecast_days, preprocessor)
            
        elif model_name == 'Prophet':
            predictions = model.predict(forecast_days, last_date)
            
        elif model_name == 'StackedEnsemble':
            # Create future dataframe
            future_df = pd.DataFrame({config.DATE_COLUMN: forecast_dates})
            # Stacked Ensemble needs history to generate features for base models like XGBoost
            predictions = model.predict(future_df, state_df, preprocessor)
            
        else:  # ARIMA
            predictions = model.predict(forecast_days)
        
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

