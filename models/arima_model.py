"""
ARIMA/SARIMA model implementation
"""
import pandas as pd
import numpy as np
from typing import Tuple
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import joblib
import config
from pathlib import Path

warnings.filterwarnings('ignore')


class ARIMAModel:
    """ARIMA/SARIMA forecasting model"""
    
    def __init__(self, state: str, use_sarima: bool = True, order: Tuple = None, seasonal_order: Tuple = None):
        self.state = state
        self.use_sarima = use_sarima
        self.order = order or config.ARIMA_ORDER
        self.seasonal_order = seasonal_order or config.SARIMA_SEASONAL_ORDER
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        
    def fit(self, train_data: pd.DataFrame):
        """Train the ARIMA/SARIMA model"""
        # Extract time series
        ts = train_data[config.TARGET_COLUMN].values
        
        try:
            if self.use_sarima:
                # SARIMA model with weekly seasonality
                self.model = SARIMAX(
                    ts,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                # ARIMA model
                self.model = ARIMA(ts, order=self.order)
            
            self.fitted_model = self.model.fit(method_kwargs={"warn_convergence": False})
            self.is_fitted = True
            
        except Exception as e:
            print(f"Error fitting ARIMA model for {self.state}: {str(e)}")
            # Fallback to simple ARIMA
            try:
                self.model = ARIMA(ts, order=(1, 1, 1))
                self.fitted_model = self.model.fit(method_kwargs={"warn_convergence": False})
                self.is_fitted = True
            except Exception as e2:
                print(f"Fallback ARIMA also failed for {self.state}: {str(e2)}")
                self.is_fitted = False
    
    def predict(self, n_periods: int) -> np.ndarray:
        """Generate forecasts"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            forecast = self.fitted_model.forecast(steps=n_periods)
            # Ensure it's a numpy array
            if not isinstance(forecast, np.ndarray):
                forecast = np.array(forecast)
            # Handle scalar case
            if forecast.ndim == 0 or forecast.size == 1:
                forecast = np.full(n_periods, float(forecast.flat[0]))
            # Ensure non-negative predictions
            forecast = np.maximum(forecast, 0)
            return forecast
        except Exception as e:
            print(f"Error in ARIMA prediction for {self.state}: {str(e)}")
            # Return last known value repeated
            last_value = self.fitted_model.fittedvalues.iloc[-1] if hasattr(self.fitted_model.fittedvalues, 'iloc') else self.fitted_model.fittedvalues[-1]
            return np.full(n_periods, max(last_value, 0))
    
    def evaluate(self, test_data: pd.DataFrame) -> dict:
        """Evaluate model on test data"""
        if not self.is_fitted:
            return {"mae": np.inf, "rmse": np.inf, "mape": np.inf}
        
        try:
            predictions = self.predict(len(test_data))
            actual = test_data[config.TARGET_COLUMN].values
            
            mae = np.mean(np.abs(predictions - actual))
            rmse = np.sqrt(np.mean((predictions - actual) ** 2))
            
            # MAPE (avoid division by zero)
            mask = actual != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((actual[mask] - predictions[mask]) / actual[mask])) * 100
            else:
                mape = np.inf
            
            return {
                "mae": mae,
                "rmse": rmse,
                "mape": mape
            }
        except Exception as e:
            print(f"Error evaluating ARIMA model for {self.state}: {str(e)}")
            return {"mae": np.inf, "rmse": np.inf, "mape": np.inf}
    
    def save(self, filepath: str):
        """Save the fitted model"""
        if self.is_fitted:
            joblib.dump(self.fitted_model, filepath)
    
    def load(self, filepath: str):
        """Load a saved model"""
        self.fitted_model = joblib.load(filepath)
        self.is_fitted = True

