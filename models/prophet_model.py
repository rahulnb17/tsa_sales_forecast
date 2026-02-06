"""
Facebook Prophet model implementation
"""
import pandas as pd
import numpy as np
from typing import Dict
from prophet import Prophet
import warnings
import joblib
import config

warnings.filterwarnings('ignore')


class ProphetModel:
    """Facebook Prophet forecasting model"""
    
    def __init__(self, state: str, prophet_config: Dict = None):
        self.state = state
        self.prophet_config = prophet_config or config.PROPHET_CONFIG
        self.model = None
        self.is_fitted = False
        
    def fit(self, train_data: pd.DataFrame):
        """Train the Prophet model"""
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_df = pd.DataFrame({
            'ds': train_data[config.DATE_COLUMN],
            'y': train_data[config.TARGET_COLUMN]
        })
        
        # Remove any negative values (Prophet doesn't handle them well)
        prophet_df['y'] = np.maximum(prophet_df['y'], 0)
        
        try:
            self.model = Prophet(**self.prophet_config)
            self.model.fit(prophet_df)
            self.is_fitted = True
        except Exception as e:
            print(f"Error fitting Prophet model for {self.state}: {str(e)}")
            # Try with simpler configuration
            try:
                self.model = Prophet(yearly_seasonality=False, weekly_seasonality=True)
                self.model.fit(prophet_df)
                self.is_fitted = True
            except Exception as e2:
                print(f"Fallback Prophet also failed for {self.state}: {str(e2)}")
                self.is_fitted = False
    
    def predict(self, n_periods: int, last_date: pd.Timestamp = None) -> np.ndarray:
        """Generate forecasts"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Create future dataframe
            if last_date is None:
                # Get last date from training data
                last_date = self.model.history['ds'].max()
            
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=n_periods,
                freq='D'
            )
            
            future_df = pd.DataFrame({'ds': future_dates})
            forecast = self.model.predict(future_df)
            
            # Extract predictions and ensure non-negative
            predictions = forecast['yhat'].values
            predictions = np.maximum(predictions, 0)
            
            return predictions
        except Exception as e:
            print(f"Error in Prophet prediction for {self.state}: {str(e)}")
            # Return last known value repeated
            last_value = self.model.history['y'].iloc[-1]
            return np.full(n_periods, max(last_value, 0))
    
    def evaluate(self, test_data: pd.DataFrame) -> dict:
        """Evaluate model on test data"""
        if not self.is_fitted:
            return {"mae": np.inf, "rmse": np.inf, "mape": np.inf}
        
        try:
            # Get last date from training
            last_train_date = self.model.history['ds'].max()
            
            # Create future dataframe for test period
            test_dates = test_data[config.DATE_COLUMN].values
            future_df = pd.DataFrame({'ds': test_dates})
            
            forecast = self.model.predict(future_df)
            predictions = forecast['yhat'].values
            predictions = np.maximum(predictions, 0)
            
            actual = test_data[config.TARGET_COLUMN].values
            
            mae = np.mean(np.abs(predictions - actual))
            rmse = np.sqrt(np.mean((predictions - actual) ** 2))
            
            # MAPE
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
            print(f"Error evaluating Prophet model for {self.state}: {str(e)}")
            return {"mae": np.inf, "rmse": np.inf, "mape": np.inf}
    
    def save(self, filepath: str):
        """Save the fitted model"""
        if self.is_fitted:
            joblib.dump(self.model, filepath)
    
    def load(self, filepath: str):
        """Load a saved model"""
        self.model = joblib.load(filepath)
        self.is_fitted = True

