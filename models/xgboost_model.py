"""
XGBoost model implementation with lag features
"""
import pandas as pd
import numpy as np
from typing import Dict
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import joblib
import config

warnings.filterwarnings('ignore')


class XGBoostModel:
    """XGBoost forecasting model with engineered features"""
    
    def __init__(self, state: str, xgboost_config: Dict = None):
        self.state = state
        self.xgboost_config = xgboost_config or config.XGBOOST_CONFIG
        self.model = None
        self.feature_columns = []
        self.is_fitted = False
        
    def fit(self, train_data: pd.DataFrame, preprocessor):
        """Train the XGBoost model"""
        # Get feature columns from preprocessor
        self.feature_columns = preprocessor.feature_columns
        
        # Prepare features and target
        X_train, y_train = preprocessor.prepare_data_for_ml(train_data, include_target=True)
        
        # Remove rows with NaN (from lag features at the beginning)
        valid_mask = ~np.isnan(X_train).any(axis=1)
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        if len(X_train) == 0:
            print(f"No valid training data for XGBoost model in {self.state}")
            self.is_fitted = False
            return
        
        try:
            self.model = XGBRegressor(**self.xgboost_config)
            self.model.fit(X_train, y_train)
            self.is_fitted = True
        except Exception as e:
            print(f"Error fitting XGBoost model for {self.state}: {str(e)}")
            self.is_fitted = False
    
    def predict(self, data: pd.DataFrame, preprocessor) -> np.ndarray:
        """Generate forecasts"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Check if features are present
            missing_cols = [col for col in self.feature_columns if col not in data.columns]
            if missing_cols:
                # Try to generate features if missing
                # We need to determine if we can generate them
                try:
                    data = preprocessor.create_all_features(data)
                except Exception:
                    # If generation fails, we might just be missing some rolling/lag 
                    # which is expected for future data, but we need to handle it
                    pass
            
            # Prepare features
            X, _ = preprocessor.prepare_data_for_ml(data, include_target=False)
            
            # Handle NaN values (fill with 0 or forward fill)
            X_df = pd.DataFrame(X)
            X_df = X_df.ffill().fillna(0)
            X = X_df.values
            
            predictions = self.model.predict(X)
            predictions = np.maximum(predictions, 0)  # Ensure non-negative
            
            return predictions
        except Exception as e:
            print(f"Error in XGBoost prediction for {self.state}: {str(e)}")
            # Return mean of training data
            return np.full(len(data), 0)
    
    def evaluate(self, test_data: pd.DataFrame, preprocessor) -> dict:
        """Evaluate model on test data"""
        if not self.is_fitted:
            return {"mae": np.inf, "rmse": np.inf, "mape": np.inf}
        
        try:
            predictions = self.predict(test_data, preprocessor)
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
            print(f"Error evaluating XGBoost model for {self.state}: {str(e)}")
            return {"mae": np.inf, "rmse": np.inf, "mape": np.inf}
    
    def save(self, filepath: str):
        """Save the fitted model"""
        if self.is_fitted:
            joblib.dump({
                'model': self.model,
                'feature_columns': self.feature_columns
            }, filepath)
    
    def load(self, filepath: str):
        """Load a saved model"""
        saved_data = joblib.load(filepath)
        self.model = saved_data['model']
        self.feature_columns = saved_data.get('feature_columns', [])
        self.is_fitted = True

