"""
Stacked Ensemble Model - Meta-learner that combines base models
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import warnings
import joblib
import config
from data_preprocessing import DataPreprocessor

warnings.filterwarnings('ignore')


class StackedEnsembleModel:
    """Stacked ensemble that combines XGBoost, Prophet, ARIMA, and LSTM predictions"""
    
    def __init__(self, state: str, base_models: Dict = None):
        self.state = state
        self.base_model_names = ['XGBoost', 'Prophet', 'ARIMA', 'LSTM']
        # Filter and copy base_models to avoid recursion or unknown models
        if base_models:
            self.base_models = {
                k: v for k, v in base_models.items() 
                if k in self.base_model_names and k != 'StackedEnsemble'
            }
        else:
            self.base_models = {}
        
        print(f"DEBUG: Initialized StackedEnsemble with base models: {list(self.base_models.keys())}")
        
        self.meta_learner = None
        self.is_fitted = False
        
    def fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame, preprocessor: DataPreprocessor = None):
        """
        Train the stacked ensemble
        
        Args:
            train_data: Training data
            test_data: Test data (used for validation split)
            preprocessor: Data preprocessor (needed for XGBoost/LSTM)
        """
        if not self.base_models:
            print(f"No base models provided for stacked ensemble in {self.state}")
            self.is_fitted = False
            return
        
        try:
            # Split train_data into train and validation for meta-learner
            split_idx = int(len(train_data) * 0.8)
            meta_train_data = train_data.iloc[:split_idx].copy()
            meta_val_data = train_data.iloc[split_idx:].copy()
            
            # Get base model predictions on validation set
            base_predictions = {}
            meta_train_target = meta_val_data[config.TARGET_COLUMN].values
            
            print(f"    Generating base model predictions for meta-learner...")
            
            for model_name, model in self.base_models.items():
                if model_name == 'StackedEnsemble':
                    continue
                    
                if not model or not hasattr(model, 'is_fitted') or not model.is_fitted:
                    print(f"      Skipping {model_name} (not fitted)")
                    continue
                
                try:
                    if model_name == 'XGBoost':
                        # XGBoost needs preprocessor
                        if preprocessor is None:
                            continue
                        predictions = model.predict(meta_val_data, preprocessor)
                    elif model_name == 'LSTM':
                        # LSTM needs preprocessor
                        if preprocessor is None:
                            continue
                        predictions = model.predict_from_data(meta_val_data, len(meta_val_data), preprocessor)
                    elif model_name == 'Prophet':
                        # Prophet needs last date
                        last_train_date = meta_train_data[config.DATE_COLUMN].max()
                        predictions = model.predict(len(meta_val_data), last_train_date)
                    else:  # ARIMA
                        predictions = model.predict(len(meta_val_data))
                    
                    # Robustly handle predictions format
                    if isinstance(predictions, (int, float)):
                        predictions = np.full(len(meta_val_data), float(predictions))
                    elif isinstance(predictions, list):
                        predictions = np.array(predictions)
                    elif not isinstance(predictions, np.ndarray):
                        predictions = np.array(predictions)
                    
                    # Handle 0-d array
                    if predictions.ndim == 0:
                        predictions = np.full(len(meta_val_data), float(predictions))
                        
                    # Ensure predictions is a numpy array with correct shape
                    if len(predictions) != len(meta_val_data):
                        # If length mismatch, pad or truncate
                        if len(predictions) == 1:
                             predictions = np.full(len(meta_val_data), float(predictions[0]))
                        elif len(predictions) < len(meta_val_data):
                            # Pad with last value
                            last_val = predictions[-1] if len(predictions) > 0 else 0
                            predictions = np.concatenate([predictions, np.full(len(meta_val_data) - len(predictions), last_val)])
                        else:
                            # Truncate
                            predictions = predictions[:len(meta_val_data)]
                    
                    # Ensure same length
                    min_len = min(len(predictions), len(meta_train_target))
                    base_predictions[model_name] = predictions[:min_len]
                    
                except Exception as e:
                    print(f"      Error getting {model_name} predictions: {str(e)}")
                    continue
            
            if len(base_predictions) < 2:
                print(f"    Need at least 2 base models, got {len(base_predictions)}")
                self.is_fitted = False
                return
            
            # Create meta-features matrix
            # Align all predictions to same length
            min_len = min([len(pred) for pred in base_predictions.values()])
            meta_train_target = meta_train_target[:min_len]
            
            meta_features = np.column_stack([
                base_predictions[name][:min_len] 
                for name in self.base_model_names 
                if name in base_predictions
            ])
            
            if meta_features.shape[0] < 10:
                print(f"    Insufficient validation data for meta-learner")
                self.is_fitted = False
                return
            
            # Train meta-learner (try Ridge first, fallback to RandomForest)
            print(f"    Training meta-learner on {meta_features.shape[0]} samples with {meta_features.shape[1]} features...")
            
            try:
                # Try Ridge regression (fast, good for combining predictions)
                self.meta_learner = Ridge(alpha=1.0, random_state=42)
                self.meta_learner.fit(meta_features, meta_train_target)
                
                # If Ridge doesn't work well, try RandomForest
                # But Ridge is usually better for stacking
                
            except Exception as e:
                print(f"    Error training meta-learner: {str(e)}")
                self.is_fitted = False
                return
            
            self.is_fitted = True
            print(f"    ✓ Stacked ensemble trained successfully")
            
        except Exception as e:
            print(f"Error fitting stacked ensemble for {self.state}: {str(e)}")
            import traceback
            traceback.print_exc()
            self.is_fitted = False
    
    def predict(self, test_data: pd.DataFrame, train_data: pd.DataFrame = None, preprocessor: DataPreprocessor = None) -> np.ndarray:
        """Generate ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Get base model predictions on test set
            base_predictions = {}
            
            for model_name, model in self.base_models.items():
                if model_name == 'StackedEnsemble':
                    continue
                    
                if not model or not hasattr(model, 'is_fitted') or not model.is_fitted:
                    continue
                
                try:
                    if model_name == 'XGBoost':
                        if preprocessor is None:
                            continue
                        
                        # XGBoost needs engineered features (lags, rolling stats) which require history
                        if train_data is not None:
                            # Combine history and future to generate features
                            # We need minimal columns to create features
                            cols = [c for c in train_data.columns if c in [config.DATE_COLUMN, config.TARGET_COLUMN, config.STATE_COLUMN, config.CATEGORY_COLUMN]]
                            train_subset = train_data[cols].copy()
                            
                            # Ensure test_data has necessary columns
                            test_subset = test_data.copy()
                            if config.TARGET_COLUMN not in test_subset.columns:
                                test_subset[config.TARGET_COLUMN] = np.nan
                            if config.STATE_COLUMN not in test_subset.columns and config.STATE_COLUMN in train_subset.columns:
                                test_subset[config.STATE_COLUMN] = train_subset[config.STATE_COLUMN].iloc[0]
                            if config.CATEGORY_COLUMN not in test_subset.columns and config.CATEGORY_COLUMN in train_subset.columns:
                                test_subset[config.CATEGORY_COLUMN] = train_subset[config.CATEGORY_COLUMN].iloc[0]
                                
                            combined = pd.concat([train_subset, test_subset], ignore_index=True)
                            
                            # Generate features on combined data
                            combined_features = preprocessor.create_all_features(combined)
                            
                            # Slice out the test part (future)
                            test_features = combined_features.iloc[-len(test_data):].copy()
                            predictions = model.predict(test_features, preprocessor)
                        else:
                            # Fallback if no history provided
                            predictions = model.predict(test_data, preprocessor)
                    elif model_name == 'LSTM':
                        if preprocessor is None:
                            continue
                        # Combine train and test for LSTM
                        if train_data is not None:
                            combined = pd.concat([train_data, test_data], ignore_index=True)
                            predictions = model.predict_from_data(combined, len(test_data), preprocessor)
                        else:
                            predictions = model.predict_from_data(test_data, len(test_data), preprocessor)
                    elif model_name == 'Prophet':
                        if train_data is not None:
                            last_train_date = train_data[config.DATE_COLUMN].max()
                        else:
                            last_train_date = None
                        predictions = model.predict(len(test_data), last_train_date)
                    else:  # ARIMA
                        predictions = model.predict(len(test_data))
                    
                    # Robustly handle predictions format
                    if isinstance(predictions, (int, float)):
                        predictions = np.full(len(test_data), float(predictions))
                    elif isinstance(predictions, list):
                        predictions = np.array(predictions)
                    elif not isinstance(predictions, np.ndarray):
                        predictions = np.array(predictions)
                        
                    # Handle 0-d array (scalar wrapped in array)
                    if predictions.ndim == 0:
                        predictions = np.full(len(test_data), float(predictions))
                    
                    # Ensure it matches test_data length
                    if len(predictions) != len(test_data):
                        if len(predictions) == 1:
                             predictions = np.full(len(test_data), float(predictions[0]))
                        elif len(predictions) < len(test_data):
                            # Pad with last value
                            last_val = predictions[-1] if len(predictions) > 0 else 0
                            predictions = np.concatenate([predictions, np.full(len(test_data) - len(predictions), last_val)])
                        else:
                            # Truncate
                            predictions = predictions[:len(test_data)]
                            
                    base_predictions[model_name] = predictions
                    
                except Exception as e:
                    print(f"      Error getting {model_name} predictions: {str(e)}")
                    continue
            
            if len(base_predictions) < 2:
                # Fallback: simple average
                all_preds = [pred for pred in base_predictions.values()]
                return np.mean(all_preds, axis=0)
            
            # Create meta-features
            min_len = min([len(pred) for pred in base_predictions.values()])
            meta_features = np.column_stack([
                base_predictions[name][:min_len] 
                for name in self.base_model_names 
                if name in base_predictions
            ])
            
            # Get meta-learner predictions
            ensemble_predictions = self.meta_learner.predict(meta_features)
            ensemble_predictions = np.maximum(ensemble_predictions, 0)  # Ensure non-negative
            
            return ensemble_predictions
            
        except Exception as e:
            print(f"Error in stacked ensemble prediction for {self.state}: {str(e)}")
            # Fallback: simple average
            all_preds = [pred for pred in base_predictions.values() if len(pred) > 0]
            if all_preds:
                return np.mean(all_preds, axis=0)
            return np.zeros(len(test_data))
    
    def evaluate(self, test_data: pd.DataFrame, train_data: pd.DataFrame, preprocessor: DataPreprocessor) -> dict:
        """Evaluate model on test data"""
        if not self.is_fitted:
            return {"mae": np.inf, "rmse": np.inf, "mape": np.inf}
        
        try:
            predictions = self.predict(test_data, train_data, preprocessor)
            actual = test_data[config.TARGET_COLUMN].values
            
            # Ensure same length
            min_len = min(len(predictions), len(actual))
            predictions = predictions[:min_len]
            actual = actual[:min_len]
            
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
            print(f"Error evaluating stacked ensemble for {self.state}: {str(e)}")
            return {"mae": np.inf, "rmse": np.inf, "mape": np.inf}
    
    def save(self, filepath: str):
        """Save the meta-learner"""
        if self.is_fitted:
            joblib.dump({
                'meta_learner': self.meta_learner,
                'base_model_names': self.base_model_names
            }, filepath)
    
    def load(self, filepath: str):
        """Load a saved meta-learner"""
        saved_data = joblib.load(filepath)
        self.meta_learner = saved_data['meta_learner']
        self.base_model_names = saved_data.get('base_model_names', ['XGBoost', 'Prophet', 'ARIMA', 'LSTM'])
        self.is_fitted = True

