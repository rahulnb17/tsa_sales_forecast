"""
Model comparison and selection system
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import config
from data_preprocessing import DataPreprocessor
from models import ARIMAModel, XGBoostModel, LSTMModel, StackedEnsembleModel, PROPHET_AVAILABLE
from model_tuning import ModelTuner

# Try to import Prophet if available
if PROPHET_AVAILABLE:
    from models import ProphetModel


class ModelComparator:
    """Compare and select the best forecasting model"""
    
    def __init__(self, enable_tuning: bool = True, resume: bool = True):
        self.results = {}
        self.best_models = {}  # Best model for each state
        self.enable_tuning = enable_tuning
        self.tuner = ModelTuner() if enable_tuning else None
        self.resume = resume
        self.checkpoint_file = config.RESULTS_DIR / "training_checkpoint.json"
        
    def load_checkpoint(self) -> Dict:
        """Load existing checkpoint to resume training"""
        if not self.resume or not self.checkpoint_file.exists():
            return {}
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"\n✓ Found checkpoint: {len(checkpoint.get('completed_states', []))} states already completed")
            return checkpoint
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {str(e)}")
            return {}
    
    def save_checkpoint(self, completed_states: List[str], results: Dict, best_models_info: Dict):
        """Save checkpoint after each state completes"""
        try:
            checkpoint = {
                'completed_states': completed_states,
                'results': results,
                'best_models': {state: {
                    'model_name': info['model_name'],
                    'metrics': info['metrics']
                } for state, info in best_models_info.items()}
            }
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {str(e)}")
    
    def check_state_completed(self, state: str) -> bool:
        """Check if a state has already been trained"""
        if not self.resume:
            return False
        
        # Check if model file exists
        model_files = list(config.MODELS_DIR.glob(f"{state}_*.joblib")) + \
                      list(config.MODELS_DIR.glob(f"{state}_*.h5"))
        
        return len(model_files) > 0
    
    def train_and_compare_models(self, df: pd.DataFrame, states: List[str] = None) -> Dict:
        """
        Train all models for each state and compare their performance
        Supports checkpoint/resume functionality
        """
        if states is None:
            states = df[config.STATE_COLUMN].unique().tolist()
        
        # Load checkpoint if resuming
        checkpoint = self.load_checkpoint() if self.resume else {}
        completed_states = set(checkpoint.get('completed_states', []))
        
        # Load existing results if resuming
        if checkpoint:
            self.results = checkpoint.get('results', {})
            # Reconstruct best_models info (models themselves will be reloaded if needed)
            best_models_info = checkpoint.get('best_models', {})
            for state, info in best_models_info.items():
                self.best_models[state] = {
                    'model_name': info['model_name'],
                    'metrics': info['metrics'],
                    'all_metrics': info.get('all_metrics', {})
                }
        
        # Filter out already completed states
        if self.resume and completed_states:
            remaining_states = [s for s in states if s not in completed_states]
            if remaining_states:
                print(f"\nResuming training: {len(remaining_states)} states remaining, {len(completed_states)} already completed")
            else:
                print(f"\n✓ All {len(completed_states)} states already completed!")
                return self.results
        else:
            remaining_states = states
        
        preprocessor = DataPreprocessor()
        
        for state in states:
            print(f"\n{'='*60}")
            print(f"Processing state: {state}")
            print(f"{'='*60}")
            
            try:
                # Prepare data for this state
                train_df, test_df = preprocessor.prepare_data_for_state(df, state)
                
                if len(train_df) < 50:  # Need minimum data points
                    print(f"Insufficient data for {state}, skipping...")
                    continue
                
                # Tune hyperparameters if enabled
                tuned_configs = {}
                if self.enable_tuning and self.tuner:
                    print("\nTuning hyperparameters...")
                    
                    # Tune ARIMA
                    try:
                        print("  Tuning ARIMA...")
                        arima_config = self.tuner.tune_arima(train_df, state)
                        tuned_configs['ARIMA'] = arima_config
                        print(f"    Best ARIMA order: {arima_config['order']}, seasonal: {arima_config['seasonal_order']}")
                    except Exception as e:
                        print(f"    ARIMA tuning failed: {str(e)}, using defaults")
                        tuned_configs['ARIMA'] = None
                    
                    # Tune Prophet
                    if PROPHET_AVAILABLE:
                        try:
                            print("  Tuning Prophet...")
                            prophet_config = self.tuner.tune_prophet(train_df, state)
                            tuned_configs['Prophet'] = prophet_config
                            print(f"    Best Prophet mode: {prophet_config['seasonality_mode']}")
                        except Exception as e:
                            print(f"    Prophet tuning failed: {str(e)}, using defaults")
                            tuned_configs['Prophet'] = None
                    
                    # Tune XGBoost
                    try:
                        print("  Tuning XGBoost...")
                        xgb_config = self.tuner.tune_xgboost(train_df, preprocessor, state)
                        tuned_configs['XGBoost'] = xgb_config
                    except Exception as e:
                        print(f"    XGBoost tuning failed: {str(e)}, using defaults")
                        tuned_configs['XGBoost'] = None
                    
                    # Tune LSTM
                    try:
                        print("  Tuning LSTM (multi-feature)...")
                        lstm_config = self.tuner.tune_lstm(train_df, state, preprocessor)
                        tuned_configs['LSTM'] = lstm_config
                    except Exception as e:
                        print(f"    LSTM tuning failed: {str(e)}, using defaults")
                        tuned_configs['LSTM'] = None
                else:
                    print("\nHyperparameter tuning disabled, using default configs...")
                
                # Initialize models with tuned configs
                arima_config = tuned_configs.get('ARIMA')
                if arima_config:
                    models = {
                        'ARIMA': ARIMAModel(
                            state, 
                            use_sarima=arima_config.get('use_sarima', True),
                            order=arima_config.get('order'),
                            seasonal_order=arima_config.get('seasonal_order')
                        ),
                        'XGBoost': XGBoostModel(state, xgboost_config=tuned_configs.get('XGBoost')),
                        'LSTM': LSTMModel(state, lstm_config=tuned_configs.get('LSTM'), use_features=True)
                    }
                else:
                    models = {
                        'ARIMA': ARIMAModel(state, use_sarima=True),
                        'XGBoost': XGBoostModel(state, xgboost_config=tuned_configs.get('XGBoost')),
                        'LSTM': LSTMModel(state, lstm_config=tuned_configs.get('LSTM'), use_features=True)
                    }
                
                # Add Prophet if available
                if PROPHET_AVAILABLE:
                    prophet_config = tuned_configs.get('Prophet')
                    models['Prophet'] = ProphetModel(state, prophet_config=prophet_config)
                else:
                    print("  Note: Prophet is not available. Skipping Prophet model.")
                
                # Train all base models
                print("\nTraining base models...")
                for model_name, model in models.items():
                    try:
                        # Special handling for XGBoost: Check if checkpoint exists
                        if model_name == 'XGBoost':
                            xgb_path = config.MODELS_DIR / f"{state}_xgboost.joblib"
                            if xgb_path.exists():
                                print(f"  Load existing XGBoost checkpoint from {xgb_path.name}")
                                try:
                                    model.load(str(xgb_path))
                                    print(f"  ✓ {model_name} loaded successfully")
                                    continue
                                except Exception as e:
                                    print(f"  ! Failed to load XGBoost checkpoint, retraining: {e}")
                                    # Fallthrough to training
                        
                        print(f"  Training {model_name}...")
                        if model_name == 'XGBoost':
                            model.fit(train_df, preprocessor)
                        elif model_name == 'LSTM':
                            model.fit(train_df, preprocessor)
                        else:
                            model.fit(train_df)
                        print(f"  ✓ {model_name} trained successfully")
                    except Exception as e:
                        print(f"  ✗ {model_name} training failed: {str(e)}")
                        import traceback
                        traceback.print_exc()
                
                # Train Stacked Ensemble (uses trained base models)
                print("\nTraining Stacked Ensemble...")
                try:
                    stacked_ensemble = StackedEnsembleModel(state, base_models=models)
                    stacked_ensemble.fit(train_df, test_df, preprocessor)
                    if stacked_ensemble.is_fitted:
                        models['StackedEnsemble'] = stacked_ensemble
                        print("  ✓ Stacked Ensemble trained successfully")
                    else:
                        print("  ✗ Stacked Ensemble training failed (insufficient base models)")
                except Exception as e:
                    print(f"  ✗ Stacked Ensemble training failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                # Evaluate all models
                print("\nEvaluating models...")
                evaluation_results = {}
                
                for model_name, model in models.items():
                    try:
                        if model_name == 'XGBoost':
                            metrics = model.evaluate(test_df, preprocessor)
                        elif model_name == 'LSTM':
                            metrics = model.evaluate(test_df, train_df, preprocessor)
                        elif model_name == 'StackedEnsemble':
                            metrics = model.evaluate(test_df, train_df, preprocessor)
                        else:
                            metrics = model.evaluate(test_df)
                        
                        evaluation_results[model_name] = metrics
                        print(f"  {model_name}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%")
                    except Exception as e:
                        print(f"  ✗ {model_name} evaluation failed: {str(e)}")
                        evaluation_results[model_name] = {
                            "mae": np.inf,
                            "rmse": np.inf,
                            "mape": np.inf
                        }
                
                # Select best model based on RMSE (lower is better)
                best_model_name = min(
                    evaluation_results.keys(),
                    key=lambda x: evaluation_results[x]['rmse']
                )
                
                self.best_models[state] = {
                    'model_name': best_model_name,
                    'model': models[best_model_name],
                    'metrics': evaluation_results[best_model_name],
                    'all_metrics': evaluation_results
                }
                
                # Save ALL models (needed for ensemble), not just the best one
                print(f"\nSaving all models...")
                for model_name, model in models.items():
                    try:
                        if model_name == 'LSTM':
                            model_path = config.MODELS_DIR / f"{state}_{model_name.lower()}.h5"
                        elif model_name == 'StackedEnsemble':
                            model_path = config.MODELS_DIR / f"{state}_stacked_ensemble.joblib"
                        else:
                            model_path = config.MODELS_DIR / f"{state}_{model_name.lower()}.joblib"
                        
                        if hasattr(model, 'is_fitted') and model.is_fitted:
                            model.save(str(model_path))
                            print(f"  ✓ Saved {model_name}")
                    except Exception as e:
                        print(f"  ✗ Failed to save {model_name}: {str(e)}")
                
                # Also save best model info
                print(f"\n✓ Best model for {state}: {best_model_name} (RMSE: {evaluation_results[best_model_name]['rmse']:.2f})")
                
                # Store results
                self.results[state] = {
                    'best_model': best_model_name,
                    'metrics': evaluation_results,
                    'train_size': len(train_df),
                    'test_size': len(test_df)
                }
                
                # Save checkpoint after each state (incremental save)
                completed_states.add(state)
                self.save_checkpoint(list(completed_states), self.results, self.best_models)
                print(f"  ✓ Checkpoint saved ({len(completed_states)}/{len(states)} states completed)")
                
            except Exception as e:
                print(f"Error processing {state}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Still save checkpoint even on error
                completed_states.add(state)  # Mark as attempted
                self.save_checkpoint(list(completed_states), self.results, self.best_models)
                continue
        
        # Final checkpoint save
        self.save_checkpoint(list(completed_states), self.results, self.best_models)
        print(f"\n✓ Training complete! Final checkpoint saved.")
        
        return self.results
    
    def get_best_model(self, state: str):
        """Get the best model for a specific state"""
        return self.best_models.get(state)
    
    def save_comparison_results(self, filepath: str = None):
        """Save comparison results to JSON"""
        if filepath is None:
            filepath = config.RESULTS_DIR / "model_comparison_results.json"
        
        # Convert to JSON-serializable format
        results_json = {}
        for state, result in self.results.items():
            results_json[state] = {
                'best_model': result['best_model'],
                'metrics': {k: {m: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                               for m, v in metrics.items()} 
                           for k, metrics in result['metrics'].items()},
                'train_size': int(result['train_size']),
                'test_size': int(result['test_size'])
            }
        
        with open(filepath, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\nComparison results saved to: {filepath}")
    
    def generate_forecasts(self, df: pd.DataFrame, states: List[str] = None) -> Dict:
        """
        Generate forecasts for the next 8 weeks using the best model for each state
        """
        if states is None:
            states = list(self.best_models.keys())
        
        preprocessor = DataPreprocessor()
        forecasts = {}
        
        for state in states:
            if state not in self.best_models:
                print(f"No trained model found for {state}, skipping...")
                continue
            
            try:
                best_model_info = self.best_models[state]
                model = best_model_info['model']
                model_name = best_model_info['model_name']
                
                # Get latest data for this state
                state_df = df[df[config.STATE_COLUMN] == state].copy()
                state_df = preprocessor.handle_missing_dates(state_df, state=None)
                state_df = preprocessor.create_all_features(state_df)
                
                # Get last date
                last_date = state_df[config.DATE_COLUMN].max()
                
                # Generate forecast dates
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=config.FORECAST_HORIZON_DAYS,
                    freq='D'
                )
                
                # Generate predictions
                if model_name == 'XGBoost':
                    # For XGBoost, we need to create future dataframe with features
                    # This is simplified - in production, you'd need to properly generate future features
                    future_df = pd.DataFrame({config.DATE_COLUMN: forecast_dates})
                    # Add state and category
                    if len(state_df) > 0:
                        future_df[config.STATE_COLUMN] = state
                        if config.CATEGORY_COLUMN in state_df.columns:
                            future_df[config.CATEGORY_COLUMN] = state_df[config.CATEGORY_COLUMN].iloc[0]
                    
                    # Create features for future dates (simplified approach)
                    future_df = preprocessor.create_time_features(future_df)
                    # Use last known values for lag features (simplified)
                    last_values = state_df[config.TARGET_COLUMN].tail(max(config.LAG_FEATURES)).values
                    for i, lag in enumerate(config.LAG_FEATURES):
                        if i < len(last_values):
                            future_df[f'lag_{lag}'] = last_values[-lag] if lag <= len(last_values) else last_values[-1]
                        else:
                            future_df[f'lag_{lag}'] = state_df[config.TARGET_COLUMN].iloc[-1]
                    
                    # Rolling features (use last known values)
                    last_rolling_mean_7 = state_df['rolling_mean_7'].iloc[-1] if 'rolling_mean_7' in state_df.columns else state_df[config.TARGET_COLUMN].iloc[-1]
                    last_rolling_mean_30 = state_df['rolling_mean_30'].iloc[-1] if 'rolling_mean_30' in state_df.columns else state_df[config.TARGET_COLUMN].iloc[-1]
                    future_df['rolling_mean_7'] = last_rolling_mean_7
                    future_df['rolling_mean_30'] = last_rolling_mean_30
                    future_df['rolling_std_7'] = state_df['rolling_std_7'].iloc[-1] if 'rolling_std_7' in state_df.columns else 0
                    future_df['rolling_std_30'] = state_df['rolling_std_30'].iloc[-1] if 'rolling_std_30' in state_df.columns else 0
                    future_df['rolling_min_7'] = state_df['rolling_min_7'].iloc[-1] if 'rolling_min_7' in state_df.columns else state_df[config.TARGET_COLUMN].iloc[-1]
                    future_df['rolling_max_7'] = state_df['rolling_max_7'].iloc[-1] if 'rolling_max_7' in state_df.columns else state_df[config.TARGET_COLUMN].iloc[-1]
                    future_df['rolling_min_30'] = state_df['rolling_min_30'].iloc[-1] if 'rolling_min_30' in state_df.columns else state_df[config.TARGET_COLUMN].iloc[-1]
                    future_df['rolling_max_30'] = state_df['rolling_max_30'].iloc[-1] if 'rolling_max_30' in state_df.columns else state_df[config.TARGET_COLUMN].iloc[-1]
                    
                    predictions = model.predict(future_df, preprocessor)
                    
                elif model_name == 'LSTM':
                    predictions = model.predict_from_data(state_df, config.FORECAST_HORIZON_DAYS, preprocessor)
                    
                elif model_name == 'StackedEnsemble':
                    # Create future dataframe for base models
                    future_df = pd.DataFrame({config.DATE_COLUMN: forecast_dates})
                    predictions = model.predict(future_df, state_df, preprocessor)
                    
                elif model_name == 'Prophet':
                    predictions = model.predict(config.FORECAST_HORIZON_DAYS, last_date)
                    
                else:  # ARIMA
                    predictions = model.predict(config.FORECAST_HORIZON_DAYS)
                
                forecasts[state] = {
                    'dates': forecast_dates.tolist(),
                    'predictions': predictions.tolist(),
                    'model_used': model_name,
                    'last_training_date': last_date.isoformat()
                }
                
                print(f"✓ Generated {len(predictions)} forecasts for {state} using {model_name}")
                
            except Exception as e:
                print(f"✗ Error generating forecasts for {state}: {str(e)}")
                continue
        
        return forecasts

