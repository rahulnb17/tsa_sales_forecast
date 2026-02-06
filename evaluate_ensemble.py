"""
Quick script to evaluate Stacked Ensemble on already trained models
This uses existing models without retraining base models
"""
import pandas as pd
from pathlib import Path
import config
from data_preprocessing import DataPreprocessor
from model_comparison import ModelComparator
from models import ARIMAModel, XGBoostModel, LSTMModel, StackedEnsembleModel, PROPHET_AVAILABLE

if PROPHET_AVAILABLE:
    from models import ProphetModel


def load_existing_model(state: str, model_name: str):
    """Load an existing trained model"""
    try:
        if model_name == 'LSTM':
            model_file = config.MODELS_DIR / f"{state}_lstm.h5"
            if model_file.exists():
                model = LSTMModel(state)
                model.load(str(model_file))
                return model
        elif model_name == 'Prophet':
            if PROPHET_AVAILABLE:
                model_file = config.MODELS_DIR / f"{state}_prophet.joblib"
                if model_file.exists():
                    model = ProphetModel(state)
                    model.load(str(model_file))
                    return model
        elif model_name == 'XGBoost':
            model_file = config.MODELS_DIR / f"{state}_xgboost.joblib"
            if model_file.exists():
                model = XGBoostModel(state)
                model.load(str(model_file))
                return model
        elif model_name == 'ARIMA':
            model_file = config.MODELS_DIR / f"{state}_arima.joblib"
            if model_file.exists():
                model = ARIMAModel(state)
                model.load(str(model_file))
                return model
    except Exception as e:
        print(f"      Error loading {model_name}: {str(e)}")
    return None


def main():
    print("="*70)
    print("Evaluating Stacked Ensemble on Existing Models")
    print("="*70)
    
    # Load data
    data_file = Path(config.DATA_DIR) / config.DATA_FILE
    if not data_file.exists():
        candidates = sorted(Path(config.DATA_DIR).glob("*.xlsx"))
        if len(candidates) == 1:
            data_file = candidates[0]
        else:
            print(f"Error: Data file not found")
            return
    
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(str(data_file))
    
    states = df[config.STATE_COLUMN].unique().tolist()
    
    results = {}
    
    for state in states:
        print(f"\n{'='*60}")
        print(f"Processing: {state}")
        print(f"{'='*60}")
        
        try:
            # Prepare data
            train_df, test_df = preprocessor.prepare_data_for_state(df, state)
            
            if len(train_df) < 50:
                continue
            
            # Load existing base models
            base_models = {}
            for model_name in ['XGBoost', 'Prophet', 'ARIMA', 'LSTM']:
                model = load_existing_model(state, model_name)
                if model and model.is_fitted:
                    base_models[model_name] = model
                    print(f"  ✓ Loaded {model_name}")
                else:
                    print(f"  ✗ {model_name} not found or not fitted")
            
            if len(base_models) < 2:
                print(f"  Need at least 2 base models, skipping {state}")
                continue
            
            # Train stacked ensemble
            print(f"\n  Training Stacked Ensemble...")
            stacked = StackedEnsembleModel(state, base_models=base_models)
            stacked.fit(train_df, test_df, preprocessor)
            
            if not stacked.is_fitted:
                print(f"  ✗ Stacked Ensemble training failed")
                continue
            
            # Evaluate
            metrics = stacked.evaluate(test_df, train_df, preprocessor)
            print(f"  Stacked Ensemble: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%")
            
            # Compare with best existing model
            # Load best model from checkpoint or find it
            best_model_file = None
            for ext in ['.joblib', '.h5']:
                for model_name in ['xgboost', 'prophet', 'arima', 'lstm']:
                    candidate = config.MODELS_DIR / f"{state}_{model_name}{ext}"
                    if candidate.exists():
                        best_model_file = candidate
                        break
                if best_model_file:
                    break
            
            if best_model_file:
                # Get model name from file
                model_name = best_model_file.stem.split('_')[-1]
                print(f"  Best existing model: {model_name}")
            
            results[state] = {
                'stacked_ensemble': metrics,
                'base_models_count': len(base_models)
            }
            
            # Save stacked ensemble
            stacked_path = config.MODELS_DIR / f"{state}_stacked_ensemble.joblib"
            stacked.save(str(stacked_path))
            print(f"  ✓ Saved to {stacked_path}")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print(f"Evaluation Complete!")
    print(f"Processed {len(results)} states")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

