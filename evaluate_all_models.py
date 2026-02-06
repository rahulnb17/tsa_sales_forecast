"""
Script to evaluate and compare ALL saved models for specific states.
Usage: python evaluate_all_models.py --states Arizona
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import config
from data_preprocessing import DataPreprocessor
from models import ARIMAModel, XGBoostModel, LSTMModel, StackedEnsembleModel, PROPHET_AVAILABLE

if PROPHET_AVAILABLE:
    from models import ProphetModel

def load_model(state, model_name):
    """Load a specific model from disk"""
    try:
        if model_name == 'ARIMA':
            path = config.MODELS_DIR / f"{state}_arima.joblib"
            if path.exists():
                model = ARIMAModel(state)
                model.load(str(path))
                return model
        elif model_name == 'XGBoost':
            path = config.MODELS_DIR / f"{state}_xgboost.joblib"
            if path.exists():
                model = XGBoostModel(state)
                model.load(str(path))
                return model
        elif model_name == 'Prophet' and PROPHET_AVAILABLE:
            path = config.MODELS_DIR / f"{state}_prophet.joblib"
            if path.exists():
                model = ProphetModel(state)
                model.load(str(path))
                return model
        elif model_name == 'LSTM':
            path = config.MODELS_DIR / f"{state}_lstm.h5"
            if path.exists():
                model = LSTMModel(state)
                model.load(str(path))
                return model
        elif model_name == 'StackedEnsemble':
            path = config.MODELS_DIR / f"{state}_stacked_ensemble.joblib"
            if path.exists():
                model = StackedEnsembleModel(state)
                model.load(str(path))
                return model
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Compare all saved models")
    parser.add_argument("--states", type=str, nargs="+", help="States to evaluate (e.g. Arizona)")
    args = parser.parse_args()

    print("="*70)
    print("Evaluating ALL Saved Models")
    print("="*70)
    
    # Load Data
    data_file = Path(config.DATA_DIR) / config.DATA_FILE
    if not data_file.exists():
        candidates = sorted([p for p in Path(config.DATA_DIR).glob("*.xlsx") if not p.name.startswith("~$")])
        if candidates:
            data_file = candidates[0]
        else:
            print("Data file not found.")
            return

    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(str(data_file))

    # Determine states
    all_states = df[config.STATE_COLUMN].unique()
    target_states = args.states if args.states else all_states

    for state in target_states:
        if state not in all_states:
            print(f"Warning: State '{state}' not found in data.")
            continue

        print(f"\nResults for state: {state}")
        print("-" * 60)

        # Prepare Data
        train_df, test_df = preprocessor.prepare_data_for_state(df, state)
        if len(train_df) < 50:
            print("Insufficient data.")
            continue

        # Load and Evaluate Models
        results = []
        model_types = ['ARIMA', 'XGBoost', 'Prophet', 'LSTM', 'StackedEnsemble']

        # We need base models loaded for StackedEnsemble to work
        base_models_cache = {}

        for name in model_types:
            model = load_model(state, name)
            if model:
                # If it's the ensemble, we must re-attach base models
                if name == 'StackedEnsemble':
                    model.base_models = {k: v for k, v in base_models_cache.items() if k != 'StackedEnsemble'}
                
                # Evaluate
                try:
                    if name == 'XGBoost':
                        metrics = model.evaluate(test_df, preprocessor)
                    elif name in ['LSTM', 'StackedEnsemble']:
                        metrics = model.evaluate(test_df, train_df, preprocessor)
                    else: # ARIMA, Prophet
                        metrics = model.evaluate(test_df)
                    results.append({
                        'Model': name,
                        'RMSE': metrics['rmse'],
                        'MAE': metrics['mae'],
                        'MAPE': metrics['mape']
                    })
                    base_models_cache[name] = model # Cache for ensemble
                except Exception as e:
                    print(f"  Error evaluating {name}: {e}")
            else:
                # print(f"  {name} not found.")
                pass

        # Sort and Print
        results.sort(key=lambda x: x['MAPE']) # Sort by Accuracy (MAPE) this time
        
        print(f"{'Model':<20} {'MAPE':<10} {'RMSE':<15} {'MAE':<15}")
        print("-" * 60)
        for r in results:
            print(f"{r['Model']:<20} {r['MAPE']:.2f}%     {r['RMSE']:<15.2f} {r['MAE']:<15.2f}")
        
        if results:
            print(f"\nBest Model for {state}: {results[0]['Model']}")

if __name__ == "__main__":
    main()
