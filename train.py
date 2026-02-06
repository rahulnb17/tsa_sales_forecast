"""
Main training script for the forecasting system
"""
import pandas as pd
import argparse
from pathlib import Path
import config
from data_preprocessing import DataPreprocessor
from model_comparison import ModelComparator


def main():
    parser = argparse.ArgumentParser(description="Train forecasting models")
    parser.add_argument(
        "--data-file",
        type=str,
        default=str(config.DATA_DIR / config.DATA_FILE),
        help="Path to the data file"
    )
    parser.add_argument(
        "--states",
        type=str,
        nargs="+",
        default=None,
        help="Specific states to train (default: all states)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.RESULTS_DIR),
        help="Output directory for results"
    )
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Disable hyperparameter tuning (use default configs)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume functionality (start from scratch, overwrite existing models)"
    )
    
    args = parser.parse_args()
    
    # Check if data file exists
    data_file = Path(args.data_file)
    if not data_file.exists():
        # Auto-detect: if user didn't override --data-file, try to find any xlsx in data/
        default_path = Path(config.DATA_DIR) / config.DATA_FILE
        if str(data_file) == str(default_path):
            candidates = sorted(Path(config.DATA_DIR).glob("*.xlsx"))
            if len(candidates) == 1:
                data_file = candidates[0]
                print(f"Note: Default data file not found. Auto-detected Excel file: {data_file.name}")
            elif len(candidates) > 1:
                print(f"Error: Default data file not found at {data_file}")
                print("Multiple Excel files found in data/. Please specify one with --data-file:")
                for c in candidates:
                    print(f"  - {c}")
                return
            else:
                print(f"Error: Data file not found at {data_file}")
                print(f"Please put your Excel file in: {config.DATA_DIR}")
                print("Or pass it explicitly: python train.py --data-file path\\to\\file.xlsx")
                return
        else:
            print(f"Error: Data file not found at {data_file}")
            print("Please provide a valid path via --data-file.")
            return
    
    print("="*70)
    print("Sales Forecasting System - Training")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from: {data_file}")
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(str(data_file))
    
    print(f"Data loaded: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Display basic statistics
    if config.STATE_COLUMN in df.columns:
        states = df[config.STATE_COLUMN].unique()
        print(f"\nStates found: {len(states)}")
        print(f"States: {', '.join(states[:10])}{'...' if len(states) > 10 else ''}")
    
    if config.DATE_COLUMN in df.columns:
        df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN])
        print(f"\nDate range: {df[config.DATE_COLUMN].min()} to {df[config.DATE_COLUMN].max()}")
    
    # Train and compare models
    print("\n" + "="*70)
    print("Training and Comparing Models")
    if not args.no_tuning:
        print("(Hyperparameter tuning ENABLED - this will take longer but produce better results)")
    else:
        print("(Hyperparameter tuning DISABLED - using default configs)")
    print("="*70)
    
    comparator = ModelComparator(
        enable_tuning=not args.no_tuning,
        resume=not args.no_resume  # Resume by default unless --no-resume is specified
    )
    results = comparator.train_and_compare_models(df, states=args.states)
    
    # Save results
    results_file = Path(args.output_dir) / "model_comparison_results.json"
    comparator.save_comparison_results(str(results_file))
    
    # Generate forecasts
    print("\n" + "="*70)
    print("Generating Forecasts")
    print("="*70)
    
    forecasts = comparator.generate_forecasts(df, states=args.states)
    
    # Save forecasts
    import json
    forecasts_file = Path(args.output_dir) / "forecasts.json"
    # Convert dates to strings for JSON serialization
    forecasts_json = {}
    for state, forecast_data in forecasts.items():
        forecasts_json[state] = {
            'dates': [str(d) for d in forecast_data['dates']],
            'predictions': [float(p) for p in forecast_data['predictions']],
            'model_used': forecast_data['model_used'],
            'last_training_date': forecast_data['last_training_date']
        }
    
    with open(forecasts_file, 'w') as f:
        json.dump(forecasts_json, f, indent=2)
    
    print(f"\nForecasts saved to: {forecasts_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)
    print(f"States processed: {len(results)}")
    print(f"Models trained: ARIMA/SARIMA, Prophet, XGBoost, LSTM")
    print(f"\nBest models by state:")
    for state, result in results.items():
        print(f"  {state}: {result['best_model']}")
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)
    print(f"\nTo start the API server, run:")
    print(f"  python api.py")
    print(f"\nOr use uvicorn:")
    print(f"  uvicorn api:app --host {config.API_HOST} --port {config.API_PORT}")


if __name__ == "__main__":
    main()

