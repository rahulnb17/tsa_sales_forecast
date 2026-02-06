"""
Retrain with Stacked Ensemble - trains all models including ensemble
This ensures all base models are saved for ensemble use
"""
import pandas as pd
from pathlib import Path
import config
from data_preprocessing import DataPreprocessor
from model_comparison import ModelComparator

def main():
    print("="*70)
    print("Retraining with Stacked Ensemble")
    print("This will train all models including Stacked Ensemble")
    print("All models will be saved (not just the best one)")
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
    
    # Train with ensemble (resume=False so it visits ALL states, but internal logic will skip XGBoost training if checkpoint exists)
    # Disable tuning to speed up training (approx 10x faster)
    comparator = ModelComparator(enable_tuning=False, resume=False)
    results = comparator.train_and_compare_models(df)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nAll models (including Stacked Ensemble) have been trained and saved.")
    print(f"Check results/model_comparison_results.json for detailed metrics.")

if __name__ == "__main__":
    main()

