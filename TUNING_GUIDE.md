# Hyperparameter Tuning Guide

## Overview

The system now includes **comprehensive hyperparameter tuning** for all models to ensure a fair comparison and maximize accuracy. This means XGBoost won't automatically win - each model gets optimized for each state!

## What's New

### 1. **ARIMA/SARIMA Auto-Tuning**
- **If `pmdarima` is installed**: Uses `auto_arima()` to automatically find the best (p,d,q)(P,D,Q,s) parameters
- **If not available**: Falls back to a grid search over common parameter combinations
- **Result**: Each state gets its own optimized ARIMA order

### 2. **Prophet Tuning**
- Tests different `seasonality_mode` (additive vs multiplicative)
- Tests with/without yearly seasonality
- Uses validation set to select best configuration
- **Result**: Prophet optimized per state

### 3. **XGBoost Tuning**
- Grid search over:
  - `max_depth`: [4, 6, 8]
  - `learning_rate`: [0.05, 0.1, 0.15]
  - `n_estimators`: [100, 200]
  - `subsample`: [0.8, 0.9]
- Uses time series validation split
- **Result**: XGBoost hyperparameters optimized per state

### 4. **LSTM Tuning**
- Tests different architectures:
  - `units`: [32, 50, 64]
  - `lookback`: [14, 30]
- Includes **EarlyStopping** and **ReduceLROnPlateau** callbacks
- **Result**: LSTM architecture optimized per state

## Usage

### Enable Tuning (Default)
```bash
python train.py
```
Tuning is **enabled by default** for best results.

### Disable Tuning (Faster, but less accurate)
```bash
python train.py --no-tuning
```
Use this if you want faster training with default hyperparameters.

## Expected Results

After tuning, you should see:
- **More diverse winners**: Not just XGBoost - some states may win with Prophet, ARIMA, or LSTM
- **Better accuracy**: All models perform better with tuned hyperparameters
- **Fairer comparison**: Each model gets optimized, so the best one truly wins

## Training Time

**With tuning enabled**: ~2-5x longer training time (but much better results)
- ARIMA: +30-60 seconds per state (if using pmdarima)
- Prophet: +10-20 seconds per state
- XGBoost: +1-3 minutes per state (grid search)
- LSTM: +2-5 minutes per state (architecture search)

**Total**: Expect ~5-10 minutes per state with tuning (vs ~2-3 minutes without)

## Optional: Install pmdarima for Better ARIMA Tuning

For the best ARIMA results, install `pmdarima`:

```bash
pip install pmdarima
```

This enables automatic ARIMA order selection using AIC, which is much better than manual grid search.

## Example Output

With tuning enabled, you'll see:

```
Tuning hyperparameters...
  Tuning ARIMA...
    Using pmdarima for auto ARIMA tuning...
    Best ARIMA order: (2, 1, 2), seasonal: (1, 1, 1, 7)
  Tuning Prophet...
    Tuning Prophet (testing 4 configs)...
    Best Prophet mode: multiplicative
  Tuning XGBoost...
    Tuning XGBoost (testing 36 configs)...
      Tested 36/36 configs...
      Best XGBoost RMSE: 1234.56
  Tuning LSTM...
    Tuning LSTM (testing 6 configs)...
      Best LSTM RMSE: 1456.78

Training models...
  Training ARIMA...
  ✓ ARIMA trained successfully
  ...
```

## Troubleshooting

**If tuning is too slow**: Use `--no-tuning` flag for faster training

**If pmdarima fails to install**: The system will fall back to manual grid search (still works, just slower)

**If memory errors during XGBoost tuning**: The grid search is limited, but you can reduce it further in `model_tuning.py`

## Next Steps

After tuning, you can:
1. Check `results/model_comparison_results.json` to see which model won for each state
2. Compare metrics - you should see more competitive results across all models
3. Some states may now win with Prophet or ARIMA instead of XGBoost!

