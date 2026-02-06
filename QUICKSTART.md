# Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- Excel file: `Forecasting Case-Study.xlsx`

## Setup (5 minutes)

1. **Install dependencies**:
   
   **Important for Python 3.12:** Install setuptools first to avoid distutils errors:
   ```bash
   pip install --upgrade setuptools wheel
   pip install -r requirements.txt
   ```
   
   Or use the provided script:
   - Windows: `install_requirements.bat`
   - Linux/Mac: `./install_requirements.sh`

2. **Place your data file**:
   - Copy `Forecasting Case-Study.xlsx` to the `data/` directory
   - The file should have columns: `State`, `Date`, `Total`, `Category`

3. **Train the models**:
   ```bash
   python train.py
   ```
   
   This will:
   - Load and preprocess your data
   - Train 4 models for each state (ARIMA, Prophet, XGBoost, LSTM)
   - Compare models and select the best one
   - Generate 8-week forecasts
   - Save everything to `models/` and `results/` directories

4. **Start the API server**:
   ```bash
   python api.py
   ```
   
   The API will start at `http://localhost:8000`

## Test the API

### Option 1: Using the example script
```bash
python example_usage.py
```

### Option 2: Using curl
```bash
# Check health
curl http://localhost:8000/health

# Get available states
curl http://localhost:8000/states

# Get forecast
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{"state": "Alabama", "weeks": 8}'
```

### Option 3: Using browser
- Visit `http://localhost:8000/docs` for interactive API documentation
- Visit `http://localhost:8000/redoc` for alternative documentation

## Expected Output

After training, you should see:
- Trained models in `models/` directory (one per state)
- Comparison results in `results/model_comparison_results.json`
- Forecasts in `results/forecasts.json`

## Troubleshooting

**Issue**: `Data file not found`
- Solution: Make sure `Forecasting Case-Study.xlsx` is in the `data/` directory

**Issue**: `No module named 'prophet'` or similar
- Solution: Run `pip install -r requirements.txt` again

**Issue**: Model training fails for a state
- Solution: Check that the state has sufficient data (at least 50 data points)

**Issue**: API returns 404 for a state
- Solution: Make sure models are trained first using `python train.py`

## Next Steps

- Read `README.md` for detailed documentation
- Customize `config.py` to adjust model parameters
- Explore the API at `http://localhost:8000/docs`

