# Quick Fix for Python 3.12 Installation Error

## The Problem
You're getting a `ModuleNotFoundError: No module named 'distutils'` error because Python 3.12 removed `distutils`.

## Quick Solution (Try This First)

Run these commands in order:

```bash
# Step 1: Upgrade pip and install setuptools
python -m pip install --upgrade pip setuptools wheel

# Step 2: Install requirements
pip install -r requirements.txt
```

## If That Doesn't Work

### Option A: Use the Installation Script (Windows)
```bash
install_requirements.bat
```

### Option B: Install Prophet Separately
If Prophet fails, install it with special flags:
```bash
pip install prophet --no-build-isolation
```

### Option C: Use Python 3.11 (Most Reliable)
Python 3.11 has better package compatibility:

1. Install Python 3.11 from python.org
2. Create a virtual environment:
   ```bash
   python3.11 -m venv venv
   ```
3. Activate it:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## System Will Still Work Without Prophet

The system has been updated to work even if Prophet fails to install. It will:
- Use ARIMA, XGBoost, and LSTM models
- Skip Prophet if it's not available
- Still select the best model from the available ones

## Verify Installation

After installation, test it:
```bash
python -c "import pandas, numpy, sklearn, statsmodels, xgboost, tensorflow, fastapi; print('Core packages OK!')"
```

If Prophet is installed:
```bash
python -c "import prophet; print('Prophet OK!')"
```

## Next Steps

Once installation is complete:
1. Place your Excel file in the `data/` directory
2. Run: `python train.py`
3. Start API: `python api.py`

For more details, see `INSTALLATION_FIX.md`

