# Installation Fix for Python 3.12

## Problem
Python 3.12 removed the `distutils` module, which causes installation errors for some packages (particularly `prophet`).

## Solutions

### Solution 1: Install setuptools first (Recommended)

```bash
# Upgrade pip and install setuptools first
python -m pip install --upgrade pip setuptools wheel

# Then install requirements
pip install -r requirements.txt
```

### Solution 2: Use installation scripts

**Windows:**
```bash
install_requirements.bat
```

**Linux/Mac:**
```bash
chmod +x install_requirements.sh
./install_requirements.sh
```

### Solution 3: Install Prophet separately with special flags

If Prophet fails to install, try:

```bash
pip install prophet --no-build-isolation
```

Or install from source:
```bash
pip install git+https://github.com/facebook/prophet.git
```

### Solution 4: Use Python 3.11 (Most Reliable)

If you continue having issues, Python 3.11 has better package compatibility:

1. Install Python 3.11
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

### Solution 5: Manual step-by-step installation

If automatic installation fails, install packages in this order:

```bash
# 1. Essential tools
pip install --upgrade pip setuptools wheel

# 2. Core data libraries
pip install pandas numpy scikit-learn statsmodels

# 3. API framework
pip install fastapi uvicorn[standard] pydantic python-multipart requests

# 4. Utilities
pip install openpyxl joblib matplotlib seaborn holidays

# 5. XGBoost
pip install xgboost

# 6. TensorFlow
pip install tensorflow

# 7. Prophet (last, as it's most problematic)
pip install prophet --no-build-isolation
```

## Verify Installation

After installation, verify all packages are installed:

```bash
python -c "import pandas, numpy, sklearn, statsmodels, prophet, xgboost, tensorflow, fastapi; print('All packages installed successfully!')"
```

## Alternative: Skip Prophet (if needed)

If Prophet continues to fail, you can modify the code to skip it:

1. Comment out Prophet imports in `models/__init__.py`
2. Modify `model_comparison.py` to skip Prophet if import fails
3. The system will still work with ARIMA, XGBoost, and LSTM models

## Troubleshooting

**Error: "No module named 'distutils'"**
- Solution: Install setuptools: `pip install --upgrade setuptools`

**Error: "Microsoft Visual C++ 14.0 or greater is required"** (Windows)
- Solution: Install Microsoft C++ Build Tools from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

**Error: Prophet compilation fails**
- Solution: Use Solution 4 (Python 3.11) or install Prophet from source

**Error: TensorFlow installation fails**
- Solution: Make sure you have a compatible Python version (3.8-3.11 recommended for TensorFlow 2.15)

