@echo off
REM Installation script for Windows with Python 3.12 compatibility

echo Installing dependencies for Python 3.12...
echo.

REM First, upgrade pip and install setuptools
echo Step 1: Upgrading pip and setuptools...
python -m pip install --upgrade pip setuptools wheel
echo.

REM Install core dependencies first
echo Step 2: Installing core dependencies...
python -m pip install pandas numpy scikit-learn statsmodels
echo.

REM Install API dependencies
echo Step 3: Installing API dependencies...
python -m pip install fastapi uvicorn[standard] pydantic python-multipart requests
echo.

REM Install utilities
echo Step 4: Installing utilities...
python -m pip install openpyxl joblib matplotlib seaborn holidays
echo.

REM Install XGBoost
echo Step 5: Installing XGBoost...
python -m pip install xgboost
echo.

REM Install TensorFlow
echo Step 6: Installing TensorFlow...
python -m pip install tensorflow
echo.

REM Install Prophet (may require special handling)
echo Step 7: Installing Prophet (this may take a while)...
python -m pip install prophet --no-build-isolation
if errorlevel 1 (
    echo.
    echo WARNING: Prophet installation failed with standard method.
    echo Trying alternative installation method...
    python -m pip install --no-build-isolation prophet
)
echo.

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo If Prophet installation failed, you may need to:
echo 1. Use Python 3.11 instead of 3.12, OR
echo 2. Install Prophet from source: pip install git+https://github.com/facebook/prophet.git
echo.

pause
