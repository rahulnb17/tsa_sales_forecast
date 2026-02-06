"""
Models package
"""
from .arima_model import ARIMAModel
from .xgboost_model import XGBoostModel
from .lstm_model import LSTMModel
from .stacked_ensemble_model import StackedEnsembleModel

# Try to import Prophet, but don't fail if it's not available
try:
    from .prophet_model import ProphetModel
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    # Create a dummy class to prevent import errors
    class ProphetModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("Prophet is not installed. Install it with: pip install prophet --no-build-isolation")

if PROPHET_AVAILABLE:
    __all__ = ['ARIMAModel', 'ProphetModel', 'XGBoostModel', 'LSTMModel', 'StackedEnsembleModel']
else:
    __all__ = ['ARIMAModel', 'XGBoostModel', 'LSTMModel', 'StackedEnsembleModel']

