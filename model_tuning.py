"""
Hyperparameter tuning module for all forecasting models
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import config

warnings.filterwarnings('ignore')


class ModelTuner:
    """Hyperparameter tuning for forecasting models"""
    
    def __init__(self, n_splits: int = 3):
        """
        Initialize tuner with time series cross-validation
        
        Args:
            n_splits: Number of splits for time series CV
        """
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
    
    def tune_arima(self, train_data: pd.DataFrame, state: str) -> Dict:
        """
        Auto-tune ARIMA/SARIMA parameters using AIC
        
        Returns:
            dict with best order and seasonal_order
        """
        ts = train_data[config.TARGET_COLUMN].values
        
        # Try to use pmdarima if available
        try:
            import pmdarima as pm
            print(f"    Using pmdarima for auto ARIMA tuning...")
            
            # Auto ARIMA
            model = pm.auto_arima(
                ts,
                start_p=0, start_q=0,
                max_p=3, max_q=3,
                seasonal=True,
                m=7,  # Weekly seasonality
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_order=5,
                information_criterion='aic'
            )
            
            order = model.order
            seasonal_order = model.seasonal_order
            
            return {
                'order': order,
                'seasonal_order': seasonal_order,
                'use_sarima': True
            }
            
        except ImportError:
            print(f"    pmdarima not available, using grid search...")
            # Manual grid search
            best_aic = np.inf
            best_order = (1, 1, 1)
            best_seasonal = (1, 1, 1, 7)
            
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            
            # Limited grid search (to save time)
            p_values = [0, 1, 2]
            d_values = [0, 1]
            q_values = [0, 1, 2]
            
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        try:
                            model = SARIMAX(
                                ts,
                                order=(p, d, q),
                                seasonal_order=(1, 1, 1, 7),
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            fitted = model.fit(disp=False, maxiter=50)
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            return {
                'order': best_order,
                'seasonal_order': best_seasonal,
                'use_sarima': True
            }
    
    def tune_prophet(self, train_data: pd.DataFrame, state: str) -> Dict:
        """
        Tune Prophet hyperparameters using time series CV
        
        Returns:
            dict with best Prophet config
        """
        from prophet import Prophet
        
        prophet_df = pd.DataFrame({
            'ds': train_data[config.DATE_COLUMN],
            'y': np.maximum(train_data[config.TARGET_COLUMN].values, 0)
        })
        
        # Split for validation
        split_idx = int(len(prophet_df) * 0.8)
        train_df = prophet_df.iloc[:split_idx]
        val_df = prophet_df.iloc[split_idx:]
        
        best_rmse = np.inf
        best_config = config.PROPHET_CONFIG.copy()
        
        # Grid search over key parameters
        seasonality_modes = ['additive', 'multiplicative']
        yearly_seasonalities = [True, False]
        
        print(f"    Tuning Prophet (testing {len(seasonality_modes) * len(yearly_seasonalities)} configs)...")
        
        for seasonality_mode in seasonality_modes:
            for yearly_seasonality in yearly_seasonalities:
                try:
                    model = Prophet(
                        yearly_seasonality=yearly_seasonality,
                        weekly_seasonality=True,
                        daily_seasonality=False,
                        seasonality_mode=seasonality_mode,
                        changepoint_prior_scale=0.05
                    )
                    model.fit(train_df)
                    
                    # Predict on validation
                    future = model.make_future_dataframe(periods=len(val_df))
                    forecast = model.predict(future)
                    
                    # Get validation predictions
                    val_forecast = forecast.iloc[-len(val_df):]['yhat'].values
                    val_actual = val_df['y'].values
                    
                    rmse = np.sqrt(mean_squared_error(val_actual, val_forecast))
                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_config = {
                            'yearly_seasonality': yearly_seasonality,
                            'weekly_seasonality': True,
                            'daily_seasonality': False,
                            'seasonality_mode': seasonality_mode,
                            'changepoint_prior_scale': 0.05
                        }
                except:
                    continue
        
        return best_config
    
    def tune_xgboost(self, train_data: pd.DataFrame, preprocessor, state: str) -> Dict:
        """
        Tune XGBoost hyperparameters using time series CV
        
        Returns:
            dict with best XGBoost config
        """
        from xgboost import XGBRegressor
        
        # Prepare data
        X, y = preprocessor.prepare_data_for_ml(train_data, include_target=True)
        valid_mask = ~np.isnan(X).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 100:  # Need enough data for CV
            return config.XGBOOST_CONFIG.copy()
        
        # Split for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        best_rmse = np.inf
        best_config = config.XGBOOST_CONFIG.copy()
        
        # Limited grid search (to save time)
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [100, 200],
            'subsample': [0.8, 0.9]
        }
        
        print(f"    Tuning XGBoost (testing {len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['n_estimators']) * len(param_grid['subsample'])} configs)...")
        
        total_configs = np.prod([len(v) for v in param_grid.values()])
        tested = 0
        
        for max_depth in param_grid['max_depth']:
            for lr in param_grid['learning_rate']:
                for n_est in param_grid['n_estimators']:
                    for subsample in param_grid['subsample']:
                        tested += 1
                        if tested % 5 == 0:
                            print(f"      Tested {tested}/{total_configs} configs...", end='\r')
                        
                        try:
                            model = XGBRegressor(
                                max_depth=max_depth,
                                learning_rate=lr,
                                n_estimators=n_est,
                                subsample=subsample,
                                colsample_bytree=0.8,
                                random_state=42,
                                n_jobs=1
                            )
                            model.fit(X_train, y_train)
                            
                            y_pred = model.predict(X_val)
                            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                            
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_config = {
                                    'max_depth': max_depth,
                                    'learning_rate': lr,
                                    'n_estimators': n_est,
                                    'subsample': subsample,
                                    'colsample_bytree': 0.8,
                                    'random_state': 42
                                }
                        except:
                            continue
        
        print(f"      Best XGBoost RMSE: {best_rmse:.2f}")
        return best_config
    
    def tune_lstm(self, train_data: pd.DataFrame, state: str, preprocessor=None) -> Dict:
        """
        Tune LSTM hyperparameters (now supports multi-feature LSTM)
        
        Returns:
            dict with best LSTM config
        """
        if len(train_data) < 100:
            return config.LSTM_CONFIG.copy()
        
        # Split for validation
        split_idx = int(len(train_data) * 0.8)
        train_df = train_data.iloc[:split_idx]
        val_df = train_data.iloc[split_idx:]
        
        best_rmse = np.inf
        best_config = config.LSTM_CONFIG.copy()
        
        # Test different architectures
        units_options = [50, 64, 80]  # Increased range
        lookback_options = [21, 30]  # Test weekly and monthly patterns
        
        print(f"    Tuning LSTM (testing {len(units_options) * len(lookback_options)} configs)...")
        
        for units in units_options:
            for lookback in lookback_options:
                if len(train_df) < lookback + 10:
                    continue
                
                try:
                    from sklearn.preprocessing import MinMaxScaler
                    from tensorflow import keras
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import LSTM, Dense, Dropout
                    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
                    from tensorflow.keras.optimizers import Adam
                    
                    # Use multi-feature if preprocessor available
                    if preprocessor is not None:
                        # Multi-feature LSTM
                        X_train_all, y_train_all = preprocessor.prepare_data_for_ml(train_df, include_target=True)
                        X_val_all, y_val_all = preprocessor.prepare_data_for_ml(val_df, include_target=True)
                        
                        # Remove NaN
                        train_mask = ~np.isnan(X_train_all).any(axis=1) & ~np.isnan(y_train_all)
                        val_mask = ~np.isnan(X_val_all).any(axis=1) & ~np.isnan(y_val_all)
                        
                        X_train_all = X_train_all[train_mask]
                        y_train_all = y_train_all[train_mask]
                        X_val_all = X_val_all[val_mask]
                        y_val_all = y_val_all[val_mask]
                        
                        if len(X_train_all) < lookback + 10:
                            continue
                        
                        # Scale
                        feature_scaler = MinMaxScaler()
                        target_scaler = MinMaxScaler()
                        
                        X_train_scaled = feature_scaler.fit_transform(X_train_all)
                        y_train_scaled = target_scaler.fit_transform(y_train_all.reshape(-1, 1)).flatten()
                        X_val_scaled = feature_scaler.transform(X_val_all)
                        y_val_scaled = target_scaler.transform(y_val_all.reshape(-1, 1)).flatten()
                        
                        # Create sequences
                        X_train_seq, y_train_seq = self._create_multi_feature_sequences(X_train_scaled, y_train_scaled, lookback)
                        X_val_seq, y_val_seq = self._create_multi_feature_sequences(X_val_scaled, y_val_scaled, lookback)
                        
                        n_features = X_train_seq.shape[2]
                    else:
                        # Univariate LSTM
                        train_ts = train_df[config.TARGET_COLUMN].values
                        val_ts = val_df[config.TARGET_COLUMN].values
                        
                        scaler = MinMaxScaler()
                        train_scaled = scaler.fit_transform(train_ts.reshape(-1, 1)).flatten()
                        val_scaled = scaler.transform(val_ts.reshape(-1, 1)).flatten()
                        
                        X_train_seq, y_train_seq = self._create_sequences(train_scaled, lookback)
                        X_val_seq, y_val_seq = self._create_sequences(val_scaled, lookback)
                        
                        X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], 1))
                        X_val_seq = X_val_seq.reshape((X_val_seq.shape[0], X_val_seq.shape[1], 1))
                        n_features = 1
                        target_scaler = scaler
                    
                    if len(X_train_seq) == 0 or len(X_val_seq) == 0:
                        continue
                    
                    # Build improved model architecture
                    model = Sequential([
                        LSTM(units, return_sequences=True, input_shape=(lookback, n_features)),
                        Dropout(0.3),
                        LSTM(units // 2, return_sequences=True),
                        Dropout(0.3),
                        LSTM(units // 4, return_sequences=False),
                        Dropout(0.2),
                        Dense(32, activation='relu'),
                        Dropout(0.2),
                        Dense(1)
                    ])
                    
                    optimizer = Adam(learning_rate=0.001)
                    model.compile(optimizer=optimizer, loss='mse')
                    
                    # Callbacks
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-7, verbose=0)
                    ]
                    
                    # Train
                    model.fit(
                        X_train_seq, y_train_seq,
                        epochs=50,
                        batch_size=32,
                        validation_data=(X_val_seq, y_val_seq),
                        callbacks=callbacks,
                        verbose=0
                    )
                    
                    # Predict
                    y_pred_scaled = model.predict(X_val_seq, verbose=0).flatten()
                    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                    y_actual = val_df[config.TARGET_COLUMN].values[:len(y_pred)]
                    
                    if len(y_pred) == len(y_actual):
                        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
                        
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_config = {
                                'units': units,
                                'lookback': lookback,
                                'epochs': 100,
                                'batch_size': 32,
                                'validation_split': 0.2
                            }
                    
                    # Clean up
                    del model
                    keras.backend.clear_session()
                    
                except Exception as e:
                    continue
        
        if best_rmse < np.inf:
            print(f"      Best LSTM RMSE: {best_rmse:.2f}")
        
        return best_config
    
    def _create_multi_feature_sequences(self, X_data: np.ndarray, y_data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """Helper to create multi-feature sequences"""
        X_seq, y_seq = [], []
        for i in range(lookback, len(X_data)):
            X_seq.append(X_data[i-lookback:i])
            y_seq.append(y_data[i])
        return np.array(X_seq), np.array(y_seq)
    
    def _create_sequences(self, data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """Helper to create sequences"""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)

