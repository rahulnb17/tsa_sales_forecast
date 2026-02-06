"""
LSTM deep learning model implementation
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import warnings
import joblib
import config

warnings.filterwarnings('ignore')


class LSTMModel:
    """LSTM deep learning forecasting model"""
    
    def __init__(self, state: str, lstm_config: Dict = None, use_features: bool = True):
        self.state = state
        self.lstm_config = lstm_config or config.LSTM_CONFIG
        self.use_features = use_features  # Use multi-feature LSTM instead of univariate
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()  # Separate scaler for features
        self.lookback = self.lstm_config.get('lookback', 30)
        self.feature_columns = []
        self.is_fitted = False
        
    def create_sequences(self, data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training (univariate)"""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def create_multi_feature_sequences(self, X_data: np.ndarray, y_data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for multi-feature LSTM"""
        X_seq, y_seq = [], []
        for i in range(lookback, len(X_data)):
            X_seq.append(X_data[i-lookback:i])
            y_seq.append(y_data[i])
        return np.array(X_seq), np.array(y_seq)
    
    def fit(self, train_data: pd.DataFrame, preprocessor=None):
        """Train the LSTM model"""
        try:
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
            from tensorflow.keras.optimizers import Adam
            
            if self.use_features and preprocessor is not None:
                # Multi-feature LSTM (uses same features as XGBoost)
                self.feature_columns = preprocessor.feature_columns
                
                # Prepare features and target
                X_all, y_all = preprocessor.prepare_data_for_ml(train_data, include_target=True)
                
                # Remove rows with NaN
                valid_mask = ~np.isnan(X_all).any(axis=1) & ~np.isnan(y_all)
                X_all = X_all[valid_mask]
                y_all = y_all[valid_mask]
                
                if len(X_all) < self.lookback + 10:
                    print(f"Insufficient data for multi-feature LSTM in {self.state}")
                    self.is_fitted = False
                    return
                
                # Scale features and target separately
                X_scaled = self.feature_scaler.fit_transform(X_all)
                y_scaled = self.scaler.fit_transform(y_all.reshape(-1, 1)).flatten()
                
                # Create sequences with multiple features
                X_seq, y_seq = self.create_multi_feature_sequences(X_scaled, y_scaled, self.lookback)
                
                n_features = X_seq.shape[2]
                
            else:
                # Univariate LSTM (original approach)
                ts = train_data[config.TARGET_COLUMN].values.reshape(-1, 1)
                ts_scaled = self.scaler.fit_transform(ts).flatten()
                
                if len(ts_scaled) < self.lookback + 1:
                    print(f"Insufficient data for LSTM model in {self.state}")
                    self.is_fitted = False
                    return
                
                X_seq, y_seq = self.create_sequences(ts_scaled, self.lookback)
                X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
                n_features = 1
            
            # Build improved architecture
            units = self.lstm_config.get('units', 64)  # Increased default
            
            self.model = Sequential([
                LSTM(units, return_sequences=True, input_shape=(self.lookback, n_features)),
                Dropout(0.3),
                LSTM(units // 2, return_sequences=True),
                Dropout(0.3),
                LSTM(units // 4, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            
            # Use Adam with learning rate schedule
            optimizer = Adam(learning_rate=0.001)
            self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            # Enhanced callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss', 
                    patience=15, 
                    restore_best_weights=True, 
                    verbose=0,
                    min_delta=1e-6
                ),
                ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.3, 
                    patience=7, 
                    min_lr=1e-7, 
                    verbose=0
                )
            ]
            
            # Train with more epochs
            epochs = self.lstm_config.get('epochs', 100)  # Increased default
            
            self.model.fit(
                X_seq, y_seq,
                epochs=epochs,
                batch_size=self.lstm_config.get('batch_size', 32),
                validation_split=self.lstm_config.get('validation_split', 0.2),
                callbacks=callbacks,
                verbose=0
            )
            
            self.is_fitted = True
            
        except Exception as e:
            print(f"Error fitting LSTM model for {self.state}: {str(e)}")
            import traceback
            traceback.print_exc()
            self.is_fitted = False
    
    def predict(self, n_periods: int, last_sequence: np.ndarray = None) -> np.ndarray:
        """Generate forecasts"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            predictions = []
            
            # Use last sequence from training if not provided
            if last_sequence is None:
                # Get last lookback values from training
                ts = self.scaler.inverse_transform(
                    self.scaler.transform(
                        np.array([0]).reshape(-1, 1)
                    )
                )
                # This is a simplified approach - in practice, we'd store the last sequence
                # For now, we'll use a workaround
                last_sequence = np.zeros(self.lookback)
            else:
                # Scale the sequence
                last_sequence = self.scaler.transform(last_sequence.reshape(-1, 1)).flatten()
            
            # Ensure we have the right shape
            if len(last_sequence) < self.lookback:
                # Pad with zeros or repeat last value
                last_sequence = np.pad(
                    last_sequence,
                    (self.lookback - len(last_sequence), 0),
                    mode='edge'
                )
            
            current_sequence = last_sequence[-self.lookback:].reshape(1, self.lookback, 1)
            
            # Generate predictions step by step
            for _ in range(n_periods):
                # Predict next value
                next_pred = self.model.predict(current_sequence, verbose=0)
                predictions.append(next_pred[0, 0])
                
                # Update sequence: remove first element, add prediction
                current_sequence = np.append(
                    current_sequence[0, 1:, :],
                    next_pred.reshape(1, 1)
                ).reshape(1, self.lookback, 1)
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions).flatten()
            predictions = np.maximum(predictions, 0)  # Ensure non-negative
            
            return predictions
        except Exception as e:
            print(f"Error in LSTM prediction for {self.state}: {str(e)}")
            return np.full(n_periods, 0)
    
    def predict_from_data(self, data: pd.DataFrame, n_periods: int, preprocessor=None) -> np.ndarray:
        """Generate forecasts using the last sequence from provided data"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.use_features and preprocessor is not None and len(self.feature_columns) > 0:
            # Multi-feature prediction
            X_all, _ = preprocessor.prepare_data_for_ml(data, include_target=False)
            
            # Handle NaN
            X_df = pd.DataFrame(X_all)
            X_df = X_df.ffill().fillna(0)
            X_all = X_df.values
            
            # Scale features
            X_scaled = self.feature_scaler.transform(X_all)
            
            # Get last sequence
            if len(X_scaled) < self.lookback:
                # Pad with last row
                padding = np.tile(X_scaled[-1:], (self.lookback - len(X_scaled), 1))
                X_scaled = np.vstack([padding, X_scaled])
            
            last_sequence = X_scaled[-self.lookback:]
            
            # Generate predictions
            predictions = []
            current_seq = last_sequence.copy()
            
            for _ in range(n_periods):
                # Predict
                pred_scaled = self.model.predict(current_seq.reshape(1, self.lookback, -1), verbose=0)[0, 0]
                predictions.append(pred_scaled)
                
                # Update sequence: shift and add prediction as new feature
                # For simplicity, we'll use the last row's features with updated target prediction
                # In production, you'd properly generate future features
                new_row = current_seq[-1].copy()
                # Shift sequence
                current_seq = np.vstack([current_seq[1:], new_row])
            
            # Inverse transform
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions).flatten()
            predictions = np.maximum(predictions, 0)
            
            return predictions
        else:
            # Univariate prediction
            ts = data[config.TARGET_COLUMN].values
            if len(ts) < self.lookback:
                ts = np.pad(ts, (self.lookback - len(ts), 0), mode='edge')
            
            last_sequence = ts[-self.lookback:]
            return self.predict(n_periods, last_sequence)
    
    def evaluate(self, test_data: pd.DataFrame, train_data: pd.DataFrame, preprocessor=None) -> dict:
        """Evaluate model on test data"""
        if not self.is_fitted:
            return {"mae": np.inf, "rmse": np.inf, "mape": np.inf}
        
        try:
            # Combine train and test for feature generation
            combined_data = pd.concat([train_data, test_data], ignore_index=True)
            
            if self.use_features and preprocessor is not None:
                # Use multi-feature prediction
                predictions = self.predict_from_data(combined_data, len(test_data), preprocessor)
            else:
                # Univariate prediction
                train_ts = train_data[config.TARGET_COLUMN].values
                if len(train_ts) < self.lookback:
                    train_ts = np.pad(train_ts, (self.lookback - len(train_ts), 0), mode='edge')
                
                last_sequence = train_ts[-self.lookback:]
                predictions = self.predict(len(test_data), last_sequence)
            
            actual = test_data[config.TARGET_COLUMN].values
            
            # Ensure same length
            min_len = min(len(predictions), len(actual))
            predictions = predictions[:min_len]
            actual = actual[:min_len]
            
            mae = np.mean(np.abs(predictions - actual))
            rmse = np.sqrt(np.mean((predictions - actual) ** 2))
            
            # MAPE
            mask = actual != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((actual[mask] - predictions[mask]) / actual[mask])) * 100
            else:
                mape = np.inf
            
            return {
                "mae": mae,
                "rmse": rmse,
                "mape": mape
            }
        except Exception as e:
            print(f"Error evaluating LSTM model for {self.state}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"mae": np.inf, "rmse": np.inf, "mape": np.inf}
    
    def save(self, filepath: str):
        """Save the fitted model"""
        if self.is_fitted:
            self.model.save(filepath)
            # Save scalers separately
            scaler_path = filepath.replace('.h5', '_scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            if self.use_features:
                feature_scaler_path = filepath.replace('.h5', '_feature_scaler.joblib')
                joblib.dump(self.feature_scaler, feature_scaler_path)
            # Save config
            config_path = filepath.replace('.h5', '_config.joblib')
            joblib.dump({
                'lookback': self.lookback,
                'use_features': self.use_features,
                'feature_columns': self.feature_columns
            }, config_path)
    
    def load(self, filepath: str):
        """Load a saved model"""
        self.model = keras.models.load_model(filepath, compile=False)
        scaler_path = filepath.replace('.h5', '_scaler.joblib')
        self.scaler = joblib.load(scaler_path)
        config_path = filepath.replace('.h5', '_config.joblib')
        config_data = joblib.load(config_path)
        self.lookback = config_data['lookback']
        self.use_features = config_data.get('use_features', False)
        self.feature_columns = config_data.get('feature_columns', [])
        if self.use_features:
            feature_scaler_path = filepath.replace('.h5', '_feature_scaler.joblib')
            self.feature_scaler = joblib.load(feature_scaler_path)
        self.is_fitted = True

