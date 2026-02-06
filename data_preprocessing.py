"""
Data preprocessing and feature engineering module
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
from typing import Dict, List, Tuple
import config


class DataPreprocessor:
    """Handles data loading, cleaning, and feature engineering"""
    
    def __init__(self):
        self.us_holidays = holidays.UnitedStates()
        self.feature_columns = []
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from Excel file"""
        df = pd.read_excel(file_path)
        return df
    
    def handle_missing_dates(self, df: pd.DataFrame, state: str = None) -> pd.DataFrame:
        """
        Fill missing dates in the time series
        """
        # Filter by state if provided
        if state:
            df = df[df[config.STATE_COLUMN] == state].copy()
        
        # Ensure Date column is datetime
        df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN])
        
        # Sort by date
        df = df.sort_values(config.DATE_COLUMN).reset_index(drop=True)
        
        # Create complete date range
        min_date = df[config.DATE_COLUMN].min()
        max_date = df[config.DATE_COLUMN].max()
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        # Create complete dataframe with all dates
        complete_df = pd.DataFrame({config.DATE_COLUMN: date_range})
        
        # Merge with original data
        df = complete_df.merge(df, on=config.DATE_COLUMN, how='left')
        
        # Forward fill missing values for state and category
        if config.STATE_COLUMN in df.columns:
            df[config.STATE_COLUMN] = df[config.STATE_COLUMN].ffill()
        if config.CATEGORY_COLUMN in df.columns:
            df[config.CATEGORY_COLUMN] = df[config.CATEGORY_COLUMN].ffill()
        
        # Interpolate missing Total values
        # Use a DatetimeIndex for time-based interpolation to avoid pandas errors
        df = df.set_index(config.DATE_COLUMN)
        df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].interpolate(method="time")
        df = df.reset_index()
        
        # Fill any remaining NaN values with 0
        df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].fillna(0)
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()
        
        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df[config.DATE_COLUMN].dt.dayofweek
        
        # Month
        df['month'] = df[config.DATE_COLUMN].dt.month
        
        # Day of month
        df['day_of_month'] = df[config.DATE_COLUMN].dt.day
        
        # Week of year
        df['week_of_year'] = df[config.DATE_COLUMN].dt.isocalendar().week
        
        # Year
        df['year'] = df[config.DATE_COLUMN].dt.year
        
        # Is weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Is holiday
        df['is_holiday'] = df[config.DATE_COLUMN].apply(
            lambda x: 1 if x in self.us_holidays else 0
        )
        
        # Cyclical encoding for day of week and month
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = config.TARGET_COLUMN) -> pd.DataFrame:
        """Create lag features"""
        df = df.copy()
        
        for lag in config.LAG_FEATURES:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = config.TARGET_COLUMN) -> pd.DataFrame:
        """Create rolling window features"""
        df = df.copy()
        
        for window in config.ROLLING_WINDOWS:
            df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std().fillna(0)
            df[f'rolling_min_{window}'] = df[target_col].rolling(window=window, min_periods=1).min()
            df[f'rolling_max_{window}'] = df[target_col].rolling(window=window, min_periods=1).max()
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all engineered features"""
        df = self.create_time_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        
        # Store feature column names (excluding target and date)
        exclude_cols = [config.DATE_COLUMN, config.TARGET_COLUMN, config.STATE_COLUMN, config.CATEGORY_COLUMN]
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]

        # Ensure all feature columns are numeric (avoid object dtypes)
        for col in self.feature_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df
    
    def split_time_series(self, df: pd.DataFrame, test_size: float = config.TEST_SIZE) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split time series data ensuring no data leakage
        Uses the last portion of data as test set
        """
        df = df.sort_values(config.DATE_COLUMN).reset_index(drop=True)
        
        split_idx = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        return train_df, test_df
    
    def prepare_data_for_state(self, df: pd.DataFrame, state: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for a specific state: handle missing dates, create features, split
        """
        # Filter by state
        state_df = df[df[config.STATE_COLUMN] == state].copy()
        
        # Handle missing dates
        state_df = self.handle_missing_dates(state_df, state=None)  # Already filtered
        
        # Create all features
        state_df = self.create_all_features(state_df)
        
        # Split into train and test
        train_df, test_df = self.split_time_series(state_df)
        
        return train_df, test_df
    
    def prepare_data_for_ml(self, df: pd.DataFrame, include_target: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for machine learning models (XGBoost, LSTM)
        Returns X (features) and y (target) arrays
        """
        # Select feature columns
        feature_cols = self.feature_columns if self.feature_columns else [
            col for col in df.columns 
            if col not in [config.DATE_COLUMN, config.TARGET_COLUMN, config.STATE_COLUMN, config.CATEGORY_COLUMN]
        ]

        # Work with a DataFrame to control dtypes, then convert to float
        X_df = df[feature_cols].copy()
        X_df = X_df.apply(pd.to_numeric, errors="coerce")
        X = X_df.to_numpy(dtype=float)
        
        if include_target:
            y = df[config.TARGET_COLUMN].values
            return X, y
        else:
            return X, None

