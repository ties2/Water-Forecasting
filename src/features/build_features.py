import pandas as pd
import numpy as np
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class FeatureEngineer:
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df['is_night'] = ((df.index.hour >= 22) | (df.index.hour <= 6)).astype(int)
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        return df

    def create_lag_features(self, df: pd.DataFrame, col: str, lags=[1, 24, 168]) -> pd.DataFrame:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        return df

    def create_rolling_features(self, df: pd.DataFrame, col: str, windows=[24, 168]) -> pd.DataFrame:
        for w in windows:
            df[f'{col}_roll_mean_{w}'] = df[col].rolling(w).mean()
        return df

    def engineer_all(self, df: pd.DataFrame, target_col: str = 'flow_rate_m3h') -> pd.DataFrame:
        df = self.create_temporal_features(df)
        df = self.create_lag_features(df, target_col)
        df = self.create_rolling_features(df, target_col)
        df = df.dropna()
        logger.info(f"Feature engineering complete: {df.shape}")
        return df