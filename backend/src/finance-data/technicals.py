"""
Feelow Backend — Technical Indicators Module
SMA, EMA, RSI, MACD, Bollinger Bands
Pure computation — no external dependencies beyond pandas/numpy.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SMA_PERIODS, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL


class TechnicalIndicators:
    """Compute technical indicators on OHLCV data."""

    @staticmethod
    def add_sma(df: pd.DataFrame, column: str = "close", periods: list = None) -> pd.DataFrame:
        periods = periods or SMA_PERIODS
        for p in periods:
            df[f"SMA_{p}"] = df[column].rolling(window=p).mean()
        return df

    @staticmethod
    def add_ema(df: pd.DataFrame, column: str = "close", span: int = 21) -> pd.DataFrame:
        df[f"EMA_{span}"] = df[column].ewm(span=span, adjust=False).mean()
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, column: str = "close", period: int = RSI_PERIOD) -> pd.DataFrame:
        delta = df[column].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_macd(df: pd.DataFrame, column: str = "close") -> pd.DataFrame:
        ema_fast = df[column].ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = df[column].ewm(span=MACD_SLOW, adjust=False).mean()
        df["MACD"] = ema_fast - ema_slow
        df["MACD_signal"] = df["MACD"].ewm(span=MACD_SIGNAL, adjust=False).mean()
        df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
        return df

    @staticmethod
    def add_bollinger(df: pd.DataFrame, column: str = "close", period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        sma = df[column].rolling(window=period).mean()
        std = df[column].rolling(window=period).std()
        df["BB_upper"] = sma + std_dev * std
        df["BB_middle"] = sma
        df["BB_lower"] = sma - std_dev * std
        return df

    @staticmethod
    def add_volume_sma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        if "volume" in df.columns:
            df["Vol_SMA"] = df["volume"].rolling(window=period).mean()
        return df

    @staticmethod
    def add_daily_return(df: pd.DataFrame, column: str = "close") -> pd.DataFrame:
        df["daily_return"] = df[column].pct_change() * 100
        return df

    @classmethod
    def add_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Add all indicators at once."""
        df = cls.add_sma(df)
        df = cls.add_ema(df)
        df = cls.add_rsi(df)
        df = cls.add_macd(df)
        df = cls.add_bollinger(df)
        df = cls.add_volume_sma(df)
        df = cls.add_daily_return(df)
        return df
