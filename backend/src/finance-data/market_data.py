"""
Feelow Backend — Market Data Provider (yfinance)
No Streamlit dependency — uses cachetools.TTLCache.
"""

from __future__ import annotations
import os, sys, logging
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf
from cachetools import TTLCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MARKET_DATA_CACHE_TTL, DEFAULT_PERIOD, DEFAULT_INTERVAL

logger = logging.getLogger(__name__)

# Module-level cache: key=(ticker, period, interval) -> DataFrame
_market_cache: TTLCache = TTLCache(maxsize=64, ttl=MARKET_DATA_CACHE_TTL)


class MarketDataLoader:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self._yf: Optional[yf.Ticker] = None

    @property
    def yf_ticker(self) -> yf.Ticker:
        if self._yf is None:
            self._yf = yf.Ticker(self.ticker)
        return self._yf

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index()
        for col_name in ("Date", "Datetime", "index"):
            if col_name in df.columns:
                df = df.rename(columns={col_name: "timestamp"})
                break
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        col_map = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep]

    def get_price_history(self, period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
        cache_key = (self.ticker, period, interval)
        if cache_key in _market_cache:
            return _market_cache[cache_key]

        try:
            df = yf.Ticker(self.ticker).history(period=period, interval=interval)
            if df.empty:
                return pd.DataFrame()
            result = self._normalise(df)
            _market_cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"Price error {self.ticker}: {e}")
            return pd.DataFrame()

    def get_current_price(self) -> float:
        try:
            info = self.yf_ticker.fast_info
            if hasattr(info, "last_price") and info.last_price:
                return float(info.last_price)
            data = self.yf_ticker.history(period="1d")
            if not data.empty:
                return float(data["Close"].iloc[-1])
        except Exception:
            pass
        return 0.0

    def get_price_change(self, days: int = 1) -> Tuple[float, float]:
        try:
            data = self.yf_ticker.history(period=f"{days + 1}d")
            if len(data) < 2:
                return 0.0, 0.0
            cur, prev = data["Close"].iloc[-1], data["Close"].iloc[0]
            return float(cur - prev), float((cur - prev) / prev * 100) if prev else (0.0, 0.0)
        except Exception:
            return 0.0, 0.0

    def get_company_info(self) -> dict:
        try:
            info = self.yf_ticker.info
            return {
                "name": info.get("shortName", self.ticker),
                "sector": info.get("sector", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "52w_high": info.get("fiftyTwoWeekHigh", 0),
                "52w_low": info.get("fiftyTwoWeekLow", 0),
            }
        except Exception:
            return {"name": self.ticker, "sector": "N/A"}
