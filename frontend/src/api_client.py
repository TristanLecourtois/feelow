"""
Feelow Frontend — API Client
HTTP client for communicating with the Feelow backend.
"""

from __future__ import annotations
import logging
from typing import Optional, Dict, Any, List

import requests

logger = logging.getLogger(__name__)


class FeelowAPI:
    """HTTP client for the Feelow FastAPI backend."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    # -----------------------------------------------------------------
    # Health & Config
    # -----------------------------------------------------------------
    def health(self) -> dict:
        try:
            r = self._session.get(f"{self.base_url}/api/health", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"status": "error", "device": str(e)}

    def get_config(self) -> dict:
        try:
            r = self._session.get(f"{self.base_url}/api/config", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Config fetch error: {e}")
            return {}

    # -----------------------------------------------------------------
    # Data Loading
    # -----------------------------------------------------------------
    def load_data(
        self,
        ticker: str,
        period: str = "1mo",
        interval: str = "1d",
        model_id: str = "ProsusAI/finbert",
    ) -> dict:
        """
        Load all base data for a ticker from the backend.
        Returns: price_data, news_data, current_price, pct_change, metrics, etc.
        """
        try:
            r = self._session.post(
                f"{self.base_url}/api/data/load",
                json={
                    "ticker": ticker,
                    "period": period,
                    "interval": interval,
                    "model_id": model_id,
                },
                timeout=60,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Data load error: {e}")
            return {
                "price_data": [],
                "news_data": [],
                "current_price": 0,
                "abs_change": 0,
                "pct_change": 0,
                "metrics": {"volume_24h": 0, "avg_sentiment": 0, "signal": "NEUTRAL"},
                "error": str(e),
            }

    # -----------------------------------------------------------------
    # Sentiment
    # -----------------------------------------------------------------
    def compare_models(self, text: str) -> list:
        """Compare all sentiment models on a single text."""
        try:
            r = self._session.post(
                f"{self.base_url}/api/sentiment/compare",
                json={"text": text},
                timeout=30,
            )
            r.raise_for_status()
            return r.json().get("results", [])
        except Exception as e:
            logger.error(f"Compare models error: {e}")
            return []

    def run_ensemble(self, headlines: List[str]) -> list:
        """Run multi-model ensemble on headlines."""
        try:
            r = self._session.post(
                f"{self.base_url}/api/sentiment/ensemble",
                json={"headlines": headlines},
                timeout=120,
            )
            r.raise_for_status()
            return r.json().get("results", [])
        except Exception as e:
            logger.error(f"Ensemble error: {e}")
            return []

    # -----------------------------------------------------------------
    # AI Analysis
    # -----------------------------------------------------------------
    def run_claude_analysis(
        self,
        ticker: str,
        sentiment_summary: str,
        price_summary: str,
        headlines: str,
        technical_summary: str = "",
        claude_key: str = "",
    ) -> dict:
        """Run single-shot Claude analysis."""
        try:
            r = self._session.post(
                f"{self.base_url}/api/analysis/claude",
                json={
                    "ticker": ticker,
                    "sentiment_summary": sentiment_summary,
                    "price_summary": price_summary,
                    "headlines": headlines,
                    "technical_summary": technical_summary,
                    "claude_key": claude_key,
                },
                timeout=60,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Claude analysis error: {e}")
            return {"analysis": f"Error: {e}", "available": False}

    # -----------------------------------------------------------------
    # Pipeline
    # -----------------------------------------------------------------
    def run_pipeline(
        self,
        ticker: str,
        period: str = "1mo",
        interval: str = "1d",
        model_id: str = "ProsusAI/finbert",
        gemini_key: str = "",
        claude_key: str = "",
    ) -> dict:
        """
        Run the full agentic pipeline (may take 30-60s).
        Returns execution trace + final report.
        """
        try:
            r = self._session.post(
                f"{self.base_url}/api/pipeline/run",
                json={
                    "ticker": ticker,
                    "period": period,
                    "interval": interval,
                    "model_id": model_id,
                    "gemini_key": gemini_key,
                    "claude_key": claude_key,
                },
                timeout=180,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return {
                "error": str(e),
                "steps": [],
                "final_report": "",
                "total_duration_ms": 0,
            }
