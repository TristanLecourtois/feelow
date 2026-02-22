"""
Feelow Frontend — UI Configuration
Colors, chart themes, categories, and display settings.
"""

from typing import List, Dict

# =============================================================================
# Backend Connection
# =============================================================================
BACKEND_URL: str = "http://localhost:8000"

# =============================================================================
# Page Settings
# =============================================================================
PAGE_TITLE: str = "Feelow — AI Sentiment × Price Intelligence"
PAGE_ICON: str = "🌊"

# =============================================================================
# Model Display Info (mirrors backend, used for sidebar selector)
# =============================================================================
MODELS: Dict[str, Dict] = {
    "ProsusAI/finbert": {"name": "FinBERT (ProsusAI)"},
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis": {"name": "DistilRoBERTa Financial"},
    "Sigma/financial-sentiment-analysis": {"name": "Sigma Financial SA"},
}

# =============================================================================
# Ticker Categories (for sidebar dropdown)
# =============================================================================
TICKER_CATEGORIES: Dict[str, List[str]] = {
    "🖥️ Tech": ["NVDA", "TSLA", "AAPL", "AMZN", "MSFT", "GOOGL", "META", "AMD", "NFLX"],
    "🏦 Finance": ["JPM", "GS", "BAC", "COIN"],
    "🪙 Crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
}

# =============================================================================
# Signal Thresholds (for display logic)
# =============================================================================
SIGNAL_THRESHOLDS: Dict[str, float] = {
    "strong_buy": 0.40,
    "buy": 0.10,
    "sell": -0.10,
    "strong_sell": -0.40,
}

# =============================================================================
# Chart / UI Colors
# =============================================================================
CHART_THEME: str = "plotly_dark"
POSITIVE_COLOR: str = "#00cc96"
NEGATIVE_COLOR: str = "#ef553b"
NEUTRAL_COLOR: str = "#ffa15a"
ACCENT_COLOR: str = "#636efa"
BG_COLOR: str = "#0e1117"
CARD_BG: str = "#1e2130"
