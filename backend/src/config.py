"""
Feelow Backend — Central Configuration
No Streamlit dependency. Used by FastAPI backend.
"""

from typing import List, Dict

# =============================================================================
# Model Configuration — Multi-Model Ensemble
# =============================================================================
MODELS: Dict[str, Dict] = {
    "ProsusAI/finbert": {
        "name": "FinBERT (ProsusAI)",
        "description": "Pre-trained on Financial PhraseBank, fine-tuned for financial sentiment",
        "label_map": {"positive": 1, "neutral": 0, "negative": -1},
        "color": "#00cc96",
    },
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis": {
        "name": "DistilRoBERTa Financial",
        "description": "DistilRoBERTa fine-tuned on financial news sentiment",
        "label_map": {"positive": 1, "neutral": 0, "negative": -1},
        "color": "#636efa",
    },
    "Sigma/financial-sentiment-analysis": {
        "name": "Sigma Financial SA",
        "description": "BERT fine-tuned on Financial PhraseBank",
        "label_map": {"positive": 1, "neutral": 0, "negative": -1},
        "color": "#ffa15a",
    },
}

DEFAULT_MODEL_ID: str = "ProsusAI/finbert"

SENTIMENT_MAP: Dict[str, int] = {
    "positive": 1, "Positive": 1,
    "neutral": 0, "Neutral": 0,
    "negative": -1, "Negative": -1,
}

SIGNAL_THRESHOLDS: Dict[str, float] = {
    "strong_buy": 0.40,
    "buy": 0.10,
    "sell": -0.10,
    "strong_sell": -0.40,
}

# =============================================================================
# Default Asset Universe
# =============================================================================
DEFAULT_TICKERS: List[str] = [
    "NVDA", "TSLA", "AAPL", "AMZN", "MSFT", "GOOGL", "META",
    "AMD", "NFLX", "COIN", "JPM", "GS", "BAC",
    "BTC-USD", "ETH-USD", "SOL-USD",
]

TICKER_CATEGORIES: Dict[str, List[str]] = {
    "Tech": ["NVDA", "TSLA", "AAPL", "AMZN", "MSFT", "GOOGL", "META", "AMD", "NFLX"],
    "Finance": ["JPM", "GS", "BAC", "COIN"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
}

# =============================================================================
# Data Sources
# =============================================================================
RSS_URL_PATTERN: str = "https://finance.yahoo.com/rss/headline?s={ticker}"

# =============================================================================
# Technical Indicators Config
# =============================================================================
SMA_PERIODS: List[int] = [7, 21, 50]
RSI_PERIOD: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9

# =============================================================================
# Chart Configuration
# =============================================================================
CHART_THEME: str = "plotly_dark"
POSITIVE_COLOR: str = "#00cc96"
NEGATIVE_COLOR: str = "#ef553b"
NEUTRAL_COLOR: str = "#ffa15a"
ACCENT_COLOR: str = "#636efa"
BG_COLOR: str = "#0e1117"
CARD_BG: str = "#1e2130"

# =============================================================================
# Cache TTL (seconds)
# =============================================================================
NEWS_CACHE_TTL: int = 300
MARKET_DATA_CACHE_TTL: int = 120

# =============================================================================
# Gemini Configuration
# =============================================================================
GEMINI_MODEL: str = "gemini-2.0-flash"
GEMINI_VISION_MODEL: str = "gemini-2.0-flash"

# =============================================================================
# Claude Agentic Configuration
# =============================================================================
CLAUDE_MODEL: str = "claude-sonnet-4-20250514"
CLAUDE_MAX_AGENT_TURNS: int = 5

# =============================================================================
# Agent Orchestrator Configuration
# =============================================================================
AGENT_STEPS: List[str] = [
    "data_collection",
    "sentiment_analysis",
    "technical_analysis",
    "gemini_visual_analysis",
    "gemini_search_grounding",
    "claude_deep_reasoning",
    "synthesis",
]

# =============================================================================
# Application Settings
# =============================================================================
PAGE_TITLE: str = "Feelow Backend API"
PAGE_ICON: str = "🌊"
DEFAULT_PERIOD: str = "1mo"
DEFAULT_INTERVAL: str = "1d"
