"""
Feelow Backend — Finance-data package.
All modules for financial data, sentiment, agents.

Modules live directly in this folder (not in a sub-folder).
config.py lives one level up at backend/src/config.py.
"""

import os, sys

# Add both this dir and parent (for config.py) to sys.path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)

for _d in (_this_dir, _parent_dir):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from sentiment_engine import MultiModelSentimentEngine
from news_ingestor import NewsIngestor
from market_data import MarketDataLoader
from technicals import TechnicalIndicators
from claude_analyst import ClaudeAnalyst, AgentToolHandler
from gemini_agent import GeminiAgent
from agent_orchestrator import AgentOrchestrator

__all__ = [
    "MultiModelSentimentEngine",
    "NewsIngestor",
    "MarketDataLoader",
    "TechnicalIndicators",
    "ClaudeAnalyst",
    "AgentToolHandler",
    "GeminiAgent",
    "AgentOrchestrator",
]
