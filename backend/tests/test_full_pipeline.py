"""
Unit tests for full_pipeline.py
================================

Tests the format bridge, pertinence normalisation, get_polymarket with
mocked PolymarketPipeline (no real Gemini/Polymarket calls), and edge cases.

Run:
    cd feelow/backend
    python -m pytest tests/test_full_pipeline.py -v
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# ─── Ensure src/ is importable ───────────────────────────────────────────────
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
SRC_DIR = os.path.abspath(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from full_pipeline import _convert_price_history, get_polymarket, log as pipeline_log

# Turn on DEBUG for the pipeline logger during tests
pipeline_log.setLevel(logging.DEBUG)

# ─── Fake market data (mimics what PolymarketPipeline.run() returns) ─────────

FAKE_RAW_MARKET_1 = {
    "id": "abc123",
    "question": "Will NVIDIA hit $200 by March 2026?",
    "event_title": "NVIDIA stock price",
    "active": True,
    "closed": False,
    "end_date": "2026-03-01T00:00:00Z",
    "volume": 80000,
    "liquidity": 15000,
    "outcomes": ["Yes", "No"],
    "outcome_prices": [0.65, 0.35],
    "url": "https://polymarket.com/event/nvidia-200",
    "pertinence_score": 85,  # 0–100 scale from agent-search
    "price_history": [
        {"timestamp": 1771536000, "price": 0.50},
        {"timestamp": 1771622400, "price": 0.55},
        {"timestamp": 1771708800, "price": 0.58},
        {"timestamp": 1771795200, "price": 0.60},
        {"timestamp": 1771881600, "price": 0.62},
        {"timestamp": 1771968000, "price": 0.65},
    ],
}

FAKE_RAW_MARKET_2 = {
    "id": "def456",
    "question": "Will NVIDIA be the largest AI chip maker in 2026?",
    "event_title": "AI chip market share",
    "active": True,
    "closed": False,
    "end_date": "2026-12-31T00:00:00Z",
    "volume": 42000,
    "liquidity": 9000,
    "outcomes": ["Yes", "No"],
    "outcome_prices": [0.80, 0.20],
    "url": "https://polymarket.com/event/nvidia-ai-chip",
    "pertinence_score": 60,
    "price_history": [
        {"timestamp": 1771536000, "price": 0.70},
        {"timestamp": 1771622400, "price": 0.72},
        {"timestamp": 1771708800, "price": 0.75},
        {"timestamp": 1771795200, "price": 0.78},
        {"timestamp": 1771881600, "price": 0.80},
    ],
}


# ═════════════════════════════════════════════════════════════════════════════
#  1. _convert_price_history
# ═════════════════════════════════════════════════════════════════════════════

class TestConvertPriceHistory:
    """Tests for the format bridge between agent-search and score_polymarket."""

    def test_converts_timestamp_price_to_t_p(self):
        """Standard case: {timestamp, price} → {t, p}."""
        market = {
            "question": "Test",
            "price_history": [
                {"timestamp": 100, "price": 0.5},
                {"timestamp": 200, "price": 0.6},
            ],
        }
        result = _convert_price_history(market)
        logging.info("Converted history: %s", result["history"])

        assert "history" in result
        assert len(result["history"]) == 2
        assert result["history"][0] == {"t": 100, "p": 0.5}
        assert result["history"][1] == {"t": 200, "p": 0.6}

    def test_already_t_p_format(self):
        """If input already uses {t, p}, it should still work."""
        market = {
            "question": "Test",
            "price_history": [
                {"t": 100, "p": 0.5},
                {"t": 200, "p": 0.6},
            ],
        }
        result = _convert_price_history(market)
        logging.info("Already t/p history: %s", result["history"])

        assert len(result["history"]) == 2
        assert result["history"][0] == {"t": 100, "p": 0.5}

    def test_empty_price_history(self):
        """Empty price_history → empty history list."""
        market = {"question": "Test", "price_history": []}
        result = _convert_price_history(market)
        logging.info("Empty history result: %s", result.get("history"))

        assert result["history"] == []

    def test_missing_price_history_key(self):
        """No price_history key at all → empty history."""
        market = {"question": "Test"}
        result = _convert_price_history(market)
        logging.info("Missing key result: history=%s", result.get("history"))

        assert result["history"] == []

    def test_malformed_entries_skipped(self):
        """Entries missing timestamp or price are silently skipped."""
        market = {
            "question": "Test",
            "price_history": [
                {"timestamp": 100, "price": 0.5},
                {"timestamp": None, "price": 0.6},  # bad
                {"foo": "bar"},  # bad
                {"timestamp": 300, "price": 0.7},
            ],
        }
        result = _convert_price_history(market)
        logging.info("Filtered history (%d entries): %s",
                      len(result["history"]), result["history"])

        assert len(result["history"]) == 2
        assert result["history"][0]["t"] == 100
        assert result["history"][1]["t"] == 300

    def test_does_not_mutate_original(self):
        """The original dict should not be modified."""
        market = {
            "question": "Test",
            "price_history": [{"timestamp": 1, "price": 0.5}],
        }
        result = _convert_price_history(market)
        assert "history" not in market  # original untouched
        assert "history" in result

    def test_non_list_price_history_ignored(self):
        """If price_history is not a list (e.g. a dict), history stays empty."""
        market = {"question": "Test", "price_history": {"bad": "format"}}
        result = _convert_price_history(market)
        logging.info("Non-list price_history result: %s", result.get("history"))

        # price_history is not a list → the if-branch is skipped, no 'history' key added
        # but the dict copy still has the original price_history
        assert result.get("history") is None or result.get("history") == []


# ═════════════════════════════════════════════════════════════════════════════
#  2. get_polymarket — with mocked PolymarketPipeline
# ═════════════════════════════════════════════════════════════════════════════

class TestGetPolymarket:
    """Tests for the main get_polymarket() entry point."""

    def test_raises_without_api_key(self):
        """Should raise ValueError if no API key is provided or in env."""
        # Clear env vars to be safe
        env = {k: v for k, v in os.environ.items()
               if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="No Gemini API key"):
                get_polymarket("NVIDIA")

    @patch("full_pipeline.PolymarketPipeline")
    def test_empty_results(self, MockPipeline):
        """If the pipeline returns no markets, get_polymarket returns empty."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = []
        MockPipeline.return_value = mock_instance

        result = get_polymarket("FakeCompany", gemini_api_key="test-key-123")

        logging.info("Empty result: %s", result)
        assert result["raw_markets"] == []
        assert result["top_markets_summary"] == []
        assert result["corr_top2"] == 0.0
        assert result["global_score"] == 0.0
        assert "No Polymarket markets found" in result["claude_block"]

    @patch("full_pipeline.PolymarketPipeline")
    def test_single_market(self, MockPipeline):
        """Single market returned → scored, no correlation possible."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = [FAKE_RAW_MARKET_1]
        MockPipeline.return_value = mock_instance

        result = get_polymarket("NVIDIA", gemini_api_key="test-key-123", top_k=1)

        logging.info("Single market — raw_markets: %d", len(result["raw_markets"]))
        logging.info("Single market — top_markets_summary: %d", len(result["top_markets_summary"]))
        logging.info("Single market — global_score: %.4f", result["global_score"])

        assert len(result["raw_markets"]) == 1
        assert len(result["top_markets_summary"]) == 1
        assert result["global_score"] > 0
        assert "Polymarket" in result["claude_block"]

        # Check the summary structure
        summary = result["top_markets_summary"][0]
        assert "question" in summary
        assert "score" in summary
        assert "engagement" in summary
        assert "metrics" in summary
        assert "advanced" in summary

    @patch("full_pipeline.PolymarketPipeline")
    def test_two_markets_full_flow(self, MockPipeline):
        """Two markets → scoring + correlation + claude_block."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = [FAKE_RAW_MARKET_1, FAKE_RAW_MARKET_2]
        MockPipeline.return_value = mock_instance

        result = get_polymarket(
            "NVIDIA",
            date="February 2026",
            gemini_api_key="test-key-123",
            top_k=2,
        )

        logging.info("Two markets — raw: %d, summaries: %d",
                      len(result["raw_markets"]), len(result["top_markets_summary"]))
        logging.info("Two markets — corr_top2: %.4f", result["corr_top2"])
        logging.info("Two markets — global_score: %.4f", result["global_score"])
        logging.info("Two markets — claude_block (first 200): %s",
                      result["claude_block"][:200])

        assert len(result["raw_markets"]) == 2
        assert len(result["top_markets_summary"]) == 2
        assert isinstance(result["corr_top2"], float)
        assert -1.0 <= result["corr_top2"] <= 1.0
        assert result["global_score"] > 0
        assert "Market #1" in result["claude_block"]
        assert "Market #2" in result["claude_block"]

    @patch("full_pipeline.PolymarketPipeline")
    def test_pertinence_normalised_from_100_to_01(self, MockPipeline):
        """Pertinence 85 (0–100 scale) → normalised to 0.85 for scoring."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = [FAKE_RAW_MARKET_1]
        MockPipeline.return_value = mock_instance

        result = get_polymarket("NVIDIA", gemini_api_key="test-key-123", top_k=1)

        # The pertinence metric in the summary should be ≈ 0.85
        summary = result["top_markets_summary"][0]
        pert = summary["metrics"]["pertinence"]
        logging.info("Normalised pertinence metric: %.4f (expected ~0.85)", pert)
        assert 0.80 <= pert <= 0.90, f"Expected ~0.85, got {pert}"

    @patch("full_pipeline.PolymarketPipeline")
    def test_pertinence_already_01_not_double_divided(self, MockPipeline):
        """Pertinence already in [0,1] should NOT be divided again."""
        market = dict(FAKE_RAW_MARKET_1)
        market["pertinence_score"] = 0.85  # already 0–1 scale

        mock_instance = MagicMock()
        mock_instance.run.return_value = [market]
        MockPipeline.return_value = mock_instance

        result = get_polymarket("NVIDIA", gemini_api_key="test-key-123", top_k=1)
        pert = result["top_markets_summary"][0]["metrics"]["pertinence"]
        logging.info("Already-normalised pertinence: %.4f (expected ~0.85)", pert)
        assert 0.80 <= pert <= 0.90, f"Expected ~0.85, got {pert}"

    @patch("full_pipeline.PolymarketPipeline")
    def test_pipeline_constructor_args(self, MockPipeline):
        """Verify PolymarketPipeline is called with correct args."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = []
        MockPipeline.return_value = mock_instance

        get_polymarket(
            "Tesla",
            date="March 2026",
            gemini_api_key="my-key",
            max_queries=2,
            limit_per_query=15,
        )

        MockPipeline.assert_called_once_with(
            api_key="my-key",
            max_queries=2,
            limit_per_query=15,
        )
        mock_instance.run.assert_called_once_with("Tesla", "March 2026")

    @patch("full_pipeline.PolymarketPipeline")
    def test_top_k_limits_summaries(self, MockPipeline):
        """top_k=1 with 2 markets → only 1 summary returned."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = [FAKE_RAW_MARKET_1, FAKE_RAW_MARKET_2]
        MockPipeline.return_value = mock_instance

        result = get_polymarket("NVIDIA", gemini_api_key="test-key-123", top_k=1)
        logging.info("top_k=1 → summaries count: %d", len(result["top_markets_summary"]))

        assert len(result["raw_markets"]) == 2  # all raw markets returned
        assert len(result["top_markets_summary"]) == 1  # only top 1

    @patch("full_pipeline.PolymarketPipeline")
    def test_output_keys_complete(self, MockPipeline):
        """Result dict always has all 5 expected keys."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = [FAKE_RAW_MARKET_1]
        MockPipeline.return_value = mock_instance

        result = get_polymarket("NVIDIA", gemini_api_key="test-key-123")
        expected_keys = {"raw_markets", "top_markets_summary", "corr_top2",
                         "global_score", "claude_block"}
        logging.info("Result keys: %s", set(result.keys()))
        assert set(result.keys()) == expected_keys

    @patch("full_pipeline.PolymarketPipeline")
    def test_advanced_metrics_present(self, MockPipeline):
        """Each summary should have advanced metrics computed."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = [FAKE_RAW_MARKET_1]
        MockPipeline.return_value = mock_instance

        result = get_polymarket("NVIDIA", gemini_api_key="test-key-123", top_k=1)
        adv = result["top_markets_summary"][0]["advanced"]

        expected_adv_keys = {
            "history_points", "history_span_hours", "p_first", "p_last",
            "net_change", "total_variation", "max_jump", "vol_dp",
            "change_count", "staleness_ratio_0_5", "time_since_last_reprice_sec",
            "slope_per_day", "slope_recent_per_day", "entropy_nats",
            "time_to_event_days", "history_quality", "composite_signal",
        }
        logging.info("Advanced keys: %s", set(adv.keys()))
        assert expected_adv_keys.issubset(set(adv.keys())), \
            f"Missing advanced keys: {expected_adv_keys - set(adv.keys())}"

        # history was converted correctly → should have data points
        assert adv["history_points"] == 6
        logging.info("Advanced metrics: history_points=%d, p_first=%.4f, p_last=%.4f",
                      adv["history_points"], adv["p_first"], adv["p_last"])
