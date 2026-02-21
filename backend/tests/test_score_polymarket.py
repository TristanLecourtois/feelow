"""
Unit tests for score_polymarket.py
===================================

Tests the helper functions, Market class, and process_polymarket_markets pipeline.

Run:
    cd feelow/backend
    python -m pytest tests/test_score_polymarket.py -v
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import sys

import numpy as np
import pytest

# ─── Import from hyphenated directory ─────────────────────────────────────────
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
SRC_DIR = os.path.abspath(SRC_DIR)

_score_path = os.path.join(SRC_DIR, "polymarket-analysis", "score_polymarket.py")
spec = importlib.util.spec_from_file_location("score_polymarket", _score_path)
score_mod = importlib.util.module_from_spec(spec)
sys.modules["score_polymarket"] = score_mod
spec.loader.exec_module(score_mod)

# Pull out everything we need
_clip01 = score_mod._clip01
_history_arrays = score_mod._history_arrays
_ols_slope_per_day = score_mod._ols_slope_per_day
_count_changes = score_mod._count_changes
_staleness_ratio = score_mod._staleness_ratio
_time_since_last_change = score_mod._time_since_last_change
_parse_iso_utc = score_mod._parse_iso_utc
_binary_entropy = score_mod._binary_entropy
Market = score_mod.Market
market_correlation = score_mod.market_correlation
process_polymarket_markets = score_mod.process_polymarket_markets

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s │ %(message)s")
log = logging.getLogger("test_score")


# ═════════════════════════════════════════════════════════════════════════════
#  SAMPLE DATA
# ═════════════════════════════════════════════════════════════════════════════

MARKET_DATA_RICH = {
    "question": "Will NVIDIA hit $200 by March?",
    "event_title": "NVIDIA stock price",
    "active": True,
    "closed": False,
    "end_date": "2026-06-01T00:00:00Z",
    "volume": 80000,
    "liquidity": 15000,
    "outcomes": ["Yes", "No"],
    "outcome_prices": [0.65, 0.35],
    "url": "https://polymarket.com/event/nvidia-200",
    "pertinence_score": 0.85,
    "history": [
        {"t": 1771536000, "p": 0.50},
        {"t": 1771622400, "p": 0.52},
        {"t": 1771708800, "p": 0.55},
        {"t": 1771795200, "p": 0.58},
        {"t": 1771881600, "p": 0.60},
        {"t": 1771968000, "p": 0.62},
        {"t": 1772054400, "p": 0.63},
        {"t": 1772140800, "p": 0.65},
    ],
}

MARKET_DATA_MINIMAL = {
    "question": "Will Bitcoin hit 150k?",
    "outcomes": ["Yes", "No"],
    "outcome_prices": [0.3, 0.7],
}

MARKET_DATA_NO_HISTORY = {
    "question": "Will Apple release AR glasses in 2026?",
    "event_title": "Apple AR",
    "volume": 5000,
    "liquidity": 1000,
    "outcomes": ["Yes", "No"],
    "outcome_prices": [0.45, 0.55],
    "pertinence_score": 0.5,
}

MARKET_DATA_STALE = {
    "question": "Stale market stuck at 0.5",
    "event_title": "Stale test",
    "volume": 100,
    "liquidity": 50,
    "outcomes": ["Yes", "No"],
    "outcome_prices": [0.5, 0.5],
    "pertinence_score": 0.3,
    "history": [
        {"t": 1000, "p": 0.5},
        {"t": 2000, "p": 0.5},
        {"t": 3000, "p": 0.5},
        {"t": 4000, "p": 0.5},
        {"t": 5000, "p": 0.5},
    ],
}


# ═════════════════════════════════════════════════════════════════════════════
#  1. HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

class TestClip01:
    def test_within_range(self):
        assert _clip01(0.5) == 0.5
        log.info("_clip01(0.5) = %.12f", _clip01(0.5))

    def test_zero_clipped(self):
        result = _clip01(0.0)
        assert result > 0
        log.info("_clip01(0.0) = %.12e (> 0)", result)

    def test_one_clipped(self):
        result = _clip01(1.0)
        assert result < 1.0
        log.info("_clip01(1.0) = %.12f (< 1)", result)

    def test_negative(self):
        result = _clip01(-0.5)
        assert result > 0
        log.info("_clip01(-0.5) = %.12e", result)

    def test_above_one(self):
        result = _clip01(1.5)
        assert result < 1.0
        log.info("_clip01(1.5) = %.12f", result)


class TestHistoryArrays:
    def test_normal(self):
        history = [{"t": 3, "p": 0.6}, {"t": 1, "p": 0.5}, {"t": 2, "p": 0.55}]
        t, p = _history_arrays(history)
        log.info("Sorted t: %s, p: %s", t, p)
        assert list(t) == [1, 2, 3]
        assert len(p) == 3
        # Should be sorted by time
        assert p[0] < p[2]

    def test_empty_list(self):
        t, p = _history_arrays([])
        assert len(t) == 0 and len(p) == 0
        log.info("Empty list → len(t)=%d, len(p)=%d", len(t), len(p))

    def test_not_a_list(self):
        t, p = _history_arrays(None)
        assert len(t) == 0
        log.info("None input → len(t)=%d", len(t))

    def test_malformed_rows_skipped(self):
        history = [
            {"t": 1, "p": 0.5},
            "bad_row",
            {"t": 2},  # missing p
            {"t": 3, "p": 0.7},
        ]
        t, p = _history_arrays(history)
        log.info("Filtered: t=%s, p=%s (from %d input rows)", t, p, len(history))
        assert len(t) == 2


class TestOlsSlopePerDay:
    def test_flat_line(self):
        t = np.array([0, 86400, 172800], dtype=float)
        p = np.array([0.5, 0.5, 0.5], dtype=float)
        slope = _ols_slope_per_day(t, p)
        log.info("Flat slope: %s", slope)
        assert slope is not None
        assert abs(slope) < 1e-10

    def test_upward_trend(self):
        t = np.array([0, 86400, 172800], dtype=float)  # 0, 1 day, 2 days
        p = np.array([0.4, 0.5, 0.6], dtype=float)
        slope = _ols_slope_per_day(t, p)
        log.info("Upward slope: %.6f per day", slope)
        assert slope is not None
        assert slope > 0
        assert abs(slope - 0.1) < 1e-6  # should be ~0.1/day

    def test_too_few_points(self):
        t = np.array([0, 86400], dtype=float)
        p = np.array([0.4, 0.5], dtype=float)
        assert _ols_slope_per_day(t, p) is None
        log.info("2 points → None (expected)")


class TestCountChanges:
    def test_all_same(self):
        p = np.array([0.5, 0.5, 0.5])
        assert _count_changes(p) == 0
        log.info("All same → %d changes", _count_changes(p))

    def test_all_different(self):
        p = np.array([0.1, 0.2, 0.3, 0.4])
        assert _count_changes(p) == 3
        log.info("All different → %d changes", _count_changes(p))

    def test_single_point(self):
        assert _count_changes(np.array([0.5])) == 0


class TestStalenessRatio:
    def test_all_at_05(self):
        p = np.array([0.5, 0.5, 0.5])
        ratio = _staleness_ratio(p)
        log.info("All at 0.5 → staleness=%.4f", ratio)
        assert ratio == 1.0

    def test_none_at_05(self):
        p = np.array([0.3, 0.7, 0.9])
        ratio = _staleness_ratio(p)
        log.info("None at 0.5 → staleness=%.4f", ratio)
        assert ratio == 0.0

    def test_empty(self):
        assert _staleness_ratio(np.array([])) is None


class TestTimeSinceLastChange:
    def test_no_large_jump(self):
        """If no jump >= threshold, returns total span."""
        t = np.array([0, 100, 200], dtype=float)
        p = np.array([0.5, 0.5, 0.5], dtype=float)
        result = _time_since_last_change(t, p, threshold=0.005)
        log.info("No jumps → time_since=%.1f (total span=200)", result)
        assert result == 200.0

    def test_recent_jump(self):
        t = np.array([0, 100, 200, 300], dtype=float)
        p = np.array([0.5, 0.5, 0.5, 0.6], dtype=float)
        result = _time_since_last_change(t, p, threshold=0.005)
        log.info("Jump at t=300 → time_since=%.1f", result)
        assert result == 0.0  # last change is at the last point

    def test_single_point(self):
        assert _time_since_last_change(np.array([0.0]), np.array([0.5])) is None


class TestParseIsoUtc:
    def test_valid_z(self):
        dt = _parse_iso_utc("2026-03-01T00:00:00Z")
        log.info("Parsed Z-date: %s", dt)
        assert dt is not None
        assert dt.year == 2026

    def test_valid_offset(self):
        dt = _parse_iso_utc("2026-03-01T00:00:00+00:00")
        assert dt is not None

    def test_invalid(self):
        assert _parse_iso_utc("not-a-date") is None

    def test_none(self):
        assert _parse_iso_utc(None) is None

    def test_empty_string(self):
        assert _parse_iso_utc("") is None


class TestBinaryEntropy:
    def test_fair_coin(self):
        h = _binary_entropy(0.5)
        log.info("Entropy at 0.5: %.6f (max=%.6f)", h, math.log(2))
        assert abs(h - math.log(2)) < 1e-6

    def test_certain_yes(self):
        h = _binary_entropy(0.999)
        log.info("Entropy at 0.999: %.6f (near 0)", h)
        assert h < 0.05

    def test_certain_no(self):
        h = _binary_entropy(0.001)
        assert h < 0.05


# ═════════════════════════════════════════════════════════════════════════════
#  2. MARKET CLASS
# ═════════════════════════════════════════════════════════════════════════════

class TestMarketInit:
    def test_full_init(self):
        m = Market(MARKET_DATA_RICH)
        log.info("Market: question=%r, volume=%.0f, history_len=%d",
                 m.question, m.volume, len(m.history))
        assert m.question == "Will NVIDIA hit $200 by March?"
        assert m.volume == 80000
        assert len(m.history) == 8

    def test_minimal_init(self):
        m = Market(MARKET_DATA_MINIMAL)
        log.info("Minimal market: question=%r, volume=%.0f, history=%s",
                 m.question, m.volume, m.history)
        assert m.question == "Will Bitcoin hit 150k?"
        assert m.volume == 0.0
        assert m.history == []


class TestMarketMetrics:
    def test_compute_metrics_keys(self):
        m = Market(MARKET_DATA_RICH)
        metrics = m.compute_metrics()
        expected = {"momentum", "volatility", "concentration", "volume", "liquidity", "pertinence"}
        log.info("Metrics keys: %s", set(metrics.keys()))
        assert set(metrics.keys()) == expected

    def test_metrics_in_01(self):
        m = Market(MARKET_DATA_RICH)
        metrics = m.compute_metrics()
        for k, v in metrics.items():
            log.info("  %s = %.4f", k, v)
            assert 0.0 <= v <= 1.0, f"{k} = {v} out of [0,1]"

    def test_momentum_with_history(self):
        m = Market(MARKET_DATA_RICH)
        mom = m.compute_momentum()
        log.info("Momentum (with history): %.4f", mom)
        assert 0.0 <= mom <= 1.0
        # Price went from 0.5 to 0.65 → momentum ≈ 0.15
        assert mom > 0.1

    def test_momentum_no_history(self):
        m = Market(MARKET_DATA_NO_HISTORY)
        mom = m.compute_momentum()
        log.info("Momentum (no history, prices=[0.45, 0.55]): %.4f", mom)
        assert 0.0 <= mom <= 1.0

    def test_volatility_with_history(self):
        m = Market(MARKET_DATA_RICH)
        vol = m.compute_volatility()
        log.info("Volatility: %.4f (min→max spread)", vol)
        assert vol > 0.0

    def test_volatility_no_history(self):
        m = Market(MARKET_DATA_NO_HISTORY)
        vol = m.compute_volatility()
        log.info("Volatility (no history): %.4f", vol)
        assert vol >= 0.0

    def test_concentration(self):
        m = Market(MARKET_DATA_RICH)
        conc = m.compute_concentration()
        log.info("Concentration (0.65/0.35): %.4f", conc)
        assert 0.0 <= conc <= 1.0
        # 0.65/0.35 is moderately concentrated
        assert conc > 0.05


class TestMarketAdvanced:
    def test_advanced_keys(self):
        m = Market(MARKET_DATA_RICH)
        adv = m.compute_advanced_metrics()
        expected = {
            "history_points", "history_span_hours", "p_first", "p_last",
            "net_change", "total_variation", "max_jump", "vol_dp",
            "change_count", "staleness_ratio_0_5", "time_since_last_reprice_sec",
            "slope_per_day", "slope_recent_per_day", "entropy_nats",
            "time_to_event_days", "history_quality", "composite_signal",
        }
        log.info("Advanced keys: %s", set(adv.keys()))
        assert expected == set(adv.keys())

    def test_history_points_count(self):
        m = Market(MARKET_DATA_RICH)
        adv = m.compute_advanced_metrics()
        log.info("history_points=%d (expected 8)", adv["history_points"])
        assert adv["history_points"] == 8

    def test_net_change_positive(self):
        m = Market(MARKET_DATA_RICH)
        adv = m.compute_advanced_metrics()
        log.info("net_change=%.6f (0.65 - 0.50 ≈ 0.15)", adv["net_change"])
        assert adv["net_change"] is not None
        assert adv["net_change"] > 0

    def test_slope_positive(self):
        m = Market(MARKET_DATA_RICH)
        adv = m.compute_advanced_metrics()
        log.info("slope_per_day=%.6f", adv["slope_per_day"])
        assert adv["slope_per_day"] > 0

    def test_stale_market_high_staleness(self):
        m = Market(MARKET_DATA_STALE)
        adv = m.compute_advanced_metrics()
        log.info("Stale market staleness_ratio=%.4f", adv["staleness_ratio_0_5"])
        assert adv["staleness_ratio_0_5"] is not None
        assert adv["staleness_ratio_0_5"] > 0.9

    def test_history_quality_range(self):
        m = Market(MARKET_DATA_RICH)
        adv = m.compute_advanced_metrics()
        hq = adv["history_quality"]
        log.info("history_quality=%.4f (expected in [0,1])", hq)
        assert hq is not None
        assert 0.0 <= hq <= 1.0


class TestMarketScore:
    def test_score_in_01(self):
        m = Market(MARKET_DATA_RICH)
        m.compute_metrics()
        score = m.compute_score()
        log.info("Score: %.4f", score)
        assert 0.0 <= score <= 1.0

    def test_engagement_in_01(self):
        m = Market(MARKET_DATA_RICH)
        m.compute_metrics()
        eng = m.compute_engagement()
        log.info("Engagement: %.4f", eng)
        assert 0.0 <= eng <= 1.0


class TestMarketSummaryText:
    def test_summary_text_contains_question(self):
        m = Market(MARKET_DATA_RICH)
        m.compute_metrics()
        m.compute_advanced_metrics()
        m.compute_score()
        m.compute_engagement()
        text = m.polymarket_summary_text()
        log.info("Summary text (first 300 chars): %s", text[:300])
        assert "NVIDIA" in text
        assert "YES probability" in text

    def test_summary_dict_keys(self):
        m = Market(MARKET_DATA_RICH)
        m.compute_metrics()
        m.compute_advanced_metrics()
        m.compute_score()
        m.compute_engagement()
        s = m.summary()
        log.info("Summary dict keys: %s", set(s.keys()))
        assert "question" in s
        assert "score" in s
        assert "engagement" in s
        assert "metrics" in s
        assert "advanced" in s


# ═════════════════════════════════════════════════════════════════════════════
#  3. CORRELATION
# ═════════════════════════════════════════════════════════════════════════════

class TestMarketCorrelation:
    def test_identical_markets(self):
        m1 = Market(MARKET_DATA_RICH)
        m2 = Market(MARKET_DATA_RICH)
        corr = market_correlation(m1, m2)
        log.info("Identical markets correlation: %.6f", corr)
        assert abs(corr - 1.0) < 1e-6

    def test_different_markets(self):
        m1 = Market(MARKET_DATA_RICH)
        m2 = Market(MARKET_DATA_STALE)
        corr = market_correlation(m1, m2)
        log.info("Rich vs Stale correlation: %s", corr)
        # NaN is expected when one series has zero variance (stale @ 0.5)
        assert (np.isnan(corr)) or (-1.0 <= corr <= 1.0)

    def test_no_history_fallback(self):
        m1 = Market(MARKET_DATA_NO_HISTORY)
        m2 = Market(MARKET_DATA_MINIMAL)
        corr = market_correlation(m1, m2)
        log.info("No-history fallback correlation: %.6f", corr)
        assert -1.0 <= corr <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
#  4. process_polymarket_markets PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

class TestProcessPolymarketMarkets:
    def test_basic_pipeline(self):
        result = process_polymarket_markets(
            [MARKET_DATA_RICH, MARKET_DATA_NO_HISTORY], top_k=2
        )
        log.info("Pipeline result keys: %s", set(result.keys()))
        assert "markets_sorted" in result
        assert "top_markets_summary" in result
        assert "corr_top2" in result
        assert "global_score" in result
        assert "claude_block" in result

    def test_pipeline_ranking(self):
        result = process_polymarket_markets(
            [MARKET_DATA_RICH, MARKET_DATA_STALE, MARKET_DATA_NO_HISTORY], top_k=3
        )
        summaries = result["top_markets_summary"]
        log.info("Ranked summaries:")
        for i, s in enumerate(summaries):
            log.info("  #%d: %s (score=%.4f, eng=%.4f)",
                     i + 1, s["question"][:40], s["score"], s["engagement"])

        # Rich market should rank higher than stale market
        scores = [s["score"] * s["engagement"] for s in summaries]
        assert scores[0] >= scores[-1], "First should be best"

    def test_pipeline_top_k(self):
        result = process_polymarket_markets(
            [MARKET_DATA_RICH, MARKET_DATA_STALE, MARKET_DATA_NO_HISTORY], top_k=1
        )
        assert len(result["top_markets_summary"]) == 1
        log.info("top_k=1 → %d summary", len(result["top_markets_summary"]))

    def test_pipeline_global_score_positive(self):
        result = process_polymarket_markets([MARKET_DATA_RICH], top_k=1)
        log.info("Global score (1 market): %.6f", result["global_score"])
        assert result["global_score"] > 0

    def test_pipeline_claude_block_content(self):
        result = process_polymarket_markets(
            [MARKET_DATA_RICH, MARKET_DATA_NO_HISTORY], top_k=2
        )
        block = result["claude_block"]
        log.info("Claude block (first 300 chars): %s", block[:300])
        assert "Polymarket" in block
        assert "Market #1" in block

    def test_single_market_pipeline(self):
        """Pipeline with a single market should not crash."""
        result = process_polymarket_markets([MARKET_DATA_RICH], top_k=1)
        assert len(result["top_markets_summary"]) == 1
        assert result["corr_top2"] == 0.0  # can't correlate 1 market
        log.info("Single market pipeline: score=%.4f", result["global_score"])
