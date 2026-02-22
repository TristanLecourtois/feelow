"""
Feelow — Unified Backend API
==============================

FastAPI backend serving:
  • Finance-data routes  (sentiment, price, technicals, agents)
  • Polymarket routes    (agent-search + scoring pipeline)
  • UI-fr routes         (GET endpoints for the Next.js dashboard)

Run:
    cd backend/src
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

UI-fr routes (used by Next.js frontend — GET):
    GET  /api/kpis?ticker=X            → price, sentiment, RSI, signal
    GET  /api/news?ticker=X            → news headlines + sentiment
    GET  /api/price-history?ticker=X   → OHLCV + technicals

Streamlit routes (used by Streamlit frontend — POST):
    GET  /api/health             → backend health
    GET  /api/config             → config constants
    POST /api/data/load          → price + news + sentiment + technicals
    POST /api/sentiment/compare  → compare 3 models on one text
    POST /api/sentiment/ensemble → multi-model ensemble on headlines
    POST /api/analysis/claude    → single-shot Claude analysis
    POST /api/pipeline/run       → full agentic pipeline

Polymarket routes:
    GET  /                       → polymarket health
    POST /get_polymarket         → full polymarket pipeline
"""

from __future__ import annotations

import os, sys, math, logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from stock_analysis.api_finbert_transformer import compute_sentiment_score, df

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ─── Path setup ──────────────────────────────────────────────────────────────
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_FINANCE_DATA_DIR = os.path.join(_SRC_DIR, "finance-data")

# Add both dirs to sys.path so modules can import config + each other
for _d in (_SRC_DIR, _FINANCE_DATA_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

# ─── Finance-data module imports ─────────────────────────────────────────────
from config import (
    MODELS, DEFAULT_MODEL_ID, SIGNAL_THRESHOLDS, DEFAULT_TICKERS,
    TICKER_CATEGORIES, CLAUDE_MODEL, GEMINI_MODEL,
    DEFAULT_PERIOD, DEFAULT_INTERVAL,
)
from sentiment_engine import MultiModelSentimentEngine
from news_ingestor import NewsIngestor
from market_data import MarketDataLoader
from technicals import TechnicalIndicators
from claude_analyst import ClaudeAnalyst
from gemini_agent import GeminiAgent
from agent_orchestrator import AgentOrchestrator

# ─── Polymarket import (optional — graceful fallback) ────────────────────────
try:
    from full_pipeline import get_polymarket
    _POLYMARKET_AVAILABLE = True
except Exception:
    _POLYMARKET_AVAILABLE = False
    logger.warning("Polymarket pipeline not available (missing deps?)")


# ─── Singleton engine ────────────────────────────────────────────────────────
_engine: Optional[MultiModelSentimentEngine] = None


def get_engine() -> MultiModelSentimentEngine:
    global _engine
    if _engine is None:
        _engine = MultiModelSentimentEngine()
    return _engine


# ─── Lifespan ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Starting Feelow backend — pre-loading sentiment engine…")
    get_engine()
    logger.info(f"Engine ready on {get_engine().device_name}")
    yield
    logger.info("Shutting down Feelow backend")


# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Feelow Unified API",
    description="Finance-data + Polymarket backend for Feelow",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _df_to_records(df: pd.DataFrame) -> list:
    """Convert DataFrame → JSON-safe list of dicts."""
    if df is None or df.empty:
        return []
    records = []
    for rec in df.to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            elif isinstance(v, pd.Timestamp):
                clean[k] = v.isoformat()
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                fv = float(v)
                clean[k] = None if math.isnan(fv) else fv
            elif isinstance(v, np.bool_):
                clean[k] = bool(v)
            else:
                clean[k] = v
        records.append(clean)
    return records


def _calc_signal(avg: float) -> str:
    if avg > SIGNAL_THRESHOLDS["strong_buy"]:
        return "STRONG BUY"
    elif avg > SIGNAL_THRESHOLDS["buy"]:
        return "BUY"
    elif avg < SIGNAL_THRESHOLDS["strong_sell"]:
        return "STRONG SELL"
    elif avg < SIGNAL_THRESHOLDS["sell"]:
        return "SELL"
    return "NEUTRAL"


# ═════════════════════════════════════════════════════════════════════════════
# Request / Response Models — Finance Data
# ═════════════════════════════════════════════════════════════════════════════
class DataLoadRequest(BaseModel):
    ticker: str
    period: str = DEFAULT_PERIOD
    interval: str = DEFAULT_INTERVAL
    model_id: str = DEFAULT_MODEL_ID


class CompareRequest(BaseModel):
    text: str


class EnsembleRequest(BaseModel):
    headlines: List[str]


class ClaudeAnalysisRequest(BaseModel):
    ticker: str
    sentiment_summary: str = ""
    price_summary: str = ""
    headlines: str = ""
    technical_summary: str = ""
    claude_key: str = ""


class PipelineRequest(BaseModel):
    ticker: str
    period: str = DEFAULT_PERIOD
    interval: str = DEFAULT_INTERVAL
    model_id: str = DEFAULT_MODEL_ID
    gemini_key: str = ""
    claude_key: str = ""


# ═════════════════════════════════════════════════════════════════════════════
# Request / Response Models — Polymarket
# ═════════════════════════════════════════════════════════════════════════════
class PolymarketRequest(BaseModel):
    company: str = Field(..., description="Company name to search for")
    date: Optional[str] = Field(None)
    max_queries: int = Field(1, ge=1, le=3)
    limit_per_query: int = Field(10, ge=1, le=50)
    top_k: int = Field(5, ge=1, le=20)


class PolymarketResponse(BaseModel):
    raw_markets: List[Dict[str, Any]] = []
    top_markets_summary: List[Dict[str, Any]] = []
    corr_top2: float = 0.0
    global_score: float = 0.0
    claude_block: str = ""


# ═════════════════════════════════════════════════════════════════════════════
# FINANCE-DATA ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def api_health():
    engine = get_engine()
    return {"status": "ok", "device": engine.device_name}


@app.get("/api/config")
async def api_config():
    return {
        "models": {mid: {"name": m["name"]} for mid, m in MODELS.items()},
        "default_model": DEFAULT_MODEL_ID,
        "tickers": DEFAULT_TICKERS,
        "ticker_categories": TICKER_CATEGORIES,
        "signal_thresholds": SIGNAL_THRESHOLDS,
    }


@app.post("/api/data/load")
async def api_data_load(req: DataLoadRequest):
    """Load price, news, sentiment, and technicals for a ticker."""
    try:
        engine = get_engine()

        # Price data
        loader = MarketDataLoader(req.ticker)
        price_df = loader.get_price_history(period=req.period, interval=req.interval)
        current_price = loader.get_current_price()
        abs_change, pct_change = loader.get_price_change(7)

        # News
        news_df = NewsIngestor(req.ticker).fetch_news()

        # Sentiment on headlines
        if not news_df.empty and "title" in news_df.columns:
            sent_results = engine.analyze_headlines(news_df["title"].tolist(), req.model_id)
            news_df = pd.concat(
                [news_df.reset_index(drop=True), sent_results.reset_index(drop=True)],
                axis=1,
            )

        # Technicals
        if not price_df.empty:
            price_df = TechnicalIndicators.add_all(price_df)

        # Metrics
        volume_24h, avg_sentiment, signal = 0, 0.0, "NEUTRAL"
        if not news_df.empty and "sentiment_numeric" in news_df.columns:
            cutoff = datetime.now() - timedelta(hours=24)
            recent = news_df[news_df["published"] > cutoff] if "published" in news_df.columns else news_df
            volume_24h = len(recent)
            if volume_24h > 0:
                avg_sentiment = float(recent["sentiment_numeric"].mean())
                signal = _calc_signal(avg_sentiment)

        return {
            "price_data": _df_to_records(price_df),
            "news_data": _df_to_records(news_df),
            "current_price": current_price,
            "abs_change": abs_change,
            "pct_change": pct_change,
            "metrics": {
                "volume_24h": volume_24h,
                "avg_sentiment": round(avg_sentiment, 4),
                "signal": signal,
            },
        }
    except Exception as e:
        logger.error(f"/api/data/load error: {e}", exc_info=True)
        return {
            "price_data": [], "news_data": [],
            "current_price": 0, "abs_change": 0, "pct_change": 0,
            "metrics": {"volume_24h": 0, "avg_sentiment": 0, "signal": "NEUTRAL"},
            "error": str(e),
        }


@app.post("/api/sentiment/compare")
async def api_sentiment_compare(req: CompareRequest):
    """Compare all sentiment models on a single text."""
    try:
        engine = get_engine()
        result_df = engine.compare_models(req.text)
        return {"results": _df_to_records(result_df)}
    except Exception as e:
        logger.error(f"/api/sentiment/compare error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sentiment/ensemble")
async def api_sentiment_ensemble(req: EnsembleRequest):
    """Run multi-model ensemble on headlines."""
    try:
        engine = get_engine()
        result_df = engine.analyze_ensemble(req.headlines)
        return {"results": _df_to_records(result_df)}
    except Exception as e:
        logger.error(f"/api/sentiment/ensemble error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class SentimentScoreRequest(BaseModel):
    company: str = Field(..., description="Company name")

class SentimentScoreResponse(BaseModel):
    company: str
    score: float
    news_count: int

@app.post("/api/sentiment/score", response_model=SentimentScoreResponse)
async def api_sentiment_score(req: SentimentScoreRequest):
    try:
        score = compute_sentiment_score(req.company)
        news_count = len(df[df['selftext'].str.contains(req.company, case=False, na=False)])
        return SentimentScoreResponse(
            company=req.company,
            score=round(score, 4),
            news_count=news_count
        )
    except Exception as e:
        logger.error(f"/api/sentiment/score error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/api/analysis/claude")
async def api_analysis_claude(req: ClaudeAnalysisRequest):
    """Run single-shot Claude analysis."""
    try:
        analyst = ClaudeAnalyst(api_key=req.claude_key or None)
        if not analyst.available:
            return {"analysis": "Claude API key not configured.", "available": False}

        analysis = analyst.generate_analysis(
            ticker=req.ticker,
            sentiment_summary=req.sentiment_summary,
            price_data_summary=req.price_summary,
            news_headlines=req.headlines,
            technical_summary=req.technical_summary,
        )
        return {"analysis": analysis, "available": True}
    except Exception as e:
        logger.error(f"/api/analysis/claude error: {e}", exc_info=True)
        return {"analysis": f"Error: {e}", "available": False}


@app.post("/api/pipeline/run")
async def api_pipeline_run(req: PipelineRequest):
    """Run the full multi-agent agentic pipeline."""
    try:
        engine = get_engine()

        # Load data (same as /api/data/load)
        loader = MarketDataLoader(req.ticker)
        price_df = loader.get_price_history(period=req.period, interval=req.interval)
        current_price = loader.get_current_price()
        _, pct_change = loader.get_price_change(7)

        news_df = NewsIngestor(req.ticker).fetch_news()
        if not news_df.empty and "title" in news_df.columns:
            sent_results = engine.analyze_headlines(news_df["title"].tolist(), req.model_id)
            news_df = pd.concat(
                [news_df.reset_index(drop=True), sent_results.reset_index(drop=True)],
                axis=1,
            )
        if not price_df.empty:
            price_df = TechnicalIndicators.add_all(price_df)

        # Build chart image for Gemini visual analysis
        chart_image_bytes = None
        try:
            from visualizer_helper import generate_chart_image
            chart_image_bytes = generate_chart_image(price_df, req.ticker)
        except Exception:
            pass  # chart image is optional

        # Init agents
        gemini_agent = GeminiAgent(api_key=req.gemini_key or os.getenv("GEMINI_API_KEY", ""))
        claude_analyst = ClaudeAnalyst(api_key=req.claude_key or None)

        orchestrator = AgentOrchestrator(
            sentiment_engine=engine,
            gemini_agent=gemini_agent if gemini_agent.available else None,
            claude_analyst=claude_analyst if claude_analyst.available else None,
        )

        state = orchestrator.run_pipeline(
            ticker=req.ticker,
            price_df=price_df,
            news_df=news_df,
            current_price=current_price,
            pct_change=pct_change,
            chart_image_bytes=chart_image_bytes,
        )

        return state.to_dict()

    except Exception as e:
        logger.error(f"/api/pipeline/run error: {e}", exc_info=True)
        return {
            "error": str(e),
            "steps": [],
            "final_report": "",
            "total_duration_ms": 0,
        }


# ═════════════════════════════════════════════════════════════════════════════
# UI-FR ENDPOINTS (GET — used by the Next.js frontend)
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/kpis")
async def api_kpis(
    ticker: str = Query("AAPL", description="Ticker symbol"),
    period: str = Query("1mo", description="Price history period"),
):
    """
    Dashboard KPIs for UI-fr section-cards:
    price, pct_change, news_volume, avg_sentiment, signal, rsi.
    """
    try:
        engine = get_engine()
        interval_map = {"7d": "1h", "1mo": "1d", "3mo": "1d", "6mo": "1d", "1y": "1wk"}
        interval = interval_map.get(period, "1d")

        loader = MarketDataLoader(ticker)
        current_price = loader.get_current_price()
        _, pct_change = loader.get_price_change(7)

        price_df = loader.get_price_history(period=period, interval=interval)
        rsi_val = 0.0
        if not price_df.empty:
            price_df = TechnicalIndicators.add_all(price_df)
            if "RSI" in price_df.columns:
                rsi_val = float(price_df["RSI"].iloc[-1])
                if math.isnan(rsi_val):
                    rsi_val = 0.0

        news_df = NewsIngestor(ticker).fetch_news()
        news_volume = 0
        avg_sentiment = 0.0
        signal = "NEUTRAL"

        if not news_df.empty and "title" in news_df.columns:
            headlines = news_df["title"].dropna().tolist()
            if headlines:
                sent_df = engine.analyze_headlines(headlines)
                if not sent_df.empty and "sentiment_numeric" in sent_df.columns:
                    news_df = news_df.reset_index(drop=True)
                    news_df["sentiment_numeric"] = sent_df["sentiment_numeric"].values[:len(news_df)]
                    cutoff = datetime.now() - timedelta(hours=24)
                    recent = news_df[news_df["published"] > cutoff] if "published" in news_df.columns else news_df
                    news_volume = len(recent)
                    if news_volume > 0:
                        avg_sentiment = float(recent["sentiment_numeric"].mean())
                        signal = _calc_signal(avg_sentiment)

        return {
            "ticker": ticker,
            "price": round(current_price, 2),
            "pct_change": round(pct_change, 2),
            "news_volume": news_volume,
            "avg_sentiment": round(avg_sentiment, 3),
            "signal": signal,
            "rsi": round(rsi_val, 1),
        }
    except Exception as e:
        logger.error(f"/api/kpis error: {e}", exc_info=True)
        return {
            "ticker": ticker, "price": 0, "pct_change": 0,
            "news_volume": 0, "avg_sentiment": 0, "signal": "NEUTRAL", "rsi": 0,
        }


@app.get("/api/news")
async def api_news(ticker: str = Query("AAPL")):
    """Raw news headlines with FinBERT sentiment for the UI-fr data-table."""
    try:
        news_df = NewsIngestor(ticker).fetch_news()
        if news_df.empty:
            return []

        if "title" in news_df.columns:
            headlines = news_df["title"].dropna().tolist()
            if headlines:
                engine = get_engine()
                sent_df = engine.analyze_headlines(headlines)
                if not sent_df.empty:
                    news_df = news_df.reset_index(drop=True)
                    if "label" in sent_df.columns:
                        news_df["sentiment"] = sent_df["label"].values[:len(news_df)]
                    if "score" in sent_df.columns:
                        news_df["confidence"] = sent_df["score"].values[:len(news_df)]
                    if "sentiment_numeric" in sent_df.columns:
                        news_df["sentiment_numeric"] = sent_df["sentiment_numeric"].values[:len(news_df)]

        records = _df_to_records(news_df.head(50))
        return records
    except Exception as e:
        logger.error(f"/api/news error: {e}", exc_info=True)
        return []


@app.get("/api/price-history")
async def api_price_history(
    ticker: str = Query("AAPL"),
    period: str = Query("1mo"),
):
    """OHLCV + technicals for the UI-fr chart-area-interactive."""
    try:
        interval_map = {"7d": "1h", "1mo": "1d", "3mo": "1d", "6mo": "1d", "1y": "1wk"}
        interval = interval_map.get(period, "1d")
        loader = MarketDataLoader(ticker)
        df = loader.get_price_history(period=period, interval=interval)
        if df.empty:
            return []
        df = TechnicalIndicators.add_all(df)
        return _df_to_records(df)
    except Exception as e:
        logger.error(f"/api/price-history error: {e}", exc_info=True)
        return []


# ═════════════════════════════════════════════════════════════════════════════
# POLYMARKET ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Polymarket health check."""
    return {"status": "ok", "service": "Feelow Unified API"}


@app.post("/get_polymarket", response_model=PolymarketResponse)
async def get_polymarket_endpoint(request: PolymarketRequest):
    """Run the full Polymarket pipeline for a given company."""
    if not _POLYMARKET_AVAILABLE:
        raise HTTPException(status_code=503, detail="Polymarket pipeline not available")
    try:
        result = get_polymarket(
            company=request.company,
            date=request.date,
            max_queries=request.max_queries,
            limit_per_query=request.limit_per_query,
            top_k=request.top_k,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
