"""
Feelow — Polymarket Backend API
================================

FastAPI backend exposing the full Polymarket analysis pipeline.

Run:
    cd feelow/backend/src
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /                → health check
    POST /get_polymarket  → full analysis pipeline
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from full_pipeline import get_polymarket

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Feelow Polymarket API",
    description=(
        "End-to-end Polymarket analysis: agent-search (Gemini LLM) → "
        "advanced scoring → Claude-ready summary."
    ),
    version="1.0.0",
)


# ─── Request / Response models ───────────────────────────────────────────────

class PolymarketRequest(BaseModel):
    company: str = Field(
        ..., description="Company name to search for (e.g. 'NVIDIA')"
    )
    date: Optional[str] = Field(
        None, description="Optional date context (e.g. 'February 2026')"
    )
    max_queries: int = Field(
        1, ge=1, le=3, description="Number of varied search queries (1–3)"
    )
    limit_per_query: int = Field(
        10, ge=1, le=50, description="Max markets returned per search query"
    )
    top_k: int = Field(
        5, ge=1, le=20, description="Number of top markets in the summary"
    )


class PolymarketResponse(BaseModel):
    raw_markets: List[Dict[str, Any]] = Field(
        description="Raw market dicts returned by agent-search"
    )
    top_markets_summary: List[Dict[str, Any]] = Field(
        description="Scored and analysed market summaries"
    )
    corr_top2: float = Field(
        description="Pearson correlation between the top 2 markets"
    )
    global_score: float = Field(
        description="Weighted global score across all markets"
    )
    claude_block: str = Field(
        description="Text block ready to inject into a Claude prompt"
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "service": "Feelow Polymarket API"}


@app.post("/get_polymarket", response_model=PolymarketResponse)
async def get_polymarket_endpoint(request: PolymarketRequest):
    """
    Run the full Polymarket pipeline for a given company.

    1. **Agent-search**: Gemini LLM searches Polymarket and scores pertinence.
    2. **Scoring**: Advanced metrics, ranking, correlation, Claude-ready block.
    """
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


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
