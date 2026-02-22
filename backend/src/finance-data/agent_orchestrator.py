"""
Feelow Backend — Agent Orchestrator
Coordinates the multi-agent pipeline with JSON-serializable output.
"""

from __future__ import annotations
import io, time, logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """Represents one step in the agentic pipeline."""
    name: str
    agent: str
    status: str = "pending"
    result: Any = None
    duration_ms: float = 0
    detail: str = ""

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        return {
            "name": self.name,
            "agent": self.agent,
            "status": self.status,
            "result": self.result if isinstance(self.result, (str, dict, list, type(None))) else str(self.result),
            "duration_ms": self.duration_ms,
            "detail": self.detail,
        }


@dataclass
class OrchestratorState:
    """Full state of the agentic pipeline execution."""
    ticker: str
    steps: List[AgentStep] = field(default_factory=list)
    final_report: str = ""
    total_duration_ms: float = 0
    gemini_available: bool = False
    claude_available: bool = False

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        return {
            "ticker": self.ticker,
            "steps": [s.to_dict() for s in self.steps],
            "final_report": self.final_report,
            "total_duration_ms": self.total_duration_ms,
            "gemini_available": self.gemini_available,
            "claude_available": self.claude_available,
        }


class AgentOrchestrator:
    """Orchestrates the multi-agent financial analysis pipeline."""

    def __init__(self, sentiment_engine=None, gemini_agent=None, claude_analyst=None):
        self.engine = sentiment_engine
        self.gemini = gemini_agent
        self.claude = claude_analyst

    def run_pipeline(
        self,
        ticker: str,
        price_df: pd.DataFrame,
        news_df: pd.DataFrame,
        current_price: float,
        pct_change: float,
        chart_image_bytes: Optional[bytes] = None,
        on_step: Optional[Callable[[AgentStep], None]] = None,
    ) -> OrchestratorState:
        start_time = time.time()

        state = OrchestratorState(
            ticker=ticker,
            gemini_available=self.gemini is not None and self.gemini.available,
            claude_available=self.claude is not None and self.claude.available,
        )

        # ---- Step 1: Data Assembly ----
        step1 = AgentStep(name="Data Assembly", agent="System")
        state.steps.append(step1)
        step1.status = "running"
        if on_step:
            on_step(step1)

        t0 = time.time()
        cached_data = self._assemble_data(ticker, price_df, news_df, current_price, pct_change)
        step1.duration_ms = (time.time() - t0) * 1000
        step1.status = "completed"
        step1.detail = f"Assembled {len(cached_data)} data fields"
        step1.result = {k: type(v).__name__ for k, v in cached_data.items()}
        if on_step:
            on_step(step1)

        # ---- Step 2: Multi-Model Sentiment Ensemble ----
        step2 = AgentStep(name="Multi-Model Sentiment Ensemble", agent="FinBERT x3")
        state.steps.append(step2)
        step2.status = "running"
        if on_step:
            on_step(step2)

        t0 = time.time()
        ensemble_result = self._run_ensemble(news_df)
        cached_data["ensemble_result"] = ensemble_result
        step2.duration_ms = (time.time() - t0) * 1000
        step2.status = "completed"
        step2.detail = f"Ensemble on {len(news_df)} headlines" if not news_df.empty else "No headlines"
        step2.result = ensemble_result
        if on_step:
            on_step(step2)

        # ---- Step 3: Gemini Visual Chart Analysis ----
        step3 = AgentStep(name="Visual Chart Analysis", agent="Gemini 2.0 Flash")
        state.steps.append(step3)
        gemini_visual_text = None

        if self.gemini and self.gemini.available and chart_image_bytes:
            step3.status = "running"
            if on_step:
                on_step(step3)

            def step3_retry(msg):
                step3.detail = msg
                if on_step:
                    on_step(step3)

            t0 = time.time()
            visual_result = self.gemini.analyze_chart_image(
                chart_image_bytes, ticker,
                context=f"Price: ${current_price:.2f} | 7d change: {pct_change:+.2f}%",
                on_retry=step3_retry,
            )
            step3.duration_ms = (time.time() - t0) * 1000

            if visual_result.get("success"):
                step3.status = "completed"
                gemini_visual_text = visual_result["analysis"]
                cached_data["gemini_visual"] = gemini_visual_text
                step3.detail = f"Visual patterns identified (model: {visual_result.get('model', '?')})"
            else:
                step3.status = "failed"
                step3.detail = visual_result.get("analysis", "Unknown error")[:150]
        else:
            step3.status = "skipped" if not chart_image_bytes else "unavailable"
            step3.detail = "Gemini API not configured" if not (self.gemini and self.gemini.available) else "No chart image available"

        step3.result = gemini_visual_text
        if on_step:
            on_step(step3)

        # ---- Step 4: Gemini Search Grounding ----
        step4 = AgentStep(name="Real-Time Search Grounding", agent="Gemini + Google Search")
        state.steps.append(step4)
        gemini_search_text = None

        if self.gemini and self.gemini.available:
            step4.status = "running"
            if on_step:
                on_step(step4)

            def step4_retry(msg):
                step4.detail = msg
                if on_step:
                    on_step(step4)

            t0 = time.time()
            search_result = self.gemini.search_grounded_analysis(
                ticker,
                query_context=f"Current price ${current_price:.2f}, sentiment signal: {cached_data.get('sentiment_summary', {}).get('signal', 'N/A')}",
                on_retry=step4_retry,
            )
            step4.duration_ms = (time.time() - t0) * 1000

            if search_result.get("success"):
                step4.status = "completed"
                gemini_search_text = search_result["analysis"]
                cached_data["gemini_search"] = gemini_search_text
                step4.detail = f"Live market context retrieved (model: {search_result.get('model', '?')})"
            else:
                step4.status = "failed"
                step4.detail = search_result.get("analysis", "Unknown error")[:150]
        else:
            step4.status = "unavailable"
            step4.detail = "Gemini API not configured"

        step4.result = gemini_search_text
        if on_step:
            on_step(step4)

        # ---- Step 5: Deep Reasoning ----
        use_claude = self.claude and self.claude.available
        reasoning_agent_name = "Claude Sonnet 4" if use_claude else "Gemini 2.0 Flash (reasoning)"
        step5 = AgentStep(name="Agentic Deep Reasoning", agent=reasoning_agent_name)
        state.steps.append(step5)

        if use_claude:
            step5.status = "running"
            if on_step:
                on_step(step5)

            t0 = time.time()

            from claude_analyst import AgentToolHandler
            tool_handler = AgentToolHandler(
                sentiment_engine=self.engine,
                market_loader=None,
                ticker=ticker,
                cached_data=cached_data,
            )

            def claude_step_callback(msg):
                step5.detail = msg
                if on_step:
                    on_step(step5)

            agent_result = self.claude.run_agentic_analysis(
                ticker=ticker,
                tool_handler=tool_handler,
                gemini_visual=gemini_visual_text,
                gemini_search=gemini_search_text,
                on_step=claude_step_callback,
            )

            step5.duration_ms = (time.time() - t0) * 1000
            step5.status = "completed"
            step5.detail = f"{agent_result['turns_used']} turns, {len(agent_result['tool_calls_made'])} tool calls"
            step5.result = agent_result

        elif self.gemini and self.gemini.available:
            step5.status = "running"
            step5.detail = "Claude unavailable — Gemini taking over deep reasoning"
            if on_step:
                on_step(step5)

            t0 = time.time()

            def gemini_step_callback(msg):
                step5.detail = msg
                if on_step:
                    on_step(step5)

            agent_result = self.gemini.run_reasoning_analysis(
                ticker=ticker,
                cached_data=cached_data,
                gemini_visual=gemini_visual_text,
                gemini_search=gemini_search_text,
                chart_image_bytes=chart_image_bytes,
                on_step=gemini_step_callback,
            )

            step5.duration_ms = (time.time() - t0) * 1000

            if agent_result.get("success", False) and agent_result.get("turns_used", 0) > 0:
                step5.status = "completed"
                model_used = agent_result.get("model_used", "Gemini")
                step5.detail = f"Gemini reasoning ({model_used}): {agent_result['turns_used']} turn(s), {len(agent_result['tool_calls_made'])} tool calls"
            else:
                step5.status = "failed"
                step5.detail = agent_result.get("analysis", "Gemini reasoning failed")[:150]

            step5.result = agent_result
        else:
            step5.status = "unavailable"
            step5.detail = "No AI API configured"
            step5.result = {"analysis": "No AI API key configured."}

        if on_step:
            on_step(step5)

        # ---- Build Final Report ----
        state.total_duration_ms = (time.time() - start_time) * 1000
        state.final_report = self._build_final_report(state, cached_data)

        return state

    def _assemble_data(self, ticker, price_df, news_df, current_price, pct_change):
        data = {}

        if not news_df.empty and "sentiment_numeric" in news_df.columns:
            avg_sent = news_df["sentiment_numeric"].mean()
            counts = news_df["label"].value_counts().to_dict() if "label" in news_df.columns else {}
            data["sentiment_summary"] = {
                "average_score": round(avg_sent, 4),
                "signal": self._score_to_signal(avg_sent),
                "news_volume": len(news_df),
                "distribution": counts,
            }
        else:
            data["sentiment_summary"] = {"average_score": 0, "signal": "NEUTRAL", "news_volume": 0}

        data["price_summary"] = {
            "ticker": ticker,
            "current_price": current_price,
            "pct_change_7d": round(pct_change, 2),
        }

        if not price_df.empty:
            last = price_df.iloc[-1]
            data["technical_summary"] = {
                "RSI": round(float(last.get("RSI", 0)), 2) if pd.notna(last.get("RSI")) else None,
                "MACD": round(float(last.get("MACD", 0)), 4) if pd.notna(last.get("MACD")) else None,
                "MACD_signal": round(float(last.get("MACD_signal", 0)), 4) if pd.notna(last.get("MACD_signal")) else None,
                "SMA_7": round(float(last.get("SMA_7", 0)), 2) if pd.notna(last.get("SMA_7")) else None,
                "SMA_21": round(float(last.get("SMA_21", 0)), 2) if pd.notna(last.get("SMA_21")) else None,
                "SMA_50": round(float(last.get("SMA_50", 0)), 2) if pd.notna(last.get("SMA_50")) else None,
                "BB_upper": round(float(last.get("BB_upper", 0)), 2) if pd.notna(last.get("BB_upper")) else None,
                "BB_lower": round(float(last.get("BB_lower", 0)), 2) if pd.notna(last.get("BB_lower")) else None,
                "daily_return_pct": round(float(last.get("daily_return", 0)), 2) if pd.notna(last.get("daily_return")) else None,
            }
        else:
            data["technical_summary"] = {}

        if not news_df.empty:
            headlines = []
            for _, row in news_df.head(15).iterrows():
                h = {"title": row.get("title", "")}
                if "label" in row:
                    h["sentiment"] = row["label"]
                if "score" in row:
                    h["confidence"] = round(float(row["score"]), 3)
                headlines.append(h)
            data["news_headlines"] = headlines
        else:
            data["news_headlines"] = []

        return data

    def _run_ensemble(self, news_df):
        if self.engine is None or news_df.empty or "title" not in news_df.columns:
            return None
        try:
            headlines = news_df["title"].tolist()
            ensemble_df = self.engine.analyze_ensemble(headlines)
            if ensemble_df.empty:
                return None
            return {
                "ensemble_avg": round(float(ensemble_df["ensemble_score"].mean()), 4),
                "model_agreement": self._calc_agreement(ensemble_df),
                "headline_count": len(headlines),
            }
        except Exception as e:
            logger.warning(f"Ensemble failed: {e}")
            return None

    @staticmethod
    def _calc_agreement(df):
        numeric_cols = [c for c in df.columns if c.endswith("_numeric")]
        if len(numeric_cols) < 2:
            return "N/A"
        signs = df[numeric_cols].apply(lambda col: col.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)))
        agreement = (signs.nunique(axis=1) == 1).mean()
        if agreement > 0.7:
            return "HIGH"
        elif agreement > 0.4:
            return "MODERATE"
        return "LOW"

    def _build_final_report(self, state, cached_data):
        parts = []
        parts.append(f"# Feelow Agentic Analysis — {state.ticker}")
        parts.append(f"*Multi-agent pipeline completed in {state.total_duration_ms/1000:.1f}s*\n")

        parts.append("## Agent Execution Trace")
        for step in state.steps:
            icon = {"completed": "OK", "failed": "FAIL", "skipped": "SKIP", "unavailable": "N/A"}.get(step.status, "?")
            time_str = f" ({step.duration_ms:.0f}ms)" if step.duration_ms > 0 else ""
            parts.append(f"- [{icon}] **{step.name}** ({step.agent}){time_str} — {step.detail}")

        parts.append("")

        reasoning_step = next((s for s in state.steps if s.name == "Agentic Deep Reasoning"), None)
        if reasoning_step and reasoning_step.status == "completed" and reasoning_step.result:
            agent_result = reasoning_step.result
            if isinstance(agent_result, dict) and agent_result.get("analysis"):
                agent_label = reasoning_step.agent
                parts.append(f"## {agent_label} — Agentic Analysis")
                parts.append(agent_result["analysis"])
                if agent_result.get("tool_calls_made"):
                    parts.append(f"\n### {agent_label} Tool Usage")
                    for tc in agent_result["tool_calls_made"]:
                        parts.append(f"- Turn {tc['turn']}: `{tc['tool']}`({tc['input']})")
        elif any(s.agent.startswith("Gemini") and s.status == "completed" for s in state.steps):
            gemini_steps = [s for s in state.steps if s.agent.startswith("Gemini") and s.status == "completed" and s.name != "Agentic Deep Reasoning"]
            for gs in gemini_steps:
                parts.append(f"\n## {gs.name}")
                if gs.result:
                    parts.append(str(gs.result))

        if not state.claude_available and not state.gemini_available:
            parts.append("\n## Configuration Required")
            parts.append("Add your API keys to enable the full agentic pipeline:")
            parts.append("- **GEMINI_API_KEY**: Multimodal chart analysis + Google Search grounding")
            parts.append("- **ANTHROPIC_API_KEY**: Agentic reasoning with tool use")

        return "\n".join(parts)

    @staticmethod
    def _score_to_signal(score):
        if score > 0.40:
            return "STRONG BUY"
        elif score > 0.10:
            return "BUY"
        elif score < -0.40:
            return "STRONG SELL"
        elif score < -0.10:
            return "SELL"
        return "NEUTRAL"
