"""
Feelow Backend — Claude Agentic Analyst Module
Uses Anthropic Claude API with TOOL USE for autonomous financial reasoning.
Identical logic to feelow/src/claude_analyst.py — no Streamlit dependency.
"""

from __future__ import annotations
import os, json, logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# =====================================================================
# Tool Definitions for Claude Agentic Loop
# =====================================================================
AGENT_TOOLS = [
    {
        "name": "get_sentiment_data",
        "description": "Get sentiment analysis results for a stock ticker. Returns average sentiment score, signal, news volume, and distribution.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., NVDA, TSLA, AAPL)"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_technical_indicators",
        "description": "Get technical analysis indicators for a stock: RSI, MACD, SMA, Bollinger Bands, daily return.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_price_action",
        "description": "Get recent price action data: current price, price change over N days.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
                "days": {"type": "integer", "description": "Number of days to look back", "default": 7},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_news_headlines",
        "description": "Get recent financial news headlines with sentiment labels and confidence scores.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
                "limit": {"type": "integer", "description": "Maximum headlines to return", "default": 10},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_company_info",
        "description": "Get company fundamentals: name, sector, market cap, P/E ratio, 52-week range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "compare_sentiment_models",
        "description": "Compare sentiment across all 3 FinBERT models on a specific headline.",
        "input_schema": {
            "type": "object",
            "properties": {
                "headline": {"type": "string", "description": "The news headline to analyze"},
            },
            "required": ["headline"],
        },
    },
    {
        "name": "get_gemini_visual_analysis",
        "description": "Get Gemini's multimodal visual analysis of the price chart.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_gemini_search_context",
        "description": "Get Gemini's Google Search-grounded real-time market context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
            },
            "required": ["ticker"],
        },
    },
]


class ClaudeAnalyst:
    """Agentic AI analyst powered by Claude with autonomous tool use."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.client = None
        self.model = "claude-sonnet-4-20250514"
        self.max_agent_turns = 5
        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Claude init failed: {e}")

    @property
    def available(self) -> bool:
        return self.client is not None

    def generate_analysis(
        self,
        ticker: str,
        sentiment_summary: str,
        price_data_summary: str,
        news_headlines: str,
        technical_summary: str = "",
    ) -> str:
        if not self.available:
            return self._fallback_analysis(ticker, sentiment_summary)

        prompt = f"""You are an elite quantitative analyst at a top hedge fund.
Analyze the following data for {ticker} and provide a concise, actionable trading brief.

## Sentiment Data
{sentiment_summary}

## Recent Headlines
{news_headlines}

## Price & Technical Summary
{price_data_summary}
{technical_summary}

Provide your analysis in this exact format:
**SIGNAL:** [STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL]
**CONVICTION:** [1-10]
**KEY INSIGHT:** [One sentence]
**SENTIMENT ANALYSIS:** [2-3 sentences]
**TECHNICAL VIEW:** [2-3 sentences]
**RISK FACTORS:** [2-3 bullet points]
**RECOMMENDATION:** [2-3 sentences]"""

        try:
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return self._fallback_analysis(ticker, sentiment_summary)

    def run_agentic_analysis(
        self,
        ticker: str,
        tool_handler: "AgentToolHandler",
        gemini_visual: Optional[str] = None,
        gemini_search: Optional[str] = None,
        on_step: Optional[callable] = None,
    ) -> Dict[str, Any]:
        if not self.available:
            return {
                "analysis": self._fallback_analysis(ticker, "N/A"),
                "tool_calls_made": [],
                "turns_used": 0,
            }

        system_prompt = f"""You are an autonomous AI financial analyst agent for the Feelow platform.
Your mission: produce the most comprehensive investment analysis for {ticker}.

You have access to tools that let you gather real-time data autonomously.

{"## Gemini Visual Chart Analysis (already completed)" + chr(10) + gemini_visual if gemini_visual else ""}
{"## Gemini Search-Grounded Market Context (already completed)" + chr(10) + gemini_search if gemini_search else ""}

WORKFLOW:
1. Gather data (sentiment, technicals, price action, news, company info)
2. Incorporate Gemini analysis if available
3. Cross-reference signals
4. Identify confluences and divergences
5. Produce final analysis

FINAL OUTPUT FORMAT:
**SIGNAL:** [STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL]
**CONVICTION:** [1-10]
**KEY INSIGHT:** [One sentence]
**DATA SOURCES USED:** [List]
**SENTIMENT ANALYSIS:** [3-4 sentences]
**TECHNICAL VIEW:** [3-4 sentences]
**CROSS-MODAL CONFLUENCE:** [2-3 sentences]
**RISK FACTORS:** [bullets]
**RECOMMENDATION:** [3-4 sentences]"""

        messages = [
            {"role": "user", "content": f"Analyze {ticker} using your available tools. Build a comprehensive investment thesis."},
        ]

        tool_calls_made = []
        turns = 0

        try:
            for turn in range(self.max_agent_turns):
                turns += 1

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    system=system_prompt,
                    tools=AGENT_TOOLS,
                    messages=messages,
                )

                if response.stop_reason == "tool_use":
                    assistant_content = response.content
                    messages.append({"role": "assistant", "content": assistant_content})

                    tool_results = []
                    for block in assistant_content:
                        if block.type == "tool_use":
                            tool_name = block.name
                            tool_input = block.input
                            tool_id = block.id

                            if on_step:
                                on_step(f"Calling tool: {tool_name}({json.dumps(tool_input)})")

                            tool_calls_made.append({
                                "tool": tool_name,
                                "input": tool_input,
                                "turn": turn + 1,
                            })

                            result = tool_handler.execute(tool_name, tool_input)

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": json.dumps(result) if isinstance(result, dict) else str(result),
                            })

                    messages.append({"role": "user", "content": tool_results})

                elif response.stop_reason == "end_turn":
                    final_text = ""
                    for block in response.content:
                        if hasattr(block, "text"):
                            final_text += block.text

                    if on_step:
                        on_step("Analysis complete")

                    return {
                        "analysis": final_text,
                        "tool_calls_made": tool_calls_made,
                        "turns_used": turns,
                    }

            # Max turns reached
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text

            return {
                "analysis": final_text or "Analysis incomplete — max agent turns reached.",
                "tool_calls_made": tool_calls_made,
                "turns_used": turns,
            }

        except Exception as e:
            logger.error(f"Claude agentic error: {e}")
            return {
                "analysis": f"Agent error: {e}",
                "tool_calls_made": tool_calls_made,
                "turns_used": turns,
            }

    @staticmethod
    def _fallback_analysis(ticker: str, sentiment_summary: str) -> str:
        return f"""**Claude API not configured** — Add your `ANTHROPIC_API_KEY` to unlock AI-powered deep analysis.

Current sentiment snapshot for {ticker}:
{sentiment_summary}

*To enable Claude reasoning, set your API key in the sidebar.*"""


class AgentToolHandler:
    """Handles tool execution for the Claude agentic loop."""

    def __init__(
        self,
        sentiment_engine=None,
        news_ingestor=None,
        market_loader=None,
        gemini_agent=None,
        ticker: str = "",
        cached_data: Optional[Dict[str, Any]] = None,
    ):
        self.engine = sentiment_engine
        self.ingestor = news_ingestor
        self.loader = market_loader
        self.gemini = gemini_agent
        self.ticker = ticker
        self.cached = cached_data or {}

    def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        handlers = {
            "get_sentiment_data": self._get_sentiment,
            "get_technical_indicators": self._get_technicals,
            "get_price_action": self._get_price,
            "get_news_headlines": self._get_news,
            "get_company_info": self._get_company,
            "compare_sentiment_models": self._compare_models,
            "get_gemini_visual_analysis": self._get_gemini_visual,
            "get_gemini_search_context": self._get_gemini_search,
        }
        handler = handlers.get(tool_name)
        if handler:
            try:
                return handler(tool_input)
            except Exception as e:
                return {"error": str(e)}
        return {"error": f"Unknown tool: {tool_name}"}

    def _get_sentiment(self, inp: dict) -> dict:
        if "sentiment_summary" in self.cached:
            return self.cached["sentiment_summary"]
        return {"info": "Sentiment data not available in cache"}

    def _get_technicals(self, inp: dict) -> dict:
        if "technical_summary" in self.cached:
            return self.cached["technical_summary"]
        return {"info": "Technical data not available in cache"}

    def _get_price(self, inp: dict) -> dict:
        if "price_summary" in self.cached:
            return self.cached["price_summary"]
        return {"info": "Price data not available in cache"}

    def _get_news(self, inp: dict) -> dict:
        if "news_headlines" in self.cached:
            limit = inp.get("limit", 10)
            headlines = self.cached["news_headlines"]
            if isinstance(headlines, list):
                return {"headlines": headlines[:limit]}
            return {"headlines": headlines}
        return {"info": "News data not available in cache"}

    def _get_company(self, inp: dict) -> dict:
        if "company_info" in self.cached:
            return self.cached["company_info"]
        if self.loader:
            try:
                return self.loader.get_company_info()
            except Exception as e:
                return {"error": str(e)}
        return {"info": "Company data not available"}

    def _compare_models(self, inp: dict) -> dict:
        headline = inp.get("headline", "")
        if self.engine and headline:
            try:
                comp = self.engine.compare_models(headline)
                return comp.to_dict(orient="records")
            except Exception as e:
                return {"error": str(e)}
        return {"info": "Sentiment engine not available"}

    def _get_gemini_visual(self, inp: dict) -> dict:
        if "gemini_visual" in self.cached:
            return {"analysis": self.cached["gemini_visual"]}
        return {"info": "Gemini visual analysis not available"}

    def _get_gemini_search(self, inp: dict) -> dict:
        if "gemini_search" in self.cached:
            return {"analysis": self.cached["gemini_search"]}
        return {"info": "Gemini search context not available"}
