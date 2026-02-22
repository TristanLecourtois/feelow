"""
Feelow Backend — Gemini Multimodal Agent
Chart image analysis, Google Search grounding, deep reasoning fallback.
Includes retry with exponential backoff and model fallback chain.
Identical logic to feelow/src/gemini_agent.py — no Streamlit dependency.
"""

from __future__ import annotations
import os, io, logging, time
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

MODEL_CHAIN = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
]

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0


class GeminiAgent:
    """Multimodal financial agent powered by Google Gemini."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.client = None
        self.model_name = MODEL_CHAIN[0]

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}")

    @property
    def available(self) -> bool:
        return self.client is not None

    def _generate_with_retry(self, contents, config=None, preferred_model=None, on_retry=None):
        models_to_try = list(MODEL_CHAIN)
        if preferred_model and preferred_model in models_to_try:
            models_to_try.remove(preferred_model)
            models_to_try.insert(0, preferred_model)

        last_error = None
        for model in models_to_try:
            for attempt in range(MAX_RETRIES):
                try:
                    kwargs = {"model": model, "contents": contents}
                    if config:
                        kwargs["config"] = config
                    response = self.client.models.generate_content(**kwargs)
                    self.model_name = model
                    return response
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        if "limit: 0" in err_str:
                            if on_retry:
                                on_retry(f"Model {model} daily quota exhausted, trying next model...")
                            break
                        delay = RETRY_BASE_DELAY * (2 ** attempt)
                        if on_retry:
                            on_retry(f"Rate limited on {model}, retrying in {delay:.0f}s (attempt {attempt+1}/{MAX_RETRIES})...")
                        time.sleep(delay)
                    else:
                        raise
        raise last_error or Exception("All Gemini models exhausted")

    def analyze_chart_image(self, chart_image_bytes, ticker, context="", on_retry=None):
        if not self.available:
            return self._fallback_visual()
        try:
            from google.genai import types
            prompt = f"""You are an elite technical analyst specializing in chart pattern recognition.
Analyze this price chart for {ticker} and provide a detailed visual technical analysis.

{f"Additional context: {context}" if context else ""}

Provide your analysis in this EXACT format:
VISUAL_PATTERNS: [List chart patterns]
TREND: [STRONG UPTREND / UPTREND / SIDEWAYS / DOWNTREND / STRONG DOWNTREND]
SUPPORT_LEVELS: [Key support levels]
RESISTANCE_LEVELS: [Key resistance levels]
VOLUME_ANALYSIS: [Volume insight]
KEY_OBSERVATION: [Most important visual insight]
VISUAL_SIGNAL: [BULLISH / NEUTRAL / BEARISH]
CONFIDENCE: [1-10]"""

            image_part = types.Part.from_bytes(data=chart_image_bytes, mime_type="image/png")
            response = self._generate_with_retry(contents=[prompt, image_part], on_retry=on_retry)
            return {"success": True, "analysis": response.text, "model": self.model_name}
        except Exception as e:
            logger.error(f"Gemini visual analysis error: {e}")
            return {"success": False, "analysis": self._clean_error(e), "model": self.model_name}

    def search_grounded_analysis(self, ticker, query_context="", on_retry=None):
        if not self.available:
            return self._fallback_search()
        try:
            from google.genai import types
            prompt = f"""You are a senior financial research analyst with access to real-time market data.
Research {ticker} using the latest available information.

{f"Focus areas: {query_context}" if query_context else ""}

Provide your research:
LATEST_NEWS: [2-3 impactful recent items]
ANALYST_CONSENSUS: [Upgrades/downgrades?]
MARKET_CONTEXT: [Broader conditions]
UPCOMING_CATALYSTS: [Events]
SECTOR_MOMENTUM: [Sector performance]
RISK_EVENTS: [Upcoming risks]
REAL_TIME_SIGNAL: [BULLISH / NEUTRAL / BEARISH]"""

            search_tool = types.Tool(google_search=types.GoogleSearch())
            config = types.GenerateContentConfig(tools=[search_tool])
            response = self._generate_with_retry(contents=prompt, config=config, on_retry=on_retry)

            text = response.text
            grounding = None
            if hasattr(response, "candidates") and response.candidates:
                if hasattr(response.candidates[0], "grounding_metadata"):
                    grounding = str(response.candidates[0].grounding_metadata)

            return {"success": True, "analysis": text, "grounding_metadata": grounding, "model": self.model_name}
        except Exception as e:
            logger.error(f"Gemini search grounding error: {e}")
            return {"success": False, "analysis": self._clean_error(e), "model": self.model_name}

    def multimodal_synthesis(self, ticker, chart_image_bytes, sentiment_summary, technical_summary, news_headlines, on_retry=None):
        if not self.available:
            return self._fallback_visual()
        try:
            from google.genai import types
            prompt = f"""You are a quantitative analyst combining visual chart analysis and quantitative data for {ticker}.

## Quantitative Data
Sentiment: {sentiment_summary}
Technicals: {technical_summary}

## Recent Headlines
{news_headlines}

Provide unified multimodal analysis:
VISUAL_TECHNICAL_CONFLUENCE: [Visual vs quantitative]
SENTIMENT_PRICE_ALIGNMENT: [Sentiment vs chart]
KEY_DIVERGENCES: [Notable divergences]
MULTIMODAL_SIGNAL: [STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL]
CONVICTION: [1-10]
SYNTHESIS: [3-4 sentences]"""

            contents = [prompt]
            if chart_image_bytes:
                image_part = types.Part.from_bytes(data=chart_image_bytes, mime_type="image/png")
                contents.append(image_part)

            search_tool = types.Tool(google_search=types.GoogleSearch())
            config = types.GenerateContentConfig(tools=[search_tool])
            response = self._generate_with_retry(contents=contents, config=config, on_retry=on_retry)
            return {"success": True, "analysis": response.text, "model": self.model_name}
        except Exception as e:
            logger.error(f"Gemini multimodal synthesis error: {e}")
            return {"success": False, "analysis": self._clean_error(e), "model": self.model_name}

    def run_reasoning_analysis(self, ticker, cached_data, gemini_visual=None, gemini_search=None, chart_image_bytes=None, on_step=None):
        if not self.available:
            return {"analysis": "Gemini API not configured.", "tool_calls_made": [], "turns_used": 0, "success": False}
        try:
            from google.genai import types
            import json

            if on_step:
                on_step("Gemini: Building comprehensive analysis prompt...")

            sentiment = cached_data.get("sentiment_summary", {})
            technicals = cached_data.get("technical_summary", {})
            price = cached_data.get("price_summary", {})
            headlines = cached_data.get("news_headlines", [])
            ensemble = cached_data.get("ensemble_result")

            headlines_text = ""
            if headlines:
                for h in headlines[:15]:
                    sent = h.get("sentiment", "?")
                    conf = h.get("confidence", 0)
                    headlines_text += f"- [{sent} {conf:.0%}] {h.get('title', '')}\n"

            prompt = f"""You are an elite quantitative analyst for the Feelow platform.
Produce the most comprehensive investment analysis for {ticker}.

## Price Data
{json.dumps(price, indent=2)}

## Sentiment Analysis
{json.dumps(sentiment, indent=2)}
{f"Ensemble agreement: {ensemble}" if ensemble else ""}

## Technical Indicators
{json.dumps(technicals, indent=2)}

## Recent Headlines
{headlines_text}

{f"## Gemini Visual Chart Analysis" + chr(10) + gemini_visual if gemini_visual else ""}
{f"## Real-Time Market Context" + chr(10) + gemini_search if gemini_search else ""}

Provide DEEP ANALYSIS:
**SIGNAL:** [STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL]
**CONVICTION:** [1-10]
**KEY INSIGHT:** [One sentence]
**DATA SOURCES USED:** [List]
**SENTIMENT ANALYSIS:** [3-4 sentences]
**TECHNICAL VIEW:** [3-4 sentences]
**CROSS-MODAL CONFLUENCE:** [2-3 sentences]
**RISK FACTORS:** [bullets]
**RECOMMENDATION:** [3-4 sentences]"""

            if on_step:
                on_step("Gemini: Generating deep reasoning analysis...")

            contents = [prompt]
            if chart_image_bytes:
                image_part = types.Part.from_bytes(data=chart_image_bytes, mime_type="image/png")
                contents.append(image_part)

            search_tool = types.Tool(google_search=types.GoogleSearch())
            config = types.GenerateContentConfig(tools=[search_tool])

            def retry_callback(msg):
                if on_step:
                    on_step(f"Gemini: {msg}")

            response = self._generate_with_retry(contents=contents, config=config, on_retry=retry_callback)

            if on_step:
                on_step(f"Gemini: Analysis complete (model: {self.model_name})")

            return {
                "analysis": response.text,
                "tool_calls_made": [
                    {"tool": "gemini_multimodal_reasoning", "input": {"ticker": ticker}, "turn": 1},
                    {"tool": "google_search_grounding", "input": {"ticker": ticker}, "turn": 1},
                ],
                "turns_used": 1,
                "success": True,
                "model_used": self.model_name,
            }
        except Exception as e:
            logger.error(f"Gemini reasoning error: {e}")
            clean = self._clean_error(e)
            return {"analysis": f"**Gemini API Error:** {clean}", "tool_calls_made": [], "turns_used": 0, "success": False}

    @staticmethod
    def _clean_error(e):
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            if "limit: 0" in msg or "PerDay" in msg:
                return "Gemini daily free-tier quota exhausted. Wait for reset or enable billing."
            return "Gemini rate limited. Retry in a few minutes."
        if "400" in msg:
            return "Bad request to Gemini API."
        if "403" in msg:
            return "Gemini API access denied."
        return msg[:200]

    @staticmethod
    def _fallback_visual():
        return {"success": False, "analysis": "Gemini API not configured.", "model": "fallback"}

    @staticmethod
    def _fallback_search():
        return {"success": False, "analysis": "Gemini API not configured.", "model": "fallback"}
