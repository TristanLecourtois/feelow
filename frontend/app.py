"""
+===================================================================+
|   FEELOW — AI Agentic Sentiment x Price Intelligence Platform     |
|   Frontend: Streamlit UI calling FastAPI backend                   |
|   Built for HackEurope 2026                                       |
+===================================================================+

Run:
  Backend:  cd backend  && uvicorn main:app --reload --port 8000
  Frontend: cd frontend && streamlit run app.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
from datetime import datetime

from config import (
    TICKER_CATEGORIES, PAGE_TITLE, PAGE_ICON,
    SIGNAL_THRESHOLDS, POSITIVE_COLOR, NEGATIVE_COLOR, NEUTRAL_COLOR,
    ACCENT_COLOR, MODELS, CARD_BG, BACKEND_URL,
)
from src.api_client import FeelowAPI
from src.visualizer import DashboardCharts

# =====================================================================
# Page Config
# =====================================================================
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide", initial_sidebar_state="expanded")

# =====================================================================
# Custom CSS (unchanged from original)
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background-color: #0e1117; }
    .stMetric { background: linear-gradient(135deg, #1e2130 0%, #161b26 100%);
        padding: 18px; border-radius: 12px; border: 1px solid #30363d; }
    .stMetric:hover { border-color: #636efa; transition: 0.3s; }
    div[data-testid="stMetricValue"] { font-size: 1.7rem; font-weight: 700; }
    .hero-title { font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #636efa, #00cc96, #ffa15a);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero-sub { color: #8b949e; font-size: 1.05rem; margin-top: -8px; }
    .card { background: #1e2130; border-radius: 12px; padding: 20px;
        border: 1px solid #30363d; margin-bottom: 12px; }
    .signal-pill { display: inline-block; padding: 6px 16px; border-radius: 20px;
        font-weight: 700; font-size: 0.95rem; letter-spacing: 0.5px; }
    .pill-green { background: #00cc96; color: #000; }
    .pill-red { background: #ef553b; color: #fff; }
    .pill-yellow { background: #ffa15a; color: #000; }
    div[data-testid="stSidebar"] { background: #161b26; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background: #1e2130; border-radius: 8px 8px 0 0;
        padding: 8px 20px; border: 1px solid #30363d; }
    .stTabs [aria-selected="true"] { background: #636efa !important; color: white !important; }
    .agent-step { background: #1e2130; border-radius: 8px; padding: 12px 16px;
        border-left: 4px solid #30363d; margin-bottom: 8px; }
    .agent-step-active { border-left-color: #636efa; }
    .agent-step-done { border-left-color: #00cc96; }
    .agent-step-fail { border-left-color: #ef553b; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# API Client
# =====================================================================
api = FeelowAPI(base_url=BACKEND_URL)

# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.markdown("## Feelow")
    st.caption("Agentic AI Sentiment x Price Intelligence")
    st.markdown("---")

    # Backend status
    health = api.health()
    if health.get("status") == "ok":
        st.success(f"Backend: ON ({health.get('device', '?')})", icon="✅")
    else:
        st.error("Backend: OFF — Start the backend first!")
        st.code("cd backend && uvicorn main:app --reload --port 8000")

    st.markdown("---")

    # Ticker selection
    st.markdown("### Asset Selection")
    category = st.selectbox("Category", list(TICKER_CATEGORIES.keys()), index=0)
    tickers_in_cat = TICKER_CATEGORIES[category]
    selected_ticker = st.selectbox("Ticker", tickers_in_cat, index=0)
    custom = st.text_input("Or enter custom ticker", placeholder="e.g. PLTR")
    if custom.strip():
        selected_ticker = custom.strip().upper()

    st.markdown("---")

    # Model selection
    st.markdown("### Sentiment Model")
    model_names = {v["name"]: k for k, v in MODELS.items()}
    chosen_name = st.selectbox("Primary Model", list(model_names.keys()))
    chosen_model_id = model_names[chosen_name]
    enable_ensemble = st.toggle("Enable Multi-Model Ensemble", value=False)

    st.markdown("---")

    # Time range
    st.markdown("### Time Range")
    period = st.selectbox("Price History", ["7d", "1mo", "3mo", "6mo", "1y"], index=1)

    st.markdown("---")

    # AI Agent Keys (sent to backend per-request)
    st.markdown("### AI Agent Configuration")
    gemini_key = st.text_input(
        "Google Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""),
        type="password", placeholder="AIza...",
    )
    claude_key = st.text_input(
        "Anthropic Claude API Key", value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password", placeholder="sk-ant-...",
    )

    col_g, col_c = st.columns(2)
    with col_g:
        if gemini_key:
            st.success("Gemini: ON", icon="✅")
        else:
            st.warning("Gemini: OFF")
    with col_c:
        if claude_key:
            st.success("Claude: ON", icon="✅")
        else:
            st.warning("Claude: OFF")

    st.markdown("---")

    if st.button("Refresh All Data", use_container_width=True):
        st.cache_data.clear()

    st.markdown("---")
    st.info(f"**Model:** {chosen_name}")

# =====================================================================
# Header
# =====================================================================
st.markdown(f'<p class="hero-title">Feelow</p>', unsafe_allow_html=True)
st.markdown(f'<p class="hero-sub">Agentic Multi-Model Sentiment x Price Intelligence for <b>{selected_ticker}</b></p>', unsafe_allow_html=True)

# =====================================================================
# Data Loading (from backend)
# =====================================================================
with st.spinner(f"Loading data for {selected_ticker}..."):
    data = api.load_data(
        ticker=selected_ticker,
        period=period,
        model_id=chosen_model_id,
    )

    if data.get("error"):
        st.error(f"Backend error: {data['error']}")

    # Convert JSON records back to DataFrames
    price_df = pd.DataFrame(data.get("price_data", []))
    news_df = pd.DataFrame(data.get("news_data", []))
    current_price = data.get("current_price", 0)
    pct_change = data.get("pct_change", 0)
    metrics = data.get("metrics", {})
    volume_24h = metrics.get("volume_24h", 0)
    avg_sentiment = metrics.get("avg_sentiment", 0)
    signal = metrics.get("signal", "NEUTRAL")

    # Parse datetime columns
    if not price_df.empty and "timestamp" in price_df.columns:
        price_df["timestamp"] = pd.to_datetime(price_df["timestamp"])
    if not news_df.empty and "published" in news_df.columns:
        news_df["published"] = pd.to_datetime(news_df["published"])

# =====================================================================
# KPI Row
# =====================================================================
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Price", f"${current_price:,.2f}", f"{pct_change:+.2f}%")
with k2:
    st.metric("News Volume (24h)", volume_24h)
with k3:
    st.metric("Avg Sentiment", f"{avg_sentiment:+.3f}")
with k4:
    sig_cls = "pill-green" if "BUY" in signal else ("pill-red" if "SELL" in signal else "pill-yellow")
    st.markdown(f"**AI Signal**")
    st.markdown(f'<span class="signal-pill {sig_cls}">{signal}</span>', unsafe_allow_html=True)
with k5:
    rsi_val = price_df["RSI"].iloc[-1] if not price_df.empty and "RSI" in price_df.columns else 0
    rsi_display = rsi_val if rsi_val and pd.notna(rsi_val) else 0
    st.metric("RSI (14)", f"{rsi_display:.1f}")

st.markdown("---")

# =====================================================================
# Main Tabs
# =====================================================================
tab_agent, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Agent Pipeline", "Price & Sentiment", "Multi-Model", "Technicals", "AI Analyst", "News Feed"
])

# ----- Tab Agent: Agentic Pipeline -----
with tab_agent:
    st.markdown("### Agentic Analysis Pipeline")
    st.markdown("*Multi-agent system: Gemini (multimodal + search) + Claude (reasoning + tool use)*")

    with st.expander("Pipeline Architecture", expanded=False):
        st.markdown("""
```
                    +------------------+
                    |   ORCHESTRATOR   |
                    | (Agent Manager)  |
                    +--------+---------+
                             |
            +----------------+----------------+
            |                |                |
   +--------v------+ +------v--------+ +-----v---------+
   | DATA ASSEMBLY | | SENTIMENT     | | GEMINI VISION |
   | (News, Price, | | ENSEMBLE      | | (Chart Image  |
   |  Technicals)  | | (FinBERT x3)  | |  Analysis)    |
   +--------+------+ +------+--------+ +-----+---------+
            |                |                |
            +----------------+----------------+
                             |
                    +--------v---------+
                    | GEMINI SEARCH    |
                    | (Google Search   |
                    |  Grounding)      |
                    +--------+---------+
                             |
                    +--------v---------+
                    | REASONING AGENT  |
                    | Claude (tool use)|
                    | OR Gemini (deep) |
                    +--------+---------+
                             |
                    +--------v---------+
                    | FINAL SYNTHESIS  |
                    | (Unified Report) |
                    +------------------+
```
        """)

    agent_status_col, agent_config_col = st.columns([2, 1])
    with agent_config_col:
        st.markdown("**Agent Status**")
        gemini_ready = bool(gemini_key)
        claude_ready = bool(claude_key)
        reasoning_ready = claude_ready or gemini_ready
        reasoning_agent = "Claude Sonnet 4" if claude_ready else ("Gemini 2.0 Flash" if gemini_ready else "None")
        agents_info = [
            ("Data Assembly", "System", True),
            ("Sentiment Ensemble", "FinBERT x3", True),
            ("Visual Analysis", "Gemini 2.0 Flash", gemini_ready),
            ("Search Grounding", "Gemini + Google", gemini_ready),
            ("Deep Reasoning", reasoning_agent, reasoning_ready),
        ]
        for name, agent, active in agents_info:
            color = POSITIVE_COLOR if active else NEUTRAL_COLOR
            st.markdown(
                f"<span style='color:{color};'>{'[OK]' if active else '[--]'}</span> **{name}** <small>({agent})</small>",
                unsafe_allow_html=True,
            )

    with agent_status_col:
        run_pipeline = st.button("🚀 Launch Agentic Analysis", use_container_width=True, type="primary")

        if not run_pipeline:
            st.info("Click **Launch Agentic Analysis** to run the full multi-agent pipeline.")

    if run_pipeline:
        with st.status("🧠 **Pipeline agentic en cours — veuillez patienter...**", expanded=True) as live_status:
            st.write("Sending request to backend...")

            result = api.run_pipeline(
                ticker=selected_ticker,
                period=period,
                model_id=chosen_model_id,
                gemini_key=gemini_key,
                claude_key=claude_key,
            )

            if result.get("error"):
                live_status.update(label=f"❌ **Pipeline error: {result['error']}**", state="error", expanded=True)
                st.error(f"Pipeline crashed: {result['error']}")
            else:
                total_s = result.get("total_duration_ms", 0) / 1000
                steps = result.get("steps", [])
                ok_count = len([s for s in steps if s.get("status") == "completed"])
                live_status.update(
                    label=f"✅ **Pipeline terminé en {total_s:.1f}s** — {ok_count}/{len(steps)} étapes réussies",
                    state="complete", expanded=True,
                )

                # Show step progress summary
                for step in steps:
                    icon_map = {"completed": "✅", "failed": "❌", "skipped": "⏭️", "unavailable": "🚫"}
                    icon = icon_map.get(step.get("status", ""), "❓")
                    dur = step.get("duration_ms", 0)
                    time_str = f" ({dur:.0f}ms)" if dur > 0 else ""
                    st.write(f"{icon} **{step['name']}** ({step['agent']}){time_str} — {step.get('detail', '')}")

        if result and not result.get("error"):
            steps = result.get("steps", [])

            # Execution trace
            st.markdown("---")
            st.markdown("#### 📊 Execution Trace")
            for step in steps:
                icon_map = {
                    "completed": ("✅", POSITIVE_COLOR),
                    "failed": ("❌", NEGATIVE_COLOR),
                    "skipped": ("⏭️", NEUTRAL_COLOR),
                    "unavailable": ("🚫", "#8b949e"),
                }
                icon, color = icon_map.get(step.get("status", ""), ("❓", "#8b949e"))
                dur = step.get("duration_ms", 0)
                time_str = f" — {dur:.0f}ms" if dur > 0 else ""
                css_cls = "done" if step.get("status") == "completed" else "fail"
                st.markdown(
                    f"<div class='agent-step agent-step-{css_cls}'>"
                    f"<span style='font-size:1.1em;'>{icon}</span> "
                    f"<b>{step['name']}</b> <small>({step['agent']}){time_str}</small><br/>"
                    f"<span style='color:#8b949e;'>{step.get('detail', '')}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(f"**⏱️ Total: {result.get('total_duration_ms', 0)/1000:.1f}s**")

            # Final report
            st.markdown("---")
            st.markdown("#### 📝 Final Agentic Report")
            st.markdown(result.get("final_report", "No report generated."))

            # Gemini results in expanders
            for step in steps:
                if step.get("name") == "Visual Chart Analysis" and step.get("status") == "completed" and step.get("result"):
                    with st.expander("🔍 Gemini Visual Chart Analysis (Full)", expanded=False):
                        st.markdown(step["result"])
                if step.get("name") == "Real-Time Search Grounding" and step.get("status") == "completed" and step.get("result"):
                    with st.expander("🌐 Gemini Search Grounding (Full)", expanded=False):
                        st.markdown(step["result"])

# ----- Tab 1: Price & Sentiment Overlay -----
with tab1:
    col_main, col_side = st.columns([2.5, 1])
    with col_main:
        fig_overlay = DashboardCharts.plot_price_sentiment_overlay(
            price_df, news_df,
            title=f"{selected_ticker} — Price & Sentiment Correlation",
        )
        st.plotly_chart(fig_overlay, use_container_width=True)
    with col_side:
        fig_gauge = DashboardCharts.plot_sentiment_gauge(avg_sentiment)
        st.plotly_chart(fig_gauge, use_container_width=True)
        fig_dist = DashboardCharts.plot_sentiment_distribution(news_df)
        st.plotly_chart(fig_dist, use_container_width=True)

    fig_timeline = DashboardCharts.plot_sentiment_timeline(news_df)
    st.plotly_chart(fig_timeline, use_container_width=True)

# ----- Tab 2: Multi-Model Comparison -----
with tab2:
    st.markdown("### Multi-Model Sentiment Comparison")
    st.caption("Compare ProsusAI/finbert, DistilRoBERTa Financial, and Sigma Financial SA")

    test_text = st.text_input(
        "Test a custom headline",
        value=f"{selected_ticker} reports record quarterly revenue, beating analyst estimates",
    )
    if test_text:
        with st.spinner("Comparing models..."):
            comp_results = api.compare_models(test_text)
        if comp_results:
            comp_df = pd.DataFrame(comp_results)
            c1, c2 = st.columns([1, 1])
            with c1:
                st.dataframe(
                    comp_df.style.format({"Confidence": "{:.1%}"}),
                    use_container_width=True, hide_index=True,
                )
            with c2:
                fig_comp = DashboardCharts.plot_model_comparison(comp_df)
                st.plotly_chart(fig_comp, use_container_width=True)

    if enable_ensemble and not news_df.empty:
        st.markdown("---")
        st.markdown("### Ensemble Results on Recent News")
        with st.spinner("Running ensemble analysis on all headlines..."):
            headlines_list = news_df["title"].tolist() if "title" in news_df.columns else []
            ensemble_results = api.run_ensemble(headlines_list)
        if ensemble_results:
            ensemble_df = pd.DataFrame(ensemble_results)
            if "ensemble_score" in ensemble_df.columns:
                avg_ens = ensemble_df["ensemble_score"].mean()
                st.metric("Ensemble Average", f"{avg_ens:+.3f}")
                st.dataframe(
                    ensemble_df[["headline", "ensemble_label", "ensemble_score"]].head(20).style.format({"ensemble_score": "{:+.3f}"}),
                    use_container_width=True, hide_index=True,
                )
    elif not enable_ensemble:
        st.info("Toggle **Enable Multi-Model Ensemble** in the sidebar to run all 3 models on every headline.")

# ----- Tab 3: Technicals -----
with tab3:
    st.markdown("### Technical Analysis")
    if price_df.empty:
        st.warning("No price data available.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig_rsi = DashboardCharts.plot_rsi(price_df)
            st.plotly_chart(fig_rsi, use_container_width=True)
        with c2:
            fig_macd = DashboardCharts.plot_macd(price_df)
            st.plotly_chart(fig_macd, use_container_width=True)

        st.markdown("### Key Indicators")
        last = price_df.iloc[-1]
        cols = st.columns(6)
        indicators = [
            ("SMA 7", last.get("SMA_7", None)),
            ("SMA 21", last.get("SMA_21", None)),
            ("SMA 50", last.get("SMA_50", None)),
            ("RSI", last.get("RSI", None)),
            ("MACD", last.get("MACD", None)),
            ("Daily Return", last.get("daily_return", None)),
        ]
        for col, (name, val) in zip(cols, indicators):
            with col:
                if val is not None and pd.notna(val):
                    fmt = f"{val:+.2f}%" if name == "Daily Return" else f"{val:.2f}"
                    st.metric(name, fmt)
                else:
                    st.metric(name, "N/A")

# ----- Tab 4: AI Analyst (Claude single-shot) -----
with tab4:
    st.markdown("### Claude AI Analyst")
    st.caption("Single-shot analysis — for the full agentic pipeline, use the **Agent Pipeline** tab")

    if not claude_key:
        st.warning("Enter your Anthropic API key in the sidebar to enable Claude analysis.")
        st.markdown("""
        **What Claude will provide:**
        - Signal with conviction score (1-10)
        - Key insight synthesis
        - Sentiment narrative interpretation
        - Technical analysis commentary
        - Risk factors assessment
        - Actionable recommendation
        """)

    if st.button("Generate AI Analysis", use_container_width=True, disabled=not claude_key):
        sent_summary = f"Average sentiment: {avg_sentiment:+.3f} | Signal: {signal} | News volume: {volume_24h}"
        if not news_df.empty and "label" in news_df.columns:
            counts = news_df["label"].value_counts().to_dict()
            sent_summary += f"\nDistribution: {counts}"

        headlines_text = ""
        if not news_df.empty and "title" in news_df.columns:
            headlines_text = "\n".join([f"- {row['title']}" for _, row in news_df.head(10).iterrows()])

        price_summary = f"Current: ${current_price:,.2f} | 7d change: {pct_change:+.2f}%"
        tech_summary = ""
        if not price_df.empty:
            last = price_df.iloc[-1]
            rsi = last.get("RSI", "N/A")
            macd = last.get("MACD", "N/A")
            tech_summary = f"RSI: {rsi:.1f} | MACD: {macd:.4f}" if isinstance(rsi, (int, float)) else ""

        with st.spinner("Claude is analyzing..."):
            result = api.run_claude_analysis(
                ticker=selected_ticker,
                sentiment_summary=sent_summary,
                price_summary=price_summary,
                headlines=headlines_text,
                technical_summary=tech_summary,
                claude_key=claude_key,
            )
        st.markdown(result.get("analysis", "No analysis returned."))

# ----- Tab 5: News Feed -----
with tab5:
    st.markdown("### Recent News & Sentiment")
    if not news_df.empty and "label" in news_df.columns:
        disp = news_df[["published", "title", "label", "score", "source"]].copy()
        disp = disp.sort_values("published", ascending=False)
        disp.columns = ["Published", "Headline", "Sentiment", "Confidence", "Source"]
        disp["Published"] = disp["Published"].dt.strftime("%Y-%m-%d %H:%M")
        disp["Confidence"] = disp["Confidence"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        disp["Sentiment"] = disp["Sentiment"].str.capitalize()

        def color_sent(val):
            c = {
                "Positive": POSITIVE_COLOR, "Negative": NEGATIVE_COLOR, "Neutral": NEUTRAL_COLOR,
            }.get(val, "white")
            return f"color: {c}; font-weight: bold;"

        styled = disp.style.map(color_sent, subset=["Sentiment"])
        st.dataframe(styled, use_container_width=True, height=500, hide_index=True)
    else:
        st.info(f"No news found for {selected_ticker}.")

# =====================================================================
# Footer
# =====================================================================
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; color:#8b949e; padding:16px 0;'>
    <p style='font-size:1rem; margin-bottom:4px;'>
        <b>Feelow</b> — Agentic AI Sentiment x Price Intelligence Platform
    </p>
    <p style='font-size:0.78rem;'>
        Agents: Gemini 2.0 Flash (multimodal + search) · Claude Sonnet 4 (reasoning + tool use)
        &nbsp;|&nbsp; NLP: ProsusAI/finbert · DistilRoBERTa Financial · Sigma/financial-sentiment-analysis
        &nbsp;|&nbsp; Data: Yahoo Finance · Finviz · Google Search
        &nbsp;|&nbsp; Built for <b>HackEurope 2026</b>
    </p>
</div>
""", unsafe_allow_html=True)
