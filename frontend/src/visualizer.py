"""
Feelow Frontend — Visualization Module
Plotly charts for the Streamlit dashboard.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    CHART_THEME, POSITIVE_COLOR, NEGATIVE_COLOR, NEUTRAL_COLOR,
    ACCENT_COLOR, SIGNAL_THRESHOLDS, MODELS,
)

COLOR_MAP = {"positive": POSITIVE_COLOR, "negative": NEGATIVE_COLOR, "neutral": NEUTRAL_COLOR}


class DashboardCharts:

    # =================================================================
    # 1. Price + Sentiment Overlay
    # =================================================================
    @staticmethod
    def plot_price_sentiment_overlay(price_df, news_df, title="Price & Sentiment") -> go.Figure:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
            vertical_spacing=0.03, subplot_titles=(title, "Volume"),
        )
        if not price_df.empty:
            if all(c in price_df.columns for c in ["open", "high", "low", "close"]):
                fig.add_trace(go.Candlestick(
                    x=price_df["timestamp"], open=price_df["open"],
                    high=price_df["high"], low=price_df["low"], close=price_df["close"],
                    name="OHLC", increasing_line_color=POSITIVE_COLOR,
                    decreasing_line_color=NEGATIVE_COLOR,
                ), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(
                    x=price_df["timestamp"], y=price_df["close"],
                    mode="lines", name="Close",
                    line=dict(color="rgba(200,200,200,0.8)", width=2),
                ), row=1, col=1)

            for col in price_df.columns:
                if col.startswith("SMA_"):
                    fig.add_trace(go.Scatter(
                        x=price_df["timestamp"], y=price_df[col],
                        mode="lines", name=col, line=dict(width=1, dash="dot"),
                    ), row=1, col=1)

            if "BB_upper" in price_df.columns:
                fig.add_trace(go.Scatter(
                    x=price_df["timestamp"], y=price_df["BB_upper"],
                    mode="lines", name="BB Upper",
                    line=dict(width=0.5, color="rgba(99,110,250,0.3)"), showlegend=False,
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=price_df["timestamp"], y=price_df["BB_lower"],
                    mode="lines", name="BB Lower", fill="tonexty",
                    line=dict(width=0.5, color="rgba(99,110,250,0.3)"),
                    fillcolor="rgba(99,110,250,0.07)", showlegend=False,
                ), row=1, col=1)

            if "volume" in price_df.columns:
                colors = [
                    POSITIVE_COLOR if c >= o else NEGATIVE_COLOR
                    for c, o in zip(price_df["close"], price_df["open"])
                ] if "open" in price_df.columns else [ACCENT_COLOR] * len(price_df)
                fig.add_trace(go.Bar(
                    x=price_df["timestamp"], y=price_df["volume"],
                    name="Volume", marker_color=colors, opacity=0.5,
                ), row=2, col=1)

        if not news_df.empty and "label" in news_df.columns and not price_df.empty:
            merged = DashboardCharts._merge_news_price(price_df, news_df)
            if not merged.empty:
                for label, color in COLOR_MAP.items():
                    sub = merged[merged["label"] == label]
                    if sub.empty:
                        continue
                    hover = sub.apply(
                        lambda r: f"<b>{str(r['title'])[:70]}</b><br>Sentiment: {r['label']}<br>Conf: {r['score']:.0%}",
                        axis=1,
                    )
                    fig.add_trace(go.Scatter(
                        x=sub["published"], y=sub["close"], mode="markers",
                        name=label.capitalize(),
                        marker=dict(color=color, size=10, line=dict(width=1.5, color="white")),
                        text=hover, hovertemplate="%{text}<extra></extra>",
                    ), row=1, col=1)

        fig.update_layout(
            template=CHART_THEME, height=560,
            legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
            margin=dict(l=0, r=0, t=60, b=0),
            xaxis_rangeslider_visible=False, hovermode="x unified",
        )
        return fig

    # =================================================================
    # 2. Sentiment Gauge
    # =================================================================
    @staticmethod
    def plot_sentiment_gauge(avg_sentiment: float, title: str = "24h Signal") -> go.Figure:
        v = max(-1.0, min(1.0, avg_sentiment))
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=v,
            number=dict(font=dict(size=38), valueformat=".2f"),
            title=dict(text=title, font=dict(size=15)),
            gauge=dict(
                axis=dict(range=[-1, 1], tickvals=[-1, 0, 1], ticktext=["-1", "0", "+1"]),
                bar=dict(color="white", thickness=0.2),
                steps=[
                    dict(range=[-1, SIGNAL_THRESHOLDS["sell"]], color=NEGATIVE_COLOR),
                    dict(range=[SIGNAL_THRESHOLDS["sell"], SIGNAL_THRESHOLDS["buy"]], color=NEUTRAL_COLOR),
                    dict(range=[SIGNAL_THRESHOLDS["buy"], 1], color=POSITIVE_COLOR),
                ],
                threshold=dict(line=dict(color="yellow", width=4), thickness=0.8, value=v),
            ),
        ))
        fig.update_layout(template=CHART_THEME, height=280, margin=dict(l=30, r=30, t=60, b=40))
        return fig

    # =================================================================
    # 3. Sentiment Distribution
    # =================================================================
    @staticmethod
    def plot_sentiment_distribution(news_df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if news_df.empty or "label" not in news_df.columns:
            fig.update_layout(template=CHART_THEME, height=260, title="Sentiment Distribution")
            return fig
        counts = news_df["label"].value_counts()
        for lbl in ["positive", "neutral", "negative"]:
            if lbl not in counts:
                counts[lbl] = 0
        fig.add_trace(go.Bar(
            x=["Positive", "Neutral", "Negative"],
            y=[counts.get("positive", 0), counts.get("neutral", 0), counts.get("negative", 0)],
            marker_color=[POSITIVE_COLOR, NEUTRAL_COLOR, NEGATIVE_COLOR],
            text=[counts.get("positive", 0), counts.get("neutral", 0), counts.get("negative", 0)],
            textposition="auto",
        ))
        fig.update_layout(template=CHART_THEME, title="Sentiment Distribution", height=260, margin=dict(l=0, r=0, t=40, b=0))
        return fig

    # =================================================================
    # 4. Model Comparison
    # =================================================================
    @staticmethod
    def plot_model_comparison(comparison_df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if comparison_df.empty:
            return fig
        models = comparison_df["Model"].tolist()
        scores = comparison_df["Confidence"].tolist()
        numerics = comparison_df["Numeric"].tolist()
        colors = [POSITIVE_COLOR if n > 0 else (NEGATIVE_COLOR if n < 0 else NEUTRAL_COLOR) for n in numerics]
        fig.add_trace(go.Bar(
            x=models, y=scores, marker_color=colors,
            text=[f"{s:.0%}" for s in scores], textposition="auto",
        ))
        fig.update_layout(
            template=CHART_THEME, title="Model Confidence Comparison",
            yaxis_title="Confidence", height=300, margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    # =================================================================
    # 5. Sentiment Timeline
    # =================================================================
    @staticmethod
    def plot_sentiment_timeline(news_df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if news_df.empty or "sentiment_numeric" not in news_df.columns:
            fig.update_layout(template=CHART_THEME, height=200, title="Sentiment Timeline")
            return fig
        df = news_df.sort_values("published")
        df["rolling_sent"] = df["sentiment_numeric"].rolling(window=5, min_periods=1).mean()
        colors = [POSITIVE_COLOR if v > 0 else (NEGATIVE_COLOR if v < 0 else NEUTRAL_COLOR)
                  for v in df["sentiment_numeric"]]
        fig.add_trace(go.Scatter(
            x=df["published"], y=df["rolling_sent"],
            mode="lines", name="Rolling Avg",
            line=dict(color=ACCENT_COLOR, width=2),
            fill="tozeroy", fillcolor="rgba(99,110,250,0.1)",
        ))
        fig.add_trace(go.Scatter(
            x=df["published"], y=df["sentiment_numeric"],
            mode="markers", name="Individual",
            marker=dict(color=colors, size=6),
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            template=CHART_THEME, title="Sentiment Timeline",
            height=240, margin=dict(l=0, r=0, t=40, b=0),
            yaxis=dict(range=[-1.2, 1.2]),
        )
        return fig

    # =================================================================
    # 6. RSI Chart
    # =================================================================
    @staticmethod
    def plot_rsi(price_df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if "RSI" not in price_df.columns:
            return fig
        fig.add_trace(go.Scatter(
            x=price_df["timestamp"], y=price_df["RSI"],
            mode="lines", name="RSI", line=dict(color=ACCENT_COLOR, width=2),
        ))
        fig.add_hline(y=70, line_dash="dash", line_color=NEGATIVE_COLOR, opacity=0.6)
        fig.add_hline(y=30, line_dash="dash", line_color=POSITIVE_COLOR, opacity=0.6)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.03)", line_width=0)
        fig.update_layout(
            template=CHART_THEME, title="RSI (14)",
            yaxis=dict(range=[0, 100]), height=220, margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    # =================================================================
    # 7. MACD Chart
    # =================================================================
    @staticmethod
    def plot_macd(price_df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        if "MACD" not in price_df.columns:
            return fig
        colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in price_df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=price_df["timestamp"], y=price_df["MACD_hist"],
            name="Histogram", marker_color=colors, opacity=0.6,
        ))
        fig.add_trace(go.Scatter(
            x=price_df["timestamp"], y=price_df["MACD"],
            mode="lines", name="MACD", line=dict(color=ACCENT_COLOR, width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=price_df["timestamp"], y=price_df["MACD_signal"],
            mode="lines", name="Signal", line=dict(color=NEUTRAL_COLOR, width=1.5, dash="dot"),
        ))
        fig.update_layout(
            template=CHART_THEME, title="MACD",
            height=220, margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    # =================================================================
    # Helper: merge news with nearest price
    # =================================================================
    @staticmethod
    def _merge_news_price(price_df, news_df):
        if news_df.empty or price_df.empty:
            return pd.DataFrame()
        n = news_df.copy()
        p = price_df.copy()
        n["published"] = pd.to_datetime(n["published"]).dt.tz_localize(None)
        p["timestamp"] = pd.to_datetime(p["timestamp"]).dt.tz_localize(None)
        return pd.merge_asof(
            n.sort_values("published"), p[["timestamp", "close"]].sort_values("timestamp"),
            left_on="published", right_on="timestamp", direction="nearest",
        )
