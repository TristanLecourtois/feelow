# Feelow 🦈

Personal Finance Agent based on Polymarket Monitoring

## Overview

Feelow analyses prediction markets on [Polymarket](https://polymarket.com) to generate financial insights for any publicly traded company. It combines LLM-powered search with quantitative scoring to surface the most relevant and active markets.

## Project Structure

```
feelow/
├── backend/          # FastAPI server + analysis pipeline
│   ├── src/          # Source code
│   └── tests/        # Unit & integration tests
└── frontend/         # (coming soon)
```

## Backend

The backend exposes a REST API that runs a two-step pipeline:

1. **Agent Search** — Gemini LLM searches Polymarket for prediction markets related to a company
2. **Advanced Scoring** — computes momentum, volatility, concentration, composite signal, and generates LLM-ready summaries

### Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn google-genai mcp pydantic numpy requests

# Run the server
cd backend/src
GEMINI_API_KEY=your_key uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Usage

```bash
curl -X POST http://localhost:8000/get_polymarket \
  -H "Content-Type: application/json" \
  -d '{"company": "NVIDIA", "date": "February 2026", "top_k": 3}'
```

### Tests

```bash
cd backend
python -m pytest tests/ -v
```

````bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
````

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

See [backend/README.md](backend/README.md) for full API reference and architecture details.



## 📊 Features

| Feature | Description | Source Repo |
|---------|-------------|-------------|
| FinBERT Sentiment | Financial text sentiment classification | ProsusAI/finBERT |
| Multi-Model Ensemble | 3 models voting for robust predictions | nickmuchi/finbert-tone, Sigma/financial-SA |
| Real-Time RSS Ingestion | Yahoo Finance + Finviz headlines | nlp-sentiment-quant-monitor |
| Candlestick + Overlay | Price chart with sentiment scatter | nlp-sentiment-quant-monitor |
| Technical Indicators | SMA, EMA, RSI, MACD, Bollinger | nlp-finance-forecast |
| Claude AI Reasoning | Deep analysis combining all signals | Anthropic Claude API |
| Model Comparison | Side-by-side model benchmarking | Custom |

---

## Expert Models Used

| Model | HuggingFace ID | F1 Score | Best For |
|-------|---------------|----------|----------|
| **FinBERT (ProsusAI)** | `ProsusAI/finbert` | ~87% | General financial sentiment |
| **FinBERT-Tone** | `nickmuchi/finbert-tone` | ~90% | Tone detection (analyst reports) |
| **Sigma Financial SA** | `Sigma/financial-sentiment-analysis` | ~98% | High-accuracy classification |

---

## Project Structure

```
feelow/
├── backend/                          # FastAPI unified API (port 8000)
│   ├── src/
│   │   ├── main.py                   # FastAPI app — all endpoints
│   │   ├── config.py                 # Central config (models, tickers, thresholds)
│   │   ├── full_pipeline.py          # Polymarket pipeline glue (agent-search → scoring)
│   │   ├── finance-data/             # Core financial modules
│   │   │   ├── sentiment_engine.py   # Multi-model FinBERT ensemble
│   │   │   ├── news_ingestor.py      # RSS headline fetching
│   │   │   ├── market_data.py        # yfinance price data loader
│   │   │   ├── technicals.py         # RSI, MACD, Bollinger, SMA, EMA
│   │   │   ├── gemini_agent.py       # Google Gemini search grounding agent
│   │   │   └── agent_orchestrator.py # Multi-step agentic pipeline orchestrator
│   │   ├── agent_search/             # Polymarket LLM search
│   │   │   ├── polymarket_pipeline.py
│   │   │   ├── orchestrator.py
│   │   │   └── scoring/              # Relevance, impact, novelty, sentiment, reliability
│   │   ├── polymarket-analysis/      # Advanced market scoring
│   │   │   └── market_scorer.py      # Momentum, volatility, concentration, composite signal
│   │   └── stock_analysis/           # Reddit-based FinBERT sentiment
│   │       └── api_finbert_transformer.py
│   └── tests/
└── webapp/
    └── UI-fr/                        # Next.js 15 dashboard (port 3000)
        ├── app/dashboard/page.tsx    # Main dashboard page
        ├── lib/ticker-context.tsx    # Global ticker state + API calls
        └── components/
            ├── section-cards.tsx           # KPI cards (price, sentiment, RSI, signal)
            ├── chart-area-interactive.tsx  # OHLCV price chart + Polymarket panel
            ├── data-table.tsx              # News headlines with sentiment badges
            └── app-sidebar.tsx             # Ticker selector (Tech / Finance / Crypto)
```

## 🏆 Hackathon Prize Targeting

- **Best Use of Data (Susquehanna €7K)** — Turns raw news + price data into trading signals
- **Best Use of Gemini (€50K credits)** — Can extend with Gemini multimodal (video/image analysis)
- **Best Stripe Integration (€3K)** — Ready for Stripe Agent Toolkit monetisation layer
- **Fintech Track (€1K)** —

---

## 👥 Team

- **Gabriel Dupuis** — ML Engineer @ Deezer, ENSTA, Stanford
- **Adrien Scazzola** — Security & AI, Microsoft, 
- **Amine Ould** — Development ENS-MVA
- **Tristan Lecourtois** — NASA, Systems Engineering- ENS MVA

---

## License

MIT — Built for HackEurope 2026 with love
