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

See [backend/README.md](backend/README.md) for full API reference and architecture details.
