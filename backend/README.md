# Feelow Backend 🔧

FastAPI backend powering Feelow's Polymarket analysis pipeline.

## Architecture

```
backend/
├── src/
│   ├── main.py                          # FastAPI app + endpoints
│   ├── full_pipeline.py                 # get_polymarket() — glue between modules
│   ├── agent-search/
│   │   └── polymarket_pipeline.py       # Gemini LLM → Polymarket search + pertinence scoring
│   ├── polymarket-analysis/
│   │   └── score_polymarket.py          # Market class, advanced metrics, ranking
│   ├── stock-analysis/                  # (reserved for future stock analysis)
│   └── config/                          # (reserved for configuration)
└── tests/
    ├── test_full_pipeline.py            # Unit tests for full_pipeline (mocked LLM)
    ├── test_score_polymarket.py         # Unit tests for scoring module
    └── test_backend.py                  # Integration tests for the HTTP API
```

## How It Works

The backend exposes a single main endpoint — `POST /get_polymarket` — that runs a **two-step pipeline**:

### Step 1 — Agent Search (`agent-search/polymarket_pipeline.py`)

Uses **Gemini LLM** with forced tool-calling to:
1. Build varied search queries from a company name (e.g. `"NVIDIA"`)
2. Call the Polymarket API to find related prediction markets
3. Score each market's **pertinence** (0–100) via structured LLM output

### Step 2 — Advanced Scoring (`polymarket-analysis/score_polymarket.py`)

Takes the raw markets and computes:
- **Momentum** — recent price trend direction
- **Volatility** — price range over history
- **Concentration** — probability skew (entropy-based)
- **Composite signal** — direction + strength scaled by pertinence, liquidity, and history quality
- **Correlation** between top markets
- **Claude-ready summary block** — text to inject into LLM prompts

### Glue — `full_pipeline.py`

The `get_polymarket()` function connects both modules:
- Handles the **format bridge** between `price_history` (agent-search) and `history` (scoring)
- Normalises pertinence from the 0–100 scale to 0–1
- Returns a unified result with raw data, scored summaries, and the Claude block

## Setup

### Prerequisites

- Python 3.10+
- A **Gemini API key** (`GEMINI_API_KEY` or `GOOGLE_API_KEY` env var)

### Install dependencies

```bash
pip install fastapi uvicorn google-genai mcp pydantic numpy requests
```

### Run the server

```bash
cd backend/src
GEMINI_API_KEY=your_key_here uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## API Reference

### `GET /`

Health check.

```json
{ "status": "ok", "service": "Feelow Polymarket API" }
```

### `POST /get_polymarket`

Run the full analysis pipeline.

**Request body:**

| Field             | Type   | Default | Description                            |
|-------------------|--------|---------|----------------------------------------|
| `company`         | string | —       | Company name (e.g. `"NVIDIA"`)         |
| `date`            | string | `null`  | Date context (e.g. `"February 2026"`)  |
| `max_queries`     | int    | `1`     | Search query variations (1–3)          |
| `limit_per_query` | int    | `10`    | Max markets per search query (1–50)    |
| `top_k`           | int    | `5`     | Top markets in the summary (1–20)      |

**Example:**

```bash
curl -X POST http://localhost:8000/get_polymarket \
  -H "Content-Type: application/json" \
  -d '{"company": "NVIDIA", "date": "February 2026", "top_k": 3}'
```

**Response:**

| Field                  | Type   | Description                                    |
|------------------------|--------|------------------------------------------------|
| `raw_markets`          | list   | All markets found by agent-search              |
| `top_markets_summary`  | list   | Scored summaries for the top-k markets         |
| `corr_top2`            | float  | Pearson correlation between top 2 markets      |
| `global_score`         | float  | Weighted global score                          |
| `claude_block`         | string | Text block ready to inject into an LLM prompt  |

## Testing

### Unit tests (no server or API key needed)

```bash
cd backend
python -m pytest tests/test_score_polymarket.py tests/test_full_pipeline.py -v
```

### Integration tests (requires running server)

```bash
# Terminal 1: start the server
cd backend/src
GEMINI_API_KEY=your_key uvicorn main:app --port 8000

# Terminal 2: run the tests
cd backend
python tests/test_backend.py
```

### Test coverage

| Test file                  | Tests | What's covered                                       |
|----------------------------|-------|------------------------------------------------------|
| `test_score_polymarket.py` | 54    | Helpers, Market class, correlation, scoring pipeline  |
| `test_full_pipeline.py`    | 17    | Format bridge, pertinence normalisation, mocked flow  |
| `test_backend.py`          | 4     | HTTP endpoints, validation, full E2E                  |
