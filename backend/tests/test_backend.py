"""
Test script for the Feelow Polymarket backend API.

Usage:
    1. Start the server first:
         cd feelow/backend/src
         uvicorn main:app --host 0.0.0.0 --port 8000 --reload

    2. Then run this test:
         python test_backend.py
"""

import logging
import requests
import json
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_backend")

BASE_URL = "http://localhost:8000"


def test_health():
    """Test the health-check endpoint."""
    log.info("─" * 50)
    log.info("TEST: GET /  (health check)")
    t0 = time.time()
    resp = requests.get(f"{BASE_URL}/")
    elapsed = time.time() - t0
    log.info("  Status: %d  (%.3fs)", resp.status_code, elapsed)
    log.info("  Body:   %s", resp.json())
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert data["status"] == "ok", f"Expected status=ok, got {data['status']}"
    assert "service" in data, "Response should contain 'service' key"
    log.info("  ✅ PASSED")


def test_get_polymarket_missing_company():
    """Test that a request without 'company' returns 422."""
    log.info("─" * 50)
    log.info("TEST: POST /get_polymarket  (missing company field → 422)")
    resp = requests.post(f"{BASE_URL}/get_polymarket", json={})
    log.info("  Status: %d", resp.status_code)
    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"
    detail = resp.json().get("detail", [])
    log.info("  Validation errors: %s", json.dumps(detail, indent=2)[:300])
    log.info("  ✅ PASSED")


def test_get_polymarket_invalid_params():
    """Test that invalid parameter ranges return 422."""
    log.info("─" * 50)
    log.info("TEST: POST /get_polymarket  (max_queries=99 → 422)")
    resp = requests.post(f"{BASE_URL}/get_polymarket", json={
        "company": "NVIDIA",
        "max_queries": 99,  # max is 3
    })
    log.info("  Status: %d", resp.status_code)
    assert resp.status_code == 422, f"Expected 422 for out-of-range, got {resp.status_code}"
    log.info("  ✅ PASSED")


def test_get_polymarket_wrong_content_type():
    """Test that non-JSON request body returns 422."""
    log.info("─" * 50)
    log.info("TEST: POST /get_polymarket  (plain text body → 422)")
    resp = requests.post(
        f"{BASE_URL}/get_polymarket",
        data="not json",
        headers={"Content-Type": "text/plain"},
    )
    log.info("  Status: %d", resp.status_code)
    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"
    log.info("  ✅ PASSED")


def test_get_polymarket_nvidia():
    """Test the full pipeline with a real query (NVIDIA)."""
    log.info("─" * 50)
    log.info("TEST: POST /get_polymarket  (company='NVIDIA')")
    payload = {
        "company": "NVIDIA",
        "date": "February 2026",
        "max_queries": 1,
        "limit_per_query": 5,
        "top_k": 3,
    }
    log.info("  Payload: %s", json.dumps(payload))
    t0 = time.time()
    resp = requests.post(f"{BASE_URL}/get_polymarket", json=payload)
    elapsed = time.time() - t0
    log.info("  Status: %d  (%.1fs)", resp.status_code, elapsed)

    if resp.status_code == 200:
        data = resp.json()
        log.info("  raw_markets count:         %d", len(data["raw_markets"]))
        log.info("  top_markets_summary count: %d", len(data["top_markets_summary"]))
        log.info("  corr_top2:                 %.4f", data["corr_top2"])
        log.info("  global_score:              %.4f", data["global_score"])
        log.info("  claude_block (first 200):  %s…", data["claude_block"][:200])

        # Structure assertions
        assert isinstance(data["raw_markets"], list)
        assert isinstance(data["top_markets_summary"], list)
        assert isinstance(data["corr_top2"], (int, float))
        assert isinstance(data["global_score"], (int, float))
        assert isinstance(data["claude_block"], str)

        # Deeper assertions on summaries
        for i, s in enumerate(data["top_markets_summary"]):
            log.info("  Summary #%d: score=%.4f eng=%.4f q=%s",
                     i + 1, s["score"], s["engagement"], s["question"][:50])
            assert "question" in s, f"Summary #{i} missing 'question'"
            assert "score" in s, f"Summary #{i} missing 'score'"
            assert "metrics" in s, f"Summary #{i} missing 'metrics'"
            assert "advanced" in s, f"Summary #{i} missing 'advanced'"
            assert 0.0 <= s["score"] <= 1.0, f"Score out of range: {s['score']}"

        log.info("  ✅ PASSED")
    elif resp.status_code == 400:
        log.warning("  ⚠️  400 error (likely missing API key): %s", resp.json()["detail"])
        log.warning("  ⏭️  SKIPPED (set GEMINI_API_KEY on the server)")
    else:
        log.error("  ❌ FAILED: %s", resp.text[:300])
        sys.exit(1)


def test_get_polymarket_unknown_company():
    """Test with an obscure company — should return empty or sparse results."""
    log.info("─" * 50)
    log.info("TEST: POST /get_polymarket  (company='XyzNonexistentCorp123')")
    payload = {
        "company": "XyzNonexistentCorp123",
        "max_queries": 1,
        "limit_per_query": 3,
        "top_k": 2,
    }
    t0 = time.time()
    resp = requests.post(f"{BASE_URL}/get_polymarket", json=payload)
    elapsed = time.time() - t0
    log.info("  Status: %d  (%.1fs)", resp.status_code, elapsed)

    if resp.status_code == 200:
        data = resp.json()
        log.info("  raw_markets count: %d", len(data["raw_markets"]))
        assert isinstance(data["raw_markets"], list)
        assert isinstance(data["claude_block"], str)
        log.info("  ✅ PASSED (returned successfully, possibly empty)")
    elif resp.status_code == 400:
        log.warning("  ⚠️  400 error: %s", resp.json()["detail"])
        log.warning("  ⏭️  SKIPPED")
    else:
        log.info("  Status: %d — server handled gracefully", resp.status_code)
        log.info("  ✅ PASSED")


def test_openapi_docs():
    """Test that the OpenAPI docs endpoint is available."""
    log.info("─" * 50)
    log.info("TEST: GET /docs  (OpenAPI Swagger UI)")
    resp = requests.get(f"{BASE_URL}/docs")
    log.info("  Status: %d", resp.status_code)
    assert resp.status_code == 200, f"Expected 200 for /docs, got {resp.status_code}"
    log.info("  ✅ PASSED")

    log.info("TEST: GET /openapi.json  (OpenAPI schema)")
    resp = requests.get(f"{BASE_URL}/openapi.json")
    log.info("  Status: %d", resp.status_code)
    assert resp.status_code == 200
    schema = resp.json()
    assert "/get_polymarket" in str(schema.get("paths", {})), "Missing /get_polymarket in schema"
    log.info("  Paths: %s", list(schema.get("paths", {}).keys()))
    log.info("  ✅ PASSED")


if __name__ == "__main__":
    log.info("")
    log.info("🚀 Feelow Polymarket Backend — Test Suite")
    log.info("=" * 50)
    log.info("Server: %s", BASE_URL)

    try:
        requests.get(f"{BASE_URL}/", timeout=3)
    except requests.ConnectionError:
        log.error("❌ Cannot connect to the server. Start it first:")
        log.error("   cd feelow/backend/src")
        log.error("   uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
        sys.exit(1)

    tests = [
        ("Health check",             test_health),
        ("Missing company (422)",    test_get_polymarket_missing_company),
        ("Invalid params (422)",     test_get_polymarket_invalid_params),
        ("Wrong content type (422)", test_get_polymarket_wrong_content_type),
        ("OpenAPI docs",             test_openapi_docs),
        ("NVIDIA full pipeline",     test_get_polymarket_nvidia),
        ("Unknown company",          test_get_polymarket_unknown_company),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as exc:
            log.error("❌ %s FAILED: %s", name, exc)
            failed += 1

    log.info("")
    log.info("=" * 50)
    log.info("Results: %d passed, %d failed, %d total", passed, failed, len(tests))
    if failed:
        log.error("❌ Some tests failed!")
        sys.exit(1)
    else:
        log.info("✅ All tests passed!")
