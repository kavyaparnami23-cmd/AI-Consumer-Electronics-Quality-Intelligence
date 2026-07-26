"""
test_api.py
───────────
Comprehensive test suite verifying every FastAPI endpoint.
Includes verification for /timeseries/anomaly and SHAP explanations.
"""

import json
import sys
import random
import httpx

BASE_URL = "http://localhost:8000"
client = httpx.Client(base_url=BASE_URL, timeout=30)

PASS = "[PASS]"
FAIL = "[FAIL]"

results = []


def check(name: str, response: httpx.Response, expected_status: int = 200):
    ok = response.status_code == expected_status
    status_str = PASS if ok else FAIL
    print(f"{status_str} [{response.status_code}] {name}")
    if not ok:
        print(f"       Response error: {response.text[:400]}")
    else:
        try:
            body = response.json()
            snippet = json.dumps(body, indent=2)
            if len(snippet) > 350:
                snippet = snippet[:350] + "\n  ..."
            print(f"       {snippet}")
        except Exception:
            pass
    results.append(ok)
    return response


# ── 1. System endpoints ───────────────────────────────────────────────────────
print("\n=== SYSTEM ENDPOINTS ===")
check("GET /", client.get("/"))
check("GET /health", client.get("/health"))
r = check("GET /models", client.get("/models"))


# ── 2. Classic ML with SHAP ───────────────────────────────────────────────────
print("\n=== CLASSICAL ML WITH SHAP EXPLANATIONS ===")
r_classic = check(
    "POST /classic/predict (with SHAP explanations)",
    client.post("/classic/predict", json={
        "features": {
            "air_temperature": 298.1,
            "process_temperature": 308.6,
            "rotational_speed": 1551.0,
            "torque": 42.8,
            "tool_wear": 0.0,
            "type_H": 0.0,
            "type_L": 0.0,
            "type_M": 1.0,
        }
    })
)

if r_classic.status_code == 200:
    data = r_classic.json()
    assert "shap_values" in data, "shap_values missing from response"
    assert "top_features" in data, "top_features missing from response"
    print("       SHAP key check passed: shap_values and top_features present in response!")

check(
    "POST /classic/predict/batch",
    client.post("/classic/predict/batch", json={
        "samples": [
            [1.0, 298.1, 308.6, 1551.0, 42.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.5, 66382.8, 0.0],
            [2.0, 305.0, 315.0, 2800.0, 75.0, 240.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 210000.0, 0.0857],
        ]
    })
)


# ── 3. Deep Learning ──────────────────────────────────────────────────────────
print("\n=== DEEP LEARNING ENDPOINTS ===")
check(
    "POST /dl/predict (LSTM)",
    client.post("/dl/predict", json={
        "features": [0.1, -0.2, 0.5, 0.3, -0.1, 0.8, 0.4, 0.2],
        "model": "lstm"
    })
)

check(
    "POST /dl/predict (CNN)",
    client.post("/dl/predict", json={
        "features": [0.1, -0.2, 0.5, 0.3, -0.1, 0.8, 0.4, 0.2],
        "model": "cnn"
    })
)


# ── 4. NLP / Sentiment ────────────────────────────────────────────────────────
print("\n=== NLP SENTIMENT ENDPOINTS ===")
check(
    "POST /nlp/sentiment (TF-IDF)",
    client.post("/nlp/sentiment", json={
        "text": "Excellent quality product! Performs beyond expectations.",
        "model": "tfidf"
    })
)

check(
    "POST /nlp/sentiment (DistilBERT)",
    client.post("/nlp/sentiment", json={
        "text": "Outstanding build quality and fast shipping.",
        "model": "distilbert"
    })
)


# ── 5. Time Series Anomaly Detection ──────────────────────────────────────────
print("\n=== TIME SERIES ANOMALY ENDPOINTS ===")

random.seed(42)
test_window = [[random.gauss(0, 0.2) for _ in range(8)] for _ in range(30)]

check(
    "POST /ts/anomaly (LSTM Autoencoder)",
    client.post("/ts/anomaly", json={"window": test_window})
)

check(
    "POST /timeseries/anomaly (Explicit /timeseries/anomaly endpoint)",
    client.post("/timeseries/anomaly", json={"window": test_window})
)

check(
    "POST /timeseries/anomaly/isolation-forest",
    client.post("/timeseries/anomaly/isolation-forest", json={"window": test_window})
)


# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(results)
total = len(results)
print(f"\n{'='*60}")
print(f"FINAL TEST SUMMARY: {passed}/{total} endpoints passed successfully")
print(f"{'='*60}\n")

if passed != total:
    sys.exit(1)
