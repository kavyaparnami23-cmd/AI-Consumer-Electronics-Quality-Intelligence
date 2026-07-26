"""
test_all_frontend_apis.py
Verify all 6 frontend-connected FastAPI endpoints respond with 200 OK.
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_apis():
    # 1. /classic/predict
    res1 = requests.post(f"{BASE_URL}/classic/predict", json={
        "features": {
            "air_temperature": 298.1,
            "process_temperature": 308.6,
            "rotational_speed": 1551.0,
            "torque": 42.8,
            "tool_wear": 0.0,
            "type_H": 0.0,
            "type_L": 0.0,
            "type_M": 1.0
        }
    })
    print("1. /classic/predict status:", res1.status_code)
    data1 = res1.json()
    print("   SHAP Values present:", "shap_values" in data1, "Top features count:", len(data1.get("top_features", [])))
    assert res1.status_code == 200

    # 2. /classic/predict/batch
    res2 = requests.post(f"{BASE_URL}/classic/predict/batch", json={
        "samples": [
            {
                "air_temperature": 298.1,
                "process_temperature": 308.6,
                "rotational_speed": 1551.0,
                "torque": 42.8,
                "tool_wear": 0.0,
                "type_M": 1.0
            },
            {
                "air_temperature": 305.0,
                "process_temperature": 315.0,
                "rotational_speed": 2800.0,
                "torque": 65.0,
                "tool_wear": 200.0,
                "type_H": 1.0
            }
        ]
    })
    print("2. /classic/predict/batch status:", res2.status_code, "Count:", res2.json().get("count"))
    assert res2.status_code == 200

    # 3. /dl/predict
    res3 = requests.post(f"{BASE_URL}/dl/predict", json={
        "features": [0.1, -0.2, 0.5, 0.3, -0.1, 0.8, 0.4, 0.2],
        "model": "lstm"
    })
    print("3. /dl/predict status:", res3.status_code, "Model used:", res3.json().get("model_used"))
    assert res3.status_code == 200

    # 4. /nlp/sentiment
    res4 = requests.post(f"{BASE_URL}/nlp/sentiment", json={
        "text": "The motor runs smoothly with excellent heat dissipation.",
        "model": "distilbert"
    })
    print("4. /nlp/sentiment status:", res4.status_code, "Sentiment:", res4.json().get("sentiment"))
    assert res4.status_code == 200

    # 5. /timeseries/anomaly
    res5 = requests.post(f"{BASE_URL}/timeseries/anomaly", json={
        "window": [[298.1, 308.6, 1550.0, 42.0, 0.0, 0.0, 0.0, 0.0] for _ in range(30)]
    })
    print("5. /timeseries/anomaly status:", res5.status_code, "Anomaly:", res5.json().get("is_anomaly"))
    assert res5.status_code == 200

    # 6. /timeseries/anomaly/isolation-forest
    res6 = requests.post(f"{BASE_URL}/timeseries/anomaly/isolation-forest", json={
        "window": [[298.1, 308.6, 1550.0, 42.0, 0.0, 0.0, 0.0, 0.0] for _ in range(30)]
    })
    print("6. /timeseries/anomaly/isolation-forest status:", res6.status_code, "Anomaly:", res6.json().get("is_anomaly"))
    assert res6.status_code == 200

    print("\n✅ ALL 6 API ENDPOINTS CONNECTED & PASSING 100%!")

if __name__ == "__main__":
    test_apis()
