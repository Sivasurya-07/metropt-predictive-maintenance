"""
Unit tests for Phase 3 Streaming & API components.
Uses FastAPI's TestClient for synchronous endpoint testing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

# ─── Sample sensor reading payload ───────────────────────────────────────────

SAMPLE_READING = {
    "timestamp": "2020-04-01T08:00:00",
    "TP2": 8.5,
    "TP3": 10.2,
    "H1": 0.0,
    "DV_pressure": 0.0,
    "Reservoirs": 8.5,
    "Oil_temperature": 56.0,
    "Motor_current": 12.3,
    "COMP": 1.0,
    "DV_eletric": 1.0,
    "TOWERS": 1.0,
    "MPG": 1.0,
    "LPS": 0.0,
    "Pressure_switch": 1.0,
    "Oil_level": 1.0,
    "Flowmeter": 4.5,
}


# ─── Health endpoint ──────────────────────────────────────────────────────────

def test_health_endpoint():
    """GET /health should return 200 with status=healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


# ─── Predict endpoint ─────────────────────────────────────────────────────────

def test_predict_endpoint_returns_horizons():
    """POST /predict should return three horizon predictions."""
    payload = {"readings": [SAMPLE_READING], "horizon": "4h"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert len(data["horizons"]) == 3
    horizons = [h["horizon"] for h in data["horizons"]]
    assert "2h" in horizons
    assert "4h" in horizons
    assert "8h" in horizons


def test_predict_horizon_fields():
    """Each horizon prediction must include required fields with valid ranges."""
    payload = {"readings": [SAMPLE_READING], "horizon": "2h"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    for h in response.json()["horizons"]:
        assert 0.0 <= h["failure_probability"] <= 1.0
        assert 0.0 <= h["confidence"] <= 1.0
        assert h["alert_level"] in {"normal", "warning", "critical", "emergency"}


def test_predict_invalid_horizon():
    """POST /predict with invalid horizon should return 422."""
    payload = {"readings": [SAMPLE_READING], "horizon": "12h"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_sensor_field():
    """POST /predict with missing required field should return 422."""
    bad_reading = {k: v for k, v in SAMPLE_READING.items() if k != "TP2"}
    payload = {"readings": [bad_reading], "horizon": "4h"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_returns_narrative():
    """POST /predict should return a non-empty narrative string."""
    payload = {"readings": [SAMPLE_READING], "horizon": "4h"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert isinstance(response.json()["narrative"], str)
    assert len(response.json()["narrative"]) > 0


# ─── Streaming Engine ─────────────────────────────────────────────────────────

def test_replay_engine_in_memory():
    """SensorReplayEngine should publish rows to in-memory buffer without MQTT."""
    from src.streaming.replay_engine import SensorReplayEngine
    from src import config

    engine = SensorReplayEngine(
        csv_path=config.RAW_DATA_DIR / config.CSV_FILENAME,
        speed_multiplier=99999.0,
    )
    engine.start(max_rows=5, blocking=True)
    assert len(engine.published_rows) == 5


def test_mqtt_client_direct_push():
    """MQTTSensorClient.push_direct fills the deque; callback fires on every push once full.
    With window_size=3 and exactly 3 pushes, the callback fires exactly once."""
    from src.streaming.mqtt_client import MQTTSensorClient

    received = []
    client_obj = MQTTSensorClient(
        window_size=3,
        on_window_ready=lambda w: received.append(w),
    )
    for _ in range(3):
        client_obj.push_direct(SAMPLE_READING)

    assert len(received) == 1  # fires exactly once when window reaches capacity
    assert len(received[0]) == 3
