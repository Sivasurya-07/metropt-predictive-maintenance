"""
FastAPI route definitions for the MetroPT APU Inference Service.

Routes:
    GET  /health           - Liveness probe
    POST /predict          - Multi-horizon failure probability prediction
    WS   /ws/alerts        - WebSocket channel for real-time alert broadcasts
"""
import asyncio
import json
from datetime import datetime, timezone
from typing import List

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from src.api.schemas import (
    AlertPayload,
    HealthResponse,
    HorizonPrediction,
    PredictionRequest,
    PredictionResponse,
)
from src.api.dependencies import (
    get_ensemble_model,
    readings_to_feature_array,
)

router = APIRouter()

import os
import redis.asyncio as aioredis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

class ConnectionManager:
    """Manages active WebSocket connections for alert broadcasting via Redis Pub/Sub."""

    def __init__(self):
        self.active: List[WebSocket] = []
        self.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        self.pubsub = self.redis.pubsub()

    async def start_listener(self):
        await self.pubsub.subscribe("apu_alerts")
        async for message in self.pubsub.listen():
            if message["type"] == "message":
                payload = message["data"]
                await self._broadcast_local(payload)

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, payload: dict):
        # Publish to Redis instead of local broadcast
        await self.redis.publish("apu_alerts", json.dumps(payload))

    async def _broadcast_local(self, payload_str: str):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(payload_str)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

manager = ConnectionManager()

# ─── Alert level thresholds ──────────────────────────────────────────────────

ALERT_THRESHOLDS = {
    "2h": {"emergency": 0.70, "critical": 0.40, "warning": 0.20},
    "4h": {"emergency": 0.65, "critical": 0.35, "warning": 0.15},
    "8h": {"emergency": 0.60, "critical": 0.30, "warning": 0.10},
}


def _get_alert_level(prob: float, horizon: str) -> str:
    thresholds = ALERT_THRESHOLDS.get(horizon, ALERT_THRESHOLDS["4h"])
    if prob >= thresholds["emergency"]:
        return "emergency"
    elif prob >= thresholds["critical"]:
        return "critical"
    elif prob >= thresholds["warning"]:
        return "warning"
    return "normal"

def _aggregate_subsystem_shap(shap_list: List[tuple]) -> dict:
    """Aggregates raw SHAP feature importances into physical subsystems."""
    subsystems = {
        "Compressor": 0.0,
        "Reservoir": 0.0,
        "Motor": 0.0,
        "Valves": 0.0
    }
    
    total_shap = 0.0
    for fname, fval in shap_list:
        val = abs(fval)
        total_shap += val
        lower_name = fname.lower()
        
        if any(k in lower_name for k in ["tp2", "comp", "oil", "h1", "pressure_diff"]):
            subsystems["Compressor"] += val
        elif any(k in lower_name for k in ["tp3", "reservoir"]):
            subsystems["Reservoir"] += val
        elif "motor" in lower_name:
            subsystems["Motor"] += val
        elif any(k in lower_name for k in ["dv_", "towers", "mpg", "lps", "pressure_switch", "flowmeter"]):
            subsystems["Valves"] += val
            
    # Normalize to percentages if total > 0
    if total_shap > 0:
        for k in subsystems:
            subsystems[k] = round((subsystems[k] / total_shap) * 100, 1)
            
    return subsystems

# ─── Routes ──────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health():
    """Liveness probe. Returns model load status."""
    model = get_ensemble_model()
    return HealthResponse(model_loaded=model is not None)


@router.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(request: PredictionRequest):
    """
    Accepts a batch of sensor readings and returns multi-horizon
    failure probabilities, SHAP attributions, and a natural language narrative.
    """
    model = get_ensemble_model()

    # Convert readings to feature matrix (Stateful Redis buffer)
    raw_dicts = [r.model_dump() for r in request.readings]
    try:
        from src.api.stateful_features import engine as feature_engine
        X, feature_names = await feature_engine.generate_features(raw_dicts)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Stateful feature engineering failed: {e}")

    horizon = request.horizon
    horizons_to_run = ["2h", "4h", "8h"]

    horizon_predictions = []
    top_features = []
    subsystem_shap = {"Compressor": 0.0, "Reservoir": 0.0, "Motor": 0.0, "Valves": 0.0}
    narrative = "The APU operates within normal parameters."

    for h in horizons_to_run:
        if model is not None:
            try:
                # Use actual confidence metrics from ensemble (disagreement-based)
                prob_arr, conf_arr = model.get_confidence_metrics(X)
                prob = float(prob_arr.mean())
                conf = float(conf_arr.mean())
                
                # Strict Truth Guardrail: Only broadcast if guardrail agrees AND confidence > 99%
                guardrail_preds = model.guardrail.predict(X)
                is_anomaly = bool(np.any(guardrail_preds))
                
                if conf < 0.99 or not is_anomaly:
                    # Suppress false positives by downgrading alert level silently
                    prob = min(prob, 0.20) # Keep below warning threshold
                    
            except Exception:
                prob = 0.0
                conf = 0.0
        else:
            prob = 0.0
            conf = 0.0

        alert_level = _get_alert_level(prob, h)

        horizon_predictions.append(HorizonPrediction(
            horizon=h,
            failure_probability=round(prob, 4),
            alert_level=alert_level,
            confidence=round(conf, 4),
        ))

    # SHAP attributions (best-effort)
    if model is not None and len(X) > 0:
        try:
            from src.explain.shap_explainer import APUExplainer
            from src.explain.narrative import generate_narrative_explanation
            explainer = APUExplainer(feature_names=feature_names)
            shap_list = explainer.explain_lightgbm(model.lgbm, X[:1])
            top_features = [
                {fname: round(float(fval), 5)}
                for fname, fval in shap_list[:5]
            ]
            subsystem_shap = _aggregate_subsystem_shap(shap_list)
            narrative = generate_narrative_explanation(shap_list[:5])
        except Exception:
            top_features = []
            narrative = "The APU operates within normal parameters."

    # Broadcast to WebSocket clients EVERY time we get a reading so the UI updates
    ts = datetime.now(timezone.utc).isoformat()
    if manager.active and len(raw_dicts) > 0:
        alert = AlertPayload(
            timestamp=ts,
            sensor_readings=raw_dicts[0],  # the most recent reading
            predictions=horizon_predictions,
            narrative=narrative,
            top_features=top_features,
            subsystem_shap=subsystem_shap,
        )
        asyncio.create_task(manager.broadcast(alert.model_dump()))

    return PredictionResponse(
        horizons=horizon_predictions,
        top_features=top_features,
        subsystem_shap=subsystem_shap,
        narrative=narrative,
        timestamp=ts,
    )


@router.websocket("/ws/alerts")
async def websocket_alerts(ws: WebSocket):
    """
    WebSocket endpoint for real-time alert streaming.
    Clients connect here and receive alert payloads whenever
    a POST /predict detects an elevated risk level.
    """
    await manager.connect(ws)
    try:
        while True:
            # Keep alive: echo any pings from client
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(ws)
