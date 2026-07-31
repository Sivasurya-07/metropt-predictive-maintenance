"""
Pydantic v2 schemas for the MetroPT APU Inference API.
Defines request/response models for sensor readings,
predictions, and WebSocket alert payloads.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """Single sensor telemetry reading from the APU."""
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    TP2: float = Field(..., description="Compressor pressure (bar)")
    TP3: float = Field(..., description="Reservoir pressure (bar)")
    H1: float = Field(..., description="Oil temperature (°C)")
    DV_pressure: float = Field(..., description="Pressure relief valve state")
    Reservoirs: float = Field(..., description="Air reservoir pressure (bar)")
    Oil_temperature: float = Field(..., description="Oil temperature sensor 2 (°C)")
    Motor_current: float = Field(..., description="Compressor motor current (A)")
    COMP: float = Field(..., description="Compressor on/off state (digital)")
    DV_eletric: float = Field(..., description="Electric valve state (digital)")
    TOWERS: float = Field(..., description="Desiccant towers state (digital)")
    MPG: float = Field(..., description="Main pneumatic gate state (digital)")
    LPS: float = Field(..., description="Low pressure switch state (digital)")
    Pressure_switch: float = Field(..., description="Pressure switch state (digital)")
    Oil_level: float = Field(..., description="Oil level sensor (digital)")
    Flowmeter: float = Field(..., description="Air flowmeter reading")


class PredictionRequest(BaseModel):
    """Batch of sensor readings for multi-horizon inference."""
    readings: List[SensorReading] = Field(
        ...,
        min_length=1,
        description="Sequence of sensor readings (minimum 1, ideally 60 for CNN)"
    )
    horizon: str = Field(
        default="4h",
        pattern="^(2h|4h|8h)$",
        description="Prediction horizon: '2h', '4h', or '8h'"
    )


class HorizonPrediction(BaseModel):
    """Prediction for a single time horizon."""
    horizon: str
    failure_probability: float = Field(..., ge=0.0, le=1.0)
    alert_level: str = Field(..., description="'normal' | 'warning' | 'critical' | 'emergency'")
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Full multi-horizon prediction response with explanations."""
    status: str = "ok"
    horizons: List[HorizonPrediction]
    top_features: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Top SHAP feature attributions"
    )
    subsystem_shap: Dict[str, float] = Field(
        default_factory=dict,
        description="Aggregated SHAP impact per APU physical subsystem"
    )
    narrative: str = Field(
        default="",
        description="Natural language diagnostic summary"
    )
    timestamp: str = ""


class AlertPayload(BaseModel):
    """WebSocket broadcast alert payload."""
    timestamp: str
    sensor_readings: Dict[str, Any]
    predictions: List[HorizonPrediction]
    narrative: str
    top_features: List[Dict[str, float]]
    subsystem_shap: Dict[str, float]


class HealthResponse(BaseModel):
    """API liveness probe response."""
    status: str = "healthy"
    model_loaded: bool
    version: str = "1.0.0"
