"""
FastAPI application factory for the MetroPT APU Inference Service.

Startup: pre-loads ensemble model into cache.
CORS: enabled for all origins (dashboard integration).
Docs: available at /docs and /redoc.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.api.dependencies import get_ensemble_model


import asyncio

async def parquet_flusher_loop():
    """Periodically flushes telemetry from Redis to Parquet for archival."""
    from src.api.stateful_features import engine as feature_engine
    while True:
        await asyncio.sleep(300) # Flush every 5 minutes
        try:
            await feature_engine.flush_to_parquet()
        except Exception as e:
            print(f"[Archiver] Parquet flush failed: {e}")

async def auto_telemetry_simulator_loop():
    """Generates continuous background APU telemetry directly in the cloud 24/7."""
    import math, random
    from datetime import datetime, timezone
    from src.api.schemas import PredictionRequest, SensorReading
    from src.api.routes import predict
    
    t_step = 0
    await asyncio.sleep(5) # Wait for model to load
    print("[AutoSimulator] Starting 24/7 cloud telemetry stream...")
    while True:
        await asyncio.sleep(2) # Emit telemetry reading every 2 seconds
        try:
            base_tp2 = 8.0 + math.sin(t_step / 10.0) * 2.0 + random.uniform(-0.1, 0.1)
            base_tp3 = 7.5 + math.cos(t_step / 15.0) * 1.5 + random.uniform(-0.1, 0.1)
            base_h1 = 8.2 + math.sin(t_step / 8.0) * 1.0 + random.uniform(-0.1, 0.1)
            
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "TP2": base_tp2,
                "TP3": base_tp3,
                "H1": base_h1,
                "DV_pressure": 0.2 + random.uniform(-0.05, 0.05),
                "Reservoirs": 8.0 + random.uniform(-0.1, 0.1),
                "Motor_current": 7.5 + math.sin(t_step / 20.0) * 0.5,
                "Oil_temperature": 65.0 + math.sin(t_step / 30.0) * 5.0,
                "COMP": 1.0 if t_step % 100 < 50 else 0.0,
                "DV_eletric": 0.0,
                "TOWERS": 0.0,
                "MPG": 1.0,
                "LPS": 0.0,
                "Pressure_switch": 1.0,
                "Oil_level": 1.0,
                "Flowmeter": 1.0,
            }
            
            req = PredictionRequest(readings=[SensorReading(**payload)], horizon="4h")
            await predict(req)
            t_step += 1
        except Exception as e:
            print(f"[AutoSimulator] Error generating telemetry: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preloads the model and narrative generator at startup, and starts Redis listener + cloud simulator."""
    from src.api.routes import manager
    listener_task = asyncio.create_task(manager.start_listener())
    flusher_task = asyncio.create_task(parquet_flusher_loop())
    simulator_task = asyncio.create_task(auto_telemetry_simulator_loop())
    
    print("[API] Loading ensemble model into memory...")
    get_ensemble_model()
    print("[API] Model ready. Inference service online.")
    yield
    listener_task.cancel()
    flusher_task.cancel()
    simulator_task.cancel()
    print("[API] Shutting down inference service.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="MetroPT APU Early-Warning Inference API",
        description=(
            "Real-time multi-horizon (2h/4h/8h) failure probability predictions "
            "for railway Air Production Unit (APU) compressors. "
            "Powered by a stacking ensemble of LightGBM, XGBoost, 1D-CNN with "
            "TreeSHAP explainability and natural language diagnostic summaries."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()

# Import websocket handler and register directly on app for bulletproof routing
from src.api.routes import websocket_alerts
app.add_api_websocket_route("/ws/alerts", websocket_alerts)
app.add_api_websocket_route("/ws/alerts/", websocket_alerts)


# ─── Runtime Diagnostic (proves what code is deployed) ────────────────────────
BUILD_ID = "2026-07-31T23:20-direct-ws"

@app.get("/debug/routes", tags=["Debug"])
async def debug_routes():
    """Lists every route registered on the running application — including the included router."""
    from src.api.routes import router as _router
    
    # Enumerate the router's own routes (the authoritative source)
    router_routes = []
    for r in _router.routes:
        router_routes.append({
            "type": type(r).__name__,
            "path": getattr(r, 'path', '?'),
            "methods": list(getattr(r, 'methods', [])) if getattr(r, 'methods', None) else "websocket",
        })

    # Enumerate app-level routes
    app_routes = []
    for r in app.routes:
        rtype = type(r).__name__
        if not hasattr(r, 'routes'):
            app_routes.append({
                "type": rtype,
                "path": getattr(r, 'path', '?'),
                "methods": list(getattr(r, 'methods', [])) if getattr(r, 'methods', None) else None,
            })

    return {
        "build_id": BUILD_ID,
        "router_routes": router_routes,
        "app_routes": app_routes,
    }

@app.get("/debug/headers", tags=["Debug"])
async def debug_headers(request: Request):
    """Inspects incoming headers passed by Railway proxy."""
    return {
        "headers": dict(request.headers),
        "url": str(request.url),
        "base_url": str(request.base_url),
        "client": request.client.host if request.client else None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
