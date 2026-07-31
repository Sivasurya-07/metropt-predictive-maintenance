"""
FastAPI application factory for the MetroPT APU Inference Service.

Startup: pre-loads ensemble model into cache.
CORS: enabled for all origins (dashboard integration).
Docs: available at /docs and /redoc.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preloads the model and narrative generator at startup, and starts Redis listener."""
    from src.api.routes import manager
    listener_task = asyncio.create_task(manager.start_listener())
    flusher_task = asyncio.create_task(parquet_flusher_loop())
    
    print("[API] Loading ensemble model into memory...")
    get_ensemble_model()
    print("[API] Model ready. Inference service online.")
    yield
    listener_task.cancel()
    flusher_task.cancel()
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

# ─── Runtime Diagnostic (proves what code is deployed) ────────────────────────
BUILD_ID = "2026-07-31T23:05-ws-debug"

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
