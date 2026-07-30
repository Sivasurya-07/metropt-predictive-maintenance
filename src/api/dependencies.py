"""
FastAPI dependency providers for the MetroPT inference service.
Handles singleton model loading and feature engineering injection.
"""
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from src import config
from src.data.features import engineer_features
from src.models.ensemble import EnsembleModel


@lru_cache(maxsize=1)
def get_ensemble_model(model_path: Optional[str] = None) -> Optional[EnsembleModel]:
    """
    Loads the trained EnsembleModel once and caches it for the process lifetime.
    Returns None if no saved model is found (cold-start / first-run).
    """
    path = Path(model_path) if model_path else config.MODELS_DIR / "ensemble_4h.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        model = pickle.load(f)
        
    # Inject ONNX models if available for blazing-fast inference
    onnx_path = config.MODELS_DIR / "lgbm_4h.onnx"
    if onnx_path.exists():
        model.lgbm.load_onnx(onnx_path)
        
    return model



def readings_to_feature_array(readings: list[dict]) -> tuple[np.ndarray, list[str]]:
    """
    Converts a list of raw sensor reading dicts to a feature matrix
    using the Polars rolling feature pipeline.

    Returns:
        X: np.ndarray of shape (N, n_features)
        feature_names: list of feature column names
    """
    import polars as pl
    df = pl.DataFrame(readings)

    # Rename timestamp column to match config
    timestamp_col = config.TIMESTAMP_COLUMN
    if "timestamp" in df.columns and timestamp_col not in df.columns:
        df = df.rename({"timestamp": timestamp_col})

    # Cast timestamp
    if df[timestamp_col].dtype == pl.Utf8:
        df = df.with_columns(pl.col(timestamp_col).str.to_datetime())

    df_feat = engineer_features(df)

    # Drop non-feature columns
    drop_cols = [timestamp_col] + [
        c for c in df_feat.columns
        if c.startswith("label_") or c in ("failure_in_next", "GPS_speed", "GPS_latitude", "GPS_longitude")
    ]
    drop_cols = [c for c in drop_cols if c in df_feat.columns]
    feature_df = df_feat.drop(drop_cols)

    feature_names = feature_df.columns
    X = feature_df.to_numpy(allow_copy=True)
    return X, list(feature_names)
