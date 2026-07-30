import numpy as np
import xgboost as xgb
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from src.models.base import BaseModel
from src.models.lightgbm_model import LightGBMModel
from src.models.cnn_model import CNN1DModel
from src.models.anomaly import IsolationForestDetector
from src import config

class XGBoostModel(BaseModel):
    """
    XGBoost classifier wrapper for a single horizon.
    """
    def __init__(self, horizon_str: str, params: Dict = None):
        self.horizon_str = horizon_str
        self.label_col = config.get_label_name(horizon_str)
        self.model = None
        
        self.params = params or {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": config.RANDOM_STATE,
            "n_jobs": -1
        }
        
    def fit(self, X, y, **kwargs) -> "XGBoostModel":
        valid_idx = (y != -1)
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        
        # Calculate class scale weight
        n_neg = np.sum(y_clean == 0)
        n_pos = np.sum(y_clean == 1)
        if n_pos > 0 and n_neg > 0:
            self.params["scale_pos_weight"] = n_neg / n_pos
            
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X_clean, y_clean)
        return self
        
    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
        
    def predict_proba(self, X) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        # Returns warning probability
        return self.model.predict_proba(X)[:, 1]
        
    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
            
    def load(self, path: Path) -> "XGBoostModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.horizon_str = obj.horizon_str
        self.label_col = obj.label_col
        self.params = obj.params
        self.model = obj.model
        return self


class EnsembleModel(BaseModel):
    """
    Ensemble Model combining LightGBM, 1D-CNN, and XGBoost using a Logistic Regression meta-model (Stacking).
    Integrates an Isolation Forest guardrail for novel pattern anomaly detection.
    """
    def __init__(self, horizon_str: str, num_features: int = 15, seq_len: int = 60):
        self.horizon_str = horizon_str
        self.horizon_idx = list(config.HORIZONS.keys()).index(horizon_str)
        self.num_features = num_features
        self.seq_len = seq_len
        
        # Level 1 Models
        self.lgbm = LightGBMModel(horizon_str=horizon_str)
        self.xgb = XGBoostModel(horizon_str=horizon_str)
        
        # Level 2 Meta-model
        self.meta_model = LogisticRegression(class_weight="balanced")
        
        # Guardrail Model
        self.guardrail = IsolationForestDetector(contamination=0.02)
        
    def fit(self, X, y, X_val=None, y_val=None, cnn_model: CNN1DModel = None, **kwargs) -> "EnsembleModel":
        """
        Fits the Level 1 models, computes predictions, and trains the Level 2 meta-model.
        Takes fitted or pre-initialized CNN1DModel as CNN training is handled globally.
        """
        # Fit Level 1: LGBM & XGBoost
        print(f"[{self.horizon_str}] Fitting LightGBM...")
        self.lgbm.fit(X, y[:, self.horizon_idx])
        
        print(f"[{self.horizon_str}] Fitting XGBoost...")
        self.xgb.fit(X, y[:, self.horizon_idx])
        
        # Fit Guardrail on healthy data only
        print(f"[{self.horizon_str}] Fitting Anomaly Guardrail...")
        self.guardrail.fit(X, y[:, self.horizon_idx])
        
        # Compute validation predictions for meta-model fitting
        # If no validation set provided, use training set predictions (with caution for overfitting)
        X_meta = X_val if X_val is not None else X
        y_meta = y_val[:, self.horizon_idx] if y_val is not None else y[:, self.horizon_idx]
        
        # Clean out neutral label -1
        valid_idx = (y_meta != -1)
        X_meta_clean = X_meta[valid_idx]
        y_meta_clean = y_meta[valid_idx]
        
        p_lgb = self.lgbm.predict_proba(X_meta_clean)
        p_xgb = self.xgb.predict_proba(X_meta_clean)
        
        if cnn_model is not None:
            # CNN outputs shape (N, 3), extract specific horizon
            p_cnn = cnn_model.predict_proba(X_meta_clean)[:, self.horizon_idx]
        else:
            p_cnn = p_lgb  # Fallback if CNN is not available
            
        # Stacking feature matrix (N, 3)
        meta_features = np.stack([p_lgb, p_cnn, p_xgb], axis=1)
        
        print(f"[{self.horizon_str}] Fitting Level 2 Meta-model...")
        if len(np.unique(y_meta_clean)) < 2:
            from sklearn.dummy import DummyClassifier
            self.meta_model = DummyClassifier(strategy="prior")
            
        self.meta_model.fit(meta_features, y_meta_clean)
        return self
        
    def predict_proba(self, X, cnn_model: CNN1DModel = None) -> np.ndarray:
        """
        Predicts calibrated probability using meta-model Stacking.
        """
        p_lgb = self.lgbm.predict_proba(X)
        p_xgb = self.xgb.predict_proba(X)
        
        if cnn_model is not None:
            p_cnn = cnn_model.predict_proba(X)[:, self.horizon_idx]
        else:
            p_cnn = p_lgb
            
        meta_features = np.stack([p_lgb, p_cnn, p_xgb], axis=1)
        meta_probs = self.meta_model.predict_proba(meta_features)
        if meta_probs.ndim == 2 and meta_probs.shape[1] > 1:
            ensemble_prob = meta_probs[:, 1]
        else:
            ensemble_prob = meta_probs.ravel()
        
        return ensemble_prob
        
    def predict(self, X, cnn_model: CNN1DModel = None, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X, cnn_model)
        preds = (probs >= threshold).astype(int)
        
        # Guardrail logic: if Isolation Forest detects a severe anomaly, force warning prediction
        # (even if supervised models missed it due to a novel feature signature)
        guardrail_anomalies = self.guardrail.predict(X)
        final_preds = np.logical_or(preds, guardrail_anomalies).astype(int)
        return final_preds
        
    def get_confidence_metrics(self, X, cnn_model: CNN1DModel = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the ensemble calibrated probability and a confidence score based on disagreement.
        Confidence = 1.0 - std_deviation(model_probabilities)
        """
        p_lgb = self.lgbm.predict_proba(X)
        p_xgb = self.xgb.predict_proba(X)
        p_cnn = cnn_model.predict_proba(X)[:, self.horizon_idx] if cnn_model is not None else p_lgb
        
        ensemble_prob = self.predict_proba(X, cnn_model)
        
        # Disagreement (standard deviation)
        probs_matrix = np.stack([p_lgb, p_cnn, p_xgb], axis=1)
        disagreement_std = np.std(probs_matrix, axis=1)
        
        # Map std deviation of 3 models to a [0, 1] confidence index
        # Max std is ~0.5 (e.g. [0.0, 1.0, 0.5]), so scale by 2.0
        confidence = 1.0 - (disagreement_std * 2.0)
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return ensemble_prob, confidence

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
            
    def load(self, path: Path) -> "EnsembleModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.horizon_str = obj.horizon_str
        self.horizon_idx = obj.horizon_idx
        self.num_features = obj.num_features
        self.seq_len = obj.seq_len
        self.lgbm = obj.lgbm
        self.xgb = obj.xgb
        self.meta_model = obj.meta_model
        self.guardrail = obj.guardrail
        return self


def run_escalation_logic(probs_2h: float, probs_4h: float, probs_8h: float) -> str:
    """
    Implements APU warning escalation logic:
    - 8h probability > 0.5: Trigger 'Warning' (Medium-term)
    - 4h probability > 0.5 AND 8h probability > 0.5: Trigger 'Critical Alert' (Short-term confirmation)
    - 2h probability > 0.5: Trigger 'Emergency Maintenance' (Imminent Action)
    """
    if probs_2h > 0.5:
        return "EMERGENCY_MAINTENANCE"
    elif probs_4h > 0.5 and probs_8h > 0.5:
        return "CRITICAL_ALERT"
    elif probs_8h > 0.5:
        return "WARNING"
    else:
        return "HEALTHY"
