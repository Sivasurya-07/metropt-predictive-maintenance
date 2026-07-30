import lightgbm as lgb
import numpy as np
import optuna
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from src.models.base import BaseModel
from src import config

class TimeSeriesPurgedSplitter:
    """
    Custom expanding window time-series splitter with a purging gap.
    Ensures that validation sets are strictly after training sets with a temporal gap
    to prevent label leakage caused by the warning windows.
    """
    def __init__(self, n_splits: int = 5, gap_seconds: int = 12 * 3600):
        self.n_splits = n_splits
        self.gap_seconds = gap_seconds
        
    def split(self, timestamps: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Splits index array into train/val folds based on timestamps.
        """
        n_samples = len(timestamps)
        indices = np.arange(n_samples)
        
        # Calculate split sizes
        # Each split will add an expanding portion of training data
        fold_size = n_samples // (self.n_splits + 1)
        splits = []
        
        for i in range(1, self.n_splits + 1):
            train_end_idx = i * fold_size
            
            # Find the actual time boundary
            train_end_time = timestamps[train_end_idx]
            gap_end_time = train_end_time + np.timedelta64(self.gap_seconds, 's')
            
            # Find the indices for validation start (after the gap)
            val_start_idx = np.searchsorted(timestamps, gap_end_time)
            val_end_idx = min(n_samples, val_start_idx + fold_size)
            
            if val_start_idx >= n_samples or val_start_idx >= val_end_idx:
                print(f"Warning: Fold {i} validation set is empty due to gap. Skipping.")
                continue
                
            train_idx = indices[:train_end_idx]
            val_idx = indices[val_start_idx:val_end_idx]
            splits.append((train_idx, val_idx))
            
        return splits

class LightGBMModel(BaseModel):
    """
    LightGBM classifier wrapping for a single failure horizon (e.g. 2h, 4h, 8h).
    """
    def __init__(self, horizon_str: str, params: Dict = None):
        self.horizon_str = horizon_str
        self.label_col = config.get_label_name(horizon_str)
        self.model = None
        self.onnx_session = None
        self.use_onnx = False
        
        # Default params
        self.params = params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
            "random_state": config.RANDOM_STATE
        }
        
    def fit(self, X, y, **kwargs) -> "LightGBMModel":
        """
        Fits the LightGBM classifier.
        Automatically calculates scale_pos_weight based on class distribution.
        """
        # Filter out neutral labels (-1) if present
        valid_idx = (y != -1)
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        
        # Calculate class weights for imbalance
        n_neg = np.sum(y_clean == 0)
        n_pos = np.sum(y_clean == 1)
        
        if n_pos > 0 and n_neg > 0:
            scale_pos_weight = n_neg / n_pos
            self.params["scale_pos_weight"] = scale_pos_weight
            
        train_data = lgb.Dataset(X_clean, label=y_clean)
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=kwargs.get("num_boost_round", 150)
        )
        return self
        
    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
        
    def predict_proba(self, X) -> np.ndarray:
        if self.use_onnx and self.onnx_session is not None:
            input_name = self.onnx_session.get_inputs()[0].name
            X_f32 = X.astype(np.float32)
            # ONNX returns [labels, probabilities_dicts]
            onnx_preds = self.onnx_session.run(None, {input_name: X_f32})[1]
            return np.array([p.get(1, p.get(1.0, 0.0)) for p in onnx_preds])

        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        # Returns probability of class 1
        return self.model.predict(X)
        
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
            
    def load(self, path: Path) -> "LightGBMModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.horizon_str = obj.horizon_str
        self.label_col = obj.label_col
        self.params = obj.params
        self.model = obj.model
        return self

    def load_onnx(self, path: Path) -> "LightGBMModel":
        import onnxruntime as ort
        self.onnx_session = ort.InferenceSession(str(path))
        self.use_onnx = True
        print(f"[ONNX] Successfully loaded LightGBM ONNX model from {path}")
        return self

def optimize_hyperparameters(X: np.ndarray, y: np.ndarray, timestamps: np.ndarray, horizon_str: str, n_trials: int = 30) -> Dict:
    """
    Runs an Optuna study to optimize hyperparameters for LightGBM.
    Uses TimeSeriesPurgedSplitter for validation.
    Optimizes validation F1 score or Recall under a precision target constraint.
    """
    valid_idx = (y != -1)
    X_clean = X[valid_idx]
    y_clean = y[valid_idx]
    ts_clean = timestamps[valid_idx]
    
    splitter = TimeSeriesPurgedSplitter(n_splits=3, gap_seconds=8 * 3600)
    splits = splitter.split(ts_clean)
    
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "verbose": -1,
            "random_state": config.RANDOM_STATE
        }
        
        scores = []
        for train_idx, val_idx in splits:
            X_train, y_train = X_clean[train_idx], y_clean[train_idx]
            X_val, y_val = X_clean[val_idx], y_clean[val_idx]
            
            # Skip if validation has no positive instances
            if np.sum(y_val == 1) == 0:
                continue
                
            model = LightGBMModel(horizon_str=horizon_str, params=params)
            model.fit(X_train, y_train, num_boost_round=100)
            
            probs = model.predict_proba(X_val)
            preds = (probs >= 0.5).astype(int)
            
            # Compute F1 score manually
            tp = np.sum((preds == 1) & (y_val == 1))
            fp = np.sum((preds == 1) & (y_val == 0))
            fn = np.sum((preds == 0) & (y_val == 1))
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            scores.append(f1)
            
        return np.mean(scores) if scores else 0.0
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Optuna HPO completed for {horizon_str}. Best Trial value (F1): {study.best_value:.4f}")
    return study.best_params
