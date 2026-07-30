import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import IsolationForest
from src.models.base import BaseModel

class ThreeSigmaDetector(BaseModel):
    """
    Baseline anomaly detector that uses the 3-sigma rule:
    Flags an anomaly if any analog sensor goes beyond 3 standard deviations from its baseline mean.
    """
    def __init__(self, threshold_std: float = 3.0):
        self.threshold_std = threshold_std
        self.means = None
        self.stds = None
        
    def fit(self, X, y=None, **kwargs) -> "ThreeSigmaDetector":
        # Expects X as a numpy array or DataFrame of shape (N, D) containing analog sensor columns
        # Fits on healthy data (where y is 0, if provided)
        if y is not None:
            healthy_idx = (y == 0)
            X_healthy = X[healthy_idx]
        else:
            X_healthy = X
            
        self.means = np.mean(X_healthy, axis=0)
        self.stds = np.std(X_healthy, axis=0)
        # Avoid division by zero
        self.stds = np.where(self.stds == 0, 1e-5, self.stds)
        return self
        
    def predict(self, X) -> np.ndarray:
        z_scores = np.abs((X - self.means) / self.stds)
        # Check if any feature has z_score > 3
        anomalies = np.any(z_scores > self.threshold_std, axis=1)
        return anomalies.astype(int)
        
    def predict_proba(self, X) -> np.ndarray:
        # Z-scores are normalized and mapped to [0, 1] range to simulate confidence/probability
        z_scores = np.abs((X - self.means) / self.stds)
        max_z = np.max(z_scores, axis=1)
        # Map 3-sigma to 0.5, higher to closer to 1.0
        probs = 1.0 / (1.0 + np.exp(-(max_z - self.threshold_std)))
        return probs
        
    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
            
    def load(self, path: Path) -> "ThreeSigmaDetector":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.means = obj.means
        self.stds = obj.stds
        self.threshold_std = obj.threshold_std
        return self

class IsolationForestDetector(BaseModel):
    """
    Baseline anomaly detector using scikit-learn's IsolationForest.
    Useful as a general guardrail for unseen anomalous patterns.
    """
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination, 
            random_state=self.random_state,
            n_jobs=-1
        )
        
    def fit(self, X, y=None, **kwargs) -> "IsolationForestDetector":
        # Fits on healthy data if possible
        if y is not None:
            healthy_idx = (y == 0)
            X_healthy = X[healthy_idx]
        else:
            X_healthy = X
            
        self.model.fit(X_healthy)
        return self
        
    def predict(self, X) -> np.ndarray:
        # Isolation forest returns -1 for outliers, 1 for inliers
        preds = self.model.predict(X)
        anomalies = (preds == -1).astype(int)
        return anomalies
        
    def predict_proba(self, X) -> np.ndarray:
        # Score_samples returns negative anomaly scores (more negative = more anomalous)
        # Map this to an approximate [0, 1] probability scale
        scores = self.model.score_samples(X)
        # Min-max map using the decision boundary (anomaly score < 0)
        # High score (close to 0) => low probability of failure
        # Low score (very negative) => high probability of failure
        probs = 1.0 / (1.0 + np.exp(scores * 10.0))  # Scale factor of 10 for logit contrast
        return probs
        
    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
            
    def load(self, path: Path) -> "IsolationForestDetector":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.model = obj.model
        self.contamination = obj.contamination
        self.random_state = obj.random_state
        return self
