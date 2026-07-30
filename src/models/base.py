from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path

class BaseModel(ABC):
    """
    Abstract base class for all predictive maintenance models.
    """
    
    @abstractmethod
    def fit(self, X, y, **kwargs) -> "BaseModel":
        """
        Fits the model on training data.
        """
        pass
        
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """
        Predicts binary classes (0: healthy, 1: warning/failure).
        """
        pass
        
    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """
        Predicts warning/failure probabilities.
        Returns array of shape (N, num_classes) or (N,) depending on output definition.
        """
        pass
        
    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Serializes and saves the model to disk.
        """
        pass
        
    @abstractmethod
    def load(self, path: Path) -> "BaseModel":
        """
        Deserializes and loads a model from disk.
        """
        pass
