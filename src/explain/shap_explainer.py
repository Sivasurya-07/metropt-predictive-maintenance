import numpy as np
import shap
from typing import Dict, List, Tuple
from src.models.lightgbm_model import LightGBMModel
from src.models.cnn_model import CNN1DModel
from src import config

try:
    import torch
    from captum.attr import GradientShap
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

class APUExplainer:
    """
    Handles local and global feature attribution explanations for the APU failure prediction models.
    """
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        
    def explain_lightgbm(self, lgbm_model: LightGBMModel, x_sample: np.ndarray) -> List[Tuple[str, float]]:
        """
        Computes SHAP values using TreeSHAP for a single sample on the LightGBM model.
        Returns a sorted list of (feature_name, shap_value) tuples.
        """
        if lgbm_model.model is None:
            raise ValueError("LightGBM model must be fitted first.")
            
        # Create TreeExplainer
        explainer = shap.TreeExplainer(lgbm_model.model)
        
        # Reshape sample if 1D
        if len(x_sample.shape) == 1:
            x_sample = x_sample.reshape(1, -1)
            
        # Compute SHAP values
        shap_values = explainer.shap_values(x_sample)
        
        # Handle list vs array output
        if isinstance(shap_values, list):
            # For binary classification, index 1 corresponds to class 1 (failure)
            shap_vals = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            # Shape might be (classes, samples, features)
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]
            
        # Map feature names to shap values
        feature_attributions = list(zip(self.feature_names, shap_vals))
        
        # Sort by absolute attribution value descending
        feature_attributions.sort(key=lambda val: abs(val[1]), reverse=True)
        return feature_attributions

    def explain_cnn(self, cnn_model: CNN1DModel, x_window: np.ndarray, horizon_idx: int) -> List[Tuple[str, float]]:
        """
        Computes attribution scores using GradientSHAP (via Captum) for the PyTorch 1D-CNN.
        Inputs:
          - x_window: 2D numpy array representing a sliding window of shape (seq_len, num_features)
          - horizon_idx: 0 (2h), 1 (4h), or 2 (8h)
        Returns:
          - Sorted list of (feature_name, mean_attribution) tuples representing feature importance across the window.
        """
        if not CAPTUM_AVAILABLE:
            print("Captum is not available. Skipping CNN SHAP explanation.")
            return []
            
        if cnn_model.lightning_module is None:
            raise ValueError("CNN model must be fitted first.")
            
        net = cnn_model.lightning_module.net
        net.eval()
        
        # Convert window to tensor of shape (1, num_features, seq_len)
        x_tensor = torch.tensor(x_window, dtype=torch.float32).t().unsqueeze(0)
        
        # We need a baseline (zeros is standard)
        baseline = torch.zeros_like(x_tensor)
        
        # Setup GradientShap explainer
        gs = GradientShap(net)
        
        # Compute attribute values
        # target represents the index of the multi-head prediction target
        attributions = gs.attribute(x_tensor, baselines=baseline, target=horizon_idx)
        
        # Shape: (1, num_features, seq_len)
        attributions_np = attributions.squeeze(0).cpu().numpy()
        
        # Average attribution over the time dimension (seq_len) to get one score per feature
        mean_attributions = np.mean(attributions_np, axis=1)
        
        feature_attributions = list(zip(self.feature_names, mean_attributions))
        feature_attributions.sort(key=lambda val: abs(val[1]), reverse=True)
        return feature_attributions
