import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List
from src import config

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

class APUExperimentTracker:
    """
    MLflow experiment tracking and model registry wrapper for MetroPT APU models.
    """
    def __init__(self, experiment_name: str = "MetroPT_APU_Predictive_Maintenance"):
        self.experiment_name = experiment_name
        self.enabled = MLFLOW_AVAILABLE
        
        if self.enabled:
            mlflow.set_experiment(self.experiment_name)
            
    def start_run(self, run_name: Optional[str] = None):
        """
        Starts a new MLflow tracking run.
        """
        if self.enabled:
            return mlflow.start_run(run_name=run_name)
        return None

    def end_run(self):
        """
        Ends current MLflow run.
        """
        if self.enabled and mlflow.active_run():
            mlflow.end_run()

    def log_params(self, params: Dict[str, Any]):
        """
        Logs hyperparameter dictionary.
        """
        if self.enabled:
            mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Logs operational metrics (Recall, Precision, F1, Lead Time, False Alarm Rate).
        """
        if self.enabled:
            mlflow.log_metrics(metrics, step=step)

    def log_shap_summary_plot(self, shap_values: np.ndarray, features: List[str], artifact_name: str = "shap_summary.png"):
        """
        Generates and logs a SHAP feature importance summary plot as an MLflow artifact.
        """
        if not self.enabled:
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        mean_abs_shap = np.abs(shap_values).mean(axis=0) if shap_values.ndim > 1 else np.abs(shap_values)
        sorted_idx = np.argsort(mean_abs_shap)[-15:]  # Top 15 features
        
        ax.barh(range(len(sorted_idx)), mean_abs_shap[sorted_idx], align='center', color='#1f77b4')
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([features[i] for i in sorted_idx])
        ax.set_xlabel("Mean |SHAP Value| (Impact on Warning Prediction)")
        ax.set_title("Top 15 Predictive Features (TreeSHAP)")
        plt.tight_layout()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_path = os.path.join(tmp_dir, artifact_name)
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(plot_path, artifact_path="explainability")

    def register_model(self, model_object: Any, artifact_path: str, model_name: str):
        """
        Logs and registers a model object to MLflow Model Registry.
        """
        if self.enabled:
            mlflow.sklearn.log_model(
                sk_model=model_object,
                artifact_path=artifact_path,
                registered_model_name=model_name
            )
