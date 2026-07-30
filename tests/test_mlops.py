import pytest
import os
import yaml
import numpy as np
from pathlib import Path
from src.utils.mlflow_tracker import APUExperimentTracker

def test_mlflow_tracker_initialization():
    tracker = APUExperimentTracker(experiment_name="Test_Experiment")
    assert tracker.experiment_name == "Test_Experiment"

def test_mlflow_tracker_logging():
    tracker = APUExperimentTracker(experiment_name="Test_Experiment")
    tracker.start_run(run_name="unit_test_run")
    
    # Test logging parameters and metrics
    tracker.log_params({"learning_rate": 0.05, "n_estimators": 100})
    tracker.log_metrics({"recall": 0.95, "precision": 0.88})
    
    # Test SHAP plot artifact generation
    shap_vals = np.array([0.5, 0.2, -0.1, 0.8])
    feature_names = ["TP2", "TP3", "H1", "Motor_current"]
    tracker.log_shap_summary_plot(shap_vals, feature_names, artifact_name="test_shap.png")
    
    tracker.end_run()

def test_dvc_yaml_structure():
    dvc_file = Path("dvc.yaml")
    assert dvc_file.exists()
    
    with open(dvc_file, "r") as f:
        dvc_config = yaml.safe_load(f)
        
    assert "stages" in dvc_config
    assert "acquire_data" in dvc_config["stages"]
    assert "engineer_features" in dvc_config["stages"]
    assert "train_models" in dvc_config["stages"]
