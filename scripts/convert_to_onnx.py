import sys
import os
import pickle
import onnxmltools
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
import warnings

# Ensure the root of the project is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

warnings.filterwarnings("ignore")

def convert_model(pkl_path: str, onnx_path: str, num_features: int = 15):
    """
    Converts a pickled EnsembleModel's LightGBM to ONNX.
    In a full production scenario, we'd convert XGBoost and meta-models as well.
    """
    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        ensemble = pickle.load(f)
        
    lgbm_model = ensemble.lgbm.model
    if lgbm_model is None:
        print("LightGBM model not found in the ensemble.")
        return

    # Define input schema
    initial_types = [('float_input', FloatTensorType([None, num_features]))]
    
    # Convert LightGBM model
    onnx_model = convert_lightgbm(lgbm_model, initial_types=initial_types, target_opset=12)
    
    # Save the ONNX model
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Successfully converted LightGBM model to {onnx_path}")

if __name__ == "__main__":
    MODELS_DIR = "models"
    for horizon in ["2h", "4h", "8h"]:
        pkl_path = f"{MODELS_DIR}/ensemble_{horizon}.pkl"
        onnx_path = f"{MODELS_DIR}/lgbm_{horizon}.onnx"
        
        if os.path.exists(pkl_path):
            convert_model(pkl_path, onnx_path)
        else:
            print(f"Model file not found: {pkl_path}")
