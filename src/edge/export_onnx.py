import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.models.cnn_model import CNN1DNet

def export_pytorch_cnn_to_onnx(
    model: nn.Module,
    output_path: Path,
    sequence_length: int = 60,
    num_features: int = 15
) -> Path:
    """
    Exports PyTorch 1D-CNN multi-head neural network model to ONNX format.
    Input tensor shape: (batch_size, num_features, sequence_length)
    """
    model.eval()
    dummy_input = torch.randn(1, num_features, sequence_length, dtype=torch.float32)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_sensor_sequence'],
        output_names=['prediction_horizons'],
        dynamic_axes={
            'input_sensor_sequence': {0: 'batch_size'},
            'prediction_horizons': {0: 'batch_size'}
        },
        dynamo=False
    )
    print(f"Successfully exported PyTorch 1D-CNN to ONNX: {output_path}")
    return output_path

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    onnx_output_path = config.MODELS_DIR / "cnn_model_edge.onnx"
    cnn_net = CNN1DNet(num_features=15, seq_len=60)
    export_pytorch_cnn_to_onnx(cnn_net, onnx_output_path)

if __name__ == "__main__":
    main()
