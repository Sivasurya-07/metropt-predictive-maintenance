import pytest
import os
import torch
from pathlib import Path

from src import config
from src.edge.export_onnx import export_pytorch_cnn_to_onnx
from src.edge.benchmark import benchmark_inference
from src.models.cnn_model import CNN1DNet

def test_pytorch_cnn_to_onnx_export(tmp_path):
    output_onnx_path = tmp_path / "test_cnn.onnx"
    cnn_net = CNN1DNet(num_features=15, seq_len=60)
    
    exported_path = export_pytorch_cnn_to_onnx(cnn_net, output_onnx_path)
    assert exported_path.exists()
    assert exported_path.stat().st_size > 0

def test_edge_benchmark_execution():
    results = benchmark_inference(num_iterations=10)
    assert "pytorch_p50_ms" in results
    assert "onnx_p50_ms" in results
    assert results["pytorch_p50_ms"] > 0
    assert results["onnx_p50_ms"] > 0
