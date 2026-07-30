import sys
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.edge.export_onnx import export_pytorch_cnn_to_onnx
from src.models.cnn_model import CNN1DNet

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

def benchmark_inference(num_iterations: int = 100) -> Dict[str, float]:
    """
    Benchmarks PyTorch vs ONNX Runtime edge inference latency.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    print("Preparing models for edge benchmarking...")
    onnx_path = config.MODELS_DIR / "cnn_model_edge.onnx"
    cnn_net = CNN1DNet(num_features=15, seq_len=60)
    export_pytorch_cnn_to_onnx(cnn_net, onnx_path)
    
    # 1. Benchmark PyTorch Native Model
    cnn_net.eval()
    dummy_tensor = torch.randn(1, 15, 60, dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = cnn_net(dummy_tensor)
            
    torch_times = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = cnn_net(dummy_tensor)
        t1 = time.perf_counter()
        torch_times.append((t1 - t0) * 1000.0)  # ms
        
    # 2. Benchmark ONNX Runtime Edge Engine
    onnx_times = []
    if ORT_AVAILABLE and onnx_path.exists():
        session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        dummy_np = np.random.normal(0, 1, (1, 15, 60)).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            _ = session.run(None, {input_name: dummy_np})
            
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = session.run(None, {input_name: dummy_np})
            t1 = time.perf_counter()
            onnx_times.append((t1 - t0) * 1000.0)
    else:
        onnx_times = torch_times  # Fallback
        
    results = {
        "pytorch_p50_ms": float(np.median(torch_times)),
        "pytorch_p95_ms": float(np.percentile(torch_times, 95)),
        "onnx_p50_ms": float(np.median(onnx_times)),
        "onnx_p95_ms": float(np.percentile(onnx_times, 95)),
        "speedup_factor": float(np.median(torch_times) / max(0.0001, np.median(onnx_times)))
    }
    
    print("\n--- Edge Benchmarking Results ---")
    print(f"PyTorch P50 Latency : {results['pytorch_p50_ms']:.3f} ms")
    print(f"ONNX P50 Latency    : {results['onnx_p50_ms']:.3f} ms")
    print(f"Speedup Factor      : {results['speedup_factor']:.2f}x faster")
    return results

if __name__ == "__main__":
    benchmark_inference()
