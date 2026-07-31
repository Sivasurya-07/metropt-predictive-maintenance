import os
import time
import psutil
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Load the PyTorch model
from src.models.cnn_model import CNN1DModel, CNN1DNet
from src import config

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_benchmarks():
    print(f"Initial Memory: {get_memory_mb():.2f} MB")
    
    # Generate some fake data to test
    np.random.seed(42)
    N_SAMPLES = 1000
    SEQ_LEN = 60
    NUM_FEATURES = 15
    
    X_val = np.random.randn(N_SAMPLES, NUM_FEATURES).astype(np.float32)
    
    # Load PyTorch CNN model (using untrained weights for benchmark parity test)
    pt_net = CNN1DNet(num_features=X_val.shape[1], seq_len=60)
    pt_net.eval()
    
    mem_after_pt = get_memory_mb()
    print(f"Memory after loading PyTorch CNN: {mem_after_pt:.2f} MB")
    
    def get_windows(X_arr, seq_len):
        windows = []
        for idx in range(len(X_arr) - seq_len + 1):
            window = X_arr[idx : idx + seq_len]
            windows.append(window.T)
        return np.stack(windows).astype(np.float32)
        
    X_windows = get_windows(X_val, 60)
    X_tensor = torch.tensor(X_windows, dtype=torch.float32)
    
    # Predict with PyTorch
    start = time.perf_counter()
    with torch.no_grad():
        pt_logits = pt_net(X_tensor)
        pt_probs = torch.sigmoid(pt_logits).numpy()
    pt_time = time.perf_counter() - start
    
    print(f"PyTorch Inference Time (1000 samples): {pt_time:.4f}s")
    
    # Export to ONNX if not exists
    onnx_path = config.MODELS_DIR / "cnn_benchmark.onnx"
    dummy_input = torch.randn(1, X_val.shape[1], 60)
    torch.onnx.export(
        pt_net,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False
    )
    
    # Load ONNX Model
    sess = ort.InferenceSession(str(onnx_path))
    mem_after_onnx = get_memory_mb()
    print(f"Memory after loading ONNX (additive): {mem_after_onnx:.2f} MB")
    
    start = time.perf_counter()
    onnx_logits = sess.run(["output"], {"input": X_windows})[0]
    onnx_probs = 1 / (1 + np.exp(-onnx_logits))
    onnx_time = time.perf_counter() - start
    
    mae = np.mean(np.abs(pt_probs - onnx_probs))
    max_diff = np.max(np.abs(pt_probs - onnx_probs))
    
    pt_classes = (pt_probs >= 0.5).astype(int)
    onnx_classes = (onnx_probs >= 0.5).astype(int)
    identical_classes = np.mean(pt_classes == onnx_classes) * 100
    
    print(f"\n--- NUMERICAL PARITY ---")
    print(f"Mean Absolute Error (MAE): {mae:.8e}")
    print(f"Max Prob Diff: {max_diff:.8e}")
    print(f"Identical Classes: {identical_classes:.2f}%")

if __name__ == "__main__":
    run_benchmarks()
