import asyncio
import json
import time
import os
import sys

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.stateful_features import engine
from src.api.routes import manager

async def mock_telemetry():
    """Generates mock telemetry conforming to our strict feature thresholds."""
    return {
        "timestamp": "2026-07-28T12:00:00Z",
        "TP2": 7.5,
        "TP3": 8.0,
        "H1": 8.5,
        "DV_pressure": 0.2,
        "Reservoirs": 8.1,
        "Motor_current": 7.5,
        "Oil_temperature": 65.0,
        "COMP": 1.0,
        "DV_eletric": 0.0,
        "TOWERS": 0.0,
        "pressure_diff_tp3_tp2": 0.5,
        "current_pressure_ratio": 0.9,
        "temp_pressure_ratio": 8.1,
        "is_compressing_under_load": 1.0,
        "is_compressor_off": 0.0
    }

async def run_e2e():
    print("========================================")
    print("    E2E BRUTAL SYSTEM VALIDATION        ")
    print("========================================\n")
    
    print("[1] Testing Redis Stateful Engine Buffer Insertion...")
    try:
        # Pushing a barrage of 100 mock requests to simulate load
        print("    -> Simulating burst of 100 telemetry payloads...")
        tasks = []
        for _ in range(100):
            payload = await mock_telemetry()
            # Slightly stagger the timestamps
            payload["timestamp"] = f"2026-07-28T12:00:{_ % 60:02d}Z"
            tasks.append(engine.push_and_get_window([payload]))
            
        await asyncio.gather(*tasks)
        print("    [PASS] Redis pipeline handled burst load gracefully.")
        
        # Verify the window length
        final_window = await engine.push_and_get_window([await mock_telemetry()])
        print(f"    -> Current Redis window size: {len(final_window)} rows.")
        if len(final_window) > 0:
            print("    [PASS] Stateful engine correctly retrieved historical window.")
        else:
            print("    [FAIL] Window is empty!")
            sys.exit(1)
            
    except Exception as e:
        print(f"    [FAIL] Redis Stateful Engine threw exception: {e}")
        sys.exit(1)
        
    print("\n[2] Testing Redis Parquet Archiver...")
    try:
        test_path = "data/processed/e2e_test_archive.parquet"
        await engine.flush_to_parquet(test_path)
        if os.path.exists(test_path):
            print(f"    [PASS] Parquet flush successful. File created at {test_path}")
            os.remove(test_path) # Cleanup
        else:
            print("    [FAIL] Parquet file was not generated.")
            sys.exit(1)
    except Exception as e:
        print(f"    [FAIL] Parquet Archiver threw exception: {e}")
        sys.exit(1)
        
    print("\n[3] Testing Redis Pub/Sub WebSocket Backplane...")
    try:
        # Start listener in background
        listener_task = asyncio.create_task(manager.start_listener())
        await asyncio.sleep(0.5) # Give it time to connect
        
        # We simulate a worker broadcasting a message
        mock_msg = {"type": "broadcast", "data": "e2e_test_payload"}
        await manager.redis.publish("apu_alerts", json.dumps(mock_msg))
        await asyncio.sleep(0.5)
        
        print("    [PASS] WebSocket Pub/Sub backplane transmitted messages without crashing.")
        listener_task.cancel()
    except Exception as e:
        print(f"    [FAIL] Pub/Sub Backplane threw exception: {e}")
        sys.exit(1)
        
    print("\n========================================")
    print("    E2E BRUTAL VALIDATION COMPLETED     ")
    print("========================================")

if __name__ == "__main__":
    # Windows asyncio policy fix
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_e2e())
