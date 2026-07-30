import requests
import time
import math
from datetime import datetime, timezone
import random

API_URL = "http://localhost:8000/predict"

def generate_telemetry(t_step: int):
    # Simulate somewhat realistic fluctuating sine wave data for pressure sensors
    base_tp2 = 8.0 + math.sin(t_step / 10.0) * 2.0 + random.uniform(-0.1, 0.1)
    base_tp3 = 7.5 + math.cos(t_step / 15.0) * 1.5 + random.uniform(-0.1, 0.1)
    base_h1 = 8.2 + math.sin(t_step / 8.0) * 1.0 + random.uniform(-0.1, 0.1)
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "TP2": base_tp2,
        "TP3": base_tp3,
        "H1": base_h1,
        "DV_pressure": 0.2 + random.uniform(-0.05, 0.05),
        "Reservoirs": 8.0 + random.uniform(-0.1, 0.1),
        "Motor_current": 7.5 + math.sin(t_step / 20.0) * 0.5,
        "Oil_temperature": 65.0 + math.sin(t_step / 30.0) * 5.0,
        "COMP": 1.0 if t_step % 100 < 50 else 0.0,
        "DV_eletric": 0.0,
        "TOWERS": 0.0,
        "MPG": 1.0,
        "LPS": 0.0,
        "Pressure_switch": 1.0,
        "Oil_level": 1.0,
        "Flowmeter": 1.0,
        "pressure_diff_tp3_tp2": abs(base_tp3 - base_tp2),
        "current_pressure_ratio": 0.9,
        "temp_pressure_ratio": 8.1,
        "is_compressing_under_load": 1.0,
        "is_compressor_off": 0.0
    }

def run_simulation():
    print("Starting edge traffic simulation...")
    t = 0
    while True:
        payload = generate_telemetry(t)
        try:
            # Wrap in list as expected by POST /predict
            req = {"readings": [payload], "horizon": "4h"}
            resp = requests.post(API_URL, json=req, timeout=2)
            if resp.status_code == 200:
                print(f"[{t}] Pushed telemetry. Status: 200 OK")
            else:
                print(f"[{t}] Failed: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"[{t}] Error connecting to backend: {e}")
            
        t += 1
        time.sleep(1)  # 1 reading per second for smooth chart preview

if __name__ == "__main__":
    run_simulation()
