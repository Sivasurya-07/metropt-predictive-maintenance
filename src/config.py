import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Dataset details
DATASET_URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
ZIP_FILENAME = "metropt_3_dataset.zip"
CSV_FILENAME = "metropt_3_dataset.csv"  # We will extract it under raw/

# Sensor Columns
ANALOG_COLUMNS = [
    "TP2",           # Compressor pressure (bar)
    "TP3",           # Reservoir pressure (bar)
    "H1",            # Pressure drop across filter (bar)
    "DV_pressure",   # Pressure drop across dryer / relief valve (bar)
    "Reservoirs",    # Reservoirs pressure (bar)
    "Motor_current", # Motor current (A)
    "Oil_temperature"# Oil temperature (C)
]

DIGITAL_COLUMNS = [
    "COMP",          # Air intake valve status (boolean/discrete)
    "DV_eletric",    # Drain valve status (boolean/discrete)
    "TOWERS",        # Tower status (drying tower 1/2)
    "MPG",           # Compressor contactor status
    "LPS",           # Low pressure switch status
    "Pressure_switch",# Pressure switch status
    "Oil_level",     # Oil level switch status
    "Flowmeter"      # Airflow status
]

GPS_COLUMNS = [
    "GPS_speed",     # Train speed (km/h)
    "GPS_latitude",
    "GPS_longitude"
]

ALL_SENSOR_COLUMNS = ANALOG_COLUMNS + DIGITAL_COLUMNS + GPS_COLUMNS
TIMESTAMP_COLUMN = "timestamp"

# Multi-Horizon Windows (in hours)
HORIZONS = {
    "2h": 2.0,
    "4h": 4.0,
    "8h": 8.0
}

# Label naming convention
def get_label_name(horizon_str: str) -> str:
    return f"label_failure_{horizon_str}"

# Model hyperparameter tuning options
RANDOM_STATE = 42
