import pytest
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from src import config
from src.data.validator import validate_raw_data
from src.data.labeling import label_data, get_failure_intervals
from src.data.features import engineer_features
from src.models.anomaly import ThreeSigmaDetector

def get_valid_mock_df(size: int = 100) -> pl.DataFrame:
    """
    Helper to generate a valid mock dataframe matching sensor specifications.
    """
    start_dt = datetime(2020, 4, 17, 12, 0)
    timestamps = [start_dt + timedelta(seconds=i*10) for i in range(size)]
    return pl.DataFrame({
        config.TIMESTAMP_COLUMN: timestamps,
        "TP2": np.random.normal(6.0, 0.5, size).tolist(),
        "TP3": np.random.normal(8.0, 0.5, size).tolist(),
        "H1": np.random.normal(8.0, 0.2, size).tolist(),
        "DV_pressure": np.random.normal(0.1, 0.05, size).tolist(),
        "Reservoirs": np.random.normal(8.0, 0.4, size).tolist(),
        "Motor_current": np.random.normal(7.0, 1.0, size).tolist(),
        "Oil_temperature": np.random.normal(60.0, 5.0, size).tolist(),
        "COMP": [1.0] * size,
        "DV_eletric": [0.0] * size,
        "TOWERS": [0.0] * size,
        "MPG": [0.0] * size,
        "LPS": [1.0] * size,
        "Pressure_switch": [1.0] * size,
        "Oil_level": [1.0] * size,
        "Flowmeter": [0.0] * size,
        "GPS_speed": np.random.normal(40.0, 10.0, size).tolist(),
        "GPS_latitude": [41.15] * size,
        "GPS_longitude": [-8.61] * size,
    })

def test_config_paths():
    assert config.RAW_DATA_DIR.exists()
    assert len(config.ANALOG_COLUMNS) == 7
    assert len(config.DIGITAL_COLUMNS) == 8

def test_data_validation_valid():
    df = get_valid_mock_df(50)
    # This should pass without raising exceptions
    validated = validate_raw_data(df)
    assert validated.shape == df.shape

def test_data_validation_invalid_ranges():
    df = get_valid_mock_df(50)
    
    # Introduce invalid value (extreme pressure)
    df_invalid = df.with_columns(pl.lit(500.0).alias("TP2"))
    
    import pandera.errors as pe
    with pytest.raises(pe.SchemaError):
        validate_raw_data(df_invalid)

def test_labeling_logic():
    # Setup mock data near failure date 2020-04-18
    # Air Leak Nr. 1 is (datetime(2020, 4, 18, 0, 0), datetime(2020, 4, 18, 23, 59))
    timestamps = [
        datetime(2020, 4, 17, 21, 0),  # 3 hours before failure start -> 0 (negative) for 2h horizon
        datetime(2020, 4, 17, 22, 30), # 1.5 hours before failure start -> 1 (warning) for 2h horizon
        datetime(2020, 4, 18, 12, 0),  # during failure -> -1 (neutral)
    ]
    df = pl.DataFrame({
        config.TIMESTAMP_COLUMN: timestamps,
        "TP2": [6.0] * 3
    })
    
    labeled_df = label_data(df)
    
    # Check 2h label column
    label_col_2h = config.get_label_name("2h")
    assert label_col_2h in labeled_df.columns
    
    labels_2h = labeled_df[label_col_2h].to_list()
    assert labels_2h[0] == 0   # 3 hours before is outside 2h horizon window
    assert labels_2h[1] == 1   # 1.5 hours before is inside 2h horizon window
    assert labels_2h[2] == -1  # during failure is neutral

def test_feature_engineering():
    # Generate 100 rows (at average 10s interval, rolling windows need at least a few samples)
    df = get_valid_mock_df(100)
    features_df = engineer_features(df)
    
    # Verify rolling feature columns exist
    assert "TP2_roll_mean_30m" in features_df.columns
    assert "TP2_roll_std_30m" in features_df.columns
    assert "pressure_diff_tp3_tp2" in features_df.columns
    assert "is_compressing_under_load" in features_df.columns
    
    # Ensure no nulls are left in the rolling features
    assert features_df["TP2_roll_mean_30m"].null_count() == 0

def test_three_sigma_detector():
    X = np.random.normal(5.0, 1.0, (100, 3))
    y = np.zeros(100)
    
    # Train detector on healthy profile
    detector = ThreeSigmaDetector()
    detector.fit(X, y)
    
    # Predict on normal profile
    preds_normal = detector.predict(X)
    assert np.mean(preds_normal == 1) < 0.1  # very few false alarms
    
    # Test on severe anomalies (e.g. value 20.0 standard deviations away)
    X_anomaly = np.array([[20.0, 20.0, 20.0]])
    pred_anomaly = detector.predict(X_anomaly)
    assert pred_anomaly[0] == 1
