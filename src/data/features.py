import polars as pl
from src import config

def compute_domain_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes domain-specific features based on physical principles of the APU:
    - Pressure differential (TP3 - TP2)
    - Motor current to pressure ratio
    - Oil temperature to pressure ratio
    - Operating state (compressor active under load)
    """
    df_features = df.with_columns([
        # Pressure differential between reservoir and compressor
        (pl.col("TP3") - pl.col("TP2")).alias("pressure_diff_tp3_tp2"),
        
        # Pressure drop across filters (H1 - DV_pressure)
        (pl.col("H1") - pl.col("DV_pressure")).alias("pressure_diff_h1_dv"),
        
        # Ratios (adding small constant to prevent division by zero)
        (pl.col("Motor_current") / (pl.col("TP3") + 1e-5)).alias("current_pressure_ratio"),
        (pl.col("Oil_temperature") / (pl.col("TP3") + 1e-5)).alias("temp_pressure_ratio"),
        
        # Operating state proxy: high current (>6A) indicates under-load compression
        (pl.when(pl.col("Motor_current") > 6.0).then(1.0).otherwise(0.0)).alias("is_compressing_under_load"),
        
        # Low current (<1A) indicates off state
        (pl.when(pl.col("Motor_current") < 1.0).then(1.0).otherwise(0.0)).alias("is_compressor_off"),
    ])
    
    return df_features

def compute_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes rolling statistics over 30m, 60m, and 120m windows.
    Since average sampling is 10s intervals:
    - 30 minutes = 180 rows
    - 60 minutes = 360 rows
    - 120 minutes = 720 rows
    """
    # Define window sizes in terms of rows
    windows = {
        "30m": 180,
        "60m": 360,
        "120m": 720
    }
    
    # Columns to apply rolling features to
    cols_to_roll = [
        "TP2", "TP3", "H1", "DV_pressure", "Reservoirs", 
        "Motor_current", "Oil_temperature", "pressure_diff_tp3_tp2"
    ]
    
    roll_exprs = []
    
    # Sort dataframe by timestamp to ensure rolling is temporally ordered
    df_sorted = df.sort(config.TIMESTAMP_COLUMN)
    
    for name, size in windows.items():
        for col in cols_to_roll:
            if col in df_sorted.columns:
                # Rolling mean
                roll_exprs.append(
                    pl.col(col).rolling_mean(window_size=size).alias(f"{col}_roll_mean_{name}")
                )
                # Rolling standard deviation
                roll_exprs.append(
                    pl.col(col).rolling_std(window_size=size).alias(f"{col}_roll_std_{name}")
                )
                # Rolling min
                roll_exprs.append(
                    pl.col(col).rolling_min(window_size=size).alias(f"{col}_roll_min_{name}")
                )
                # Rolling max
                roll_exprs.append(
                    pl.col(col).rolling_max(window_size=size).alias(f"{col}_roll_max_{name}")
                )
                # Rate of change: current value minus rolling mean
                roll_exprs.append(
                    (pl.col(col) - pl.col(col).rolling_mean(window_size=size)).alias(f"{col}_diff_mean_{name}")
                )
                
    # Add digital signal sliding frequencies (duty cycles)
    for name, size in windows.items():
        for col in ["COMP", "DV_eletric", "TOWERS"]:
            if col in df_sorted.columns:
                # Rolling mean of boolean/digital values represents active percentage
                roll_exprs.append(
                    pl.col(col).rolling_mean(window_size=size).alias(f"{col}_duty_cycle_{name}")
                )
                
    # Evaluate all rolling expressions
    df_features = df_sorted.with_columns(roll_exprs)
    
    # Fill null values resulting from rolling window startup
    df_features = df_features.fill_null(strategy="backward")
    
    return df_features

def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Runs the entire feature engineering pipeline on a raw dataframe.
    """
    print("Engineering features...")
    df_domain = compute_domain_features(df)
    df_all = compute_rolling_features(df_domain)
    
    # Drop rows that still have nulls due to windowing if any
    df_all = df_all.drop_nulls(subset=["TP2_roll_mean_30m"])
    
    print(f"Feature engineering completed. New shape: {df_all.shape}")
    return df_all

if __name__ == "__main__":
    # Test feature pipeline with mock data
    import polars as pl
    import numpy as np
    from datetime import datetime, timedelta
    
    start_dt = datetime(2020, 4, 17, 12, 0)
    timestamps = [start_dt + timedelta(seconds=i*10) for i in range(1000)]
    mock_df = pl.DataFrame({
        config.TIMESTAMP_COLUMN: timestamps,
        "TP2": np.random.normal(6.0, 0.5, len(timestamps)).tolist(),
        "TP3": np.random.normal(8.0, 0.5, len(timestamps)).tolist(),
        "H1": np.random.normal(8.0, 0.2, len(timestamps)).tolist(),
        "DV_pressure": np.random.normal(0.1, 0.05, len(timestamps)).tolist(),
        "Reservoirs": np.random.normal(8.0, 0.4, len(timestamps)).tolist(),
        "Motor_current": np.random.normal(7.0, 1.0, len(timestamps)).tolist(),
        "Oil_temperature": np.random.normal(60.0, 5.0, len(timestamps)).tolist(),
        "COMP": [1.0] * len(timestamps),
        "DV_eletric": [0.0] * len(timestamps),
        "TOWERS": [0.0] * len(timestamps)
    })
    
    features_df = engineer_features(mock_df)
    print("Generated features shape:", features_df.shape)
    print("Example engineered columns:", [c for c in features_df.columns if "roll" in c][:10])
