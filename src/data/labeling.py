import polars as pl
from datetime import datetime, timedelta
from typing import List, Tuple
from src import config

# List of documented failure intervals in the MetroPT-3 dataset (Feb 2020 - Aug 2020)
# These represent the start and end times of known failures/maintenance events.
FAILURE_INTERVALS: List[Tuple[datetime, datetime]] = [
    (datetime(2020, 4, 18, 0, 0), datetime(2020, 4, 18, 23, 59)),   # Air Leak Nr. 1
    (datetime(2020, 5, 29, 23, 30), datetime(2020, 5, 30, 6, 0)),   # Air Leak Nr. 2 (May 29-30)
    (datetime(2020, 6, 5, 10, 0), datetime(2020, 6, 7, 14, 30)),     # Air Leak Nr. 3 (June 5-7)
    (datetime(2020, 7, 15, 14, 30), datetime(2020, 7, 15, 19, 0)),  # Air Leak Nr. 4 (July 15)
]

def get_failure_intervals() -> List[Tuple[datetime, datetime]]:
    """
    Returns the list of known failure intervals.
    """
    return FAILURE_INTERVALS

def label_data(df: pl.DataFrame, horizons_hours: dict = None) -> pl.DataFrame:
    """
    Applies the multi-horizon labeling strategy to the dataframe.
    For each horizon:
      - 1 (positive): within the horizon window before a failure starts
      - -1 (neutral/ignore): inside the failure interval (active failure/repair)
      - 0 (negative): normal operating conditions
    """
    if horizons_hours is None:
        horizons_hours = config.HORIZONS
        
    labeled_df = df.clone()
    timestamp_col = config.TIMESTAMP_COLUMN
    
    # Initialize label columns with 0 (default: healthy)
    for horizon_name in horizons_hours.keys():
        label_col = config.get_label_name(horizon_name)
        labeled_df = labeled_df.with_columns(pl.lit(0).alias(label_col))
        
    # We will process each failure interval and assign positive and neutral labels
    for start_time, end_time in FAILURE_INTERVALS:
        # Mark neutral window (inside the active failure / maintenance window)
        for horizon_name in horizons_hours.keys():
            label_col = config.get_label_name(horizon_name)
            # Assign -1 to timestamps within [start_time, end_time]
            labeled_df = labeled_df.with_columns(
                pl.when(
                    (pl.col(timestamp_col) >= start_time) & 
                    (pl.col(timestamp_col) <= end_time)
                )
                .then(-1)
                .otherwise(pl.col(label_col))
                .alias(label_col)
            )
            
            # Mark warning window (before the failure starts)
            horizon_duration = timedelta(hours=horizons_hours[horizon_name])
            warning_start = start_time - horizon_duration
            
            # Assign 1 to timestamps within [warning_start, start_time)
            # (Note: we use strict less than for start_time so it doesn't overlap the neutral window)
            labeled_df = labeled_df.with_columns(
                pl.when(
                    (pl.col(timestamp_col) >= warning_start) & 
                    (pl.col(timestamp_col) < start_time) &
                    (pl.col(label_col) != -1)  # don't overwrite neutral if there was overlap
                )
                .then(1)
                .otherwise(pl.col(label_col))
                .alias(label_col)
            )
            
    # Print label distribution
    print("Label distributions generated:")
    for horizon_name in horizons_hours.keys():
        label_col = config.get_label_name(horizon_name)
        counts = labeled_df.group_by(label_col).len().sort(label_col)
        print(f"Horizon {horizon_name} distribution:")
        for label, count in counts.iter_rows():
            print(f"  Label {label}: {count} samples")
        
    return labeled_df

if __name__ == "__main__":
    # Test labeling helper with mock dataset
    start_dt = datetime(2020, 4, 17, 12, 0)
    timestamps = [start_dt + timedelta(minutes=i*10) for i in range(250)]
    mock_df = pl.DataFrame({
        config.TIMESTAMP_COLUMN: timestamps,
        "TP2": [1.0] * len(timestamps)
    })
    labeled = label_data(mock_df)
    print(labeled.filter(pl.col("label_failure_2h") != 0).head())
