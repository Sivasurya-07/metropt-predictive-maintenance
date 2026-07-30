import pandera.polars as pa
import polars as pl
from src import config

# Raw data validation schema
raw_schema = pa.DataFrameSchema({
    config.TIMESTAMP_COLUMN: pa.Column(pl.Datetime, required=True),
    "TP2": pa.Column(pl.Float64, checks=pa.Check.between(-1.0, 20.0), required=True),
    "TP3": pa.Column(pl.Float64, checks=pa.Check.between(-1.0, 20.0), required=True),
    "H1": pa.Column(pl.Float64, checks=pa.Check.between(-1.0, 20.0), required=True),
    "DV_pressure": pa.Column(pl.Float64, checks=pa.Check.between(-1.0, 20.0), required=True),
    "Reservoirs": pa.Column(pl.Float64, checks=pa.Check.between(-1.0, 20.0), required=True),
    "Motor_current": pa.Column(pl.Float64, checks=pa.Check.between(-1.0, 50.0), required=True),
    "Oil_temperature": pa.Column(pl.Float64, checks=pa.Check.between(-10.0, 150.0), required=True),
    
    # Digital columns - might be loaded as Float64, Int64, or Boolean, so we relax checks to be numeric and check range [0, 1]
    "COMP": pa.Column(pl.Float64, checks=pa.Check.between(0.0, 1.1), required=False),
    "DV_eletric": pa.Column(pl.Float64, checks=pa.Check.between(0.0, 1.1), required=False),
    "TOWERS": pa.Column(pl.Float64, checks=pa.Check.between(0.0, 1.1), required=False),
    "MPG": pa.Column(pl.Float64, checks=pa.Check.between(0.0, 1.1), required=False),
    "LPS": pa.Column(pl.Float64, checks=pa.Check.between(0.0, 1.1), required=False),
    "Pressure_switch": pa.Column(pl.Float64, checks=pa.Check.between(0.0, 1.1), required=False),
    "Oil_level": pa.Column(pl.Float64, checks=pa.Check.between(0.0, 1.1), required=False),
    "Flowmeter": pa.Column(pl.Float64, checks=pa.Check.between(0.0, 1.1), required=False),
    
    "GPS_speed": pa.Column(pl.Float64, checks=pa.Check.between(0.0, 160.0), required=False),
    "GPS_latitude": pa.Column(pl.Float64, checks=pa.Check.between(-90.0, 90.0), required=False),
    "GPS_longitude": pa.Column(pl.Float64, checks=pa.Check.between(-180.0, 180.0), required=False),
})

def validate_raw_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Validates raw dataframe against schema.
    Returns validated dataframe, or raises SchemaError.
    """
    # Ensure digital and GPS columns are double to prevent type mismatches
    cast_dict = {}
    for col in config.DIGITAL_COLUMNS + config.GPS_COLUMNS:
        if col in df.columns:
            cast_dict[col] = pl.Float64
            
    if cast_dict:
        df = df.with_columns([pl.col(c).cast(t) for c, t in cast_dict.items()])
        
    print("Validating raw data schema...")
    validated_df = raw_schema.validate(df)
    print("Raw data schema validation passed.")
    return validated_df
