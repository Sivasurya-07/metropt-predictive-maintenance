import json
import os
from typing import List, Tuple
import numpy as np
import polars as pl
from datetime import datetime
import redis.asyncio as aioredis
from src.data.features import engineer_features
from src import config

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

class StatefulFeatureEngine:
    def __init__(self):
        self.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        self.stream_key = "apu_telemetry_stream"
        self.window_seconds = 7200  # 120 minutes is the max rolling window size

    async def push_and_get_window(self, raw_dicts: List[dict]) -> List[dict]:
        """
        Pushes new readings to the Redis sorted set and retrieves the historical
        window necessary to compute rolling features.
        """
        if not raw_dicts:
            return []

        # Use the timestamp of the last reading as the current time
        latest_ts_str = raw_dicts[-1].get(config.TIMESTAMP_COLUMN, datetime.utcnow().isoformat())
        try:
            latest_ts = datetime.fromisoformat(latest_ts_str.replace("Z", "+00:00")).timestamp()
        except Exception:
            latest_ts = datetime.utcnow().timestamp()

        # Pipeline for atomic operations
        async with self.redis.pipeline(transaction=True) as pipe:
            for r in raw_dicts:
                # Ensure unique member string even if timestamps collide exactly
                member = json.dumps(r) + f"_{np.random.randint(0, 10000)}"
                # Default timestamp fallback
                ts_str = r.get(config.TIMESTAMP_COLUMN)
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp() if ts_str else latest_ts
                except Exception:
                    ts = latest_ts
                
                pipe.zadd(self.stream_key, {member: ts})

            # Trim the window to (latest_ts - window_seconds)
            min_ts = latest_ts - self.window_seconds
            pipe.zremrangebyscore(self.stream_key, "-inf", min_ts)
            
            # Fetch the entire required window
            pipe.zrange(self.stream_key, 0, -1)
            
            results = await pipe.execute()
            
        # The last result in the pipeline is the zrange output
        window_entries = results[-1]
        
        # Parse JSON and remove the random entropy tag used for uniqueness
        window_dicts = [json.loads(e.rsplit("_", 1)[0]) for e in window_entries]
        return window_dicts

    async def generate_features(self, raw_dicts: List[dict]) -> Tuple[np.ndarray, List[str]]:
        """
        Stateful alternative to dependencies.readings_to_feature_array
        """
        # 1. Fetch full stateful window from Redis
        window_dicts = await self.push_and_get_window(raw_dicts)
        
        if len(window_dicts) == 0:
            raise ValueError("No valid telemetry data in window.")

        df = pl.DataFrame(window_dicts)

        timestamp_col = config.TIMESTAMP_COLUMN
        if "timestamp" in df.columns and timestamp_col not in df.columns:
            df = df.rename({"timestamp": timestamp_col})

        if df[timestamp_col].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col(timestamp_col).str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.f", strict=False)
            )

        # 2. Compute heavy rolling features
        df_feat = engineer_features(df)

        # 3. Only keep the rows corresponding to the NEW raw_dicts injected
        # We slice the last N rows where N is len(raw_dicts)
        num_new = len(raw_dicts)
        df_new_feat = df_feat.tail(num_new)

        drop_cols = [timestamp_col] + [
            c for c in df_new_feat.columns
            if c.startswith("label_") or c in ("failure_in_next", "GPS_speed", "GPS_latitude", "GPS_longitude")
        ]
        drop_cols = [c for c in drop_cols if c in df_new_feat.columns]
        feature_df = df_new_feat.drop(drop_cols)

        feature_names = feature_df.columns
        X = feature_df.to_numpy(allow_copy=True)
        return X, list(feature_names)

    async def flush_to_parquet(self, filepath: str = "data/processed/telemetry_archive.parquet"):
        """Flushes the current Redis window to a highly compressed Parquet file for historical archiving."""
        entries = await self.redis.zrange(self.stream_key, 0, -1)
        if not entries:
            return
            
        window_dicts = [json.loads(e.rsplit("_", 1)[0]) for e in entries]
        df = pl.DataFrame(window_dicts)
        
        timestamp_col = config.TIMESTAMP_COLUMN
        if "timestamp" in df.columns and timestamp_col not in df.columns:
            df = df.rename({"timestamp": timestamp_col})
            
        if df[timestamp_col].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col(timestamp_col).str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.f", strict=False)
            )
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if os.path.exists(filepath):
            df_existing = pl.read_parquet(filepath)
            df_combined = pl.concat([df_existing, df]).unique(subset=[timestamp_col], keep="last")
            df_combined.write_parquet(filepath)
        else:
            df.write_parquet(filepath)
        print(f"[Archiver] Flushed {len(df)} rows to {filepath}")

engine = StatefulFeatureEngine()
