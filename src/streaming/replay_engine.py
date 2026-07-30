"""
Streaming Replay Engine for MetroPT sensor data.
Reads the raw CSV chronologically and publishes rows as JSON
to MQTT topic 'apu/sensor_data' at a configurable replay speed.
"""
import sys
import json
import time
import threading
from pathlib import Path
from typing import Optional

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

MQTT_TOPIC = "apu/sensor_data"
SENSOR_INTERVAL_SECONDS = 10  # MetroPT records at 10-second intervals


class SensorReplayEngine:
    """
    Replays MetroPT CSV as a real-time MQTT stream.

    Args:
        csv_path: Path to the raw MetroPT CSV file.
        host: MQTT broker host (default localhost).
        port: MQTT broker port (default 1883).
        speed_multiplier: Replay speed relative to real-time.
                          1.0 = real-time, 10.0 = 10x faster.
        start_row: Row index to begin replay from.
    """

    def __init__(
        self,
        csv_path: Optional[Path] = None,
        host: str = "localhost",
        port: int = 1883,
        speed_multiplier: float = 600.0,  # Default 600x = fast demo mode
        start_row: int = 0,
    ):
        self.csv_path = csv_path or (config.RAW_DATA_DIR / config.CSV_FILENAME)
        self.host = host
        self.port = port
        self.speed_multiplier = speed_multiplier
        self.start_row = start_row

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._client: Optional[object] = None
        self._connected = False

        # In-memory buffer: published rows for non-MQTT environments
        self.published_rows: list[dict] = []

    def _connect_mqtt(self) -> bool:
        """Tries to connect to MQTT broker. Returns True on success."""
        if not MQTT_AVAILABLE:
            return False
        try:
            self._client = mqtt.Client(client_id="apu_replay_engine")
            self._client.connect(self.host, self.port, keepalive=60)
            self._client.loop_start()
            self._connected = True
            return True
        except Exception:
            self._connected = False
            return False

    def _publish(self, payload: dict):
        """Publishes a sensor reading. Falls back to in-memory buffer if MQTT unavailable."""
        self.published_rows.append(payload)
        if self._connected and self._client:
            self._client.publish(MQTT_TOPIC, json.dumps(payload), qos=0)

    def _replay_loop(self, df: pl.DataFrame, delay: float, max_rows: Optional[int]):
        """Main replay loop executed in a background thread."""
        rows = df.iter_rows(named=True)
        count = 0
        for row in rows:
            if self._stop_event.is_set():
                break
            if max_rows and count >= max_rows:
                break
            self._publish(row)
            count += 1
            time.sleep(delay)

    def start(self, max_rows: Optional[int] = None, blocking: bool = False):
        """
        Starts the replay engine.

        Args:
            max_rows: Limits total rows replayed. None = full dataset.
            blocking: If True, runs in current thread (for testing).
        """
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")

        df = pl.read_csv(self.csv_path, try_parse_dates=True)
        if self.start_row > 0:
            df = df.slice(self.start_row)

        delay = SENSOR_INTERVAL_SECONDS / self.speed_multiplier

        self._connect_mqtt()
        self._stop_event.clear()

        if blocking:
            self._replay_loop(df, delay, max_rows)
        else:
            self._thread = threading.Thread(
                target=self._replay_loop,
                args=(df, delay, max_rows),
                daemon=True,
            )
            self._thread.start()

    def stop(self):
        """Signals the replay loop to stop and disconnects MQTT."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._connected and self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False
