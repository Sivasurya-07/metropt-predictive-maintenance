"""
MQTT subscriber client that buffers incoming sensor windows
for feeding into the inference pipeline.
"""
import json
import threading
from collections import deque
from typing import Optional, Callable

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

MQTT_TOPIC = "apu/sensor_data"


class SensorWindowBuffer:
    """
    Thread-safe sliding window buffer that accumulates
    sensor readings for a fixed window length (e.g. 60 timesteps).
    """

    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self._buffer: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def push(self, reading: dict):
        with self._lock:
            self._buffer.append(reading)

    def get_window(self) -> list[dict]:
        with self._lock:
            return list(self._buffer)

    @property
    def is_full(self) -> bool:
        with self._lock:
            return len(self._buffer) == self.window_size


class MQTTSensorClient:
    """
    Subscribes to the MQTT sensor topic and fills a SensorWindowBuffer.

    Args:
        host: MQTT broker host.
        port: MQTT broker port.
        window_size: Number of readings to accumulate per inference window.
        on_window_ready: Optional callback fired when buffer reaches window_size.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 1883,
        window_size: int = 60,
        on_window_ready: Optional[Callable[[list[dict]], None]] = None,
    ):
        self.host = host
        self.port = port
        self.buffer = SensorWindowBuffer(window_size=window_size)
        self.on_window_ready = on_window_ready
        self._client = None
        self._connected = False

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            client.subscribe(MQTT_TOPIC, qos=0)

    def _on_message(self, client, userdata, msg):
        try:
            reading = json.loads(msg.payload.decode("utf-8"))
            self.buffer.push(reading)
            if self.buffer.is_full and self.on_window_ready:
                self.on_window_ready(self.buffer.get_window())
        except Exception:
            pass

    def connect(self):
        if not MQTT_AVAILABLE:
            return
        self._client = mqtt.Client(client_id="apu_sensor_consumer")
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        try:
            self._client.connect(self.host, self.port, keepalive=60)
            self._client.loop_start()
        except Exception:
            self._connected = False

    def disconnect(self):
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False

    def push_direct(self, reading: dict):
        """
        Directly push a reading without MQTT (for testing or offline mode).
        """
        self.buffer.push(reading)
        if self.buffer.is_full and self.on_window_ready:
            self.on_window_ready(self.buffer.get_window())
