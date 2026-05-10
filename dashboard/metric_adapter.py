"""
METRIC ADAPTER
==============
This is the bridge between your two external components
(the ones that already collect CPU / queue metrics from live microservices)
and this prediction system.

HOW IT WORKS
------------
Your external components push metrics via HTTP POST to:
    POST /api/ingest
with a JSON body (see schema below).

Alternatively, if your components cannot push, this adapter
can PULL from them on a configurable interval.

INGESTION SCHEMA (what your components must send)
--------------------------------------------------
{
  "service":          "python-api",      # any unique string
  "cpu_percent":      43.7,              # float 0-100
  "in_flight_queue":  5.0,               # jobs currently in queue
  "incoming_rate":    40.0,              # requests/jobs per second arriving
  "processing_rate":  35.0,              # requests/jobs per second processed
  "queue_growth_rate": 5.0,             # (incoming - processing) delta
  "cpu_trend_5min_ma": 12.3,            # optional: 5-min moving avg if your component computes it
  "cpu_trend_10min_ma": 8.1,            # optional: 10-min moving avg
  "timestamp":        1778081050.0       # optional: unix epoch, defaults to now
}

PULL MODE
---------
If PULL_URLS is configured below, the adapter will call each
URL every PULL_INTERVAL_SEC seconds and expect the same JSON schema.
Set PULL_URLS = [] to disable pull mode and rely on push only.
"""

import time
import threading
import requests
import logging
from collections import defaultdict, deque
from datetime import datetime

logger = logging.getLogger("metric_adapter")

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
# Add the URLs your two components expose here.
# Example: "http://component-1:8080/metrics"
PULL_URLS: list[str] = [
    # "http://localhost:8081/metrics",   # Component 1 (CPU collector)
    # "http://localhost:8082/metrics",   # Component 2 (Queue / memory collector)
]

PULL_INTERVAL_SEC = 5          # how often to pull (seconds)
HISTORY_PER_SERVICE = 200      # rows kept in memory per service
WARMUP_ROWS = 10               # rows needed before predictions are made
# ────────────────────────────────────────────────────────────────────────────────


class ServiceBuffer:
    """
    Holds a rolling window of metric rows for ONE service.
    Thread-safe via a simple lock.
    """
    def __init__(self, name: str):
        self.name    = name
        self.rows    = deque(maxlen=HISTORY_PER_SERVICE)
        self.lock    = threading.Lock()
        self.warmed  = False

    def push(self, row: dict):
        with self.lock:
            row["service"] = self.name
            row.setdefault("timestamp", time.time())
            row.setdefault("cpu_trend_5min_ma",  0.0)
            row.setdefault("cpu_trend_10min_ma", 0.0)
            row.setdefault("queue_growth_rate",
                           row.get("incoming_rate", 0) - row.get("processing_rate", 0))
            self.rows.append(row)
            if not self.warmed and len(self.rows) >= WARMUP_ROWS:
                self.warmed = True
                logger.info(f"Service '{self.name}' passed warm-up ({WARMUP_ROWS} rows)")

    def snapshot(self) -> list[dict]:
        """Return a copy of all buffered rows (oldest → newest)."""
        with self.lock:
            return list(self.rows)

    def latest(self) -> dict | None:
        with self.lock:
            return self.rows[-1] if self.rows else None

    @property
    def row_count(self) -> int:
        with self.lock:
            return len(self.rows)


class MetricStore:
    """
    Central store for all service buffers.
    One instance lives for the lifetime of the Flask app.
    """
    def __init__(self):
        self._buffers: dict[str, ServiceBuffer] = {}
        self._lock    = threading.Lock()
        self._puller  = None

    def get_or_create(self, service_name: str) -> ServiceBuffer:
        with self._lock:
            if service_name not in self._buffers:
                self._buffers[service_name] = ServiceBuffer(service_name)
                logger.info(f"New service registered: {service_name}")
            return self._buffers[service_name]

    def ingest(self, payload: dict) -> tuple[bool, str]:
        """
        Accept a metric payload dict.
        Returns (success, error_message).
        """
        service = payload.get("service") or payload.get("service_name")
        if not service:
            return False, "Missing 'service' field"

        required = ["cpu_percent"]
        for f in required:
            if f not in payload:
                return False, f"Missing required field: {f}"

        buf = self.get_or_create(service)
        buf.push({
            "service":           service,
            "project_id":        payload.get("project_id", "unknown"),
            "cpu_percent":       float(payload.get("cpu_percent", 0)),
            "in_flight_queue":   float(payload.get("in_flight_queue", 0)),
            "incoming_rate":     float(payload.get("incoming_rate", 0)),
            "processing_rate":   float(payload.get("processing_rate", 0)),
            "queue_growth_rate": float(payload.get("queue_growth_rate", 0)),
            "cpu_trend_5min_ma": float(payload.get("cpu_trend_5min_ma", 0)),
            "cpu_trend_10min_ma":float(payload.get("cpu_trend_10min_ma", 0)),
            "timestamp":         float(payload.get("timestamp", time.time())),
            "label":             int(payload.get("label", 0)),
        })
        return True, ""

    def all_services(self) -> list[str]:
        with self._lock:
            return list(self._buffers.keys())

    def get_buffer(self, service: str) -> ServiceBuffer | None:
        with self._lock:
            return self._buffers.get(service)

    def start_puller(self):
        """Start background thread that pulls from PULL_URLS."""
        if not PULL_URLS:
            logger.info("Pull mode disabled (PULL_URLS is empty). Waiting for push.")
            return
        self._puller = threading.Thread(target=self._pull_loop, daemon=True)
        self._puller.start()
        logger.info(f"Puller started — polling {len(PULL_URLS)} URLs every {PULL_INTERVAL_SEC}s")

    def _pull_loop(self):
        while True:
            for url in PULL_URLS:
                try:
                    resp = requests.get(url, timeout=4)
                    resp.raise_for_status()
                    data = resp.json()
                    # Support both single dict and list of dicts
                    rows = data if isinstance(data, list) else [data]
                    for row in rows:
                        ok, err = self.ingest(row)
                        if not ok:
                            logger.warning(f"Bad payload from {url}: {err}")
                except Exception as e:
                    logger.warning(f"Pull failed for {url}: {e}")
            time.sleep(PULL_INTERVAL_SEC)


# Singleton — imported by app.py
metric_store = MetricStore()
