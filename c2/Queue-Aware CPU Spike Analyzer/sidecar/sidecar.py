"""

Adds on top of commit 1:
  - `state`  — runtime counters
  - `state_lock` — asyncio lock protecting state
  - `incoming_timestamps` / `processing_timestamps` — deques for rate windows
  - `cleanup_old_timestamps()` — keeps only the last-second window
  - `get_cpu_allocation()` — reads cgroup quota or falls back to cpu_count
  - `calculate_error_rate()` — ratio helper
  - `telemetry_middleware` — tracks active requests, latency, errors, timeouts
"""

import asyncio
import os
import time
import logging
from collections import deque

import psutil
from aiohttp import web


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SIDECAR] %(message)s"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET_HOST = os.environ.get("TARGET_HOST", "127.0.0.1")
TARGET_PORT = int(os.environ.get("TARGET_PORT", 8080))

SIDECAR_PORT = int(os.environ.get("SIDECAR_PORT", os.environ.get("PORT", 8091)))

SERVICE_NAME = os.environ.get("SERVICE_NAME", "microservice")

PLATFORM_NAME = os.environ.get("PLATFORM", "azure")

START_TIME = time.time()

TARGET_HEALTH_PATH = os.environ.get("TARGET_HEALTH_PATH")


# =============================================================================
# RUNTIME STATE  ← NEW
# =============================================================================

state = {
    "incoming_total": 0,
    "processing_total": 0,

    "active_requests": 0,

    "http_errors_total": 0,
    "timeouts_total": 0,

    "last_latency_ms": 0.0,

    "last_health_check_failed": 0,

    "last_service_unreachable": 0,
}

# Request timestamps for rate calculation (last-1-second window).
incoming_timestamps: deque = deque()
processing_timestamps: deque = deque()


# =============================================================================
# LOCK  ← NEW
# =============================================================================

state_lock = asyncio.Lock()


# =============================================================================
# HELPERS  ← NEW
# =============================================================================

def cleanup_old_timestamps():
    """
    Keep only timestamps from the previous second.

    This allows incoming_requests and processing_requests
    to represent requests/second rather than lifetime totals.
    """
    now = time.monotonic()

    while incoming_timestamps:
        if now - incoming_timestamps[0] <= 1.0:
            break
        incoming_timestamps.popleft()

    while processing_timestamps:
        if now - processing_timestamps[0] <= 1.0:
            break
        processing_timestamps.popleft()


def get_cpu_allocation():
    """
    Detect CPU allocation from the Linux cgroup when available.
    Falls back to the number of CPUs visible to the runtime.
    """
    # Docker / Linux cgroup v2
    try:
        quota_path = "/sys/fs/cgroup/cpu.max"
        if os.path.exists(quota_path):
            with open(quota_path, "r") as f:
                values = f.read().strip().split()
            if len(values) >= 2:
                quota, period = values[0], values[1]
                if quota != "max":
                    cpus = float(quota) / float(period)
                    return f"{round(cpus, 2)}vCPU"
    except Exception:
        pass

    # Docker / Linux cgroup v1
    try:
        quota_path = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
        period_path = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
        if os.path.exists(quota_path) and os.path.exists(period_path):
            with open(quota_path, "r") as f:
                quota = float(f.read().strip())
            with open(period_path, "r") as f:
                period = float(f.read().strip())
            if quota > 0 and period > 0:
                cpus = quota / period
                return f"{round(cpus, 2)}vCPU"
    except Exception:
        pass

    # Runtime-visible CPU count fallback.
    cpu_count = psutil.cpu_count()
    if cpu_count:
        return f"{cpu_count}vCPU"

    return "unknown"


def calculate_error_rate():
    """Return HTTP error ratio against lifetime incoming total."""
    total = state["incoming_total"]
    if total <= 0:
        return 0.0
    return state["http_errors_total"] / total


# =============================================================================
# REQUEST MIDDLEWARE  ← NEW
# =============================================================================

@web.middleware
async def telemetry_middleware(request, handler):
    """
    Intercept every non-internal request to track:
      - incoming count and rate
      - active (in-flight) requests
      - completed count and rate
      - per-request latency
      - HTTP error count
      - timeout count
    """
    internal_paths = {"/metrics", "/health", "/ping", "/spike"}

    if request.path in internal_paths:
        return await handler(request)

    start_time = time.monotonic()

    async with state_lock:
        state["incoming_total"] += 1
        state["active_requests"] += 1
        incoming_timestamps.append(time.monotonic())

    try:
        response = await handler(request)

        if response.status >= 400:
            async with state_lock:
                state["http_errors_total"] += 1

        return response

    except asyncio.TimeoutError:
        async with state_lock:
            state["timeouts_total"] += 1
            state["http_errors_total"] += 1
        raise

    except Exception:
        async with state_lock:
            state["http_errors_total"] += 1
        raise

    finally:
        latency_ms = (time.monotonic() - start_time) * 1000

        async with state_lock:
            state["processing_total"] += 1
            state["active_requests"] -= 1
            state["last_latency_ms"] = latency_ms
            processing_timestamps.append(time.monotonic())


# =============================================================================
# /PING
# =============================================================================

async def ping(request):
    return web.json_response({
        "status": "ok",
        "service": SERVICE_NAME,
        "platform": PLATFORM_NAME,
        "sidecar": True
    })


# =============================================================================
# APPLICATION
# =============================================================================

app = web.Application(middlewares=[telemetry_middleware])

app.router.add_get("/ping", ping)


# =============================================================================
# START
# =============================================================================

if __name__ == "__main__":
    logging.info(f"Sidecar '{SERVICE_NAME}' starting")
    logging.info(f"Target: {TARGET_HOST}:{TARGET_PORT}")
    logging.info(f"Listening on port {SIDECAR_PORT}")

    web.run_app(
        app,
        host="0.0.0.0",
        port=SIDECAR_PORT,
        access_log=None
    )

