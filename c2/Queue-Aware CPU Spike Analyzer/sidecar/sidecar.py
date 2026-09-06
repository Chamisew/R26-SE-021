"""
Sidecar Proxy — Queue-Aware Runtime Telemetry


  Logic:
    • If TARGET_HEALTH_PATH is unset  → only checks TCP reachability.
    • If TARGET_HEALTH_PATH is set    → also checks HTTP status on that path;
                                        status >= 500 marks health_check_failed.
    • 404 is NOT treated as unreachable (service may not expose root).
    • Updates `last_service_unreachable` and `last_health_check_failed`
      so /metrics always reflects the latest health state.
"""

import asyncio
import os
import time
import logging
from collections import deque
from datetime import datetime, timezone

import aiohttp
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
# RUNTIME STATE
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

incoming_timestamps: deque = deque()
processing_timestamps: deque = deque()

state_lock = asyncio.Lock()


# =============================================================================
# HELPERS
# =============================================================================

def cleanup_old_timestamps():
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
    try:
        quota_path = "/sys/fs/cgroup/cpu.max"
        if os.path.exists(quota_path):
            with open(quota_path, "r") as f:
                values = f.read().strip().split()
            if len(values) >= 2 and values[0] != "max":
                cpus = float(values[0]) / float(values[1])
                return f"{round(cpus, 2)}vCPU"
    except Exception:
        pass

    try:
        quota_path = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
        period_path = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
        if os.path.exists(quota_path) and os.path.exists(period_path):
            with open(quota_path, "r") as f:
                quota = float(f.read().strip())
            with open(period_path, "r") as f:
                period = float(f.read().strip())
            if quota > 0 and period > 0:
                return f"{round(quota / period, 2)}vCPU"
    except Exception:
        pass

    cpu_count = psutil.cpu_count()
    return f"{cpu_count}vCPU" if cpu_count else "unknown"


def calculate_error_rate():
    total = state["incoming_total"]
    if total <= 0:
        return 0.0
    return state["http_errors_total"] / total


# =============================================================================
# MIDDLEWARE
# =============================================================================

@web.middleware
async def telemetry_middleware(request, handler):
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
# /METRICS
# =============================================================================

async def metrics(request):
    async with state_lock:
        cleanup_old_timestamps()

        incoming_rate = len(incoming_timestamps)
        processing_rate = len(processing_timestamps)
        queue_length = state["active_requests"]
        http_errors = state["http_errors_total"]
        timeouts = state["timeouts_total"]
        last_latency = state["last_latency_ms"]

    cpu_percent = psutil.cpu_percent(interval=0.1)

    error_rate = 0.0
    if incoming_rate > 0:
        error_rate = http_errors / max(state["incoming_total"], 1)

    return web.json_response({
        "cpu_usage_percent": round(cpu_percent, 2),
        "queue_length": queue_length,
        "incoming_requests": incoming_rate,
        "processing_requests": processing_rate,
        "latency_ms": round(last_latency, 2),
        "service_unreachable": int(state["last_service_unreachable"]),
        "health_check_failed": int(state["last_health_check_failed"]),
        "request_timeout": int(timeouts > 0),
        "http_errors_per_sec": float(http_errors),
        "error_rate": round(error_rate, 6),
        "platform": PLATFORM_NAME,
        "service": SERVICE_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cpu_allocation": get_cpu_allocation(),
        "measurement_method": "sidecar_runtime_and_proxy",
        "uptime_seconds": round(time.time() - START_TIME, 2),
    })


# =============================================================================
# /HEALTH  ← NEW
# =============================================================================

async def health(request):
    """
    Check whether the target microservice is reachable.

    If TARGET_HEALTH_PATH is set, the response status of that path is
    also evaluated; HTTP 5xx is treated as health_check_failed.

    404 on the root path is deliberately ignored — many services do not
    expose a root or /health route and that must not be treated as down.
    """
    target_url = f"http://{TARGET_HOST}:{TARGET_PORT}"

    if TARGET_HEALTH_PATH:
        target_url += TARGET_HEALTH_PATH

    start_time = time.monotonic()

    target_reachable = False
    health_failed = False

    try:
        timeout = aiohttp.ClientTimeout(total=5)

        async with aiohttp.ClientSession() as session:
            async with session.get(target_url, timeout=timeout) as response:
                target_reachable = True

                if TARGET_HEALTH_PATH:
                    health_failed = response.status >= 500
                else:
                    health_failed = False

    except asyncio.TimeoutError:
        target_reachable = False
        health_failed = True

    except aiohttp.ClientError:
        target_reachable = False
        health_failed = True

    except Exception:
        target_reachable = False
        health_failed = True

    latency_ms = (time.monotonic() - start_time) * 1000

    async with state_lock:
        state["last_service_unreachable"] = int(not target_reachable)
        state["last_health_check_failed"] = int(health_failed)

    return web.json_response({
        "status": "ok" if target_reachable and not health_failed else "failed",
        "service_id": SERVICE_NAME,
        "target_reachable": target_reachable,
        "health_check_failed": int(health_failed),
        "latency_ms": round(latency_ms, 2),
        "target": target_url,
    })


# =============================================================================
# APPLICATION
# =============================================================================

app = web.Application(middlewares=[telemetry_middleware])

app.router.add_get("/ping", ping)
app.router.add_get("/metrics", metrics)
app.router.add_get("/health", health)


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

