"""
Sidecar Proxy — Queue-Aware Runtime Telemetry

The sidecar:
1. Proxies requests to the real microservice.
2. Measures CPU utilization from the sidecar/container runtime.
3. Measures active/in-flight requests.
4. Calculates incoming and processing rates from real requests.
5. Measures real request latency.
6. Tracks HTTP errors.
7. Tracks request timeouts.
8. Detects target-service reachability.
9. Provides /ping, /metrics, /health and /spike.

No artificial metric values are generated.

Configuration is supplied through environment variables.
"""

import asyncio
import math
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


# Health endpoint of the real service.
#
# If TARGET_HEALTH_PATH is supplied, that endpoint is checked.
# Otherwise the sidecar only checks whether the target can be reached.
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


# Request timestamps used to calculate real request rates.
incoming_timestamps = deque()
processing_timestamps = deque()


# =============================================================================
# LOCK
# =============================================================================

state_lock = asyncio.Lock()


# =============================================================================
# HELPERS
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

    This avoids returning a hardcoded value such as '1vCPU'.

    Falls back to the number of CPUs visible to the runtime.
    """

    # Docker / Linux cgroup v2
    try:
        quota_path = "/sys/fs/cgroup/cpu.max"

        if os.path.exists(quota_path):
            with open(quota_path, "r") as f:
                values = f.read().strip().split()

            if len(values) >= 2:
                quota = values[0]
                period = values[1]

                if quota != "max":
                    quota_value = float(quota)
                    period_value = float(period)

                    if period_value > 0:
                        cpus = quota_value / period_value

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

    # Runtime-visible CPU count.
    cpu_count = psutil.cpu_count()

    if cpu_count:
        return f"{cpu_count}vCPU"

    return "unknown"


def calculate_error_rate():
    """
    Calculate HTTP error ratio during the current one-second window.
    """

    total = state["incoming_total"]

    if total <= 0:
        return 0.0

    return state["http_errors_total"] / total


# =============================================================================
# REQUEST MIDDLEWARE
# =============================================================================

@web.middleware
async def telemetry_middleware(request, handler):

    # Do not count sidecar telemetry endpoints as application traffic.
    internal_paths = {
        "/metrics",
        "/health",
        "/ping",
        "/spike"
    }

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

    # Real CPU utilization of the runtime/container.
    cpu_percent = psutil.cpu_percent(interval=0.1)

    error_rate = 0.0

    if incoming_rate > 0:
        error_rate = http_errors / max(
            state["incoming_total"],
            1
        )

    return web.json_response({

        # =========================================================
        # Required telemetry fields
        # =========================================================

        "cpu_usage_percent": round(cpu_percent, 2),

        "queue_length": queue_length,

        "incoming_requests": incoming_rate,

        "processing_requests": processing_rate,

        "latency_ms": round(last_latency, 2),

        "service_unreachable": int(
            state["last_service_unreachable"]
        ),

        "health_check_failed": int(
            state["last_health_check_failed"]
        ),

        "request_timeout": int(
            timeouts > 0
        ),

        "http_errors_per_sec": float(http_errors),

        "error_rate": round(error_rate, 6),

        # =========================================================
        # Runtime information
        # =========================================================

        "platform": PLATFORM_NAME,

        "service": SERVICE_NAME,

        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "cpu_allocation": get_cpu_allocation(),

        "measurement_method": "sidecar_runtime_and_proxy",

        "uptime_seconds": round(
            time.time() - START_TIME,
            2
        )
    })


# =============================================================================
# /HEALTH
# =============================================================================

async def health(request):

    target_url = f"http://{TARGET_HOST}:{TARGET_PORT}"

    if TARGET_HEALTH_PATH:
        target_url += TARGET_HEALTH_PATH

    start_time = time.monotonic()

    target_reachable = False
    health_failed = False

    try:

        timeout = aiohttp.ClientTimeout(
            total=5
        )

        async with aiohttp.ClientSession() as session:

            async with session.get(
                target_url,
                timeout=timeout
            ) as response:

                target_reachable = True

                # A response means the service is reachable.
                #
                # 404 is NOT treated as unreachable.
                # This is important because your Python service
                # currently does not define '/' or '/health'.

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

    latency_ms = (
        time.monotonic() - start_time
    ) * 1000

    async with state_lock:

        state["last_service_unreachable"] = int(
            not target_reachable
        )

        state["last_health_check_failed"] = int(
            health_failed
        )

    return web.json_response({

        "status": (
            "ok"
            if target_reachable and not health_failed
            else "failed"
        ),

        "service_id": SERVICE_NAME,

        "target_reachable": target_reachable,

        "health_check_failed": int(
            health_failed
        ),

        "latency_ms": round(
            latency_ms,
            2
        ),

        "target": target_url
    })


# =============================================================================
# /SPIKE
# =============================================================================

async def spike(request):

    duration_parameter = request.rel_url.query.get(
        "duration"
    )

    if duration_parameter is None:
        return web.json_response(
            {
                "error": "duration parameter is required"
            },
            status=400
        )

    try:
        duration = float(duration_parameter)

    except ValueError:

        return web.json_response(
            {
                "error": "duration must be numeric"
            },
            status=400
        )

    if duration <= 0:

        return web.json_response(
            {
                "error": "duration must be greater than zero"
            },
            status=400
        )

    logging.info(
        f"[{SERVICE_NAME}] CPU spike requested "
        f"for {duration} seconds"
    )

    loop = asyncio.get_running_loop()

    await loop.run_in_executor(
        None,
        _cpu_burn,
        time.monotonic() + duration
    )

    return web.json_response({
        "message": "CPU spike completed",
        "service_id": SERVICE_NAME,
        "duration_seconds": duration
    })


def _cpu_burn(end_time):

    while time.monotonic() < end_time:

        math.sqrt(
            64 ** 5
        )


# =============================================================================
# TRANSPARENT PROXY
# =============================================================================

async def proxy(request):

    target_url = (
        f"http://{TARGET_HOST}:"
        f"{TARGET_PORT}"
        f"{request.path_qs}"
    )

    start_time = time.monotonic()

    try:

        timeout = aiohttp.ClientTimeout(
            total=30
        )

        async with aiohttp.ClientSession() as session:

            async with session.request(
                method=request.method,
                url=target_url,

                headers={
                    key: value
                    for key, value in request.headers.items()
                    if key.lower() != "host"
                },

                data=await request.read(),

                timeout=timeout

            ) as response:

                body = await response.read()

                latency_ms = (
                    time.monotonic() - start_time
                ) * 1000

                # -------------------------------------------------
                # Update actual runtime measurements
                # -------------------------------------------------

                async with state_lock:

                    state["last_service_unreachable"] = 0

                    state["last_latency_ms"] = latency_ms

                    if response.status >= 400:
                        state["http_errors_total"] += 1

                # -------------------------------------------------
                # Forward response
                # -------------------------------------------------

                skip_headers = {
                    "transfer-encoding",
                    "content-encoding",
                    "connection",
                    "keep-alive"
                }

                forwarded_headers = {
                    key: value
                    for key, value in response.headers.items()
                    if key.lower() not in skip_headers
                }

                return web.Response(
                    body=body,
                    status=response.status,
                    headers=forwarded_headers
                )

    except asyncio.TimeoutError:

        async with state_lock:

            state["timeouts_total"] += 1
            state["http_errors_total"] += 1
            state["last_service_unreachable"] = 1

        logging.error(
            f"Request timeout -> {target_url}"
        )

        return web.json_response(
            {
                "error": "Target service request timed out",
                "target": target_url
            },
            status=504
        )

    except aiohttp.ClientError as error:

        async with state_lock:

            state["http_errors_total"] += 1
            state["last_service_unreachable"] = 1

        logging.error(
            f"Target service unreachable -> "
            f"{target_url}: {error}"
        )

        return web.json_response(
            {
                "error": "Target service unreachable",
                "target": target_url
            },
            status=502
        )

    except Exception as error:

        async with state_lock:

            state["http_errors_total"] += 1

        logging.error(
            f"Proxy error -> {target_url}: {error}"
        )

        return web.json_response(
            {
                "error": "Proxy error",
                "target": target_url
            },
            status=502
        )


# =============================================================================
# APPLICATION
# =============================================================================

app = web.Application(
    middlewares=[
        telemetry_middleware
    ]
)


app.router.add_get(
    "/ping",
    ping
)

app.router.add_get(
    "/metrics",
    metrics
)

app.router.add_get(
    "/health",
    health
)

app.router.add_get(
    "/spike",
    spike
)

app.router.add_route(
    "*",
    "/{path_info:.*}",
    proxy
)


# =============================================================================
# START
# =============================================================================

if __name__ == "__main__":

    logging.info(
        f"Sidecar '{SERVICE_NAME}' starting"
    )

    logging.info(
        f"Target: {TARGET_HOST}:{TARGET_PORT}"
    )

    logging.info(
        f"Listening on port {SIDECAR_PORT}"
    )

    web.run_app(
        app,
        host="0.0.0.0",
        port=SIDECAR_PORT,
        access_log=None
    )