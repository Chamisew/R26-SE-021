"""
Sidecar Proxy — Queue-Aware CPU Spike Analyzer
==============================================
Deploys alongside ANY microservice on Azure (any language, any framework).
Adds /ping, /metrics, and /spike without touching the original service.
All other requests are transparently proxied to the target service.

Azure Container Apps Usage:
  Run this as a second container in the same Container App.
  Point ingress to SIDECAR_PORT (8091). The sidecar forwards traffic to the
  real service on TARGET_PORT (e.g. 8080).

Environment Variables:
  TARGET_HOST   - Host of the real service      (default: localhost)
  TARGET_PORT   - Port of the real service       (default: 8080)
  SIDECAR_PORT  - Port this sidecar listens on   (default: 8091)
  SERVICE_NAME  - Friendly name for this service (default: auto)
"""

import asyncio
import math
import os
import time
import logging
import platform

import psutil
import aiohttp
from aiohttp import web

logging.basicConfig(level=logging.INFO, format='%(asctime)s [SIDECAR] %(message)s')

# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_HOST  = os.getenv("TARGET_HOST",  "localhost")
TARGET_PORT  = int(os.getenv("TARGET_PORT",  "8080"))
SIDECAR_PORT = int(os.getenv("SIDECAR_PORT", "8091"))
SERVICE_NAME = os.getenv("SERVICE_NAME", f"service-{TARGET_PORT}")
START_TIME   = time.time()

# ── Request counter ───────────────────────────────────────────────────────────
total_requests = 0

@web.middleware
async def request_counter(request, handler):
    global total_requests
    total_requests += 1
    return await handler(request)

# ── /ping ─────────────────────────────────────────────────────────────────────
async def ping(request):
    return web.json_response({
        "status":     "ok",
        "service_id": SERVICE_NAME,
        "message":    "pong",
        "sidecar":    True
    })

# ── /metrics ──────────────────────────────────────────────────────────────────
async def metrics(request):
    cpu_pct = psutil.cpu_percent(interval=0.1)
    mem     = psutil.virtual_memory()
    disk    = psutil.disk_usage('/')

    return web.json_response({
        "service_id":      SERVICE_NAME,
        "sidecar":         True,
        "language":        "sidecar",
        "cpu_percent":     round(cpu_pct, 2),
        "memory_percent":  round(mem.percent, 2),
        "memory_used_mb":  round(mem.used  / 1024 / 1024, 2),
        "memory_total_mb": round(mem.total / 1024 / 1024, 2),
        "disk_used_gb":    round(disk.used  / 1024 / 1024 / 1024, 2),
        "disk_total_gb":   round(disk.total / 1024 / 1024 / 1024, 2),
        "uptime_seconds":  round(time.time() - START_TIME, 2),
        "total_requests":  total_requests,
        "cpu_count":       psutil.cpu_count(),
        "platform":        platform.system(),
        "target":          f"{TARGET_HOST}:{TARGET_PORT}"
    })

# ── /spike ────────────────────────────────────────────────────────────────────
async def spike(request):
    duration = int(request.rel_url.query.get("duration", 10))
    logging.info(f"[{SERVICE_NAME}] CPU spike triggered for {duration}s")

    # Run CPU burn in a thread pool so the event loop stays responsive
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _cpu_burn, time.time() + duration)

    return web.json_response({
        "message":    f"CPU spiked for {duration} seconds",
        "service_id": SERVICE_NAME
    })

def _cpu_burn(end_time):
    """Blocking CPU-intensive loop — runs in a thread pool."""
    while time.time() < end_time:
        math.sqrt(64 ** 5)

# ── /health ───────────────────────────────────────────────────────────────────
async def health(request):
    """Checks if the proxied target service is reachable."""
    target_ok = False
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                f"http://{TARGET_HOST}:{TARGET_PORT}/",
                timeout=aiohttp.ClientTimeout(total=1)
            ) as r:
                target_ok = r.status < 500
    except Exception:
        pass

    return web.json_response({
        "status":           "ok",
        "sidecar":          True,
        "service_id":       SERVICE_NAME,
        "target_reachable": target_ok,
        "target":           f"{TARGET_HOST}:{TARGET_PORT}"
    })

# ── Transparent proxy for all other routes ───────────────────────────────────
async def proxy(request):
    """Forwards any unrecognised request to the real target service."""
    target_url = f"http://{TARGET_HOST}:{TARGET_PORT}{request.path_qs}"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.request(
                method  = request.method,
                url     = target_url,
                headers = {k: v for k, v in request.headers.items()
                           if k.lower() != 'host'},
                data    = await request.read(),
                timeout = aiohttp.ClientTimeout(total=30)
            ) as resp:
                body = await resp.read()
                # Strip hop-by-hop headers that cannot be forwarded
                skip = {'transfer-encoding', 'content-encoding', 'connection'}
                fwd_headers = {k: v for k, v in resp.headers.items()
                               if k.lower() not in skip}
                return web.Response(body=body, status=resp.status,
                                    headers=fwd_headers)
    except Exception as e:
        logging.error(f"Proxy error → {target_url}: {e}")
        return web.json_response(
            {"error": "Target service unreachable", "target": target_url},
            status=502
        )

# ── App wiring ────────────────────────────────────────────────────────────────
app = web.Application(middlewares=[request_counter])
app.router.add_get("/ping",           ping)
app.router.add_get("/metrics",        metrics)
app.router.add_get("/spike",          spike)
app.router.add_get("/health",         health)
app.router.add_route("*", "/{path_info:.*}", proxy)   # catch-all proxy

if __name__ == "__main__":
    logging.info(f"Sidecar for '{SERVICE_NAME}' listening on port {SIDECAR_PORT}")
    logging.info(f"Proxying all other traffic → {TARGET_HOST}:{TARGET_PORT}")
    web.run_app(app, host="0.0.0.0", port=SIDECAR_PORT, access_log=None)
