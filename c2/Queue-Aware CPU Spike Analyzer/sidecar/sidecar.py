"""

Adds:
  - Logging setup
  - All environment-variable configuration
  - /ping endpoint
  - App bootstrap (web.run_app)
"""

import os
import time
import logging

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
# If TARGET_HEALTH_PATH is supplied, that endpoint is checked.
# Otherwise the sidecar only checks whether the target can be reached.
TARGET_HEALTH_PATH = os.environ.get("TARGET_HEALTH_PATH")


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

app = web.Application()

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

