"""
Workload generator for CPU-overload experiments.

Adds:
  - LEVELS preset mapping (low, medium, high, very_high)
  - Target service loader `load_targets()` reading from services_config.json
  - Async worker `hammer(session, url, path, rps_per_service, duration)`
      • Computes pacing interval = 1.0 / max(rps_per_service, 0.1)
      • Executes paced GET requests with timeout handling
      • Returns total sent request count
"""

import asyncio
import json
import os
import time

import aiohttp

LEVELS = {"low": 10, "medium": 50, "high": 100, "very_high": 200}

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "services_config.json",
)


def load_targets():
    """
    Parses services_config.json and extracts base URLs for all
    enabled HTTP microservice targets.
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return [
        s["base_url"].rstrip("/")
        for s in cfg.get("services", [])
        if s.get("enabled", True) and s.get("base_url", "").startswith("http")
    ]


async def hammer(session, url, path, rps_per_service, duration):
    """
    Sends continuous paced HTTP requests to target URL at the assigned rate.
    """
    interval = 1.0 / max(rps_per_service, 0.1)
    end = time.monotonic() + duration
    sent = 0
    while time.monotonic() < end:
        try:
            async with session.get(url + path, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                await resp.read()
        except Exception:
            pass
        sent += 1
        await asyncio.sleep(interval)
    return sent
