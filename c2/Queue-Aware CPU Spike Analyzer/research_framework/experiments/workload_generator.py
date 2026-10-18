"""
Workload generator for CPU-overload experiments.

Sends configurable-rate HTTP traffic to every enabled service in
services_config.json so the sidecars record realistic
incoming/processing rates, latency, and CPU behavior.

Usage examples:
    python workload_generator.py --level low     --duration 120
    python workload_generator.py --level high    --duration 300
    python workload_generator.py --rps 200       --duration 180
    python workload_generator.py --level medium  --duration 60 --path /

Levels: low=10 rps, medium=50 rps, high=100 rps, very_high=200 rps
(rate is split evenly across enabled services)
"""

import argparse
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
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return [
        s["base_url"].rstrip("/")
        for s in cfg.get("services", [])
        if s.get("enabled", True) and s.get("base_url", "").startswith("http")
    ]


async def hammer(session, url, path, rps_per_service, duration):
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


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", choices=list(LEVELS), default="medium")
    p.add_argument("--rps", type=float, default=None, help="Total requests/sec across all services")
    p.add_argument("--duration", type=float, default=60, help="Seconds")
    p.add_argument("--path", default="/")
    args = p.parse_args()

    total_rps = args.rps if args.rps else LEVELS[args.level]
    targets = load_targets()
    if not targets:
        print("No enabled services found in services_config.json")
        return

    per_service = total_rps / len(targets)
    print(f"[workload] {total_rps} rps total -> {per_service:.1f} rps x {len(targets)} services for {args.duration}s")

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[
            hammer(session, t, args.path, per_service, args.duration) for t in targets
        ])
    print(f"[workload] done. requests sent per service: {results}")


if __name__ == "__main__":
    asyncio.run(main())
