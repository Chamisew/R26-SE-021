"""
CPU spike injector — triggers the sidecar /spike endpoint on enabled services.

Usage:
    python spike_injector.py --duration 60
    python spike_injector.py --duration 180 --service service-go
    python spike_injector.py --duration 30 --all
"""

import argparse
import asyncio
import json
import os

import aiohttp

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "services_config.json",
)


def load_services():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return [
        s for s in cfg.get("services", [])
        if s.get("enabled", True) and s.get("base_url", "").startswith("http")
    ]


async def trigger(session, svc, duration):
    url = f"{svc['base_url'].rstrip('/')}/spike?duration={duration}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=duration + 30)) as resp:
            body = await resp.text()
            print(f"[spike] {svc['name']}: HTTP {resp.status} {body[:120]}")
    except Exception as e:
        print(f"[spike] {svc['name']}: ERROR {e}")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--service", default=None, help="Only this service name")
    p.add_argument("--all", action="store_true", help="Spike all enabled services")
    args = p.parse_args()

    services = load_services()
    if args.service:
        services = [s for s in services if s["name"].strip() == args.service]
    if not services:
        print("No matching enabled services.")
        return

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[trigger(session, s, args.duration) for s in services])


if __name__ == "__main__":
    asyncio.run(main())
