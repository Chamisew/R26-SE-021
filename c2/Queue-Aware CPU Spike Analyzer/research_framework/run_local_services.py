import asyncio
import os
import sys
import logging
from aiohttp import web

# Add sidecar directory to python path
sidecar_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "micro services", "sidecar"))
sys.path.insert(0, sidecar_dir)

import sidecar

async def start_service_instance(name, port):
    app = sidecar.app
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    logging.info(f"Local Microservice '{name}' active on http://localhost:{port}")
    return runner

async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [LOCAL-SERVICES] %(message)s')
    logging.info("Starting 5 Local Microservices (Go, Node, Python, Ruby, PHP)...")
    
    services = [
        ("go-azure", 5001),
        ("node-azure", 5002),
        ("python-azure", 5003),
        ("ruby-azure", 5004),
        ("php-azure", 5005),
    ]
    
    runners = []
    for name, port in services:
        runner = await start_service_instance(name, port)
        runners.append(runner)
        
    logging.info("All 5 Local Microservices are online on ports 5001-5005!")
    
    # Keep running forever
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
