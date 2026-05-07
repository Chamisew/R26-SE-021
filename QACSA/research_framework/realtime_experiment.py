import asyncio
import aiohttp
from aiohttp import web
import aiohttp_cors
import time
import subprocess
import logging
from collections import deque
import numpy as np
import csv
import os
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Configuration
NORMAL_RATE = 5 
SPIKE_DURATION = 20
SPIKE_INTERVAL = 300 
CSV_FILE = "final_research_dataset.csv"

PORT_MAP = {
    "Python": 5001,
    "Node.js": 5002,
    "Go": 5003,
    "Ruby": 5004,
    "Php": 5005
}

# Global State
stats = {"in_flight": 0, "total_sent": 0, "total_completed": 0}
history_cpu = {} 
running_service_names = []
current_metrics = None # INITIALIZED TO PREVENT NameError

# Research State Engine
incident_id_counter = 0
active_incident_id = 0
incident_service = "None"
system_state = "NORMAL" 
incident_phase = "NONE" 
cooldown_timer = 0

async def get_all_container_stats(containers):
    if not containers: return {}
    try:
        cmd = f'docker stats {" ".join(containers)} --no-stream --format "{{{{.Name}}}}:{{{{.CPUPerc}}}}"'
        proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, _ = await proc.communicate()
        results = {}
        for line in stdout.decode().strip().split('\n'):
            if ':' in line:
                name, cpu = line.split(':')
                results[name] = min(float(cpu.replace('%', '')), 100.0)
        return results
    except: return {}

async def fetch(session, url):
    stats["in_flight"] += 1; stats["total_sent"] += 1
    try:
        async with session.get(url, timeout=10) as response: await response.text()
    except: pass
    finally: stats["in_flight"] -= 1; stats["total_completed"] += 1

async def load_generator(session):
    while True:
        if running_service_names:
            target = random.choice(running_service_names)
            port = PORT_MAP.get(target, 5002)
            tasks = [fetch(session, f"http://localhost:{port}/ping") for _ in range(NORMAL_RATE)]
            await asyncio.gather(*tasks)
        await asyncio.sleep(1)

async def spike_trigger(session):
    await asyncio.sleep(60) 
    while True:
        if running_service_names:
            target = random.choice(running_service_names)
            port = PORT_MAP.get(target, 5002)
            logging.info(f"--- TRIGGERING RESEARCH SPIKE: {target} ---")
            asyncio.create_task(fetch(session, f"http://localhost:{port}/spike?duration={SPIKE_DURATION}"))
        await asyncio.sleep(SPIKE_DURATION + SPIKE_INTERVAL)

async def metrics_collector():
    global active_incident_id, incident_id_counter, incident_service, system_state, incident_phase, cooldown_timer, running_service_names, current_metrics
    last_sent = 0; last_completed = 0
    
    csv_keys = [
        "incident_id", "timestamp", "time", "system_state", "incident_phase", 
        "failing_service", "cpu_percent", "in_flight_queue", "incoming_rate", 
        "processing_rate", "queue_growth_rate", "cpu_trend_5min_ma", 
        "cpu_trend_10min_ma", "overload_flag", "queue_pressure_index", "label"
    ]
    
    if not os.path.isfile(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='') as f:
            csv.DictWriter(f, fieldnames=csv_keys).writeheader()

    while True:
        start_loop = time.time()
        
        try:
            cmd = 'docker ps --filter "name=microservices-service" --format "{{.Names}}"'
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            containers = [c for c in res.stdout.strip().split('\n') if c]
        except: containers = []
        container_stats = await get_all_container_stats(containers)

        curr_sent = stats["total_sent"]; curr_comp = stats["total_completed"]
        inc_rate = float(curr_sent - last_sent); proc_rate = float(curr_comp - last_completed)
        last_sent = curr_sent; last_completed = curr_comp
        queue = float(stats["in_flight"])
        queue_growth = inc_rate - proc_rate
        
        # New Metrics
        overload_flag = 1 if inc_rate > proc_rate else 0
        # Simple real-time QPI: (queue/20 * 0.5) + (overload * 0.5)
        qpi = min(1.0, (queue / 20.0 * 0.5) + (overload_flag * 0.5))
        
        highest_cpu = 0.0; top_service = "None"
        services_summary = []
        current_running = []
        for c_id in containers:
            name = c_id.replace('microservices-service-', '').replace('-1', '').capitalize()
            if name == "Node": name = "Node.js"
            current_running.append(name)
            cpu = container_stats.get(c_id, 0.0)
            if name not in history_cpu: history_cpu[name] = deque(maxlen=600)
            history_cpu[name].append(cpu)
            if cpu > highest_cpu: highest_cpu = cpu; top_service = name
            services_summary.append({"name": name, "cpu": cpu})
        
        running_service_names = current_running

        is_failed = (highest_cpu >= 85.0 or queue >= 20.0)
        is_degraded = (highest_cpu >= 40.0 or queue >= 10.0)
        is_healthy = (highest_cpu < 25.0 and queue < 2.0)

        numeric_label = 0
        if is_failed:
            if system_state != "FAILED":
                if system_state == "NORMAL": incident_id_counter += 1
                active_incident_id = incident_id_counter
                system_state = "FAILED"; incident_phase = "PEAK"; incident_service = top_service
            numeric_label = 1; cooldown_timer = 0
        elif is_degraded:
            if system_state == "NORMAL":
                incident_id_counter += 1; active_incident_id = incident_id_counter
                system_state = "DEGRADED"; incident_phase = "START"; incident_service = top_service
            elif system_state == "FAILED":
                system_state = "RECOVERING"; incident_phase = "RECOVERY"
            numeric_label = 1; cooldown_timer = 0
        elif is_healthy:
            if system_state != "NORMAL":
                cooldown_timer += 1
                if cooldown_timer >= 20:
                    system_state = "NORMAL"; incident_phase = "NONE"; active_incident_id = 0; incident_service = "None"; cooldown_timer = 0; numeric_label = 0
                    stats["total_sent"] = 0; stats["total_completed"] = 0; last_sent = 0; last_completed = 0
                else:
                    system_state = "RECOVERING"; incident_phase = "RECOVERY"; numeric_label = 0
            else: numeric_label = 0
        else:
            if system_state == "FAILED" or system_state == "RECOVERING" or system_state == "DEGRADED":
                system_state = "RECOVERING"; incident_phase = "RECOVERY"; numeric_label = 0
            cooldown_timer = 0

        node_hist = history_cpu.get("Node.js", deque([0.0]))
        trend_5m = np.mean(list(node_hist)[-300:]) if len(node_hist) > 0 else 0.0
        trend_10m = np.mean(list(node_hist)) if len(node_hist) > 0 else 0.0

        current_metrics = {
            "incident_id": active_incident_id,
            "timestamp": float(time.time()),
            "time": time.strftime('%H:%M:%S', time.localtime()),
            "system_state": system_state,
            "incident_phase": incident_phase,
            "failing_service": incident_service,
            "services": services_summary,
            "cpu_percent": node_hist[-1] if len(node_hist) > 0 else 0.0,
            "in_flight_queue": queue,
            "incoming_rate": inc_rate,
            "processing_rate": proc_rate,
            "queue_growth_rate": queue_growth,
            "cpu_trend_5min_ma": trend_5m,
            "cpu_trend_10min_ma": trend_10m,
            "overload_flag": overload_flag,
            "queue_pressure_index": qpi,
            "label": numeric_label
        }

        with open(CSV_FILE, mode='a', newline='') as f:
            csv.DictWriter(f, fieldnames=csv_keys).writerow({k: current_metrics[k] for k in csv_keys})

        elapsed = time.time() - start_loop
        await asyncio.sleep(max(0, 1.0 - elapsed))

async def get_data(request):
    if current_metrics is None:
        return web.json_response({"status": "initializing", "services": [], "system_state": "STARTING", "incident_id": 0, "in_flight_queue": 0, "incoming_rate": 0, "processing_rate": 0, "label": 0})
    return web.json_response(current_metrics)

async def download_csv(request): return web.FileResponse(CSV_FILE, headers={'Content-Disposition': f'attachment; filename="{CSV_FILE}"'})

from queue_analyzer import process_pipeline as refresh_ml_pipeline

async def ml_auto_refresh_task():
    """Background task that keeps the ML-ready dataset synchronized."""
    await asyncio.sleep(10) # Initial wait for data to accumulate
    while True:
        try:
            # Run the pipeline in a thread to avoid blocking the event loop
            await asyncio.to_thread(refresh_ml_pipeline)
        except Exception as e:
            logging.error(f"ML Refresh Error: {e}")
        await asyncio.sleep(10)

async def start_background_tasks(app):
    session = aiohttp.ClientSession()
    app['session'] = session
    app['load_task'] = asyncio.create_task(load_generator(session))
    app['spike_task'] = asyncio.create_task(spike_trigger(session))
    app['metrics_task'] = asyncio.create_task(metrics_collector())
    app['ml_refresh_task'] = asyncio.create_task(ml_auto_refresh_task())

async def cleanup_background_tasks(app):
    app['load_task'].cancel(); app['spike_task'].cancel(); app['metrics_task'].cancel()
    app['ml_refresh_task'].cancel()
    await app['session'].close()

app = web.Application()
app.router.add_get('/api/data', get_data); app.router.add_get('/api/download', download_csv)
cors = aiohttp_cors.setup(app, defaults={"*": aiohttp_cors.ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*")})
for route in list(app.router.routes()): cors.add(route)
app.on_startup.append(start_background_tasks); app.on_cleanup.append(cleanup_background_tasks)

if __name__ == "__main__":
    web.run_app(app, port=8080)
