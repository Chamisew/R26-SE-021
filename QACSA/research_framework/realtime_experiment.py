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

# Constants for Advanced Detection
THRESHOLD_WARNING = 60.0
THRESHOLD_OVERLOAD = 75.0
THRESHOLD_FAILED = 90.0
CONFIRMATION_WINDOW = 3 # Seconds sustained before state change
RECOVERY_COOLDOWN = 15 # Seconds stable before NORMAL
SPIKE_VELOCITY = 40.0 # CPU % jump per second to trigger SPIKE state

# Global State
stats = {"in_flight": 0, "total_sent": 0, "total_completed": 0}
history_cpu = {} 
running_service_names = []
current_metrics = None 
discovered_ports = {} 
global_container_list = []
global_container_stats = {}

# Incident Tracking
active_incident_id = 0
incident_id_counter = 0
system_state = "NORMAL"
incident_phase = "NONE"
incident_service = "None"
patient_zero = "None"
incident_start_time = 0
state_timers = {"sustained": 0, "recovery": 0}
cooldown_timer = 0
# Global CPU History for Sliding Windows (10 min = 600 samples @ 1Hz)
global_cpu_history = deque(maxlen=600)

async def discovery_task():
    """Background task to poll docker ps and discover ports."""
    global global_container_list, discovered_ports
    logging.info("Discovery Task Started")
    while True:
        try:
            proc = await asyncio.create_subprocess_shell(
                'docker ps --format "{{.Names}}"',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            names = [n for n in stdout.decode().strip().split('\n') if n]
            
            if names:
                for c_id in names:
                    name = c_id.replace('microservices-service-', '').replace('-1', '').capitalize()
                    if name not in discovered_ports:
                        port_proc = await asyncio.create_subprocess_shell(
                            f"docker inspect {c_id}",
                            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                        )
                        out, _ = await port_proc.communicate()
                        try:
                            import json
                            inspect_data = json.loads(out.decode())[0]
                            ports = inspect_data.get("NetworkSettings", {}).get("Ports", {})
                            for p_info in ports.values():
                                if p_info:
                                    discovered_ports[name] = int(p_info[0]["HostPort"])
                                    logging.info(f"Discovery: Found port {discovered_ports[name]} for {name}")
                                    break
                        except: pass
                
                global_container_list = names
                logging.info(f"Discovery Task: Found {len(names)} containers, {len(discovered_ports)} ports mapped")
        except Exception as e:
            logging.error(f"Discovery Task Exception: {e}")
        await asyncio.sleep(5)

async def stats_polling_task():
    """Background task to poll docker stats."""
    global global_container_stats
    logging.info("Stats Task Started")
    while True:
        try:
            proc = await asyncio.create_subprocess_shell(
                'docker stats --no-stream --format "{{.Name}}:{{.CPUPerc}}"',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if stderr:
                logging.error(f"Docker Stats Error: {stderr.decode()}")
            new_stats = {}
            for line in stdout.decode().strip().split('\n'):
                if ':' in line:
                    try:
                        name, cpu = line.split(':')
                        new_stats[name] = min(float(cpu.replace('%', '')), 100.0)
                    except: continue
            if new_stats:
                logging.info(f"Stats Task updated {len(new_stats)} containers")
                global_container_stats = new_stats
        except Exception as e:
            logging.error(f"Stats Task Exception: {e}")
        await asyncio.sleep(2)

async def fetch(session, url):
    stats["in_flight"] += 1; stats["total_sent"] += 1
    try:
        async with session.get(url, timeout=5) as response: 
            await response.text()
    except Exception as e:
        if stats["total_sent"] % 50 == 0:
            logging.warning(f"Fetch failed for {url}: {e}")
    finally: stats["in_flight"] -= 1; stats["total_completed"] += 1

async def load_generator(session):
    """Generates a steady heartbeat to measure system responsiveness (λ vs μ)"""
    logging.info("Starting Research Heartbeat (Steady Load)...")
    count = 0
    while True:
        if running_service_names:
            for target in running_service_names:
                port = discovered_ports.get(target)
                if port:
                    # Send 2 pings per second per service (Steady λ)
                    for _ in range(2):
                        asyncio.create_task(fetch(session, f"http://localhost:{port}/ping"))
            
            if count % 10 == 0:
                logging.info(f"Heartbeat: Monitoring {len(running_service_names)} services")
        else:
            if count % 10 == 0:
                logging.warning("Heartbeat: Waiting for services to be discovered...")
        
        count += 1
        await asyncio.sleep(1)

async def spike_trigger(session):
    """Periodically triggers complex research spikes and cascading failures"""
    await asyncio.sleep(45) 
    while True:
        if running_service_names and len(running_service_names) >= 2:
            mode = random.choice(["SINGLE", "CASCADE", "SLOW"])
            targets = random.sample(running_service_names, 2)
            
            if mode == "SINGLE":
                logging.info(f"--- TRIGGERING RESEARCH SPIKE: {targets[0]} ---")
                asyncio.create_task(fetch(session, f"http://localhost:{discovered_ports[targets[0]]}/spike?duration=20"))
            
            elif mode == "CASCADE":
                logging.info(f"--- TRIGGERING CASCADING FAILURE: {targets[0]} -> {targets[1]} ---")
                asyncio.create_task(fetch(session, f"http://localhost:{discovered_ports[targets[0]]}/spike?duration=15"))
                await asyncio.sleep(8)
                asyncio.create_task(fetch(session, f"http://localhost:{discovered_ports[targets[1]]}/spike?duration=15"))
            
            elif mode == "SLOW":
                logging.info(f"--- TRIGGERING SLOW DEGRADATION: {targets[0]} ---")
                for _ in range(5):
                    asyncio.create_task(fetch(session, f"http://localhost:{discovered_ports[targets[0]]}/spike?duration=5"))
                    await asyncio.sleep(4)
                    
        await asyncio.sleep(SPIKE_INTERVAL)

async def metrics_collector():
    global active_incident_id, incident_id_counter, system_state, incident_phase, incident_service, patient_zero, current_metrics, incident_start_time, running_service_names, discovered_ports
    
    logging.info("Starting Advanced Metrics Collector...")
    last_sent = 0
    last_completed = 0
    
    # Initialize CSV if not exists
    csv_keys = [
        "incident_id", "timestamp", "time", "system_state", "incident_phase", 
        "failing_service", "patient_zero", "cpu_percent", "cpu_velocity", 
        "cpu_trend_5min", "cpu_trend_10min",
        "in_flight_queue", "incoming_rate", "processing_rate", "queue_growth_rate", 
        "overload_flag", "queue_pressure_index", "incident_duration", "label"
    ]
    
    if not os.path.isfile(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        with open(CSV_FILE, mode='w', newline='') as f:
            csv.DictWriter(f, fieldnames=csv_keys).writeheader()

    while True:
        start_loop = time.time()
        try:
            containers = global_container_list
            container_stats = global_container_stats

            # Basic Rate Calculation
            curr_sent = stats["total_sent"]; curr_comp = stats["total_completed"]
            inc_rate = float(curr_sent - last_sent); proc_rate = float(curr_comp - last_completed)
            last_sent = curr_sent; last_completed = curr_comp
            queue = float(stats["in_flight"])
            queue_growth = inc_rate - proc_rate
            
            # CPU Analytics
            highest_cpu = 0.0
            failing_list = []
            current_running = []
            service_details = []
            
            for c_id in containers:
                # Clean name for UI
                name = c_id.replace('microservices-service-', '').replace('-1', '').capitalize()
                
                # If we have a port, it's a valid target for our Load Generator
                if name in discovered_ports:
                    current_running.append(name)

                cpu = container_stats.get(c_id, 0.0)
                service_details.append({"name": name, "cpu": cpu})
                if cpu >= THRESHOLD_WARNING: failing_list.append(name)
                
                if name not in history_cpu: history_cpu[name] = deque(maxlen=60)
                history_cpu[name].append(cpu)
                if cpu > highest_cpu: highest_cpu = cpu
            
            # Calculate Moving Averages for Research Drift
            global_cpu_history.append(highest_cpu)
            cpu_ma_5 = float(np.mean(list(global_cpu_history)[-300:])) if global_cpu_history else 0.0
            cpu_ma_10 = float(np.mean(list(global_cpu_history))) if global_cpu_history else 0.0

            running_service_names = current_running
            
            # Calculate Velocity (Anomaly Detection)
            if 'system_peak' not in history_cpu: history_cpu['system_peak'] = deque([0.0], maxlen=60)
            prev_cpu = history_cpu['system_peak'][-1]
            history_cpu['system_peak'].append(highest_cpu)
            cpu_velocity = highest_cpu - prev_cpu
            
            # Advanced QPI and Overload Flag
            overload_flag = 1 if highest_cpu >= THRESHOLD_OVERLOAD or inc_rate > proc_rate else 0
            qpi = min(1.0, (queue / 20.0 * 0.4) + (overload_flag * 0.6))
            
            # --- Advanced State Machine Logic ---
            target_state = "NORMAL"
            if highest_cpu >= THRESHOLD_FAILED or queue >= 25: target_state = "FAILED"
            elif highest_cpu >= THRESHOLD_OVERLOAD or queue >= 15: target_state = "OVERLOADED"
            elif highest_cpu >= THRESHOLD_WARNING: target_state = "WARNING"
            
            # Handle Instant SPIKE Anomaly
            if cpu_velocity >= SPIKE_VELOCITY and system_state == "NORMAL":
                system_state = "FAILED"; incident_phase = "SPIKE"
                incident_id_counter += 1; active_incident_id = incident_id_counter
                incident_start_time = time.time()
                patient_zero = failing_list[0] if failing_list else "Unknown"

            # Sustained State Transition
            if target_state != "NORMAL" and target_state != system_state:
                state_timers["sustained"] += 1
                if state_timers["sustained"] >= CONFIRMATION_WINDOW:
                    if system_state == "NORMAL":
                        incident_id_counter += 1; active_incident_id = incident_id_counter
                        incident_start_time = time.time()
                        patient_zero = failing_list[0] if failing_list else "Unknown"
                    system_state = target_state
                    incident_phase = "STABLE_HIGH" if target_state == "FAILED" else "TRANSITION"
                    state_timers["sustained"] = 0
            else:
                state_timers["sustained"] = 0

            # Recovery Stabilization
            if target_state == "NORMAL" and system_state != "NORMAL":
                system_state = "RECOVERING"; incident_phase = "COOLDOWN"
                state_timers["recovery"] += 1
                if state_timers["recovery"] >= RECOVERY_COOLDOWN:
                    system_state = "NORMAL"; incident_phase = "NONE"
                    active_incident_id = 0; patient_zero = "None"
                    state_timers["recovery"] = 0
                    # Reset simulation stats for fresh start
                    stats["total_sent"] = 0; stats["total_completed"] = 0
            else:
                state_timers["recovery"] = 0

            incident_duration = time.time() - incident_start_time if active_incident_id > 0 else 0
            incident_service = ", ".join(failing_list) if failing_list else "None"
            numeric_label = 1 if system_state in ["WARNING", "OVERLOADED", "FAILED"] else 0

            current_metrics = {
                "incident_id": active_incident_id,
                "services": service_details,
                "discovered_targets": list(discovered_ports.keys()),
                "timestamp": float(time.time()),
                "time": time.strftime('%H:%M:%S', time.localtime()),
                "system_state": system_state,
                "incident_phase": incident_phase,
                "failing_service": incident_service,
                "patient_zero": patient_zero,
                "cpu_percent": highest_cpu,
                "cpu_velocity": cpu_velocity,
                "cpu_trend_5min": round(cpu_ma_5, 2),
                "cpu_trend_10min": round(cpu_ma_10, 2),
                "in_flight_queue": queue,
                "incoming_rate": inc_rate,
                "processing_rate": proc_rate,
                "queue_growth_rate": queue_growth,
                "overload_flag": overload_flag,
                "queue_pressure_index": qpi,
                "incident_duration": round(incident_duration, 2),
                "label": numeric_label
            }
            
            # Write to CSV (ignore extra keys like 'services' which are for UI only)
            with open(CSV_FILE, mode='a', newline='') as f:
                csv.DictWriter(f, fieldnames=csv_keys, extrasaction='ignore').writerow(current_metrics)
                
            logging.info(f"[{system_state}] Containers: {len(containers)} | CPU: {highest_cpu}% | QPI: {qpi:.2f}")
        except Exception as e:
            logging.error(f"Metrics Loop Error: {e}")
            
        elapsed = time.time() - start_loop
        await asyncio.sleep(max(0, 1.0 - elapsed))

async def load_generator(session):
    """Generates a steady heartbeat to measure system responsiveness (λ vs μ)"""
    logging.info("Starting Research Heartbeat (Steady Load)...")
    count = 0
    while True:
        if running_service_names:
            # We target ALL running services to get a global health measure
            for target in running_service_names:
                port = discovered_ports.get(target)
                if port:
                    # Send 2 pings per second per service (Steady λ)
                    for _ in range(2):
                        asyncio.create_task(fetch(session, f"http://localhost:{port}/ping"))
            
            if count % 10 == 0:
                logging.info(f"Heartbeat: Monitoring {len(running_service_names)} services")
        else:
            if count % 10 == 0:
                logging.warning("Heartbeat: Waiting for services to be discovered...")
        
        count += 1
        await asyncio.sleep(1)

async def spike_trigger(session):
    """Periodically triggers complex research spikes and cascading failures"""
    # Note: This is disabled by default to allow manual testing
    await asyncio.sleep(45) 
    while True:
        if running_service_names and len(running_service_names) >= 2:
            mode = random.choice(["SINGLE", "CASCADE", "SLOW"])
            targets = random.sample(running_service_names, 2)
            
            if mode == "SINGLE":
                logging.info(f"--- TRIGGERING RESEARCH SPIKE: {targets[0]} ---")
                asyncio.create_task(fetch(session, f"http://localhost:{discovered_ports[targets[0]]}/spike?duration=20"))
            
            elif mode == "CASCADE":
                logging.info(f"--- TRIGGERING CASCADING FAILURE: {targets[0]} -> {targets[1]} ---")
                asyncio.create_task(fetch(session, f"http://localhost:{discovered_ports[targets[0]]}/spike?duration=15"))
                await asyncio.sleep(8)
                asyncio.create_task(fetch(session, f"http://localhost:{discovered_ports[targets[1]]}/spike?duration=15"))
            
            elif mode == "SLOW":
                logging.info(f"--- TRIGGERING SLOW DEGRADATION: {targets[0]} ---")
                for _ in range(5):
                    asyncio.create_task(fetch(session, f"http://localhost:{discovered_ports[targets[0]]}/spike?duration=5"))
                    await asyncio.sleep(4)
                    
        await asyncio.sleep(SPIKE_INTERVAL)

async def get_data(request):
    if current_metrics is None:
        return web.json_response({"status": "initializing", "services": [], "system_state": "STARTING", "incident_id": 0, "in_flight_queue": 0, "incoming_rate": 0, "processing_rate": 0, "label": 0})
    return web.json_response(current_metrics)

async def handle_metrics(request):
    return web.json_response(current_metrics, headers={'Access-Control-Allow-Origin': '*'})

async def download_csv(request):
    """Serves the full research CSV for download"""
    if os.path.exists(CSV_FILE):
        return web.FileResponse(CSV_FILE, headers={
            'Content-Disposition': 'attachment; filename="final_research_dataset.csv"',
            'Access-Control-Allow-Origin': '*'
        })
    return web.Response(text="CSV File not found", status=404, headers={'Access-Control-Allow-Origin': '*'}),

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
    # app['spike_task'] = asyncio.create_task(spike_trigger(session)) # Disabled for manual control
    app['metrics_task'] = asyncio.create_task(metrics_collector())
    app['discovery_task'] = asyncio.create_task(discovery_task())
    app['stats_task'] = asyncio.create_task(stats_polling_task())
    app['ml_refresh_task'] = asyncio.create_task(ml_auto_refresh_task())

async def cleanup_background_tasks(app):
    app['load_task'].cancel(); app['spike_task'].cancel(); app['metrics_task'].cancel()
    app['discovery_task'].cancel(); app['stats_task'].cancel()
    app['ml_refresh_task'].cancel()
    await app['session'].close()

# --- ROBUST ROUTING TABLE ---
routes = web.RouteTableDef()

@routes.get('/')
async def test_root(request):
    return web.json_response({"status": "Backend is Online", "port": 8081})

@routes.get('/api/data')
async def get_data_wrapper(request):
    return await get_data(request)

@routes.get('/api/download')
async def download_csv_wrapper(request):
    return await download_csv(request)

# --- FOOLPROOF MANUAL CORS MIDDLEWARE ---
@web.middleware
async def cors_middleware(request, handler):
    try:
        response = await handler(request)
    except web.HTTPException as ex:
        response = ex
    except Exception as e:
        logging.error(f"Server Error: {e}")
        response = web.Response(status=500, text=str(e))
    
    # Force CORS headers on EVERY response
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    
    if request.method == 'OPTIONS':
        return web.Response(status=200, headers=response.headers)
        
    return response

# Initialize the Web App
app = web.Application(middlewares=[cors_middleware])
app.add_routes(routes)

# Register background task handlers
app.on_startup.append(start_background_tasks)
app.on_cleanup.append(cleanup_background_tasks)

if __name__ == "__main__":
    logging.info("Advanced Experiment Backend starting on http://localhost:8081")
    web.run_app(app, host='0.0.0.0', port=8081)
