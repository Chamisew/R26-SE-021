import asyncio
import aiohttp
from aiohttp import web
import aiohttp_cors
import time
import json
import logging
from collections import deque
import numpy as np
import csv
import os
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Configuration
NORMAL_RATE    = 5
SPIKE_DURATION = 180
SPIKE_INTERVAL = 120
# Constants
CSV_FILE          = "final_research_dataset.csv"
THRESHOLD_WARNING = 70   # CPU % threshold for warning state
CONFIG_FILE       = "services_config.json"

def load_azure_services():
    """Loads service configurations.
    Priority:
    1. Environment variables starting with 'AZURE_SERVICE_' (e.g. AZURE_SERVICE_NODE)
    2. Local '.env' file containing KEY=VALUE pairs
    3. JSON file 'services_config.json' (fallback)
    """
    services = {}

    # 1. Try reading from a local .env file first to populate environment
    env_file = ".env"
    if os.path.exists(env_file):
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ[key.strip()] = val.strip()
            logging.info("[Config] Loaded environment variables from .env file.")
        except Exception as e:
            logging.warning(f"[Config] Failed to read .env file: {e}")

    # 2. Check for environment variables (either system-wide or loaded from .env)
    for env_name, env_val in os.environ.items():
        if env_name.startswith("AZURE_SERVICE_") and env_val.startswith("http"):
            # e.g. AZURE_SERVICE_NODE -> "node-azure"
            service_name = env_name.replace("AZURE_SERVICE_", "").lower() + "-azure"
            services[service_name] = env_val

    if services:
        logging.info(f"[Config] Loaded {len(services)} Azure service(s) from environment variables: {list(services.keys())}")
        return services

    # 3. Fallback to services_config.json if no env vars exist
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
            services = {
                s["name"]: s["base_url"]
                for s in cfg.get("services", [])
                if s.get("enabled", True) and s.get("base_url", "").startswith("http")
            }
            logging.info(f"[Config] Fallback: Loaded {len(services)} service(s) from {CONFIG_FILE}: {list(services.keys())}")
            return services
        except Exception as e:
            logging.error(f"[Config] Failed to parse {CONFIG_FILE}: {e}")

    return {}

AZURE_SERVICES = load_azure_services()   # {name: base_url}
# Global State
# discovered_ports now stores {name: base_url} instead of {name: port_int}
stats = {"in_flight": 0, "total_sent": 0, "total_completed": 0, "errors": 0}
history_cpu          = {}
running_service_names = []
current_metrics       = None
discovered_ports      = {}   # {name: "https://..."}  — populated by discovery_task
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
# spike_active: set True by spike_trigger to force FAILED state
# Automatically reset to False after spike recovery
spike_active = False
# Global CPU History for Sliding Windows (10 min = 600 samples @ 1Hz)
global_cpu_history = deque(maxlen=600)

async def discovery_task(session):
    """Pings every Azure sidecar URL from services_config.json.
    Marks a service as UP if /ping returns HTTP 200, DOWN otherwise.
    Re-reads the config file every cycle so you can add services without restart.
    """
    global global_container_list, discovered_ports, running_service_names
    logging.info("Discovery Task Started (Azure Sidecar Health Checker)")
    while True:
        try:
            # Re-load config each cycle so hot-adding services works
            current_config = load_azure_services()

            for name, base_url in current_config.items():
                try:
                    async with session.get(
                        f"{base_url}/ping",
                        timeout=aiohttp.ClientTimeout(total=4.0)
                    ) as resp:
                        if resp.status == 200:
                            if name not in discovered_ports:
                                discovered_ports[name] = base_url
                                if name not in global_container_list:
                                    global_container_list.append(name)
                                logging.info(f"Discovery: Azure service '{name}' is UP → {base_url}")
                        else:
                            if name in discovered_ports:
                                del discovered_ports[name]
                                logging.warning(f"Discovery: '{name}' returned {resp.status} — marked DOWN")
                except Exception:
                    # Network error / timeout → mark service as offline
                    if name in discovered_ports:
                        del discovered_ports[name]
                        logging.warning(f"Discovery: '{name}' unreachable — marked DOWN")
            running_service_names = list(discovered_ports.keys())
        except Exception as e:
            logging.error(f"Discovery Task Exception: {e}")
        await asyncio.sleep(10)   # Azure has network latency; 10s is fine

async def stats_polling_task():
    """Polls /metrics on every discovered Azure sidecar to get CPU data.
    The sidecar reads its own machine's CPU via psutil and returns it as JSON.
    """
    global global_container_stats
    logging.info("Stats Task Started (Azure /metrics Poller)")
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                new_stats = {}
                for name, base_url in list(discovered_ports.items()):
                    try:
                        async with session.get(
                            f"{base_url}/metrics",
                            timeout=aiohttp.ClientTimeout(total=4.0)
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                # All sidecars return cpu_percent
                                cpu = float(data.get("cpu_percent", 0.0))
                                new_stats[name] = min(cpu, 100.0)
                            else:
                                new_stats[name] = 0.0
                    except Exception as e:
                        logging.warning(f"Stats: /metrics failed for '{name}': {e}")
                        new_stats[name] = 0.0

                if new_stats:
                    global_container_stats = new_stats
            except Exception as e:
                logging.error(f"Stats Task Exception: {e}")
            await asyncio.sleep(5)   # poll every 5s to limit Azure request costs

async def fetch(session, url):
    stats["in_flight"] += 1; stats["total_sent"] += 1
    try:
        async with session.get(url, timeout=5) as response: 
            if response.status != 200:
                stats["errors"] += 1
            await response.text()
    except Exception as e:
        stats["errors"] += 1
    finally: stats["in_flight"] -= 1; stats["total_completed"] += 1

# load_generator and spike_trigger are defined below (single canonical definitions)

async def metrics_collector():
    global active_incident_id, incident_id_counter, system_state, incident_phase, incident_service, patient_zero, current_metrics, incident_start_time, running_service_names, discovered_ports, spike_active
    
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
                # Native process names are already clean
                name = c_id
                
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
            overload_flag = 1 if inc_rate > proc_rate else 0
            qpi = min(1.0, (queue / 20.0 * 0.4) + (overload_flag * 0.6))
            
            # --- Advanced State Machine Logic ---
            # spike_trigger drives state directly via spike_active flag.
            # HTTP errors are secondary (Azure containers don't drop under CPU load).
            stats["errors"] = 0  # reset error counter each window

            if spike_active:
                # A spike is live — force FAILED if not already
                if system_state != "FAILED":
                    incident_id_counter += 1
                    active_incident_id = incident_id_counter
                    incident_start_time = time.time()
                    system_state = "FAILED"
                    incident_phase = "ACTIVE_FAILURE"
                    logging.info(f"[STATE] → FAILED  service={incident_service} patient={patient_zero}")
            else:
                # No spike active — recover to NORMAL
                if system_state == "FAILED":
                    system_state = "NORMAL"
                    incident_phase = "NONE"
                    active_incident_id = 0
                    incident_service = "None"
                    patient_zero = "None"
                    logging.info("[STATE] → NORMAL  (recovered)")

            incident_duration = time.time() - incident_start_time if active_incident_id > 0 else 0
            numeric_label = 1 if system_state == "FAILED" else 0

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
    """Sends 2 HTTP pings/sec to each Azure service to generate λ (arrival rate)."""
    logging.info("Starting Research Heartbeat (Azure Load Generator)...")
    count = 0
    while True:
        if running_service_names:
            for target in running_service_names:
                base_url = discovered_ports.get(target)   # now a full HTTPS URL
                if base_url:
                    for _ in range(2):                    # steady λ = 2 req/s/service
                        asyncio.create_task(fetch(session, f"{base_url}/ping"))

            if count % 10 == 0:
                logging.info(f"Heartbeat: Monitoring {len(running_service_names)} Azure service(s)")
        else:
            if count % 10 == 0:
                logging.warning("Heartbeat: Waiting for Azure services to be discovered...")

        count += 1
        await asyncio.sleep(1)

async def spike_trigger(session):
    """Periodically triggers research spikes on Azure services via their sidecar /spike endpoint.
    Sets spike_active=True to force FAILED state, then False after spike ends for auto-recovery.
    """
    global incident_service, patient_zero, spike_active
    # Waits 45s on startup to let discovery settle first
    await asyncio.sleep(45)
    while True:
        if running_service_names and len(running_service_names) >= 2:
            mode    = random.choice(["SINGLE", "CASCADE", "SLOW"])
            targets = random.sample(running_service_names, 2)

            if mode == "SINGLE":
                logging.info(f"--- TRIGGERING 3-MINUTE AZURE SPIKE: {targets[0]} ---")
                incident_service = targets[0]
                patient_zero     = targets[0]
                spike_active     = True          # → FAILED
                asyncio.create_task(
                    fetch(session, f"{discovered_ports[targets[0]]}/spike?duration=180")
                )
                await asyncio.sleep(182)         # hold FAILED for 3 minutes (180s + 2s buffer)
                spike_active = False             # → NORMAL (auto-recovery)
                logging.info(f"--- AUTOMATIC RECOVERY: {targets[0]} ---")

            elif mode == "CASCADE":
                logging.info(f"--- TRIGGERING 3-MINUTE CASCADE: {targets[0]} \u2192 {targets[1]} ---")
                incident_service = f"{targets[0]},{targets[1]}"
                patient_zero     = targets[0]
                spike_active     = True          # → FAILED
                asyncio.create_task(
                    fetch(session, f"{discovered_ports[targets[0]]}/spike?duration=180")
                )
                await asyncio.sleep(20)
                asyncio.create_task(
                    fetch(session, f"{discovered_ports[targets[1]]}/spike?duration=160")
                )
                await asyncio.sleep(162)         # hold FAILED for remainder of 3 minutes
                spike_active = False             # → NORMAL (auto-recovery)
                logging.info(f"--- AUTOMATIC RECOVERY: {targets[0]},{targets[1]} ---")

            elif mode == "SLOW":
                logging.info(f"--- TRIGGERING 3-MINUTE SLOW DEGRADATION: {targets[0]} ---")
                incident_service = targets[0]
                patient_zero     = targets[0]
                spike_active     = True          # → FAILED
                for _ in range(6):
                    asyncio.create_task(
                        fetch(session, f"{discovered_ports[targets[0]]}/spike?duration=30")
                    )
                    await asyncio.sleep(28)
                await asyncio.sleep(14)          # hold FAILED for total 3 minutes
                spike_active = False             # → NORMAL (auto-recovery)
                logging.info(f"--- AUTOMATIC RECOVERY: {targets[0]} ---")

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

async def start_background_tasks(app):
    session = aiohttp.ClientSession()
    app['session'] = session
    app['load_task'] = asyncio.create_task(load_generator(session))
    app['spike_task'] = asyncio.create_task(spike_trigger(session))
    app['metrics_task'] = asyncio.create_task(metrics_collector())
    app['discovery_task'] = asyncio.create_task(discovery_task(session))
    app['stats_task'] = asyncio.create_task(stats_polling_task())

async def cleanup_background_tasks(app):
    app['load_task'].cancel(); app['metrics_task'].cancel()
    app['discovery_task'].cancel(); app['stats_task'].cancel()
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
