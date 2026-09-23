app = FastAPI(
    title="Memory Failure Detection API",
    description="FastAPI wrapper providing REST and WebSocket APIs for microservice memory-failure pipeline monitoring.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SharedState:
    """Thread-safe in-memory state used across all API endpoints and background workers."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.status = "stopped"          # Options: stopped | running | collecting
        self.mode = PIPELINE_MODE

        self.raw_rows: list[dict[str, Any]] = []
        self.windows_rows: list[dict[str, Any]] = []

        self.total_rows_collected = 0
        self.total_windows_generated = 0
        self.last_global_window_id = 0
        self.active_containers: list[str] = []
        self.last_error: str | None = None


STATE = SharedState()

SERVICES_CONFIG_PATH = Path(__file__).resolve().parent / "services_config.json"


def _read_services_config() -> dict[str, Any]:
    """Safely load configured services from services_config.json."""
    try:
        with SERVICES_CONFIG_PATH.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        return {"services": []}
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Cannot read services configuration: {exc}"
        ) from exc

    if not isinstance(config, dict) or not isinstance(config.get("services", []), list):
        raise HTTPException(
            status_code=500, detail="services_config.json has an invalid format"
        )
    return config

    def _write_services_config(config: dict[str, Any]) -> None:
    """Safely write updated service configuration back to services_config.json."""
    try:
        with SERVICES_CONFIG_PATH.open("w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=2)
            config_file.write("\n")
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Cannot save services configuration: {exc}"
        ) from exc

SERVICE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class AzureServiceInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, description="Service container name")
    base_url: HttpUrl = Field(..., description="HTTP base endpoint URL of sidecar")
    enabled: bool = Field(default=True, description="Service enabled flag")


    @app.get("/api/azure-services")
def api_azure_services() -> list[dict[str, Any]]:
    """Return all configured Azure microservice sidecars."""
    return _read_services_config()["services"]


@app.get("/api/services-config")
def api_services_config() -> dict[str, Any]:
    """Read-only view of services_config.json for configuration tab."""
    return _read_services_config()

    @app.post("/api/azure-services")
def api_add_azure_service(service: AzureServiceInput) -> dict[str, Any]:
    """Validate, health check, and register a new Azure sidecar service."""
    name = service.name.strip()
    if not SERVICE_NAME_PATTERN.fullmatch(name):
        raise HTTPException(
            status_code=422,
            detail="Service name may contain only letters, numbers, dots, underscores, and hyphens.",
        )

    base_url = str(service.base_url).rstrip("/")
    verified = False
    verification_notes = []

    try:
        metrics_resp = requests.get(f"{base_url}/metrics", timeout=6.0)
        if metrics_resp.status_code == 200:
            verified = True
            verification_notes.append("/metrics returned HTTP 200")
        else:
            verification_notes.append(f"/metrics returned HTTP {metrics_resp.status_code}")
    except requests.RequestException as exc:
        verification_notes.append(f"/metrics unreachable: {exc}")

    config = _read_services_config()
    config["services"] = [
        item for item in config.get("services", [])
        if isinstance(item, dict) and item.get("name") != name
    ]
    config["services"].append({"name": name, "base_url": base_url, "enabled": service.enabled})
    _write_services_config(config)

    with STATE.lock:
        if name not in STATE.active_containers and service.enabled:
            STATE.active_containers = sorted(STATE.active_containers + [name])

    msg = f"Service '{name}' saved to services_config.json."
    if verified:
        msg += " Endpoint verified and active."
    else:
        msg += f" Note: remote check details: {'; '.join(verification_notes)}"

    return {
        "status": "saved",
        "verified": verified,
        "service": {"name": name, "base_url": base_url, "enabled": service.enabled},
        "message": msg,
    }

    @app.delete("/api/azure-services/{service_name}")
def api_delete_azure_service(service_name: str) -> dict[str, Any]:
    """Delete a service entry from services_config.json."""
    name = service_name.strip()
    config = _read_services_config()
    services_list = config.get("services", [])
    before_len = len(services_list)
    config["services"] = [s for s in services_list if isinstance(s, dict) and s.get("name") != name]

    if len(config["services"]) == before_len:
        raise HTTPException(
            status_code=404, detail=f"Service '{name}' not found in services_config.json."
        )

    _write_services_config(config)
    with STATE.lock:
        STATE.active_containers = [s for s in STATE.active_containers if s != name]

    return {
        "status": "deleted",
        "service_name": name,
        "message": f"Service '{name}' removed from services_config.json.",
    }