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