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