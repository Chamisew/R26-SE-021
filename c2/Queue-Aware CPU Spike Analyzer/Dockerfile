# Production Python Backend, Telemetry Engine & Web Console (serves API + frontend on port 8081)
FROM python:3.11-slim AS runner
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements & install
COPY research_framework/requirements.txt ./
COPY research_framework/requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt

# Copy backend research framework, web console and root files
COPY research_framework/ ./research_framework/
COPY frontend/dist/ ./frontend/dist/
COPY services_config.json ./services_config.json
COPY services_config.json ./research_framework/services_config.json

# Environment variables
ENV PORT=8081
EXPOSE 8081

WORKDIR /app/research_framework
CMD ["python", "realtime_experiment.py"]
