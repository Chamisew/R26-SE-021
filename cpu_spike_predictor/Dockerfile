FROM python:3.11-slim

WORKDIR /app

# Install build dependencies for scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY . .

# Expose CPU Predictor API port
EXPOSE 8000

# Start ML Prediction FastAPI server
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
