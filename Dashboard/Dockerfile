FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

# Copy dashboard application
COPY . .

# Expose Gateway API port
EXPOSE 8766

# Run Gateway server
CMD ["python", "server.py"]
