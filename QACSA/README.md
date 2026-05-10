# Queue-Aware CPU Spike Analyzer

A comprehensive research framework and real-time visualization dashboard for analyzing CPU spikes with queue-awareness. This project combines a Python-based research backend for data collection and analysis with a modern React frontend for real-time monitoring and visualization.

## Project Structure

- `research_framework/`: Python scripts for CPU performance monitoring, queue analysis, and data collection.
- `frontend/`: A React + Vite dashboard for visualizing CPU metrics and spike patterns.

## Features

- **Real-time CPU Monitoring**: Tracks CPU usage across multiple cores.
- **Queue-Aware Analysis**: Analyzes process queues to identify the root causes of CPU spikes.
- **ML-Ready Datasets**: Generates structured datasets for machine learning training.
- **Interactive Dashboard**: Premium UI for monitoring system health and performance spikes.

## Setup

### Backend (Research Framework)
1. Navigate to `research_framework/`.
2. Create a virtual environment: `python -m venv venv`.
3. Install dependencies: `pip install -r requirements.txt`.
4. Run experiments: `python realtime_experiment.py`.

### Frontend
1. Navigate to `frontend/`.
2. Install dependencies: `npm install`.
3. Start dev server: `npm run dev`.
