<div align="center">

# Sentinel AIOps

### A Predictive Site-Reliability Command Center

**Machine-learning-driven failure prediction · Explainable AI · Human-in-the-loop remediation**

`Python 3.11` · `FastAPI` · `scikit-learn` · `XGBoost` · `SHAP` · `React` · `Vite`

</div>

---

## Overview

**Sentinel** is a 24/7 autonomous guard for microservice platforms. Where conventional monitoring reacts *after* an outage, Sentinel predicts failures *before* they occur.

The platform ingests live logs and telemetry, applies a suite of machine-learning models to forecast memory-leak and CPU-spike incidents, and surfaces each prediction to site-reliability engineers with a plain-English explanation and a recommended remediation step (for example, load shedding or a controlled restart). Every decision is captured in an auditable event log.

The end-to-end loop is:

> **Telemetry → Prediction → Explanation → Recommended Action → Human Decision**

---

## System Architecture

```
                        ┌──────────────────────────────────────────────┐
   Microservices  ────> │  Component 1 · Ingestion API          :8001   │  Log & metric mining
   + queue sidecars     │  (Drain3 templates · raw telemetry)           │  FastAPI + WebSocket
                        └───────────────────────┬──────────────────────┘
                                                │  telemetry
             ┌──────────────────────────────────┼──────────────────────────────────┐
             ▼                                  ▼                                  ▼
  ┌───────────────────────┐   ┌───────────────────────────┐   ┌───────────────────────────────┐
  │  Component 2           │   │  Component 3 · MUP         │   │  Component 4 · CPU Predictor    │
  │  Queue-Aware Analyzer  │   │  Memory-Leak Predictor     │   │  FastAPI :8000 +                │
  │  Services :5001-5005   │   │  LOPO Random Forest + SHAP │   │  Live Inference Engine          │
  │  Console  :8081        │   │  (feeds :8000)             │   │  Time-to-Failure / MTTA models   │
  └───────────┬───────────┘   └─────────────┬─────────────┘   └────────────────┬────────────────┘
              │  predictions                 │  predictions                     │  predictions
              └──────────────────────────────┴──────────────────────────────────┘
                                                ▼
                        ┌──────────────────────────────────────────────┐
                        │  Sentinel Dashboard · Gateway + React  :8766   │
                        │  KPIs · failure probability · service cards   │
                        │  alerts · recommended actions · audit log      │
                        └────────────────────────────────────────────────┘
```

Predictors publish results to the Component 4 REST API (`POST /predict`) and to CSV feeds (`MUP/data/*.csv`), which the dashboard gateway reads and renders in real time.

---

## Components

| # | Component | Responsibility | Technology | Endpoint |
|---|-----------|----------------|------------|----------|
| 1 | [Ingestion Pipeline](component1/) | Collects raw logs and metrics, mines log templates with Drain3, and serves REST/WebSocket APIs. | FastAPI · pandas · scikit-learn | `:8001` |
| 2 | [Queue-Aware Spike Analyzer](c2/) | Runs the target microservices and queue-aware sidecars that generate realistic load and CPU-spike telemetry. | Python · .NET · React | `:5001–5005`, `:8081` |
| 3 | [Memory-Leak Predictor (MUP)](MUP/) | Predicts out-of-memory failures from leak-physics features and streams live predictions. | Random Forest · LOPO CV · SHAP | feeds `:8000` |
| 4 | [CPU Spike Predictor](cpu_spike_predictor/) | Predicts CPU spikes and time-to-failure/MTTA, and runs continuous live inference. | FastAPI · RF/XGBoost · joblib | `:8000` |
| — | [Sentinel Dashboard](Dashboard/) | Command center for alerts, explanations, recommended actions, and the audit trail. | Python gateway · React (Vite) | `:8766` (`:5173` dev) |

---

## Prerequisites

- **Python 3.11+**
- **Node.js** — required only for the optional Vite development server; the gateway already serves the production UI bundle.

---

## Quick Start

### Launch the full stack (Windows)

```bat
start_all_backends.bat
```

This starts Components 1–4, the dashboard gateway, and (optionally) the Vite development server, then opens <http://localhost:8766>.

> **Important:** Component 3 must be trained before the stack runs, otherwise every memory prediction is `0.0`. Execute `MUP\start.bat` — or `cd MUP && python scripts/train_model.py` — once beforehand.

### Launch individual components

```bash
# Component 3 — train the model, then start the live feed
cd MUP
pip install -r requirements.txt
python scripts/train_model.py
python scripts/simulate_live.py

# Component 4 — prediction API and live inference engine
cd cpu_spike_predictor
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
python -m src.live_inference

# Sentinel dashboard gateway (serves the API and the built React UI)
cd Dashboard
python server.py            # http://localhost:8766
```

---

## Machine-Learning Methodology

| Technique | Description |
|-----------|-------------|
| **Feature engineering** | Raw telemetry is transformed into leak- and pressure-physics features (for example `heap_fill_ratio`, `gc_pressure_index`, `memory_acceleration`, `time_to_oom_slope`). |
| **Leave-One-Project-Out (LOPO) validation** | Each fold trains on nine projects and tests on a fully unseen project, demonstrating that the model generalizes rather than memorizing a single service's behaviour. |
| **Metric selection** | Failures are rare, so accuracy is misleading. Models are ranked on **F1, Recall, Precision, AUC, and LOPO variance** — not accuracy. |
| **Threshold optimization** | The alarm threshold (Component 3: `0.28`) is tuned on the LOPO folds to prioritize detecting real leaks over suppressing false alarms. |
| **Explainability (SHAP)** | Every prediction carries a plain-English rationale, ranked by cross-fold feature-importance consistency. |
| **Hot reload** | The live simulator detects and loads retrained model bundles without interrupting the data feed. |

---

## Project Structure

```
.
├── component1/            # Component 1 — Ingestion API and log/metric pipeline
├── c2/                    # Component 2 — Queue-Aware CPU Spike Analyzer and microservices
├── MUP/                   # Component 3 — Memory-Leak Predictor
├── cpu_spike_predictor/   # Component 4 — CPU Spike / Time-to-Failure Predictor
├── Dashboard/             # Sentinel AIOps gateway and React UI
├── start_all_backends.bat # One-command full-stack launcher (Windows)
└── start_all_backends.ps1 # PowerShell launcher
```

---

## Testing

Each component provides its own `pytest` suite:

```bash
cd MUP                 && python -m pytest tests/   # Component 3
cd cpu_spike_predictor && python -m pytest tests/   # Component 4
cd Dashboard           && python -m pytest tests/   # Dashboard
```

---

## Configuration Reference

| Setting | Value | Defined in |
|---------|-------|------------|
| Dashboard gateway / UI | `http://localhost:8766` | `Dashboard/server.py` |
| Component 4 prediction API | `http://127.0.0.1:8000/predict` | `cpu_spike_predictor/src/api.py` |
| Component 1 ingestion API | `http://localhost:8001/docs` | `component1/app.py` |
| Component 2 microservices | ports `5001–5005` | `c2/**/run_local_services.py` |
| Memory alarm threshold | `0.28` | `MUP/models/model_metadata.json` |

---

## Notes

- Model binaries (`*.pkl`, `*.joblib`) and generated data feeds (`data/*.csv`, `logs/*`) are git-ignored and fully regenerable. Train before running.
- The root-level `app.py` is an unrelated utility and is not part of Sentinel AIOps.
