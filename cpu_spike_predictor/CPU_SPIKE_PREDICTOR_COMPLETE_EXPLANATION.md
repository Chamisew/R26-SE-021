# CPU Spike Predictor: Complete Explanation

## 1. Purpose

`cpu_spike_predictor` is Component 4 of the AIOps project. It predicts imminent microservice failure from CPU and queue telemetry, combines the CPU prediction with memory-risk information from Component 3, generates an operational decision, produces an explanation and runbook, and exposes results through an API, live CSV inference engine, and dashboard-compatible files.

The documented research question is:

> Can CPU and queue behavior provide enough advance warning to detect an approaching microservice failure and give an SRE system time to respond?

The primary guide is `PRESENTATION_GUIDE.md`.

## 2. Main data flow

```text
Component 2 CPU/queue dataset
        |
        v
CPU ingestion and schema validation
        |
        v
Forward-looking imminent-failure label
        |
        v
Balanced Random Forest
        |
        v
CPU failure probability
        |
        +---------------------------+
        |                           |
        v                           v
Component 3 memory predictions   Decision engine
        |                           |
        +-------------+-------------+
                      |
                      v
       CPU + memory alarm policy
                      |
        +-------------+-------------+
        |             |             |
        v             v             v
      XAI/RCA     Mitigation     API/dashboard
```

The CPU model uses CPU/queue features only. Memory predictions are decision context and are not model features, reducing possible target leakage.

## 2.1 Step-by-step data flow and technologies

### Step 1: C2 produces CPU and queue telemetry

Component 2, the Queue-Aware CPU Spike Analyzer, saves its output as a CSV file:

```text
c2/Queue-Aware CPU Spike Analyzer/research_framework/final_research_dataset.csv
```

The file contains CPU utilization, CPU trends, queue depth, request rates, overload information, timestamps, service information, and incident labels. CSV is used as a simple file-based data exchange format between components.

### Step 2: MUP produces memory predictions

The Memory Usage Predictor (MUP) saves its predictions as:

```text
MUP/data/memory_predictions.csv
```

MUP provides `memory_prob`, `alert`, and `pred_label`, together with `timestamp`, `service_name`, and `project_id`. These values describe memory risk; they are not used to train the CPU model.

### Step 3: Python reads both CSV files

`src/ingestion.py` uses **Python**, the **pandas** data-processing library, and `pandas.read_csv()` to load the two files. The default paths are calculated with Python's `os.path` functions, so the files can be found whether the commands are run from the workspace root or the `cpu_spike_predictor` directory.

### Step 4: Input schemas are checked

The ingestion layer checks required columns using Python sets and pandas DataFrames. If C2 or MUP is missing a required column, the program raises `ValueError` instead of silently creating an invalid input. Timestamps are converted with pandas to a common `timestamp_dt` field, and missing numeric values are filled with safe defaults.

### Step 5: C2 receives a forward-looking target

Using pandas operations, the loader creates `imminent_failure`. A row is labelled positive when a failure occurs on that row or within the next five rows. This target is used during model training; it is not required for live inference.

### Step 6: C2 and MUP records are integrated

`merge_datasets()` uses pandas grouping and sorting. Records are grouped by `project_id` and `service_name`, then ordered by timestamp. The current implementation matches records by position within each group. The result can be saved as CSV or Apache Parquet in `data/processed/`.

Apache Parquet, supported through pandas and PyArrow, provides a faster and more compact option for loading processed telemetry. If Parquet is unavailable, the system falls back to CSV.

### Step 7: The CPU model is trained

`src/model.py` uses **scikit-learn** and its `RandomForestClassifier`. Only the ten C2 CPU/queue features are passed to the model. The model learns the relationship between those features and `imminent_failure`. The trained model is serialized with **Joblib** to:

```text
outputs/cpu_rf_model.joblib
```

MUP data is loaded and merged for downstream analysis, but it is deliberately excluded from training features to reduce target leakage.

### Step 8: The model generates a CPU risk probability

For new C2 telemetry, scikit-learn's `predict_proba()` returns `cpu_failure_prob`, a value from `0.0` to `1.0`. A probability of at least `0.60` creates the CPU alarm.

### Step 9: Memory and CPU risks are combined

The decision layer uses normal Python control flow to evaluate the CPU probability and MUP's memory probability, alert, and label. It creates a joint alarm and selects an operational action such as load shedding, pod restart, or traffic rerouting.

### Step 10: Explanations and mitigation output are generated

The **XAI** module uses the model's feature importances and baseline deviations to create approximate feature contributions, an RCA narrative, a runbook, and an incident ticket payload. The **mitigation** module creates recommended Kubernetes and Envoy CLI commands. These commands are recorded as recommendations and are not executed against a real cluster.

### Step 11: Results are exposed to other systems

There are two runtime delivery methods:

- **Live CSV mode:** Python periodically polls both files, using `time`, pandas, and a JSON checkpoint to process new rows. Results are written as JSON Lines (`.jsonl`) and a status JSON file.
- **API mode:** **FastAPI** with **Uvicorn** hosts the REST service. **Pydantic** validates incoming JSON requests at `POST /predict` or `POST /batch_predict`. C2 and MUP can send their values directly without writing a CSV for every prediction.

The API response is JSON and can be consumed by the dashboard, connector scripts, or another service. The API and live CSV mode are alternative runtime paths; they are not required to run simultaneously.

### Technology summary

| Pipeline step | Main technology | Role |
|---|---|---|
| File exchange | CSV | Transfers C2 and MUP data |
| Data loading and cleaning | Python, pandas, NumPy | Reads, validates, transforms, and aligns records |
| Processed storage | CSV, Apache Parquet, PyArrow | Stores integrated telemetry |
| Machine learning | scikit-learn | Trains and runs the Random Forest |
| Model storage | Joblib | Saves and loads the trained model |
| REST service | FastAPI, Uvicorn, Pydantic | Receives JSON and returns predictions |
| Live processing | Python polling, JSON checkpoint | Detects and processes new CSV rows |
| Explanation | Python XAI module | Produces RCA, contributions, and runbook |
| Response output | JSON, JSONL | Feeds dashboards and downstream services |

## 3. Package contents

```text
cpu_spike_predictor/
├── src/
│   ├── ingestion.py
│   ├── model.py
│   ├── validation.py
│   ├── live_inference.py
│   ├── api.py
│   ├── xai.py
│   ├── mitigation.py
│   ├── mtta.py
│   ├── visualization.py
│   └── cli.py
├── tests/
│   ├── test_pipeline.py
│   └── test_live_inference.py
├── data/processed/aligned_telemetry.csv
├── outputs/
├── comp4_connector.py
├── _verify_data.py
├── _test_integration.py
├── requirements.txt
└── PRESENTATION_GUIDE.md
```

## 4. Input datasets

### 4.1 Component 2 CPU dataset

The expected file is:

```text
c2/Queue-Aware CPU Spike Analyzer/research_framework/final_research_dataset.csv
```

That Component 2 directory is not present in the current workspace. The expected columns are:

```text
incident_id, timestamp, time, system_state, incident_phase,
failing_service, patient_zero, cpu_percent, cpu_velocity,
cpu_trend_5min, cpu_trend_10min, in_flight_queue, incoming_rate,
processing_rate, queue_growth_rate, overload_flag,
queue_pressure_index, incident_duration, label
```

The data contains CPU utilization, CPU changes, trends, queue state, request rates, overload state, incident information, and ground-truth labels.

### 4.2 Component 3 memory dataset

The expected file is:

```text
MUP/data/memory_predictions.csv
```

Required columns:

```text
timestamp, service_name, project_id, memory_prob, alert, pred_label
```

The loader derives these additional fields:

```text
memory_leak_prob
timestamp_dt
alert_bool
```

`pred_label` is normally `NORMAL`, `WARNING`, or `FAILURE`.

## 5. `ingestion.py`

`ingestion.py` loads and prepares both datasets.

### Main responsibilities

- Define default input paths.
- Validate required schemas.
- Load CPU/queue telemetry.
- Load memory predictions.
- Derive missing project and service identifiers.
- Parse timestamps.
- Fill missing feature values.
- Create the forward-looking target.
- Merge CPU and memory records for evaluation and decisions.

### Strict validation

Missing required columns cause a `ValueError`. The code does not silently manufacture missing input columns.

### Project and service derivation

If `project_id` is absent, the loader tries to extract a project number from `failing_service`. If it cannot, it falls back to `Proj_01`.

If `service_name` is absent, it uses `failing_service` or creates a generic backend service name.

### Missing values

Numeric CPU fields are filled with `0.0`. `overload_flag` is filled with integer `0`.

## 6. Lead-time labeling

The target is `imminent_failure`.

The default CLI lead time is five, but the implementation uses a five-row forward-looking window, not five elapsed minutes. A row is positive if a failure is present now or appears within the following five rows.

```text
imminent_failure[i] = max(failure[i], failure[i+1], ..., failure[i+5])
```

The phrase `five-minute warning` is valid only when the source data contains one row per minute. If records arrive at another interval, the real elapsed warning time is different.

## 7. CPU model features

The model uses exactly these ten features:

1. `cpu_percent`: current CPU utilization.
2. `cpu_velocity`: rate of CPU change.
3. `cpu_trend_5min`: short-term CPU trend.
4. `cpu_trend_10min`: longer CPU trend.
5. `in_flight_queue`: current queue depth.
6. `incoming_rate`: request arrival rate.
7. `processing_rate`: request processing throughput.
8. `queue_growth_rate`: rate at which the queue grows.
9. `queue_pressure_index`: queue back-pressure indicator.
10. `overload_flag`: binary overload indicator.

These are declared in `src/model.py` as `COMP2_CPU_FEATURE_COLS`.

## 8. `model.py`

`model.py` trains, saves, loads, and runs the CPU model.

### Random Forest configuration

```python
RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

`class_weight="balanced"` helps compensate for imbalanced failure labels. `random_state=42` makes training deterministic for the same data and environment. `n_jobs=-1` uses available CPU workers.

### Model output

The model returns `cpu_failure_prob`, a value from `0.0` to `1.0`.

The CPU alarm threshold is:

```text
cpu_failure_prob >= 0.60
```

### Functions

- `get_features_and_target()` extracts the ten CPU features and `imminent_failure`.
- `train_model()` trains the Random Forest.
- `save_model()` writes a Joblib model file.
- `load_model()` loads the Joblib model.
- `predict_probabilities()` returns positive-class probabilities and handles missing feature columns defensively.
- `get_feature_importances()` returns Random Forest feature importance values.

The production model is stored as:

```text
outputs/cpu_rf_model.joblib
```

## 9. Training workflow

The production training workflow is:

```text
Load Component 2 data
        |
Load Component 3 data
        |
Create CPU failure labels
        |
Merge CPU and memory context
        |
Extract only ten CPU features
        |
Train Random Forest on all available data
        |
Save cpu_rf_model.joblib
```

Memory data is loaded and merged, but it is not passed into CPU model training.

## 10. `validation.py`

`validation.py` evaluates model generalization.

### LOPO validation

LOPO means Leave-One-Project-Out validation.

For each project:

1. Hold that project out.
2. Train using all other projects.
3. Predict the held-out project.
4. Calculate metrics.
5. Repeat for every project.

This tests transfer to an unseen project and avoids placing rows from the same project in both training and testing.

If only one project exists, the code falls back to an 80/20 train/test split.

### Metrics

The validation output includes:

- Precision.
- Recall.
- F1 score.
- False-alarm rate.
- True negatives.
- False positives.
- False negatives.
- True positives.

## 11. Current CPU validation results

The stored artifact is `outputs/lopo_results.csv`.

| Project | Precision | Recall | F1 | False-alarm rate |
|---|---:|---:|---:|---:|
| Proj_01 | 0.2719 | 0.9998 | 0.4275 | 0.9987 |
| Proj_02 | 1.0000 | 0.2727 | 0.4285 | 0.0000 |
| Proj_03 | 1.0000 | 0.5096 | 0.6752 | 0.0000 |
| Proj_04 | 0.8750 | 0.7778 | 0.8235 | 0.5000 |
| Proj_05 | 1.0000 | 0.6667 | 0.8000 | 0.0000 |
| Proj_06 | 1.0000 | 0.6923 | 0.8182 | 0.0000 |

The results are uneven. `Proj_01` detects almost every positive but produces many false alarms. `Proj_02` has perfect precision but misses many failures. The artifact demonstrates the evaluation pipeline, but it does not prove robust generalization across all projects.

## 12. Evaluation formulas

Precision:

```text
TP / (TP + FP)
```

Recall:

```text
TP / (TP + FN)
```

F1:

```text
2 * precision * recall / (precision + recall)
```

False-alarm rate:

```text
FP / (TN + FP)
```

## 13. `mtta.py`

`mtta.py` evaluates advance warning time.

The code identifies a failure start when `ground_truth_failure` changes from `0` to `1`. It searches up to 20 minutes before the failure and finds the earliest alarm.

```text
warning time = failure time - earliest alarm time
```

It evaluates four strategies:

1. CPU-only.
2. Memory-only.
3. CPU AND memory.
4. CPU OR memory.

The CPU threshold is `0.60`. The MTTA module uses a memory threshold of `0.60`, while the API uses `0.70`; this inconsistency should be resolved.

## 14. `visualization.py`

The module creates `outputs/cpu_mtta_chart.png` containing:

1. A strategy comparison bar chart.
2. A project-level CPU-only warning-time box plot.

The second chart sets a maximum y-axis of six minutes, although stored MTTA values can approach twenty minutes. Values may therefore be clipped.

## 15. `live_inference.py`

This module performs real-time inference from live-updating CSV files.

It does not:

- Retrain the model.
- Modify source CSV files.
- Generate fake source data.

It does:

- Load a previously trained model.
- Validate both input schemas.
- Detect new records.
- Preprocess new records.
- Align CPU and memory data.
- Predict CPU failure probability.
- Apply CPU-memory decisions.
- Generate RCA output.
- Write JSONL predictions.
- Write per-service status JSON.
- Track processed records.

### Startup requirements

The model and both input files must exist. Startup fails if the model is missing or a required input schema is invalid.

### Safe CSV reading

`safe_read_csv()` retries parser, empty-file, and permission failures. It also drops an all-null final row, which may indicate a partially written CSV.

### Checkpointing

Checkpoint file:

```text
outputs/.inference_checkpoint.json
```

Component 2 keys use incident ID and timestamp:

```text
c2:<incident_id>:<timestamp>
```

Component 3 keys use timestamp, service, and project:

```text
c3:<timestamp>:<service_name>:<project_id>
```

The checkpoint is saved using atomic file replacement and keeps up to 10,000 keys.

### Live alignment

The live engine groups both datasets by `project_id` and `service_name`, sorts by timestamp, and aligns records by row position.

It does not perform an exact timestamp or event-ID join. Different sampling rates, missing rows, or different file offsets can attach the wrong memory result to a CPU record.

### Live output

Each prediction can include:

```text
prediction_timestamp
source_timestamp
incident_id
service_name
project_id
cpu_failure_prob
cpu_alarm
memory_prob
memory_alert
memory_pred_label
joint_alarm
action_recommended
rca_narrative
feature_contributions
shapley_phi_values
sre_runbook_steps
incident_ticket_payload
```

The prediction JSONL file is rotated to approximately 2,000 lines to avoid unbounded growth.

### Live limitations

When all CPU rows are marked processed, the current code resets the checkpoint and can reprocess the entire file, creating duplicate predictions. The live positional memory alignment can also use incorrect historical rows for new CPU records.

## 16. `api.py`

The FastAPI service defaults to:

```text
http://127.0.0.1:8000
```

### Endpoints

- `GET /health`: process liveness.
- `GET /healthz/liveness`: liveness alias.
- `GET /healthz/readiness`: confirms the model is loaded.
- `GET /metrics`: JSON or Prometheus-style counters.
- `POST /predict`: one service prediction.
- `POST /batch_predict`: multiple service predictions.

### Request structure

```json
{
  "service_name": "srv-payment-backend",
  "project_id": "Proj_01",
  "comp2_cpu": {
    "cpu_percent": 87.5,
    "cpu_velocity": 5.2,
    "cpu_trend_5min": 12.3,
    "cpu_trend_10min": 20.1,
    "in_flight_queue": 150.0,
    "incoming_rate": 250.0,
    "processing_rate": 230.0,
    "queue_growth_rate": 20.0,
    "overload_flag": 0,
    "queue_pressure_index": 3.0
  },
  "comp3_memory": {
    "memory_prob": 0.95,
    "alert": "TRUE",
    "pred_label": "FAILURE"
  }
}
```

Pydantic validates ranges and applies defaults.

### Decision policy

CPU alarm:

```text
CPU probability >= 0.60
```

Memory alarm if any condition is true:

```text
memory alert is TRUE
memory probability >= 0.70
memory label is WARNING or FAILURE
```

| CPU alarm | Memory alarm | Recommended action |
|---|---|---|
| No | No | `NO_ACTION` |
| Yes | No | `TRIGGER_LOAD_SHEDDING` |
| No | Yes | `TRIGGER_PROACTIVE_POD_RESTART` |
| Yes | Yes | `CRITICAL_RESTART_AND_TRAFFIC_REROUTE` |

The joint alarm is CPU AND memory, not CPU OR memory.

### API response

The response contains probabilities, alarm states, recommended action, warning message, RCA, feature contributions, runbook steps, ticket payload, mitigation result, latency, and timestamp.

The response field `memory_alert` represents the computed memory alarm and not always the original raw Component 3 alert value.

## 17. `xai.py`

The XAI module generates:

- Feature contributions.
- Primary and secondary likely causes.
- Severity.
- Impact radius.
- RCA narrative.
- SRE runbook.
- Incident-ticket payload.

It compares current values with fixed baseline values and weights positive deviations by Random Forest feature importance.

Conceptually:

```text
weight(feature) = positive_baseline_deviation * model_feature_importance
```

Weights are normalized into percentages.

### Important accuracy statement

The CPU predictor's `shapley_phi_values` are not exact TreeSHAP values. They are an approximate attribution based on feature importance, baseline deviations, and predicted probability. They should be presented as approximate feature attribution.

### RCA categories

The generated RCA can identify:

- High CPU velocity.
- CPU resource exhaustion.
- Queue backup.
- Overload condition.
- Normal operation.
- Elevated risk.

The ticket ID uses Python's built-in `hash()`, so it may vary between processes.

## 18. `mitigation.py`

The mitigation module maintains process-local per-service state and a three-minute cooldown.

### Tier 1: load shedding

```text
envoy-cli rate-limit set --service=<service> --drop-ratio=0.25
```

### Tier 2: scale or replace

```text
kubectl scale deployment/<service> --replicas=+2
```

or:

```text
kubectl delete pod -l app=<service> --grace-period=30
```

### Tier 3: restart and drain

```text
kubectl rollout restart deployment/<service>
kubectl label pod -l app=<service> status=drain
```

The commands are returned and logged. They are not executed against a real Kubernetes or Envoy environment.

Limitations include process-local state, cooldown suppression of non-`NO_ACTION` actions including critical actions, lack of project namespacing, and a typo in `TIER_2_HORIZONAL_SCALE_OUT`.

## 19. `cli.py`

### Validate

```powershell
python -m src.cli validate
```

Runs ingestion, merging, LOPO validation, MTTA calculation, and chart generation.

### Train

```powershell
python -m src.cli train
```

Trains the production model and saves `outputs/cpu_rf_model.joblib`.

### Serve

```powershell
python -m src.cli serve
```

Starts the FastAPI server.

### Live

```powershell
python -m src.cli live
```

Starts the live CSV inference engine. The CLI default polling interval is five seconds; the module-level default is one second.

## 20. `comp4_connector.py`

This helper sends data to the API.

Functions:

- `send_to_component4()`: sends combined CPU and memory data.
- `send_comp2_cpu_data()`: sends CPU data with normal memory defaults.
- `send_comp3_memory_data()`: sends memory data with zero CPU defaults.
- `send_batch_to_component4()`: sends multiple services to `/batch_predict`.

Default endpoints:

```text
http://localhost:8000/predict
http://localhost:8000/batch_predict
```

The connector has limited retry handling, returns empty objects/lists on many failures, and does not send timestamps or incident IDs.

## 21. Tests

`tests/test_pipeline.py` tests:

- CPU ingestion.
- Memory ingestion.
- Required feature columns.
- Lead-time labels.
- CPU-memory merge.
- Model training.
- Probability bounds.
- Model saving and loading.
- LOPO validation.
- MTTA calculation.
- FastAPI health and predictions.
- RCA and runbook output.

`tests/test_live_inference.py` is a manual integration script. It tests startup, reading records, preprocessing, alignment, output creation, checkpoint updates, and a second read.

Run the standard tests from the project directory:

```powershell
pytest
```

## 22. Requirements

The project uses:

- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- FastAPI
- Uvicorn
- pytest
- requests
- Streamlit
- Joblib
- HTTPX
- Polars
- PyArrow

See `requirements.txt` for version constraints.

## 23. Current MUP evidence

The stored MUP metadata reports:

```text
Best model: Random Forest
Mean F1: 0.8571
Mean AUC: 0.9057
Rows: 24,585
Projects: 5
Threshold: 0.60
```

A separate historical comparison artifact reports Gradient Boosting with F1 `0.8415`. These are conflicting runs and must not be combined without a fresh reproducible experiment.

## 24. Most important limitations

1. The Component 2 source directory is missing from the current workspace.
2. CPU-memory records are aligned by row position rather than exact timestamp or event ID.
3. Live incremental alignment may attach incorrect memory context.
4. Five-minute lead time is implemented as five rows.
5. Live ground truth is unavailable, so live F1 is not valid.
6. CPU XAI is approximate, not exact TreeSHAP.
7. Mitigation commands are generated but not executed.
8. Mitigation state is process-local and lost on restart.
9. API and live engine can write shared files concurrently.
10. Checkpoint reset can replay existing records.
11. API and MTTA memory thresholds differ.
12. Stored model artifacts represent different historical runs.
13. LOPO folds have uneven class distributions.
14. At least one fold has a false-alarm rate near 99.87%.
15. No controlled numerical baseline comparison is available.
16. Dashboard reliability data includes hard-coded/fallback values.
17. Production security, authorization, auditability, and business impact are not evidenced.

## 25. Correct project contribution

The repository supports the following contribution claim:

> The project converts CPU and queue telemetry into an evaluated early-warning probability, combines that prediction with independent memory-risk context, generates an explainable operational decision and runbook, and exposes the result through API, live inference, and dashboard outputs.

It does not currently prove universal accuracy, superiority over a measured baseline, real production incident reduction, automatic Kubernetes recovery, or exact SHAP explanations.

## 26. Recommended run sequence

From `cpu_spike_predictor`:

```powershell
python -m src.cli train
python -m src.cli validate
python -m src.cli serve
```

In another terminal:

```powershell
python -m src.cli live
```

API checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/healthz/readiness
Invoke-RestMethod http://127.0.0.1:8000/metrics
```

The training and validation commands require the expected Component 2 dataset to be available.

## 27. Viva explanation

`cpu_spike_predictor` is an AIOps prediction and response component for microservices. It receives CPU and queue telemetry from Component 2 and memory predictions from Component 3. A balanced Random Forest uses ten CPU and queue features to predict imminent failure without using memory predictions as training features. The system evaluates generalization with Leave-One-Project-Out validation. At runtime, CPU probability is combined with memory probability, alert, and label. The decision engine classifies the service and recommends no action, load shedding, proactive pod replacement, or critical restart and traffic rerouting. The system also generates approximate feature attribution, RCA text, runbook steps, ticket data, API responses, live JSONL predictions, and dashboard status. It is a research prototype because alignment is row-based, live ground truth is unavailable, CPU attribution is approximate, and mitigation commands are generated rather than executed.
