# Canonical Dataset Data Dictionary: `final_research_dataset.csv`
**Queue-Aware CPU Spike Analyzer Research Framework**

This document serves as the authoritative specification and data dictionary for `final_research_dataset.csv`. It details the 10 functional telemetry sections, data types, semantic definitions, and operational caveats for machine learning and statistical evaluation.

---

## 1. Core Labeling Architecture: Predictions vs. Ground Truth

A critical distinction in this dataset is the decoupling of **Online Model Predictions** from **Objective Physical Ground Truth**.

### 1.1 `actual_failure` — Objective Platform Ground Truth ($0/1$)
* **Data Type**: Binary Integer (`0` or `1`)
* **Semantic Definition**: An unambiguous, physically observed platform outage event. It is **zero-threshold-based** and derived strictly from transport-layer and orchestrator signals:
  $$\text{actual\_failure} = \mathbb{I}\left(\text{service\_unreachable} = 1 \lor \text{health\_check\_failed} = 1 \lor \text{request\_timeout} = 1\right)$$
  - `service_unreachable = 1`: HTTP 5xx error, TCP socket connection refused, or connection timeout while polling `/metrics`.
  - `health_check_failed = 1`: Non-200 response or `target_reachable: false` returned by the `/health` endpoint.
  - `request_timeout = 1`: Client `/ping` probe timed out under end-to-end network latency.
* **Guarantee**: `actual_failure` never incorporates CPU utilization, queue length, or latency thresholds. It is pure ground truth.

### 1.2 `predicted_state` — Online Heuristic Detector Verdict
* **Data Type**: Categorical String (`HEALTHY`, `DEGRADED`, `FAILED`, `RECOVERING`, `CALIBRATING`)
* **Semantic Definition**: The real-time heuristic state machine classification emitted by the live telemetry analyzer. It compares instantaneous telemetry ($Z_{\text{cpu}}$, $Z_{\text{queue}}$, $Z_{\text{QPI}}$, and rolling dynamic P95/P99 thresholds) against baseline calibration profiles.
  - Requires a 3-tick debounce (`abnormal_count >= 3`) to transition into `FAILED`.
  - Requires a bounded hysteresis window (up to 10 ticks) to unwind back to `HEALTHY`.

### 1.3 Interpreting Discrepancies (False Positives & False Negatives)
Disagreements between `predicted_state` and `actual_failure` are **not data-quality bugs**; they represent the exact classification outcomes required to evaluate operational monitoring systems:
1. **`predicted_state == HEALTHY` with `actual_failure == 1` (False Negative / Detection Lag)**:
   - **Debounce Window**: If a crash or partition occurs abruptly at $t$, `actual_failure` flips immediately to `1`, but the detector requires 3 consecutive ticks of observed anomalies before asserting `FAILED`.
   - **Crash / Telemetry Blind Spot**: When a container process is killed instantly (e.g., OOM kill or SIGKILL), its CPU utilization drops to `0.0%`. A CPU-only monitor sees low CPU and remains `HEALTHY` while the platform reports the service unreachable.
2. **`predicted_state == FAILED` with `actual_failure == 0` (False Positive / Pre-Crash Overload)**:
   - **Capacity Saturation**: A microservice undergoing a severe load surge ($Z_{\text{cpu}} > 3.0$ or $Z_{\text{QPI}} > 3.0$) is flagged as `FAILED` (operational exhaustion) even though its TCP socket is still answering pings (`actual_failure == 0`).
   - **Recovery Hysteresis**: While the service is recovering from an incident, the detector intentionally remains in `FAILED` or `RECOVERING` to prevent flapping, even if platform pings have just begun succeeding again.

> [!IMPORTANT]
> If `predicted_state` matched `actual_failure` on 100% of rows, the predictive pipeline would suffer from tautological target leakage. Evaluating ROC-AUC, Precision, Recall, and F1 requires these natural classification discrepancies.

---

## 2. Telemetry Ingestion Caveats & Filtering Rules

### 2.1 Downed Container Telemetry (`cpu_percent == 0.0`)
* When a service is unreachable or its container crashes, the background telemetry scraper cannot sample process counters. In accordance with safety defaults, missing telemetry fields default to `0.0`.
* Consequently, **100% of rows with `service_unreachable == 1` report `cpu_percent == 0.0`**.
* **Filtering Rule for CPU Analysis**: When analyzing active CPU distributions, velocity, or workload modeling, filter for active running containers:
  ```python
  df_active = df[df['service_unreachable'] == 0]
  ```

### 2.2 Queue Staleness & QPI Clamping (`queue_stale == 0`)
* The Queue Pressure Index (QPI) is formulated as:
  $$\text{QPI} = \min\left(1.0, \frac{Q}{Q_{\text{ref}}} \times \frac{\lambda}{\max(\mu, \epsilon)}\right)$$
* For uninstrumented microservices or static test stubs, `in_flight_queue` may report a static constant placeholder (e.g., exactly `1.0`). Because baseline $Q_{\text{ref}} = \max(1.0, Q_{\text{baseline\_p95}})$, an arrival rate $\lambda \ge \mu$ causes QPI to clamp at `1.0`.
* The pipeline automatically detects frozen queue metrics via `update_queue_staleness()` and flags them with `queue_stale = 1`.
* **Filtering Rule for QPI Analysis**: To evaluate legitimate queue dynamics and prevent distortion from mock placeholders, filter for active non-stale queues:
  ```python
  df_qpi = df[df['queue_stale'] == 0]
  ```

---

## 3. Exhaustive 10-Section Schema Field Reference

| Section | Column Name | Type | Description |
| :--- | :--- | :--- | :--- |
| **1. Identifiers** | `timestamp` | Float | Epoch timestamp in seconds (float precision) |
| | `time` | String | Local wall-clock time (`HH:MM:SS`) |
| | `incident_id` | UUID | Unique incident identifier for correlation |
| | `service` | String | Microservice name (e.g., `microservice-04-azure`, `auth-service`) |
| | `cloud_provider` | String | Host cloud provider (`aws`, `azure`, `gcp`, `local`) |
| | `region` | String | Deployment cloud region (e.g., `eastus`, `us-east-1`) |
| | `instance_id` | String | Physical/virtual container instance identifier |
| | `instance_count` | Integer | Active horizontal replica count |
| **2. Raw Telemetry** | `cpu_percent` | Float | Process CPU utilization percentage ($0.0 - 100.0$) |
| | `in_flight_queue` | Float | Current queued or pending request depth |
| | `incoming_rate` | Float | Arrival rate $\lambda$ (requests/second) |
| | `processing_rate` | Float | Service completion rate $\mu$ (requests/second) |
| | `latency_ms` | Float | Observed response time / ping latency in milliseconds |
| | `workload_level` | String | Inferred workload category (`LOW`, `NOMINAL`, `HIGH`, `BURST`) |
| **3. Normalized Features** | `cpu_percent_norm` | Float | Z-score normalized CPU relative to healthy baseline |
| | `in_flight_queue_norm` | Float | Queue normalized by baseline P99 ($Q / Q_{\text{p99}}$) |
| | `incoming_rate_norm` | Float | Arrival rate normalized by baseline $Q_{\text{ref}}$ |
| | `processing_rate_norm`| Float | Processing rate normalized by baseline $Q_{\text{ref}}$ |
| | `cpu_velocity_norm` | Float | CPU velocity normalized by baseline standard deviation |
| **4. Queue Dynamics** | `queue_growth_rate` | Float | First derivative of queue length ($dQ / dt$) |
| | `arrival_processing_diff`| Float | Net rate imbalance ($\lambda - \mu$) |
| | `processing_efficiency`| Float | Ratio of processing to arrival capacity ($\min(1.0, \mu / \lambda)$) |
| | `overload_flag` | Binary | $1$ if $\lambda > \mu$, else $0$ |
| **4b. Quality Flags** | `queue_stale` | Binary | **$1$ if queue length is frozen/placeholder; $0$ if dynamic** |
| | `incoming_counter_mode`| String | Counter classification (`cumulative`, `windowed`, `unknown`) |
| | `processing_counter_mode`| String | Counter classification (`cumulative`, `windowed`, `unknown`) |
| **5. Derivatives** | `cpu_velocity` | Float | Instantaneous CPU rate of change ($d\text{CPU}/dt$ in %/s) |
| | `cpu_acceleration` | Float | Second derivative of CPU ($d^2\text{CPU}/dt^2$ in %/s²) |
| | `cpu_trend_5min` | Float | 5-minute rolling average CPU utilization |
| | `cpu_trend_10min` | Float | 10-minute rolling average CPU utilization |
| | `cpu_5min_avg` | Float | Short-term smoothed CPU average |
| | `predicted_cpu_5min` | Float | Linear regression projected CPU in $+5$ minutes |
| | `predicted_cpu_10min` | Float | Linear regression projected CPU in $+10$ minutes |
| | `predicted_cpu_15min` | Float | Linear regression projected CPU in $+15$ minutes |
| **6. Pressure Indices**| `qref` | Float | Healthy calibrated baseline queue scale $Q_{\text{ref}}$ |
| | `qpi` | Float | **Queue Pressure Index** ($\min(1.0, (Q/Q_{\text{ref}}) \times (\lambda/\mu))$) |
| | `cpu_pressure_index` | Float | Normalized CPU pressure ($\text{CPU} / 100.0$) |
| | `overall_cpu_queue_pressure` | Float | Joint pressure metric ($\max(\text{QPI}, \text{CPU}/100.0)$) |
| **7. Baseline Stats** | `baseline_tier` | String | Calibration tier (`CALIBRATING`, `PROVISIONAL`, `STATISTICAL`) |
| | `cpu_mean` | Float | Baseline mean CPU utilization |
| | `cpu_std` | Float | Baseline CPU standard deviation |
| | `cpu_p95` | Float | Baseline 95th percentile CPU |
| | `cpu_p99` | Float | Baseline 99th percentile CPU |
| | `cpu_zscore` | Float | Standardized CPU score ($(\text{CPU} - \mu) / \sigma$; NaN if $\sigma \le 0.001$) |
| | `queue_mean` | Float | Baseline mean queue length |
| | `queue_std` | Float | Baseline queue standard deviation |
| | `queue_p95` | Float | Baseline 95th percentile queue length |
| | `queue_p99` | Float | Baseline 99th percentile queue length |
| | `queue_zscore` | Float | Standardized queue score ($Z_{\text{queue}}$) |
| | `qpi_mean` | Float | Baseline mean QPI |
| | `qpi_std` | Float | Baseline QPI standard deviation |
| | `qpi_p95` | Float | Baseline 95th percentile QPI |
| | `qpi_p99` | Float | Baseline 99th percentile QPI |
| | `qpi_zscore` | Float | Standardized QPI score ($Z_{\text{QPI}}$) |
| | `abnormal_duration_seconds` | Float | Consecutive duration in severe/abnormal state (seconds) |
| **8. Fault Metadata** | `fault_injected` | Binary | $1$ if active chaos fault is targeting this service, else $0$ |
| | `is_patient_zero` | Binary | $1$ if this service is the initial root-cause victim |
| | `failing_service` | String | Target service undergoing injected chaos |
| | `patient_zero` | String | Service name of the original failure initiator |
| | `incident_duration` | Float | Elapsed time since incident onset (seconds) |
| **9a. Platform Signals**| `service_unreachable` | Binary | **$1$ if `/metrics` poll failed (HTTP 5xx, timeout, socket closed)** |
| | `health_check_failed`| Binary | **$1$ if `/health` check failed or reported unreachable** |
| | `request_timeout` | Binary | **$1$ if synthetic client `/ping` timed out** |
| | `http_errors_per_sec`| Float | Non-200 HTTP responses observed per second |
| | `error_rate` | Float | Proportion of failed requests relative to total sent ($0.0 - 1.0$) |
| **9b. System Outcomes**| `actual_degradation` | Binary | Soft failure ($1$ if reachable but errors $> 0$ or error rate $> 0$) |
| | `actual_failure` | Binary | **Objective Ground Truth ($1$ if unreachable/timeout/health fail)** |
| | `predicted_state` | String | **Online Detector Classification (`HEALTHY`, `DEGRADED`, `FAILED`)** |
| | `system_state` | String | Global cluster state derived from failed service ratio |
| | `workload_state` | String | System-wide workload phase |
| | `incident_phase` | String | Incident timeline (`NORMAL`, `ONSET`, `PEAK`, `RECOVERY`) |
| **9c. Overload Target**| `overload` | Binary | Combined overload target ($1$ if `actual_failure == 1` or `cpu >= 90%`) |
| **10. ML Target** | `ml_label` | Float | **30-second forward-looking target**: $0=\text{NORMAL}$, $1=\text{PRE\_FAILURE}$, $2=\text{FAILURE}$ |
