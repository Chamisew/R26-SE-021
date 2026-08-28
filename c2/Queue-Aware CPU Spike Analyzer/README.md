# Queue-Aware CPU Spike Analyzer

A comprehensive research framework and real-time visualization dashboard for analyzing CPU spikes with queue-awareness (Queueing Theory, $\lambda - \mu$ dynamics, and Queue Pressure Index - QPI). This project combines an asynchronous Python-based research backend for cloud telemetry collection with a modern React frontend for real-time failure visualization.

---

## 📁 Repository Structure

* **`research_framework/`**: Python asynchronous engine for sidecar polling, dynamic Azure service discovery, queue telemetry calculation, and ML dataset generation (`final_research_dataset.csv`).
* **`frontend/`**: React + Vite real-time monitoring dashboard with glassmorphism UI, interactive telemetry charts, and live incident tracking.
* **`micro services/`**: Stack-agnostic microservices (`go`, `node`, `python`, `ruby`, `php`) with sidecar monitoring proxies for Azure Container Apps deployment.

---

## 🌿 Git & Branching Strategy

This project strictly adheres to **GitHub Flow** to guarantee that the `main` branch remains stable, tested, and always in a deployable state.

### 1. Main Branch Policy
* The `main` branch represents production-ready, stable code.
* **Direct pushes to `main` are strictly prohibited.** All code changes must enter `main` exclusively through vetted Pull Requests (PRs).

### 2. Branch Naming Conventions
All development work takes place on dedicated topic branches cut from `main`. Use structured prefixes according to change category:

| Prefix | Usage | Example Branch Name |
| :--- | :--- | :--- |
| `feature/` | New functionality or research features | `feature/queue-pressure-index` |
| `bugfix/` | Resolving bugs or telemetry issues | `bugfix/service-discovery-names` |
| `docs/` | Documentation, README, or schema updates | `docs/git-branching-strategy` |
| `refactor/` | Code structure improvements without functional change | `refactor/metrics-collector-loop` |
| `research/` | Machine learning model development & data scripts | `research/xgboost-failure-predictor` |

---

## 📝 Commit Granularity & Guidelines

To maintain clean, reviewable history, commits must be **atomic** (one logical change per commit) and follow the **Conventional Commits** standard:

### Commit Format
```text
<type>(<scope>): <short descriptive summary in imperative mood>

[optional body explaining motivation and technical context]
```

### Commit Types
* `feat`: A new feature added to backend or frontend.
* `fix`: A bug fix or patch.
* `docs`: Documentation changes only.
* `refactor`: Code refactoring without changing functionality.
* `test`: Adding or modifying test suites.
* `chore`: Build configuration, dependencies, or tool settings.

### Examples
* `feat(telemetry): implement Queue Pressure Index (QPI) math`
* `fix(discovery): populate running_service_names in discovery task`
* `docs(readme): add dated version log and PR guidelines`

---

## 🔀 Pull Request (PR) & Merge Procedures

1. **Create Topic Branch**: Cut a branch from updated `main` (`git checkout -b feature/my-feature`).
2. **Commit Granularly**: Make small, self-contained commits with clear commit messages.
3. **Open Pull Request**: Open a PR targeting `main`. Direct pushes to `main` are blocked.
4. **Required PR Description Template**:
   All PRs must include a structured description covering:
   * **Summary**: What this change accomplishes.
   * **Motivation / Problem**: Why this change is necessary.
   * **Key Changes**: Bulleted technical details.
   * **Verification**: Evidence of testing (e.g. log output, screenshots, test execution).
5. **Review & Merge**: Review the PR, squash & merge or merge commit into `main`, and delete the feature branch.

---

## 📜 Dated History & Merge Record Log

This section maintains an audit log of merged branches, dated feature changes, and critical fixes:

| Date of Merge | Feature / Change | Merged Branch | Brief Summary |
| :--- | :--- | :--- | :--- |
| **2026-08-13** | Service Discovery Fix | `bugfix/service-discovery-names` | Updated `discovery_task` to populate active service lists, enabling automatic 3-minute failure triggers. |
| **2026-08-13** | Local Sidecar Runner | `feature/local-microservices-runner` | Added `run_local_services.py` to enable offline 5-microservice research data generation on ports 5001–5005. |
| **2026-07-25** | 3-Minute Failure Topology | `feature/3min-spike-topology` | Configured `spike_trigger` to hold anomalies for 180s followed by 120s automatic recovery cycles. |
| **2026-07-25** | Queue-Aware Telemetry Vector | `feature/qpi-telemetry-vector` | Refactored failure detection to use Queueing Theory metrics ($\lambda - \mu$, QPI, CPU Velocity). |
| **2026-07-13** | Azure ACA Migration | `feature/azure-container-apps` | Migrated Go, Node, Python, Ruby, and PHP microservices with sidecars to Azure Container Apps. |

---

## 🚀 Quick Start Guide

### 1. Backend Setup (`research_framework/`)
```powershell
# Navigate to framework
cd "research_framework"

# Create & activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Option A: Run with local microservices (offline)
python run_local_services.py   # Terminal 1
python realtime_experiment.py  # Terminal 2

# Option B: Run with Azure Cloud (online)
# Edit .env with your Azure Container App URLs
python realtime_experiment.py
```

### 2. Frontend Setup (`frontend/`)
```powershell
cd "frontend"
npm install
npm run dev
```
Open `http://localhost:5173` to view the live research dashboard.
