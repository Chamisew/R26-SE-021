# CPU Spike Predictor (Component 4)

AIOps prediction and mitigation layer for microservices that ingests CPU/queue telemetry and memory-risk predictions to forecast impending failures, compute Mean Time To Alert (MTTA), and initiate automated mitigation workflows.

---

## 1. Commit Granularity Guidelines

To maintain a clean, bisectable, and reviewable Git history, adhere to the following commit practices:

* **Atomic Commits:** Each commit must represent a single logical change or cohesive unit of work (e.g., adding a specific feature, fixing one bug, or refactoring a single module).
* **Self-Contained & Working State:** Code must compile/run and pass existing unit tests at each commit. Avoid committing broken or half-implemented states.
* **Separation of Concerns:** Keep formatting/linting changes, documentation updates, and functional code modifications in distinct commits.
* **Conventional Commit Messages:** Follow the format: `<type>(<scope>): <subject>`
  * **Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`
  * **Examples:**
    * `feat(ingestion): add schema validation for queue pressure metrics`
    * `fix(model): handle zero-division in MTTA calculation`
    * `test(api): add endpoint test for /batch_predict`
    * `docs(readme): add branching strategy and merge changelog`

---

## 2. Branching Strategy & Naming Rules

This repository follows **GitHub Flow** (or lightweight **Git Flow**) tailored for continuous integration and stability.

```
       (feature/cpup-12-lead-time)
             o---o---o (PR)
            /         \
main  -----o-----------o-----------------o (Stable & Deployable)
                        \               /
                         o---o (PR)----/
                    (fix/handle-nan-mtta)
```

### Branch Naming Conventions
All branch names must be lowercase, hyphen-separated, and use standard prefixes with ticket/feature identifiers:

| Branch Type | Prefix / Pattern | Example |
| :--- | :--- | :--- |
| **Feature** | `feature/<ticket-or-name>` or `feat/<name>` | `feature/cpup-10-queue-telemetry` |
| **Bug Fix** | `bugfix/<ticket-or-name>` or `fix/<name>` | `fix/mtta-division-by-zero` |
| **Hotfix** | `hotfix/<ticket-or-name>` | `hotfix/live-inference-null-pointer` |
| **Documentation** | `docs/<name>` | `docs/update-architecture-guide` |
| **Refactoring** | `refactor/<name>` | `refactor/modularize-mitigation-engine` |
| **Testing** | `test/<name>` | `test/add-lopo-validation-tests` |

---

## 3. Main Branch Stability Policy

The `main` branch represents production-ready, deployable code at all times.

* **Direct Push Prohibited:** Pushing commits directly to `main` is strictly forbidden. Branch protection rules must enforce this.
* **Mandatory CI Checks:** All automated unit and integration tests (`pytest tests/`) must pass prior to merge.
* **No Broken Builds:** If `main` breaks, fixing it takes top priority over developing new features.

---

## 4. Pull Request (PR) & Code Review Guidelines

All changes enter `main` exclusively through Pull Requests.

### PR Requirements
1. **Title:** Clear, concise summary matching commit conventions (e.g., `feat(api): expose /health and /metrics endpoints`).
2. **Description Template:** Every PR must include:
   * **Summary / Objective:** What this PR does and why it is needed.
   * **Key Changes:** Bulleted list of modified modules and new behaviors.
   * **Testing Performed:** Commands run (e.g., `pytest tests/test_pipeline.py`) and verification results.
   * **Breaking Changes / Dependencies:** Any new libraries added to `requirements.txt` or schema changes.
3. **Peer Review:** At least one code review approval before merging.

### Merge Procedures
* **Squash and Merge:** Recommended for multi-commit feature branches to maintain a clean, linear history on `main`.
* **Rebase and Merge:** Permitted for branches with clean, atomic commit histories.
* **Branch Cleanup:** Delete the feature/fix branch immediately upon merging.

---

## 5. Dated Merge Record & Changelog

| Date (YYYY-MM-DD) | PR / Branch | Change Type | Summary of Changes |
| :--- | :--- | :--- | :--- |
| **2026-08-27** | `docs/add-git-workflow-docs` | `docs` | Added repository README with branching strategy, commit rules, PR procedures, and merge changelog. |
| **2026-08-15** | `feature/progressive-mitigation` | `feat` | Implemented 3-tier automated mitigation engine (cooldown, circuit breaker, rate limit) in `mitigation.py`. |
| **2026-08-01** | `feature/live-inference-engine` | `feat` | Added 5-second polling CSV live inference engine and API integration in `live_inference.py`. |
| **2026-07-20** | `feature/mtta-eval-framework` | `feat` | Implemented LOPO validation and Mean Time To Alert (MTTA) warning-time metrics across CPU and memory signals. |
| **2026-07-05** | `feature/cpu-rf-pipeline` | `feat` | Built balanced Random Forest training pipeline, lead-time labeling, and data ingestion for Component 2 & 3. |
| **2026-06-15** | `feature/initial-scaffolding` | `chore` | Initialized repository structure, base dependencies, and test harness for `cpu_spike_predictor`. |

---

## 6. How to Follow This Workflow

### Step-by-Step Developer Workflow

1. **Pull latest `main`:**
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Create a structured branch:**
   ```bash
   git checkout -b feature/cpup-15-feature-importance-viz
   ```

3. **Develop with atomic commits:**
   ```bash
   git add src/visualization.py
   git commit -m "feat(viz): add feature importance bar chart generator"
   ```

4. **Run test suite before pushing:**
   ```bash
   pytest tests/
   ```

5. **Push and open a Pull Request:**
   ```bash
   git push -u origin feature/cpup-15-feature-importance-viz
   ```
   Fill in the PR template with description, changes, and verification output.

6. **Review, approve, and merge to `main`.**
