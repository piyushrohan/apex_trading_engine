# APEX Testing & Validation Runbook

This runbook outlines the single-command workflows for local institutional-grade testing of the APEX trading engine. The goal is complete parity between local validation and CI/CD pipelines.

## Bootstrapping the Environment
When cloning the repository or onboarding a new developer, run:
```bash
bash scripts/setup_dev.sh
```
This initializes the virtual environment, installs all dependencies (including `pytest-xdist` for parallel testing), sets up the `.pre-commit` hooks, and validates the `Makefile`.

## Single-Command CI Parity
Our local engineering philosophy guarantees that **local green = CI green**. We enforce identical pytest markers, mypy configs, lint rules, and dependency resolution.

Before pushing any code, you must ensure it passes the full strict validation suite. Run:
```bash
make ci-local
```
This executes formatting (`black`, `isort`), linting (`ruff`, `flake8`), strict type-checking (`mypy`), the critical risk-engine safety gate, and the complete parallel unit test suite.

## Granular Testing Commands
To rapidly iterate, use these granular targets.

### 1. Risk Engine Safety Gate
The most critical part of the system. Validates Kelly sizing, leverage limits, and the kill-switch.
```bash
make test-risk
```

### 2. Core Unit Tests
Fast (millisecond) execution of business logic across ML, Execution, and Data layers. Runs in parallel via `pytest-xdist`.
```bash
make test-unit
```

### 3. Integration Tests
Tests requiring mocked streams or local infrastructure.
> Note: Start local infra using `docker-compose -f docker-compose.test.yml up -d` before running.
```bash
make test-integration
```

### 4. Deterministic Replay Tests
Heavy historical simulations validating edge detection and equity changes over mocked datasets.
```bash
make test-replay
```

## Debugging UX

If a test fails, use Pytest's advanced debugging commands:

- **Run only failed tests from the last run:**
  ```bash
  pytest --lf
  ```
- **Run all tests, but run the failures first:**
  ```bash
  pytest --ff
  ```
- **Run tests matching a specific keyword (e.g., only "slippage" tests):**
  ```bash
  pytest -k slippage
  ```
- **Run in highly verbose mode without capturing standard output (to see logger.debug prints):**
  ```bash
  pytest -vv -s
  ```

## Coverage Visualizations
To see exactly which lines of code are not covered by the test suite, run:
```bash
make coverage
```
This will print a summary to the terminal and generate an interactive `htmlcov/index.html` report. Open this file in your browser to explore the source code line-by-line.

## Local Observability & Telemetry (Future-Ready)
Future releases will support local Grafana dashboards mapped to Prometheus metrics emitted by the local test suites to visualize replay behavior.

## Advanced Replay & Simulation Debugging
The `BacktestEngine` supports deterministic historical replay. To step through a specific historical session (e.g., investigating a postmortem or model drift):
```bash
make replay-debug SESSION=2026-05-16
```
This forces Pytest to drop into a verbose logging state (`-s -vv --log-cli-level=DEBUG`) while replaying the mock or live Parquet datasets.
