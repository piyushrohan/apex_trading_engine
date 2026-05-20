# APEX Institutional AI Trading System

APEX is a production-oriented autonomous trading engine for **Binance USD-M Futures on ETHUSDC**. It combines live market-data ingestion, maker-only execution, strict risk controls, explainable model decisions, MLOps shadow lanes, hedge-strategy selection, and a read-only operator API/terminal.

The project is currently implemented through the roadmap's Milestone 9: paper/live operator modes, MLOps shadow lanes, live startup hardening, contextual hedge-bandit support, observability hooks, reporting, and a high-coverage local/CI validation suite.

## Current Status

Latest local validation snapshot:

```bash
make ci-local
# format + lint + typecheck warning pass-through + risk gate + parallel tests + coverage
# 227 passed
# total coverage: 96% in make coverage output
```

Strict local coverage gate:

```bash
venv/bin/pytest -m "not slow and not replay" \
  --cov=src \
  --cov-report=term-missing \
  --cov-fail-under=95
# Required test coverage of 95% reached. Total coverage: 95.91%
```

The risk-only safety gate intentionally selects only tests marked `risk`:

```bash
make test-risk
# collected 227 items / 216 deselected / 11 selected
# 11 passed
```

`216 deselected` is expected. Pytest collects the full suite, then runs only tests carrying `@pytest.mark.risk`.

## Core Principles

- **USDC only**: The system targets Binance USD-M `ETHUSDC` and rejects USDT assumptions in source/config/tests.
- **Maker-first execution**: Live and paper execution use post-only GTX-style order behavior wherever applicable.
- **Paper before live**: Live startup is gated behind paper-mode validation unless explicitly configured otherwise.
- **Shadow is MLOps, not operator mode**: Paper/live are the two operator modes. Shadow lanes are internal virtual books for candidate model evaluation.
- **Explain everything**: Decisions are journaled with regime, confidence, risk factors, hedge context, book role, and model id.
- **Risk gates first**: Kelly sizing, exposure caps, drawdown limits, hedge limits, slippage rules, and kill-switch behavior are tested separately through `make test-risk`.

## Implemented Capabilities

### Data Layer

- Async Binance REST client for historical OHLCV backfills, premium index, open interest, listen-key lifecycle, signed account/order helpers, and retry/error handling.
- Binance WebSocket ingestion for aggTrade, depth, and mark-price streams.
- DuckDB-backed cache for OHLCV, ticks, features, market snapshots, and paper equity snapshots.
- Incremental ingestion service with REST bootstrap, gap repair, funding/OI polling, tick buffering, and live stream append.
- Market-state service that builds latest inference snapshots from cached market data and persists state-vector features.

### Execution And Risk

- Shared execution adapter contract for paper and live execution.
- `PaperExecutionAdapter` for virtual maker orders, partial fills, fill-rate metrics, and virtual order flattening.
- `LiveExecutionAdapter` wrapping signed Binance REST order placement, fills, cancels, and emergency flattening.
- `RiskEngine` enforcing Kelly sizing, leverage caps, drawdown limits, hedge constraints, and kill-switch behavior.
- `AccountSynchronizer` for Binance user-data streams, USDC wallet balance sync, position leg parsing, order updates, and reconnect/keepalive paths.

### Operator Pipelines

- `TradingPipeline` is the shared loop for both operator modes.
- `paper` mode uses the primary virtual book and paper adapter.
- `live` mode uses signed live execution for the primary book after live startup validation.
- Live startup can set hedge mode and leverage before account sync.
- Paper/live paths publish runtime status, paper snapshots, explanations, hedge bandit decisions, shadow lane decisions, and metrics.

### MLOps And Model Registry

- GBM and PPO agents support train/save/load flows.
- `ModelRegistry` supports model registration, metric updates, state transitions, active prod/shadow pointers, rollback, and reproducibility manifests.
- `AutoRetrainPipeline` trains candidates from DuckDB OHLCV, evaluates out-of-sample performance, writes manifests, and promotes safe candidates to SHADOW.
- `ShadowLaneRunner` evaluates candidate models in virtual books alongside the primary operator loop without placing live orders.
- `PromotionService` compares shadow metrics against primary metrics and promotes, rejects, or rolls back models.

### Hedge Strategy System

- Multi-strategy hedge plugin registry with rule-based selection.
- Implemented hedge plugins include signal disagreement, regime straddle, protective hedge, ETH/BTC relative strength, sweep dual leg, maker grid hedge, and funding bias hedge.
- Contextual LinUCB bandit selector can take over after enough decision history exists.
- Hedge decision logs preserve both selected strategy and all candidate scores for attribution and bandit training.

### Explainability, API, Terminal, And Reports

- Explainability v2 adds confidence buckets, market narrative, position lifecycle context, risk factors, and portfolio sync explanations.
- Read-only FastAPI service exposes:
  - `GET /health`
  - `GET /status`
  - `GET /explain/latest`
  - `GET /portfolio`
  - `GET /positions`
  - `GET /metrics`
  - `GET /metrics/paper`
  - `WS /ws/status`
- Static frontend terminal in `frontend/` reads API status, explainability, portfolio, and metrics views.
- Paper report summarizes paper equity snapshots, Sharpe, max drawdown, directional decisions, fills, and fill rate.
- Hedge report aggregates selected strategies, score observations, average score, and hedge PnL.
- Prometheus-compatible metrics expose inference latency, PnL, paper fill rate, and ingestion health.

## Repository Structure

```text
apex_trading_engine/
├── .github/workflows/        # CI, paper validation, model evaluation, shadow checks
├── configs/                  # Base config and risk profiles
├── data_lake/                # Local DuckDB/Parquet/runtime artifacts (git-ignored)
├── docs/                     # Roadmap, progress, testing runbook
├── frontend/                 # Static operator terminal
├── ops/                      # Prometheus config
├── scripts/                  # Dev setup helpers
├── src/
│   ├── api/                  # FastAPI status/explainability endpoints
│   ├── core/                 # Config, logging, encrypted credentials
│   ├── data/                 # Binance REST/WS, DuckDB cache, ingestion, features
│   ├── execution/            # Adapters, portfolio, risk, live gate, account sync
│   ├── mlops/                # Auto-retrain, registry, promotion, shadow lanes
│   ├── models/               # PPO, GBM, meta-controller, regime detector
│   ├── observability/        # Prometheus/no-op metrics facade
│   ├── pipelines/            # Paper/live/shared trading loops and backtest
│   ├── reports/              # Paper and hedge reports
│   └── strategies/hedge/     # Hedge plugins and contextual bandit selector
└── tests/                    # Unit, risk, integration, MLOps, API, report tests
```

## Quick Start

### 1. Create The Environment

Recommended bootstrap:

```bash
bash scripts/setup_dev.sh
source venv/bin/activate
```

Manual bootstrap:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

### 2. Configure Credentials

Generate a Fernet key for encrypted local credential storage:

```bash
python - <<'PY'
from cryptography.fernet import Fernet
print(Fernet.generate_key().decode())
PY
```

Export it before running tools that use encrypted credentials:

```bash
export APEX_MASTER_KEY="<generated_fernet_key>"
```

For live signed exchange access, prefer environment variables during local testing:

```bash
export BINANCE_API_KEY="<your_binance_key>"
export BINANCE_API_SECRET="<your_binance_secret>"
```

The default config keeps live disabled:

```yaml
execution:
  operator_mode: paper

live:
  enabled: false
```

### 3. Run The Local Validation Suite

Full local gate:

```bash
make ci-local
```

Fast parallel non-slow suite:

```bash
make test
```

Safety-critical risk gate:

```bash
make test-risk
```

Strict total coverage gate:

```bash
venv/bin/pytest -m "not slow and not replay" \
  --cov=src \
  --cov-report=term-missing \
  --cov-fail-under=95
```

Coverage HTML report:

```bash
make coverage
open htmlcov/index.html
```

### 4. Run Paper Mode

Paper is the safe default operator mode. It uses live market data with a virtual primary book:

```bash
source venv/bin/activate
python -m src.pipelines.paper_trade
```

The module currently defaults to `configs/base.yaml`; direct programmatic callers can pass a config path through `main(config_path)`.

### 5. Run Live Mode

Live mode places real signed orders. Keep this behind paper validation and operator review:

```bash
export APEX_MASTER_KEY="<generated_fernet_key>"
export BINANCE_API_KEY="<your_binance_key>"
export BINANCE_API_SECRET="<your_binance_secret>"
```

Set the config deliberately:

```yaml
execution:
  operator_mode: live
  position_mode: hedge

live:
  enabled: true
  skip_paper_gate: false
  leverage: 3
```

Start live mode:

```bash
python -m src.pipelines.live_trade
```

### 6. Run The API And Terminal

API:

```bash
python -m src.api.server
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/status
curl http://127.0.0.1:8080/explain/latest
curl http://127.0.0.1:8080/portfolio
curl http://127.0.0.1:8080/metrics/paper
```

Static terminal:

```bash
python -m http.server 5173 --directory frontend
# open http://127.0.0.1:5173
```

Run a paper or live pipeline in another terminal so `/status`, `/portfolio`, and `/explain/latest` have runtime data.

### 7. Run Reports

Paper report:

```bash
python -m src.reports.paper_report --config configs/base.yaml
```

Hedge attribution report:

```bash
python -m src.reports.hedge_report --days 7 --config configs/base.yaml
```

### 8. Run MLOps Retraining

The retraining path reads cached OHLCV from DuckDB, writes an experiment run ledger, trains a candidate, runs OOS and stress gates, writes an immutable manifest, and promotes only safe candidates to the MLOps shadow lane. Bootstrap data first through ingestion/paper runs, then execute:

```bash
python -m src.mlops.auto_retrain
```

Candidate output is written under the configured model registry directory, usually:

```text
data_lake/models/
```

Experiment history is appended to:

```text
data_lake/mlops/experiments.jsonl
```

Live mode is blocked unless `active_prod` is a `PROD` registry model with a saved artifact, manifest, git hash, and data snapshot id. See [Model Governance And Retraining Discipline](docs/MODEL_GOVERNANCE.md) for the full train -> validate -> backtest -> stress -> shadow -> approve -> prod workflow.

Useful model-governance endpoints:

```bash
curl -s http://127.0.0.1:8080/models
curl -s http://127.0.0.1:8080/models/lifecycle
curl -s http://127.0.0.1:8080/models/promotion/status
```

### 9. Run Observability Stack

Prometheus/Grafana support is wired through Docker Compose:

```bash
docker compose -f docker-compose.observability.yml up -d
docker compose -f docker-compose.observability.yml logs -f
docker compose -f docker-compose.observability.yml down
```

Metrics are served by the app when `observability.metrics.enabled` is true:

```yaml
observability:
  metrics:
    enabled: true
    port: 9108
```

## Testing Documentation

Detailed testing commands, marker behavior, coverage policy, debugging recipes, and CI notes are in:

```text
docs/testing.md
```

Useful entry points:

```bash
make test-risk
make test
make coverage
make ci-local
venv/bin/pytest tests/pipelines/test_trading_pipeline_edges.py -q
venv/bin/pytest tests/data/test_ingestion_service.py -q
```

## Safety Notes

- Do not enable `live.enabled: true` until paper validation and risk checks pass.
- Do not bypass the paper gate except for controlled local development.
- Treat shadow lanes as virtual-only MLOps evaluation. They must never send exchange orders.
- Keep `data_lake/`, `.coverage`, `coverage.xml`, `htmlcov/`, `.pytest_cache/`, and runtime logs out of commits.
- If Black or Ruff modifies files during pre-commit, stage those formatter changes before committing again.

## Risk Disclaimer

This software is for research and automated execution experimentation. Cryptocurrency futures are volatile and can produce significant financial loss. Review configuration, exchange permissions, paper results, risk gates, and live startup settings before using real capital.
