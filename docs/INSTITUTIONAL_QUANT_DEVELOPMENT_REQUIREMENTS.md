# Institutional Quant Development Requirements

This document converts the latest quant-trader critique into concrete
development requirements for APEX. Treat this as the next source of truth after
the current production hardening backlog: the system is now structurally and
operationally serious, but the next frontier is sustainable alpha, realistic
market simulation, multi-asset intelligence, and research-grade operator tools.

## Executive Verdict

APEX already has the correct professional separation:

```text
Data -> Features -> Models -> Risk -> Hedge -> Execution -> Explainability -> Governance
```

That structure is the right shape for a serious trading system. The codebase is
already stronger than a typical retail crypto bot because strategy, execution,
risk, shadow evaluation, and governance are separated instead of mixed together.

The main weakness is no longer engineering discipline. The main weakness is
alpha depth: the current feature set and model stack are useful but still early.
The next roadmap must prioritize research quality before live capital ambition.

## Current-State Audit

| Area | Current Assessment | Evidence In Codebase | Requirement Status |
| --- | --- | --- | --- |
| Data | Separate ingestion, REST, websocket, DuckDB cache, freshness checks. | `src/data/`, `src/reports/data_freshness_check.py` | Strong foundation |
| Features | Dedicated feature engine and market-state builder, but feature set is compact. | `src/data/feature_engine.py`, `src/data/market_state.py` | Needs alpha expansion |
| Models | PPO, GBM, meta-controller, registry artifacts, transformer package stub. | `src/models/`, `src/models/meta_controller.py` | Intermediate |
| Risk | Independent risk layer with Kelly, drawdown, leverage, hedge caps, kill switches. | `src/execution/risk_engine.py`, `src/execution/kill_switch.py` | Strong but not portfolio-complete |
| Hedge | Modular hedge strategies and selector exist. | `src/strategies/hedge/` | Needs multi-asset hedge graph |
| Execution | Paper/live adapter separation exists; shadow uses paper adapter only. | `src/execution/adapters/`, `src/execution/factory.py`, `src/mlops/shadow_lane.py` | Strong foundation |
| Explainability | Dedicated explainability and position lifecycle logic exist. | `src/mlops/explainability.py`, `src/mlops/position_lifecycle.py` | Strong operator value |
| Governance | Registry, manifests, experiment ledger, promotion, rollback, gates. | `src/mlops/registry.py`, `src/mlops/auto_retrain.py`, `src/mlops/promotion_service.py` | Strong |
| Frontend | Operational cockpit, controls, readiness, lifecycle, live feed, replay foundation. | `frontend/`, `src/api/server.py` | Strong ops UI, weak research UI |

## Confirmed Strengths To Preserve

### Architecture Separation

Preserve these boundaries:

- Data collection must not contain strategy decisions.
- Feature generation must not place orders.
- Models must output probabilities, confidence, and context, not final exposure.
- Risk must be deterministic and allowed to veto models.
- Hedge logic must be explicit and separately auditable.
- Execution adapters must own exchange/paper mechanics only.
- Explainability must describe decisions without mutating execution state.
- Governance must decide model lifecycle, not live trading loops directly.

Acceptance criteria:

- No model class imports a live exchange adapter.
- No feature class imports order placement code.
- No execution adapter imports model code.
- Risk tests can run without loading models.
- Paper, live, and shadow remain separate by `execution.mode`, `book.role`, and
  `model_id`.

### MLOps Discipline

APEX already has the right institutional MLOps primitives:

- shadow lanes
- model registry
- promotion lifecycle
- rollback states
- immutable manifests
- experiment ledgers
- out-of-sample validation
- stress testing
- calibration gates
- label quality gates
- walk-forward validation
- git hash linkage

Required preservation rules:

- No model goes directly from training to live.
- Candidate models must pass offline gates before shadow.
- Shadow evidence must exist before production promotion.
- Production promotion must require a manifest-backed artifact.
- Registry events must remain append-only.
- Generated model artifacts must include data snapshot id, config hash, git hash,
  feature reference, and evaluation metrics.
- Runtime artifact safety checks must block crash-prone artifacts from loading.

### Risk System

APEX already enforces:

- Kelly sizing
- leverage caps
- drawdown kill switches
- hedge constraints
- gross and net leverage caps
- hedge ratio caps
- manual flattening
- operator kill-switch lanes
- paper-to-live gate
- model governance before live

Confirmed facts to preserve:

- Shadow lanes must never place exchange orders.
- Live startup must remain blocked without paper evidence unless an explicit
  emergency override is configured and audited.
- Live startup must remain blocked without a ready production model unless an
  explicit non-production test mode is used.
- Risk must remain stronger than the model layer. A model recommendation is only
  an input; risk approval is the final sizing authority.

### Frontend Thinking

The frontend is already more than a chart:

- explainability
- operator controls
- audit logs
- model lifecycle visibility
- workflow tabs
- replay/history foundation
- shadow metrics
- readiness posture
- websocket live feeds

The replay/history layer is especially valuable and should become a central
research tool, not only an operational view.

## Biggest Missing Pieces

### 1. Alpha Research Is Still Weak

Current features are useful starters:

- momentum
- ATR
- volume
- liquidity sweep
- ETH/BTC z-score
- CVD/net volume where available

Required upgrade:

- build a feature research framework
- build a factor library
- add market microstructure signals
- add order-flow toxicity signals
- add volatility forecasting
- add liquidation/funding pressure models
- add cross-market and cross-exchange context
- add feature stability and drift analytics
- add label research tools

This is priority 1. Better engineering will not compensate for weak alpha.

### 2. Single-Symbol Limitation

The system is currently ETHUSDC-centric. That is good for early discipline but
weak for institutional robustness.

Required target universe:

- `ETHUSDC` futures
- `BTCUSDC` futures
- `ETHBTC` futures or synthetic ETH/BTC cross where direct futures support is
  unavailable

Required portfolio behavior:

- The portfolio contains all three instruments.
- ETHUSDC, BTCUSDC, and ETHBTC can hedge each other.
- Risk is computed at portfolio level, not only per symbol.
- Capital allocation accounts for correlation, volatility, liquidity, funding,
  and regime.
- The frontend shows symbol-level and portfolio-level exposure.

### 3. Model Architecture Is Still Intermediate

Current model stack:

- PPO warm-start model
- GBM/LightGBM candidate
- regime switcher/meta-controller

Required upgrades:

- Transformer or Temporal Fusion Transformer sequence model
- DeepLOB-style order-book model after L2 replay exists
- multi-horizon probabilistic forecasting
- uncertainty-aware inference
- ensemble forecast combiner
- regime-specific specialists
- calibration by regime and horizon

No model should be promoted because it looks good in one backtest. Promotion
requires OOS, walk-forward, stress, calibration, runtime safety, and shadow
evidence.

### 4. Missing True Market Simulator

This is critical for maker-only systems. Current paper simulation is useful but
not enough to trust maker profitability.

Required simulator:

- L2 order book replay
- queue position simulation
- matching-engine approximation
- maker/taker classification
- latency simulation
- cancel/replace timing
- partial fill realism
- adverse selection modeling
- spread capture and post-fill drift measurement

Acceptance gate:

- Any maker strategy must pass the true replay simulator before it can influence
  live promotion decisions.

### 5. No Portfolio-Level Intelligence Yet

Required portfolio layer:

- portfolio optimizer
- dynamic capital allocation
- regime-based exposure rotation
- volatility targeting
- risk parity baseline
- correlation-aware hedge sizing
- cross-strategy allocation
- per-symbol and portfolio-level drawdown limits

### 6. Frontend Needs Quant Research Terminal Features

The cockpit is operationally strong. It now needs research-grade tools.

Required modules:

- feature inspector
- model diagnostics
- research lab
- real-time risk heatmaps
- multi-layer replay engine

### 7. Missing Self-Healing Operations

Current automation is strong:

- freshness checks
- governance reports
- ledger auditing
- shadow sanity monitoring

Required self-healing:

- automatic process restart
- degraded-mode operation
- ingestion failover
- websocket recovery with backoff
- exchange/account desync recovery
- dynamic throttling
- stale data kill-switch lane activation
- automatic recovery audit events

### 8. Future Distributed Infrastructure

Do not prematurely distribute the system, but plan for it.

Required future path:

- ClickHouse or TimescaleDB for high-frequency market history
- Kafka or Redis Streams for market/event buses
- object storage for raw replay data and artifacts
- feature store for reproducible training data
- Kubernetes or process supervisor when local cockpit becomes insufficient

Trigger point:

- Move beyond single-machine DuckDB when multi-symbol L2 replay, cross-exchange
  feeds, or large-scale ML experiments make local storage a bottleneck.

## Development Roadmap

### Phase 0 - Preserve Existing Safety Before Expanding

Goal: Make sure new research features cannot weaken risk or governance.

Tasks:

- Add architecture-boundary tests for forbidden imports.
- Add config validation for allowed symbols and operator modes.
- Keep shadow lanes virtual-only.
- Keep live startup blocked by paper gate and production model readiness.
- Keep generated model artifacts out of commits unless explicitly reviewed.
- Keep runtime artifact preflight for LightGBM/Torch safety.

Acceptance criteria:

- `make ci-local` passes.
- No architecture-boundary violations.
- `/ops/readiness` reports production blockers accurately.
- Paper can run while API/frontend are up.
- Training can read DuckDB while API/frontend are up.

### Phase 1 - Alpha Research Framework

Goal: Make alpha research repeatable, measurable, and governed.

Tasks:

- Create `src/research/` package.
- Add `FactorSpec` interface with:
  - name
  - version
  - required inputs
  - lookback
  - output columns
  - leakage policy
  - stability metrics
- Add factor registry.
- Add factor materialization into DuckDB `features`.
- Add factor evaluation report:
  - IC
  - rank IC
  - turnover
  - feature drift
  - feature decay
  - regime-sliced performance
  - correlation to existing factors
- Add label research report:
  - horizon sensitivity
  - threshold sensitivity
  - near-threshold ratio
  - class balance
  - forward return distribution
- Add CLI:

```bash
venv/bin/python -m src.research.factor_report --symbol ETHUSDC --timeframe 3m
venv/bin/python -m src.research.label_report --symbol ETHUSDC --timeframe 3m
```

Initial factors to implement:

- order-flow imbalance
- CVD slope and divergence
- spread z-score
- realized volatility forecast
- volatility-of-volatility
- funding pressure
- open-interest shock
- ETH/BTC relative strength
- BTC lead/lag
- liquidity sweep persistence
- absorption proxy
- adverse selection proxy

Acceptance criteria:

- Every factor has tests and a report row.
- No factor uses future data.
- Feature report identifies redundant and unstable factors.
- Auto-retrain stores factor set version in model manifest.

### Phase 2 - Multi-Asset Portfolio Intelligence

Goal: Move from ETH-only decisions to a three-asset hedged portfolio.

Target instruments:

- `ETHUSDC`
- `BTCUSDC`
- `ETHBTC` or synthetic ETH/BTC cross

Tasks:

- Change config from single `target_symbol` to `portfolio.symbols`.
- Keep backwards compatibility for `data.target_symbol`.
- Extend ingestion to collect all portfolio symbols.
- Extend `MarketStateService` to produce:
  - per-symbol state vectors
  - cross-symbol feature matrix
  - portfolio regime
- Extend `PortfolioService` to track positions by symbol and side.
- Add `PortfolioRiskEngine`:
  - portfolio gross exposure
  - portfolio net exposure
  - beta-adjusted exposure
  - correlation matrix
  - volatility targeting
  - per-symbol drawdown
  - portfolio drawdown
- Add hedge graph:
  - ETHUSDC hedged with BTCUSDC
  - ETHUSDC hedged with ETHBTC
  - BTCUSDC hedged with ETHBTC
- Add optimizer:
  - equal risk contribution baseline
  - volatility target
  - max concentration
  - correlation cap
  - funding-aware adjustment

Acceptance criteria:

- One paper run can track all three assets.
- Risk can reject a trade because portfolio exposure is too high even if the
  single-symbol trade looks safe.
- Frontend shows portfolio exposure and symbol-level exposure.
- Reports separate symbol PnL, hedge PnL, and portfolio PnL.

### Phase 3 - True Replay Market Simulator

Goal: Stop trusting optimistic maker fills.

Tasks:

- Add raw L2 order book capture:
  - snapshots
  - incremental depth updates
  - sequence validation
  - gap repair
- Store replay data separately from candle data.
- Add `src/simulation/` package.
- Implement replay clock.
- Implement order book reconstruction.
- Implement queue model:
  - estimated queue ahead
  - queue depletion
  - cancel uncertainty
  - partial fill probability
- Implement latency model:
  - data latency
  - decision latency
  - order submission latency
  - cancel latency
- Implement adverse selection metrics:
  - post-fill drift
  - spread capture
  - markout at 1s, 5s, 30s, 3m
- Add simulator report:

```bash
venv/bin/python -m src.simulation.replay_report --session latest
```

Acceptance criteria:

- Paper fill assumptions can be compared against replay fill assumptions.
- Maker strategies that only work under optimistic fills are rejected.
- Promotion gates include replay simulator metrics before live approval.

### Phase 4 - Advanced Model Stack

Goal: Move from moderate model sophistication to research-grade probabilistic
forecasting.

Tasks:

- Add sequence dataset builder.
- Add model interfaces for:
  - point forecast
  - class probabilities
  - uncertainty intervals
  - horizon-specific outputs
- Add Transformer/TFT baseline.
- Add DeepLOB model after L2 data exists.
- Add ensemble combiner:
  - GBM tabular
  - PPO policy
  - Transformer sequence
  - regime specialist
- Add uncertainty gate:
  - abstain when entropy is high
  - reduce size when uncertainty is high
  - require calibration by regime
- Add confusion matrix by regime.
- Add calibration curves by regime and horizon.

Acceptance criteria:

- Model diagnostics show whether improved models beat GBM after costs and
  stress.
- No advanced model is promoted without outperforming simpler baselines.
- The system can explain "no trade" due to uncertainty.

### Phase 5 - Quant Research Cockpit

Goal: Turn the frontend into a research terminal: Bloomberg + Grafana + MLflow +
TradingView style, but focused on APEX.

Feature Inspector:

- live feature values
- feature drift charts
- feature correlations
- feature stability
- SHAP or native feature importance
- feature contribution by decision

Model Diagnostics:

- calibration curves
- prediction distribution
- confidence entropy
- false positive analysis
- confusion matrix by regime
- model comparison table
- champion vs challenger charts

Research Lab:

- factor explorer
- strategy builder
- parameter sweeps
- walk-forward visualizer
- Monte Carlo stress testing
- label sensitivity UI

Risk Heatmaps:

- liquidation-distance map
- leverage heatmap
- exposure map
- drawdown waterfall
- regime heatmap
- correlation heatmap

Replay Engine:

- candles
- L2 order book
- features
- model confidence
- hedge decisions
- execution events
- latency
- fills
- risk changes
- synchronized timeline playback

Acceptance criteria:

- A trader can replay a session and answer:
  - what did the market do?
  - what features changed?
  - what did each model believe?
  - why did risk approve or reject?
  - what hedge was selected?
  - what happened to orders and fills?
  - did latency or queue position hurt the trade?

### Phase 6 - Self-Healing Operations

Goal: Make the system recover from common runtime failures without hiding them.

Tasks:

- Add process supervisor state machine.
- Add restart policies:
  - API
  - frontend
  - paper
  - training
  - ingestion
- Add degraded modes:
  - API read-only
  - market-data-only
  - no-new-orders
  - risk-only
- Add websocket recovery:
  - heartbeat monitor
  - exponential backoff
  - sequence gap detection
  - automatic resubscribe
- Add exchange desync recovery:
  - account snapshot reconciliation
  - open order reconciliation
  - forced cancel-and-resync
- Add throttling:
  - API rate limit awareness
  - order rate guard
  - reconnect guard
- Add audit events for every automatic recovery action.

Acceptance criteria:

- A killed paper process can be restarted automatically when enabled.
- A stale data feed activates data kill-switch lane.
- Recovery actions are visible in frontend logs and audit history.
- The system never silently resumes trading after a severe desync without a
  recorded recovery reason.

### Phase 7 - Distributed Infrastructure Path

Goal: Prepare for scale without overcomplicating the current local setup.

Tasks:

- Define storage abstraction:
  - DuckDB local
  - ClickHouse/Timescale production
  - object storage for raw replay
- Define event bus abstraction:
  - in-process/local
  - Redis Streams
  - Kafka
- Define feature store abstraction.
- Add migration document for:
  - market data
  - features
  - order lifecycle
  - model artifacts
  - audit logs
- Add deployment topology:
  - local research
  - single VPS paper
  - production cluster

Acceptance criteria:

- Existing code still works locally.
- Storage/event abstractions do not force Kubernetes for development.
- Production path is clear when data volume requires it.

## Priority Order

| Priority | Workstream | Why |
| --- | --- | --- |
| P0 | Alpha research framework | Sustainable alpha is the core missing edge. |
| P0 | True replay simulator | Maker strategies are unsafe without realistic fills. |
| P0 | Multi-asset portfolio/risk foundation | ETH-only logic limits robustness and hedging. |
| P1 | Advanced models | Better models only matter after better data/features/simulator. |
| P1 | Quant research cockpit | Makes research and diagnostics usable by a trader. |
| P1 | Self-healing operations | Needed before long unattended runs. |
| P2 | Distributed infrastructure | Needed when local DuckDB and single-machine flow become bottlenecks. |

## Required Acceptance Gates Before Any Live Capital

Live trading must remain blocked until all of these pass:

- Manifest-backed `PROD` model exists.
- Active production artifact passes runtime safety preflight.
- Paper gate passes required duration, decisions, Sharpe, drawdown, and fill
  quality.
- Shadow evidence exists for the candidate model.
- Replay simulator validates maker fill quality.
- Portfolio risk accepts all configured assets.
- Exchange/account synchronization is healthy.
- Kill-switch lanes are clear.
- Frontend readiness has zero critical findings.
- Operator can explain the latest decision from feature to model to risk to
  execution.

## Suggested Implementation Sequence

1. Add architecture-boundary tests.
2. Add `src/research/` factor framework.
3. Add factor and label reports.
4. Expand features with order-flow, funding, OI, spread, volatility, and
   cross-asset signals.
5. Add multi-symbol config and ingestion.
6. Add portfolio-level state, risk, and reports.
7. Add L2 data capture.
8. Add replay simulator and realistic maker fill scoring.
9. Add model diagnostics and uncertainty gates.
10. Add Transformer/TFT baseline only after sequence datasets are stable.
11. Add research cockpit tabs for feature, model, risk, and replay inspection.
12. Add self-healing supervisor and recovery audit events.
13. Plan distributed storage/event migration only after local scale breaks.

## Development Commands

Run local quality gates:

```bash
venv/bin/ruff check .
venv/bin/pytest tests/ -q
make ci-local
```

Run the runtime bug-hunt demo:

```bash
venv/bin/python scripts/runtime_bug_hunt_demo.py --live-market --start-paper --paper-seconds 30
```

Start the cockpit:

```bash
venv/bin/python -m src.ops.cockpit
```

Start cockpit with paper trading:

```bash
venv/bin/python -m src.ops.cockpit --paper
```

Train a candidate:

```bash
venv/bin/python -m src.mlops.auto_retrain
```

Check model lifecycle:

```bash
curl -s http://127.0.0.1:8080/models/lifecycle?limit=5
```

Check readiness:

```bash
curl -s http://127.0.0.1:8080/ops/readiness
```

## Non-Negotiable Engineering Rules

- Do not weaken risk to make a model look better.
- Do not promote models without manifests.
- Do not bypass paper/shadow gates for normal live operation.
- Do not allow shadow lanes to place exchange orders.
- Do not trust maker backtests without replay validation.
- Do not add advanced models before the data, labels, and simulator can judge
  them.
- Do not hide runtime failures; surface them in readiness, audit logs, and the
  frontend.
- Do not treat frontend polish as a substitute for alpha research.

## Final Target State

APEX should become a governed multi-asset crypto quant platform with:

- research-grade factor discovery,
- realistic replay simulation,
- portfolio-level risk and hedging,
- uncertainty-aware model selection,
- transparent operator controls,
- synchronized replay/history,
- self-healing runtime operations,
- and strict model governance from training to shadow to production.

The next real milestone is not "make it trade more." The next milestone is:

```text
prove that the system has durable, explainable, risk-adjusted alpha after costs,
latency, fills, stress, and portfolio constraints.
```
