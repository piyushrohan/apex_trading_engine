# APEX Implementation Progress

Tracks completed implementation milestones and current work. For the active
forward-looking requirements, use
[Institutional Quant Development Requirements](./INSTITUTIONAL_QUANT_DEVELOPMENT_REQUIREMENTS.md)
and [Trader Production Backlog](./TRADER_PRODUCTION_BACKLOG.md).

**Legend:** `DONE` | `IN_PROGRESS` | `NOT_STARTED`

---

## Current focus

| Item | Status |
|------|--------|
| **Milestone 5** — Explainability v2 + API | `DONE` |
| **Milestone 6** — ML & MLOps | `DONE` |
| **Milestone 7** — Observability & CI | `DONE` |
| **Milestone 8** — Terminal | `DONE` |
| **Milestone 9** — Contextual bandit selector | `DONE` |
| **Phase 0** — Foundations | `DONE` |
| **Trader production hardening** — order telemetry, drift, controls, cockpit replay | `DONE` |
| **Operator simplification** — one-command cockpit, process runbook, live market chart | `DONE` |

---

## Backlog — Model Quality / Alpha Discipline

| Item | Status | Notes |
|------|--------|-------|
| Remove future leakage from supervised features | DONE | `future_returns` is now used only for labels; training features are causal/current-bar only |
| Real LightGBM local training stability | DONE | Added `scikit-learn`, macOS `libomp` setup guidance, and safer LightGBM defaults |
| Fee-adjusted horizon labels | DONE | Auto-retrain labels now use `label_horizon_bars` plus fee/slippage buffer before assigning SHORT/FLAT/LONG |
| Label stability diagnostics | DONE | Candidate metrics include directional ratio, dominant label ratio, entropy, and near-threshold sensitivity |
| Probability calibration diagnostics | DONE | Candidate metrics include OOS accuracy, Brier score, expected calibration error, confidence, and trade-signal coverage |
| Model quality promotion gate | DONE | Shadow promotion can now be blocked by short history, unstable labels, class imbalance, or weak classifier calibration |
| Forward paper/shadow evidence before PROD | IN_PROGRESS | New candidates should remain SHADOW until paper/live-forward journal evidence confirms offline metrics |
| Richer alpha feature research | IN_PROGRESS | Quality milestone hardens labels/calibration; next slice should add order-flow, spread, funding, cross-asset, and regime-aware features |
| Stricter promotion gates after leakage fix | DONE | Base config now requires longer history and quality evidence before shadow promotion |

---

## Milestone 5 — Explainability v2 + API (P0)

| Task | Status | Notes |
|------|--------|-------|
| Confidence buckets | DONE | trend / momentum / liquidity / regime + tier |
| Market narrative | DONE | regime, sweeps, vol z, funding |
| Position lifecycle | DONE | `position_lifecycle.py` — why flat/open, invalidation |
| Portfolio sync explanations | DONE | `decode_portfolio_state()` on account sync |
| Schema v2 journal | DONE | `schema_version: 2` on all payloads |
| FastAPI server | DONE | `src/api/server.py` |
| `/health` | DONE | Liveness |
| `/status` | DONE | Mode, regime, kill switch, portfolio summary |
| `/explain/latest` | DONE | Runtime store or journal fallback |
| `/portfolio` | DONE | Runtime + paper equity snapshots |
| Status store | DONE | `src/api/status_store.py`, pipeline publishes each tick |
| Tests | DONE | explainability v2, lifecycle, API |

### Milestone 5 exit criteria

- [x] Confidence decomposition (trend/momentum/liquidity/regime)
- [x] Why flat / why open / invalidation in payload
- [x] FastAPI read-only: `/health`, `/status`, `/explain/latest`, `/portfolio`
- [x] Tests pass (**129** unit/integration + 8 risk)

### Run API

```bash
pip install -r requirements.txt
python -m src.api.server
# GET http://localhost:8080/health
```

Run the trading pipeline in another terminal so `/status` and `/explain/latest` populate from live ticks.

---

## Milestone 4 — Live trading minimal (P0)

**Status:** `DONE`

---

## Milestone 3 — Paper trading E2E (P0)

**Status:** `DONE`

---

## Milestone 1–2

**Status:** `DONE`

---

## Milestone 6 — ML & MLOps (P1)

| Task | Status | Notes |
|------|--------|-------|
| GBM train/save/load | DONE | `GBMAgent.train()`, `save()`, `load()` with LightGBM when available and deterministic fallback when native LightGBM libs are unavailable |
| PPO train/save/load | DONE | `PPOAgent.train()` supervised warm-start plus checkpoint `save()` / `load()` |
| Registry metric/status updates | DONE | `update_model_metrics()`, `set_model_status()`, `rollback_prod()` |
| Auto-retrain real data path | DONE | Reads DuckDB OHLCV, builds causal supervised features plus next-bar labels, trains candidate, saves artifact, runs OOS backtest, evaluates safety, promotes to SHADOW or rejects |
| Model quality upgrade | DONE | `AutoRetrainPipeline` records fee-adjusted label quality, class-balanced GBM weights, OOS probability calibration, and quality-gate blockers |
| Shadow lane runner | DONE | `ShadowLaneRunner` runs candidate models through shared `PaperExecutionAdapter` with `book.role=shadow` and logs decisions |
| TradingPipeline shadow integration | DONE | Pipeline invokes shadow lanes when `shadow.enabled` is true under paper or live operator mode |
| Promotion service | DONE | `src/mlops/promotion_service.py` compares shadow vs primary metrics and promotes, rejects, or rolls back independently of paper→live gate |
| Hedge T-C plugins | DONE | `eth_btc_rs_hedge`, `sweep_dual_leg`, `funding_bias_hedge` implemented and registered |
| All-seven hedge score logging | DONE | Registry covers T1-T7; primary and shadow journal payloads include `hedge.candidates` for bandit training |
| Active PROD loading | DONE | `TradingPipeline` now loads active registry artifacts into the primary `MetaController` before inference |
| Reproducibility manifests | DONE | Model registry writes `manifest.json` with git hash, data snapshot id, hyperparams, and metrics |
| Shadow metrics | DONE | Shadow decisions log timestamp, equity, PnL, and bandit context for promotion windows |
| Model governance discipline | DONE | Registry lifecycle states, experiment run ledger, immutable manifests, configurable gates, and live startup model readiness guard |
| Active feature drift evidence | DONE | Retrain stores training feature references and `/models/drift` compares current DuckDB features with active model evidence |

### Milestone 6 exit criteria

- [x] GBM/PPO candidates can train, save, and load registry artifacts
- [x] Auto-retrain no longer uses mocked PnL/trades for candidate evaluation
- [x] Shadow lane uses shared paper simulator and shadow book tags
- [x] Promotion service separates shadow→prod from operator paper→live
- [x] Shadow lanes run continuously in the shared paper/live `TradingPipeline` loop when `shadow.enabled` is true
- [x] Hedge T-C plugins and all-seven strategy score logging are complete
- [x] Train → shadow → promote → rollback paths are covered by automated MLOps tests
- [x] Live mode refuses unregistered or incomplete production model evidence
- [x] Candidate promotion captures label stability and probability calibration evidence

---

## Milestone 7 — Observability & CI (P1)

| Task | Status | Notes |
|------|--------|-------|
| Prometheus metrics | DONE | `src/observability/metrics.py` exposes paper/live PnL, inference latency, and WS/ingestion health with a no-op fallback |
| Pipeline metrics wiring | DONE | `TradingPipeline` records inference latency, mode/book PnL, and ingestion health |
| Observability compose | DONE | `docker-compose.observability.yml` plus `ops/prometheus.yml` for Prometheus + Grafana |
| Paper validation workflow | DONE | `.github/workflows/paper-trading-validation.yml` runs bounded paper tests and uploads journal/report artifacts |
| Hedge T-D maker grid | DONE | `maker_grid_hedge.py` plus `src/execution/grid_adapter.py` |
| Hedge attribution report | DONE | `src/reports/hedge_report.py` aggregates selected strategy counts, scores, and hedge PnL |
| Live exchange prep | DONE | Live startup can set hedge mode and symbol leverage before account sync |
| Paper fill realism | DONE | Paper adapter supports partial fills; paper fills are journaled and reported as fill-rate metrics |
| CI model/shadow workflows | DONE | Model evaluation and shadow deployment workflows now run real tests/pipeline checks instead of placeholder `echo` steps |

### Milestone 7 exit criteria

- [x] Prometheus-compatible metrics module exists for PnL, inference latency, and WS health
- [x] Observability stack compose file exists
- [x] CI paper workflow runs real tests and archives reports/journals
- [x] Maker-grid hedge strategy and grid order planner are implemented
- [x] Hedge report produces per-strategy attribution over an N-day window

### Verification

```bash
venv/bin/ruff check src tests
venv/bin/pytest -q
# 149 passed
```

---

## Milestone 8 — Terminal (P2)

| Task | Status | Notes |
|------|--------|-------|
| Terminal frontend | DONE | `frontend/` static terminal shows mode, market/risk, explainability, paper performance, and hedge selector state |
| API support | DONE | FastAPI exposes `/status`, `/explain/latest`, `/portfolio`, `/positions`, `/metrics`, `/metrics/paper`, and `/ws/status` |
| Mode indicator | DONE | Terminal displays PAPER/LIVE banner from runtime status |
| Hedge panel | DONE | Terminal renders selected strategy, bandit arm, exploration state, and candidate scores |

---

## Milestone 9 — Contextual bandit selector (P2)

| Task | Status | Notes |
|------|--------|-------|
| LinUCB selector | DONE | `ContextualBanditSelector` selects hedge arms after `hedge.bandit.min_decisions` |
| Activation gate | DONE | Falls back to rule-based selector until every enabled arm has enough decision history |
| Rule-score shadow logging | DONE | Bandit mode preserves rule scores as `candidates_rule_shadow` for explainability |
| Reward update path | DONE | Selector can update persisted bandit state from decision rows containing `hedge_reward` |
| Rollback switch | DONE | Config can return `hedge.selection` to `rule_based` without code changes |

---

## Trader Production Readiness Backlog

| Task | Status | Notes |
|------|--------|-------|
| Quant critic backlog | DONE | `docs/TRADER_PRODUCTION_BACKLOG.md` converts the demo critique into P0/P1/P2 production work |
| Shadow artifact quarantine | DONE | Shadow lane preflights GBM artifacts in a child process and disables unsafe candidates instead of crashing startup |
| Trader readiness API | DONE | `/ops/readiness` summarizes live blockers, paper gate, fill evidence, model readiness, runtime freshness, and data diagnostics |
| Cockpit Ops tab | DONE | Frontend adds `Ops` with guardrail findings, data freshness, live-gate snapshot, and next actions |
| Exchange rule validation foundation | DONE | Binance REST client validates `exchangeInfo` constraints for USDC margin, `GTX`, price filter, lot size, and min notional |
| Full order lifecycle analytics | DONE | Paper/live adapters persist submit/open/fill/cancel/reject lifecycle events to DuckDB/JSONL; API summarizes fill rate, queue age, cancel/replace ratio, rejects, and post-fill drift |
| Feature drift and label stability reports | DONE | Retrain records label stability, classifier calibration, and training feature references; `/models/drift` and `/ops/readiness` expose active-model drift |
| Separate kill-switch lanes | DONE | Manual, model, data, execution, and account-sync lanes are normalized, persisted, surfaced in status/readiness, and controllable from the cockpit |
| Chart overlays and replay mode | DONE | Frontend history view adds decision replay chart, probability history, fill markers, and order-fill timeline from lifecycle telemetry |
| One-command cockpit and live tape | DONE | `make start`, `src.ops.cockpit`, `/ops/workflow`, `/ops/processes`, browser Runbook controls, and direct `/ws/market` live chart are implemented |
| Full frontend runtime control catalog | DONE | Runbook can start/stop/restart paper, live, shadow, training, governance, health, freshness, audit, contract, and report jobs from the allow-listed process manager |
| Coverage expansion | DONE | Strict full-suite coverage now passes `--cov-fail-under=98` at 98.12% |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-05-18 | Milestones 1–4 complete |
| 2026-05-18 | **Milestone 5 complete** — explainability v2, FastAPI read-only API |
| 2026-05-19 | **Milestone 6 started** — model artifact persistence, real-data auto-retrain path, shadow lane runner, and promotion service |
| 2026-05-19 | **Milestone 6 complete** — shadow lanes integrated, hedge T-C plugins registered, and all-seven hedge scores logged |
| 2026-05-19 | **Milestone 7 complete** — metrics, observability compose, paper CI workflow, maker grid, and hedge attribution report |
| 2026-05-19 | **Milestones 8–9 complete** — terminal API/frontend, active PROD loading, live prep hardening, partial paper fills, registry manifests, CI workflow wiring, and contextual bandit reward update path |
| 2026-05-20 | **Model governance hardening** — experiment tracking, strict registry lifecycle, stress gates, readiness API/frontend, and live model evidence guard |
| 2026-05-21 | **Trader production readiness** — added quant-critic backlog, shadow artifact quarantine, `/ops/readiness`, cockpit Ops tab, and exchange-rule validation foundation |
| 2026-05-21 | **Model quality upgrade** — added fee-adjusted horizon labels, label stability diagnostics, probability calibration diagnostics, and quality gates for shadow promotion |
| 2026-05-21 | **Trader hardening completion** — added order lifecycle telemetry, feature drift API/readiness checks, lane-specific kill switches, replay overlays, frontend contract updates, and 98.12% strict coverage |
| 2026-05-21 | **Operator cockpit simplification** — added `python -m src.ops.cockpit`, browser Runbook process controls, `/ops/workflow`, `/ops/processes`, direct `/ws/market` price streaming, and lower-latency defaults |
| 2026-05-21 | **Frontend control center expansion** — added `make start`, frontend-first cockpit startup, guarded live process control, report/evaluation job controls, restart actions, and process log snippets in the Runbook |
