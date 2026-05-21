# Trader Production Backlog

This backlog converts the crypto quant demo critique into concrete engineering
work. The goal is not to make APEX trade more often. The goal is to make it
trade only when data, model evidence, execution quality, and risk controls are
strong enough to justify live market exposure.

> Next requirements source: [Institutional Quant Development Requirements](./INSTITUTIONAL_QUANT_DEVELOPMENT_REQUIREMENTS.md)
> expands this backlog into the next institutional development program:
> alpha research, multi-asset portfolio intelligence, true replay simulation,
> advanced models, research cockpit tooling, self-healing operations, and the
> future distributed infrastructure path.

## Implemented In This PR

| Area | Status | Detail |
| --- | --- | --- |
| Shadow artifact resilience | DONE | Shadow lanes preflight GBM artifacts in a child process before loading them in the runtime. A crashing native artifact is quarantined instead of crashing paper/live startup. |
| Trader readiness API | DONE | `GET /ops/readiness` reports live blockers, paper gate status, PROD model readiness, fill evidence, runtime staleness, low-conviction warnings, journal freshness, and DuckDB diagnostic availability. |
| Trader cockpit Ops tab | DONE | Frontend adds an `Ops` tab with live readiness, critical/warning counts, guardrail findings, data freshness, live-gate snapshot, and next actions. |
| Exchange rule validation foundation | DONE | Binance REST client can fetch `exchangeInfo`, extract symbol filters, and validate USDC margin, `GTX` post-only support, min notional, price filter, and lot-size filter availability. |
| Frontend/API smoke coverage | DONE | Contract smoke now includes `/ops/readiness`; frontend static tests cover the new operator readiness tab. |
| Order lifecycle telemetry | DONE | Paper/live adapters record submit, open, fill, cancel, reject, queue age, fill price, and post-fill mark context to DuckDB/JSONL; `/orders/lifecycle` summarizes execution quality. |
| Feature drift visibility | DONE | Auto-retrain stores active training feature references; `/models/drift` and `/ops/readiness` compare current features against model evidence and warn on drift. |
| Lane-specific kill switches | DONE | Manual, model, data, execution, and account-sync kill-switch lanes are persisted independently and visible in status/readiness/cockpit controls. |
| Trader replay overlays | DONE | The browser history view now combines market path, decisions, fills, probability history, and order-fill timeline for post-session review. |
| One-command cockpit | DONE | `python -m src.ops.cockpit --paper` starts the API, frontend, and optional paper/training child processes with logs under `logs/`. |
| Browser runbook controls | DONE | `GET /ops/workflow`, `GET /ops/processes`, and `POST /ops/processes/{process_name}` let the frontend start/stop allow-listed paper and training processes after API runtime starts. |
| Live price tape | DONE | `WS /ws/market` streams Binance mark, trade, and depth events directly into a browser live chart with latency fields. |
| 98% coverage gate | DONE | Full strict local suite passes `venv/bin/pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=98` at 98.12%. |

## P0 Before Live

| Item | Why It Matters | Acceptance Gate |
| --- | --- | --- |
| Active PROD model | Live mode must never run on `unregistered` or shadow-only models. | `/models/lifecycle` shows manifest-backed `PROD`; `/ops/readiness` has no `prod_model_not_ready` critical finding. |
| Continuous shadow lane | Candidate models must run beside primary without placing real orders. | Shadow lane records fresh decisions for the active shadow model without artifact load failures. |
| Maker fill evidence | A maker-first strategy is only real if it can get filled without adverse selection. | Paper run shows minimum fills, non-zero fill rate, spread capture, and acceptable post-fill drift. |
| Paper-to-live gate | Live should require sustained forward evidence. | Configured `paper.min_days`, `paper.min_trades`, Sharpe, drawdown, and fill-quality gates pass. |
| Data health | Trading on stale or partial crypto feeds is worse than no trading. | OHLCV, market snapshots, features, and paper equity are fresh; tick gaps are explained. |

## P1 Execution Quality

| Item | Target Behavior |
| --- | --- |
| Idempotent client order ids | Every order, retry, cancel, and replace has a deterministic auditable client id. |
| Full order lifecycle journal | DONE: persist submit, open, fill, cancel, and reject events today; replace/timeout can build on the same schema when adapters emit them. |
| Maker quote analytics | DONE: track queue age, cancel/replace ratio, fill rate, rejects, and post-fill drift; spread capture can be added when order-book snapshots are attached to fills. |
| Exchange rule startup gate | Validate symbol status, margin asset, tick size, step size, min notional, order limits, rate limits, and `GTX` support before live starts. |
| User stream health | Track listen-key age, keepalive success, order-update heartbeat, reconnect count, and account-sync lag. |

## P1 Alpha And Model Governance

| Item | Target Behavior |
| --- | --- |
| Champion/challenger discipline | Current PROD remains champion while SHADOW candidates compete forward-only. |
| Regime-sliced walk-forward | Validation reports pass/fail by trend, mean reversion, volatility expansion, funding stress, and liquidity sweep regimes. |
| Feature drift report | DONE: `/models/drift` compares current DuckDB features with active model training references and feeds `/ops/readiness`. |
| Label stability report | DONE: retrain records near-threshold sensitivity, label balance, entropy, calibration, and trade-signal coverage in candidate metrics. |
| No-trade intelligence | Meta-controller should explain when the best decision is no trade because the market is not worth trading. |

## P1 Risk And Survival

| Item | Target Behavior |
| --- | --- |
| Liquidation-distance guard | Block new exposure when projected liquidation distance or margin buffer is too thin. |
| Funding and open-interest shock guard | Reduce or block exposure when funding or open-interest moves into abnormal regimes. |
| Spread and stale-mark guard | Block orders when spread widens, mark price is stale, or exchange data disagrees. |
| Rolling loss limits | Separate loss limits by hour, day, and rolling 24h window. |
| Separate kill switches | DONE: model, execution, data, account-sync, and manual operator lanes are independently visible and persisted. |

## P2 Frontend Trader Experience

| Item | Target Behavior |
| --- | --- |
| Candles with decisions | DONE foundation: history replay chart overlays decisions and fills over recent market path; true OHLC candles/order-book ladder remain a visual upgrade. |
| Order book ladder | PARTIAL: `/ws/market` now exposes best bid/ask and spread bps; full ladder depth and quote placement overlay remain. |
| Fill timeline | DONE: order lifecycle table shows event, order id, side, quantity, queue age, fill price, and post-fill drift. |
| Probability history | DONE: history view charts recent confidence/probability trajectory from persisted decisions. |
| Shadow vs primary comparison | Show candidate PnL, drawdown, fill quality, decisions, and divergence from primary. |
| Replay mode | DONE foundation: browser history view supports session-style review from persisted decisions, market rows, and lifecycle events; step-through playback remains a future interaction polish. |
| Live chart | DONE: cockpit Live tab charts direct websocket price ticks without waiting for DuckDB history refresh. |

## Production Efficiency Target

APEX should reach production efficiency only when it can run continuously with:

- fresh causal market data,
- a manifest-backed PROD model,
- live-blocking model governance,
- sustained paper and shadow evidence,
- maker execution telemetry,
- exchange/account stream health,
- auditable controls,
- and frontend transparency good enough for a trader to explain every decision.
