# Trader Production Backlog

This backlog converts the crypto quant demo critique into concrete engineering
work. The goal is not to make APEX trade more often. The goal is to make it
trade only when data, model evidence, execution quality, and risk controls are
strong enough to justify live market exposure.

## Implemented In This PR

| Area | Status | Detail |
| --- | --- | --- |
| Shadow artifact resilience | DONE | Shadow lanes preflight GBM artifacts in a child process before loading them in the runtime. A crashing native artifact is quarantined instead of crashing paper/live startup. |
| Trader readiness API | DONE | `GET /ops/readiness` reports live blockers, paper gate status, PROD model readiness, fill evidence, runtime staleness, low-conviction warnings, journal freshness, and DuckDB diagnostic availability. |
| Trader cockpit Ops tab | DONE | Frontend adds an `Ops` tab with live readiness, critical/warning counts, guardrail findings, data freshness, live-gate snapshot, and next actions. |
| Exchange rule validation foundation | DONE | Binance REST client can fetch `exchangeInfo`, extract symbol filters, and validate USDC margin, `GTX` post-only support, min notional, price filter, and lot-size filter availability. |
| Frontend/API smoke coverage | DONE | Contract smoke now includes `/ops/readiness`; frontend static tests cover the new operator readiness tab. |

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
| Full order lifecycle journal | Persist submit, ack, open, partial fill, cancel, replace, fill, reject, and timeout events. |
| Maker quote analytics | Track queue age, spread capture, cancel/replace ratio, fill latency, and mid-price drift after fill. |
| Exchange rule startup gate | Validate symbol status, margin asset, tick size, step size, min notional, order limits, rate limits, and `GTX` support before live starts. |
| User stream health | Track listen-key age, keepalive success, order-update heartbeat, reconnect count, and account-sync lag. |

## P1 Alpha And Model Governance

| Item | Target Behavior |
| --- | --- |
| Champion/challenger discipline | Current PROD remains champion while SHADOW candidates compete forward-only. |
| Regime-sliced walk-forward | Validation reports pass/fail by trend, mean reversion, volatility expansion, funding stress, and liquidity sweep regimes. |
| Feature drift report | Compare live feature distributions with the training snapshot used by the active model. |
| Label stability report | Show how many labels sit near decision thresholds and how sensitive they are to small price moves. |
| No-trade intelligence | Meta-controller should explain when the best decision is no trade because the market is not worth trading. |

## P1 Risk And Survival

| Item | Target Behavior |
| --- | --- |
| Liquidation-distance guard | Block new exposure when projected liquidation distance or margin buffer is too thin. |
| Funding and open-interest shock guard | Reduce or block exposure when funding or open-interest moves into abnormal regimes. |
| Spread and stale-mark guard | Block orders when spread widens, mark price is stale, or exchange data disagrees. |
| Rolling loss limits | Separate loss limits by hour, day, and rolling 24h window. |
| Separate kill switches | Model, execution, data, account-sync, and manual operator kill switches should be visible independently. |

## P2 Frontend Trader Experience

| Item | Target Behavior |
| --- | --- |
| Candles with decisions | Overlay LONG/SHORT/FLAT, fills, cancels, and shadow decisions on recent candles. |
| Order book ladder | Show spread, best bid/ask, depth imbalance, and quote placement. |
| Fill timeline | Explain each simulated or live fill with order id, queue age, price, side, and post-fill drift. |
| Probability history | Chart model action probabilities and confidence buckets over time. |
| Shadow vs primary comparison | Show candidate PnL, drawdown, fill quality, decisions, and divergence from primary. |
| Replay mode | Let the operator replay a historical session and inspect what the model saw at each step. |

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
