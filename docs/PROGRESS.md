# APEX Implementation Progress

Tracks delivery against [PRODUCTION_CHECKLIST_AND_ROADMAP.md](./PRODUCTION_CHECKLIST_AND_ROADMAP.md).

**Legend:** `DONE` | `IN_PROGRESS` | `NOT_STARTED`

---

## Current focus

| Item | Status |
|------|--------|
| **Milestone 5** — Explainability v2 + API | `DONE` |
| **Milestone 6** — ML & MLOps | `NOT_STARTED` |
| **Phase 0** — Foundations | `DONE` |

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

## Milestone 6 — Next

- Real GBM + PPO train/save/load
- Shadow lane runner inside `TradingPipeline`
- `promotion_service.py`

---

## Changelog

| Date | Change |
|------|--------|
| 2026-05-18 | Milestones 1–4 complete |
| 2026-05-18 | **Milestone 5 complete** — explainability v2, FastAPI read-only API |
