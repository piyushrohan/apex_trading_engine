# Operational Automations

APEX includes six operator-grade automation reports. They are designed to run
both locally during paper/shadow sessions and daily in GitHub Actions.

## Paper Trading Health Watchdog

Checks whether the paper operator loop is alive, explained, and producing usable
evidence.

```bash
python -m src.reports.paper_health_watchdog \
  --config configs/base.yaml \
  --runtime-status-path data_lake/runtime_status.json \
  --format markdown
```

Strict mode is useful during an active paper trading session:

```bash
python -m src.reports.paper_health_watchdog \
  --config configs/base.yaml \
  --strict \
  --max-status-age-minutes 15 \
  --max-decision-age-minutes 30
```

It watches for stale runtime status, active kill switch state, missing latest
explanations, stale or missing paper decisions, missing equity snapshots, zero
paper fill rate, and malformed journal rows.

## Shadow Lane Sanity Monitor

Checks whether shadow lanes are configured, virtual-only, and producing decision
evidence for candidate models.

```bash
python -m src.reports.shadow_sanity_monitor \
  --config configs/base.yaml \
  --format markdown
```

Strict mode treats missing active shadow evidence as a failure:

```bash
python -m src.reports.shadow_sanity_monitor \
  --config configs/base.yaml \
  --strict \
  --max-decision-age-minutes 60
```

It verifies active shadow registry state, artifact and manifest presence,
shadow book tagging, decision freshness, active-shadow evidence, and hedge
candidate score maps.

## Daily Model Governance Report

Summarizes model discipline across registry state, production readiness,
experiment runs, paper metrics, shadow metrics, and promotion posture.

```bash
python -m src.reports.model_governance_report \
  --config configs/base.yaml \
  --format markdown
```

The report recommends one of the following postures:

- `train_and_promote_prod_candidate`
- `fix_active_prod_readiness`
- `review_shadow_for_prod_promotion`
- `continue_shadow_data_collection`
- `hold`

## Experiment Ledger Auditor

Audits the append-only experiment ledger and checks that successful training
runs are reproducible and linked to registry evidence.

```bash
python -m src.reports.experiment_ledger_auditor \
  --config configs/base.yaml \
  --format markdown
```

Strict mode is useful before approving a shadow or production promotion:

```bash
python -m src.reports.experiment_ledger_auditor \
  --config configs/base.yaml \
  --strict \
  --max-running-age-minutes 240
```

It checks malformed JSONL rows, missing run starts, duplicate starts or
completions, stale `RUNNING` runs, failed steps, required candidate retrain
steps, registry linkage, model artifacts, immutable manifests, and data
snapshot ids.

## Frontend/API Contract Smoke Test

Statically checks that the browser terminal still matches the FastAPI control
surface.

```bash
python -m src.reports.frontend_api_contract_smoke \
  --format markdown
```

To smoke-test a running API from the browser contract, start the API first and
then enable live probes:

```bash
python -m src.api.server
python -m src.reports.frontend_api_contract_smoke \
  --api-base-url http://127.0.0.1:8080 \
  --live-api \
  --format markdown
```

It verifies expected GET endpoints, guarded control POSTs, the status websocket,
the `?api=` local override, the React root, and JSON responses from the live API
when `--live-api` is supplied.

## Data Freshness And DuckDB Integrity Check

Scans the local DuckDB data lake for missing tables, empty tables, stale
timestamps, OHLCV gaps, duplicate keys, bad OHLCV values, invalid feature JSON,
and missing freshness signals.

```bash
python -m src.reports.data_freshness_check \
  --config configs/base.yaml \
  --format markdown
```

Strict mode is the right choice during an active data ingestion, paper, or
shadow session:

```bash
python -m src.reports.data_freshness_check \
  --config configs/base.yaml \
  --strict \
  --max-ohlcv-age-minutes 30 \
  --max-tick-age-minutes 30 \
  --max-market-age-minutes 60 \
  --max-feature-age-minutes 60 \
  --max-equity-age-minutes 60
```

You can override the inspected market and cache path:

```bash
python -m src.reports.data_freshness_check \
  --db-path data_lake/apex_market_data.duckdb \
  --symbol ETHUSDC \
  --timeframe 3m \
  --format markdown
```

## GitHub Automation

The scheduled workflow lives at:

```text
.github/workflows/operational-automations.yml
```

It runs daily at `02:30 UTC` and can also be triggered manually from GitHub
Actions. The workflow uploads JSON and Markdown artifacts under:

```text
reports/automation/
```

The daily workflow runs in non-strict mode so that an empty CI data lake still
produces useful artifacts instead of failing. Use strict mode locally when a
paper, shadow, retraining, or data-ingestion session is actively expected to be
producing fresh telemetry.
