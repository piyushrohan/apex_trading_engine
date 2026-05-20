# APEX Testing And Validation Runbook

This runbook is the operational reference for validating the APEX trading engine locally and understanding the CI gates. It reflects the recent test coverage expansion across REST/WS data paths, ingestion, DuckDB cache behavior, market state, trading pipeline edge paths, MLOps registry/promotion, model governance, reports, frontend static checks, and contextual hedge bandit logic.

## Latest Validated Snapshot

Local validation after the recent coverage work:

```bash
make ci-local
# includes formatter/lint/typecheck, frontend static checks, risk tests, main suite, and coverage
```

Strict local coverage enforcement:

```bash
venv/bin/pytest -m "not slow and not replay" \
  --cov=src \
  --cov-report=term-missing \
  --cov-fail-under=95
# Required test coverage of 95% reached.
#
# Latest strict full-suite check:
# venv/bin/pytest tests/ -v --cov=src --cov-fail-under=95 --cov-report=xml
# 244 passed
# Total coverage: 95.16%
```

Risk gate:

```bash
make test-risk
# collected 227 items / 216 deselected / 11 selected
# 11 passed
```

The `216 deselected` count is normal. `make test-risk` runs `pytest -m risk`, so Pytest collects every test and then deselects anything without the `risk` marker.

## Environment Bootstrapping

Recommended one-command setup:

```bash
bash scripts/setup_dev.sh
source venv/bin/activate
```

Manual setup:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

On macOS, real LightGBM training also needs the OpenMP runtime:

```bash
brew install libomp
```

Confirm tooling is available:

```bash
which python
python --version
which pytest
which black
which ruff
which mypy
```

When using non-interactive shells or the Codex terminal, prefer the venv binaries directly:

```bash
venv/bin/pytest --version
venv/bin/ruff --version
venv/bin/black --version
```

## Make Targets

The current `Makefile` targets are:

```bash
make format
make lint
make typecheck
make test-risk
make test
make test-unit
make test-integration
make test-replay
make replay-debug SESSION=latest
make coverage
make ci-local
```

### `make ci-local`

Full local validation:

```bash
make ci-local
```

Execution order:

```text
make format      -> black src tests; isort src tests
make lint        -> ruff check src tests; flake8 E9/F63/F7/F82 checks
make typecheck   -> mypy src/ --ignore-missing-imports || echo warning
make frontend-test -> node --check frontend/app.js; node --test tests/frontend/*.test.mjs
make test-risk   -> pytest -m risk
make test        -> pytest -m "not slow and not replay" -n auto
make coverage    -> pytest --cov=src --cov-report=html --cov-report=term-missing
```

Important: the current Makefile treats mypy as advisory because the command ends with `|| echo "Mypy warnings found..."`. A duplicate-module mypy warning may appear locally without failing `make ci-local`.

### `make test-risk`

Runs only safety-critical tests:

```bash
make test-risk
```

Equivalent raw command:

```bash
pytest -m risk
```

Currently selected risk tests cover:

```text
tests/chaos/test_disasters.py
tests/execution/test_risk_engine.py
tests/execution/test_risk_hedge_limits.py
tests/execution/test_risk_slippage_edges.py
```

Expected output shape:

```text
collected 227 items / 216 deselected / 11 selected
11 passed
```

### `make test`

Runs the main fast suite in parallel:

```bash
make test
```

Equivalent raw command:

```bash
pytest -m "not slow and not replay" -n auto
```

Use this before pushing when you do not need coverage output.

### `make test-unit`

Runs tests marked `unit`:

```bash
make test-unit
```

Equivalent raw command:

```bash
pytest -m unit -n auto
```

### `make test-integration`

Runs tests marked `integration`:

```bash
make test-integration
```

Equivalent raw command:

```bash
pytest -m integration
```

If a future integration test requires local infrastructure, start the test compose stack first:

```bash
docker compose -f docker-compose.test.yml up -d
make test-integration
docker compose -f docker-compose.test.yml down
```

### `make test-replay`

Runs deterministic replay tests:

```bash
make test-replay
```

Equivalent raw command:

```bash
pytest -m replay
```

Verbose replay debugging:

```bash
make replay-debug SESSION=2026-05-16
```

Equivalent raw command:

```bash
pytest -m replay -s -vv --log-cli-level=DEBUG
```

### `make coverage`

Generates terminal and HTML coverage:

```bash
make coverage
```

Equivalent raw command:

```bash
pytest --cov=src --cov-report=html --cov-report=term-missing
```

Open the report:

```bash
open htmlcov/index.html
```

Note: `make coverage` currently reports coverage but does not set a local fail-under threshold. Use the strict command below when you need to enforce the 95% local gate.

## Strict Coverage Commands

Fast strict coverage gate used after the recent work:

```bash
venv/bin/pytest -m "not slow and not replay" \
  --cov=src \
  --cov-report=term-missing \
  --cov-fail-under=95
```

Full strict coverage including replay/slow tests when present:

```bash
venv/bin/pytest \
  --cov=src \
  --cov-report=html \
  --cov-report=term-missing \
  --cov-fail-under=95
```

Execution/risk granular gate similar to CI:

```bash
pytest tests/ -m "unit or risk" \
  --cov=src/execution \
  --cov-fail-under=95
```

Full XML coverage artifact for CI-style upload/debugging:

```bash
pytest tests/ \
  --cov=src \
  --cov-report=xml \
  --cov-report=term-missing
```

## Pytest Markers

Markers are registered in `pytest.ini`:

| Marker | Purpose | Typical command |
|--------|---------|-----------------|
| `unit` | Fast business logic tests | `pytest -m unit -n auto` |
| `integration` | Component interaction tests with mocked/local infra | `pytest -m integration` |
| `replay` | Historical deterministic simulation tests | `pytest -m replay` |
| `slow` | Long-running paper trading or heavy data tests | `pytest -m slow` |
| `risk` | Safety-critical execution, sizing, and kill-switch tests | `pytest -m risk` |
| `mlops` | Model evaluation, registry, promotion, and state transitions | `pytest -m mlops` |
| `chaos` | Failure, disconnect, and catastrophe simulations | `pytest -m chaos` |

List collected tests without running them:

```bash
pytest --collect-only -q
pytest -m risk --collect-only -q
pytest -m "unit or integration or mlops or chaos" --collect-only -q
```

## Recent Coverage Expansion

The latest coverage work added and expanded tests in these areas:

| Area | Coverage focus |
|------|----------------|
| `src/data/binance_rest.py` | signed request signing/retry/error handling, signed account helpers, public market helpers, listen-key success/failure paths |
| `src/data/binance_ws.py` | stream URL construction, timeout handling, cancel handling, legacy connect shutdown, callback/no-callback paths |
| `src/data/cache_manager.py` | empty writes, duplicate inserts, feature JSON handling, failure swallowing, snapshots, gap detection, parquet backup |
| `src/data/ingestion_service.py` | disabled no-op paths, bootstrap totals, current-cache skip, gap repair, funding loop recovery, WS mark/depth/aggTrade handling, `_main` cleanup |
| `src/data/market_state.py` | empty cache, empty features, latest snapshot fallbacks, feature persistence failure, owned-cache close |
| `src/pipelines/trading_pipeline.py` | live prep, startup validation, account bootstrap, status publishing, bandit decision logging, signal execution, hedge/grid orders, paper fills, kill switch, stop cleanup |
| `src/mlops/registry.py` | manifest writes, git hash fallback, metric/status updates, shadow archival, rollback edges |
| `src/mlops/promotion_service.py` | insufficient history, small edge hold, drawdown reject, metrics from decision logs, rollback breach/no-breach |
| `src/strategies/hedge/bandit_selector.py` | empty selector, missing rule scores, tie/exploration, reward updates, state load/save, decision bootstrap |
| `src/reports/*.py` | missing/empty journal paths, empty cache reports, fill counts, invalid timestamps, CLI main functions |
| `src/pipelines/live_trade.py` | build pipeline and KeyboardInterrupt stop behavior |

Current high-confidence modules include data cache, market state, registry, bandit selector, live trade, position sync, portfolio, live gate, risk engine, order manager, slippage, and hedge registry.

## Targeted Test Commands

Run one file:

```bash
pytest tests/data/test_ingestion_service.py -q
pytest tests/pipelines/test_trading_pipeline_edges.py -q
pytest tests/mlops/test_registry_edges.py -q
```

Run one test:

```bash
pytest tests/data/test_binance_rest_signed.py::test_signed_request_retries_errors_and_rejects_unknown_methods -q
```

Run a keyword:

```bash
pytest -k slippage
pytest -k "live_gate or paper_gate"
pytest -k "bandit and reward"
```

Run with logs and print output:

```bash
pytest -vv -s --log-cli-level=DEBUG tests/pipelines/test_trading_pipeline_edges.py
```

Run without xdist when debugging race-sensitive behavior:

```bash
pytest -m "not slow and not replay" -n 0
pytest tests/execution/test_live_gate.py -n 0 -vv
```

Run last failures:

```bash
pytest --lf
pytest --ff
```

Drop into debugger on failure:

```bash
pytest -x --pdb tests/data/test_ingestion_service.py
```

## Parallel Test Safety

The main test suite runs with xdist:

```bash
pytest -m "not slow and not replay" -n auto
```

Rules for new tests:

- Use `tmp_path` for DuckDB files.
- Override `mock_config["data"]["storage"]["db_path"]` when a test calls production code that opens DuckDB.
- Do not let tests share `data_lake/apex_market_data.duckdb`.
- Close `DuckDBCacheManager` instances in tests.
- Prefer fake REST/cache/session objects for edge branches instead of real network calls.
- Patch sleeps with `AsyncMock()` or a small deterministic coroutine.
- For background tasks, cancel/await them and close un-awaited coroutines in fake task factories.

Example isolated DuckDB config:

```python
def test_gate_uses_isolated_db(mock_config, tmp_path):
    mock_config["data"]["storage"] = {
        "db_path": str(tmp_path / "paper_gate.duckdb")
    }
    ...
```

This pattern prevents xdist workers from locking the same DuckDB database.

## Pre-Commit And Commit-Time Checks

Run the same hooks before committing:

```bash
venv/bin/pre-commit run --all-files
```

Individual formatting/linting:

```bash
black src tests
isort src tests
ruff check src tests
ruff check src tests --fix
flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics
```

Commit sequence:

```bash
make ci-local
git status --short
git add README.md docs/testing.md tests src
git commit -m "test: expand coverage and document validation workflow"
```

If a hook says "files were modified by this hook", stage the modified files and commit again:

```bash
git status --short
git add <modified-files>
git commit -m "..."
```

## CI Workflows

Important workflows under `.github/workflows/`:

| Workflow | Purpose |
|----------|---------|
| `ci.yml` | Core risk, unit, integration, mlops, chaos, coverage checks |
| `paper-trading-validation.yml` | Bounded paper validation and artifacts |
| `model-evaluation.yml` | MLOps/model evaluation checks |
| `shadow-deployment.yml` | Shadow lane validation |
| `nightly-validation.yml` | Slow/replay/nightly coverage artifacts |

Useful GitHub CLI commands:

```bash
gh run list --limit 10
gh run view <run-id> --log
gh run watch <run-id>
gh pr view --web
gh pr checks
```

## Troubleshooting

### `make test-risk` shows many deselected tests

Expected:

```text
collected 227 items / 216 deselected / 11 selected
```

Reason: `pytest -m risk` selects only tests marked `risk`.

### `make ci-local` prints a mypy duplicate-module warning

Current Makefile command:

```bash
mypy src/ --ignore-missing-imports || echo "Mypy warnings found. Run with strictness locally."
```

Because of the `|| echo`, this warning is advisory and does not fail `make ci-local`. If you want to investigate strictly:

```bash
mypy src/ --ignore-missing-imports --explicit-package-bases
```

### DuckDB locking or flaky xdist behavior

Run the suspected file serially:

```bash
pytest tests/execution/test_live_gate.py -n 0 -vv
```

Then check whether the test uses a shared DB path. Fix by using `tmp_path`.

### Coverage is below 95%

Print missing lines:

```bash
venv/bin/pytest -m "not slow and not replay" \
  --cov=src \
  --cov-report=term-missing \
  --cov-fail-under=95
```

Open HTML details:

```bash
make coverage
open htmlcov/index.html
```

Start with the biggest miss-count modules and add branch-level tests around real behavior. Prefer tests over `pragma: no cover`; reserve no-cover only for true CLI guards or unreachable defensive branches.

### Black passes locally but commit still fails

Pre-commit may run from a slightly different file set after staging. Run:

```bash
black src tests
ruff check src tests --fix
git status --short
git add <modified-files>
venv/bin/pre-commit run --all-files
```

### Need a clean local test artifact state

Most runtime artifacts are git-ignored. To inspect them:

```bash
git status --ignored --short
```

Common generated paths:

```text
.coverage
coverage.xml
htmlcov/
.pytest_cache/
.ruff_cache/
.mypy_cache/
data_lake/
logs/
```

Do not commit these artifacts.

## Adding New Tests

Checklist for new tests:

- Mark each test with the narrowest marker: `unit`, `integration`, `risk`, `mlops`, `chaos`, `replay`, or `slow`.
- Keep risk tests small and deterministic.
- Use fakes/mocks for Binance network calls.
- Use temp DuckDB paths for anything that touches storage.
- Assert behavior, not implementation trivia.
- Add regression tests for every bug fix.
- Run the relevant targeted file first.
- Run `make test-risk` if risk/execution behavior changed.
- Run `make test` for the parallel suite.
- Run the strict coverage command before pushing coverage-sensitive changes.

Recommended command sequence for a coverage PR:

```bash
black src tests
isort src tests
ruff check src tests
pytest tests/<changed-area> -q
make test-risk
make test
venv/bin/pytest -m "not slow and not replay" \
  --cov=src \
  --cov-report=term-missing \
  --cov-fail-under=95
make ci-local
```
