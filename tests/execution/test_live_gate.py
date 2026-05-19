import pandas as pd
import pytest

from src.execution import live_gate
from src.execution.live_gate import (
    check_api_credentials,
    evaluate_paper_gate,
    validate_live_startup,
)


@pytest.mark.unit
def test_paper_gate_fails_without_snapshots(mock_config, tmp_path):
    mock_config["data"]["storage"] = {"db_path": str(tmp_path / "empty_gate.duckdb")}

    result = evaluate_paper_gate(mock_config)

    assert result.passed is False
    assert any("snapshots" in r for r in result.reasons)


@pytest.mark.unit
def test_paper_gate_passes_when_metrics_met(mock_config, tmp_path):
    from src.data.cache_manager import DuckDBCacheManager

    db_path = str(tmp_path / "gate.duckdb")
    mock_config["data"]["storage"] = {"db_path": db_path}
    mock_config["paper"] = {
        "enabled": True,
        "min_days": 0,
        "min_trades": 0,
        "min_sharpe": 0,
        "max_drawdown": 1.0,
    }
    cache = DuckDBCacheManager(db_path=db_path)
    cache.insert_paper_equity_snapshot(
        book_id="primary",
        equity=1000.0,
        long_qty=0.0,
        short_qty=0.0,
        mark_price=3500.0,
        regime="MEAN_REVERSION",
    )
    cache.close()

    result = evaluate_paper_gate(mock_config)
    assert result.passed is True


@pytest.mark.unit
def test_validate_live_startup_requires_enabled(mock_config):
    mock_config["live"] = {"enabled": False}
    with pytest.raises(RuntimeError, match="live.enabled"):
        validate_live_startup(mock_config)


@pytest.mark.unit
def test_validate_live_startup_blocks_failed_paper_gate(mock_config, tmp_path):
    mock_config["data"]["storage"] = {"db_path": str(tmp_path / "blocked_gate.duckdb")}
    mock_config["live"] = {"enabled": True, "skip_paper_gate": False}
    mock_config.setdefault("paper", {})["min_days"] = 30

    with pytest.raises(RuntimeError, match="paper gate"):
        validate_live_startup(mock_config)


@pytest.mark.unit
def test_paper_gate_can_be_skipped(mock_config):
    result = evaluate_paper_gate(mock_config, skip_gate=True)

    assert result.passed is True
    assert result.metrics["gate"] == "skipped"


@pytest.mark.unit
def test_paper_gate_reports_drawdown_breach(mock_config, monkeypatch):
    monkeypatch.setattr(live_gate, "_paper_run_days", lambda config, book_id: 2.0)
    monkeypatch.setattr(
        live_gate,
        "generate_paper_report",
        lambda config, book_id, journal_path: {
            "directional_decisions": 10,
            "sharpe": 2.0,
            "max_drawdown": 0.25,
            "snapshots": 3,
        },
    )
    mock_config["paper"] = {
        "enabled": True,
        "min_days": 1,
        "min_trades": 1,
        "min_sharpe": 1,
        "max_drawdown": 0.10,
    }

    result = evaluate_paper_gate(mock_config)

    assert result.passed is False
    assert any("drawdown" in reason for reason in result.reasons)


class FakeCache:
    def __init__(self, db_path):
        self.db_path = db_path
        self.closed = False

    def load_paper_equity_snapshots(self, book_id):
        return pd.DataFrame(
            {"timestamp": ["2026-05-17T00:00:00Z", "2026-05-19T12:00:00Z"]}
        )

    def close(self):
        self.closed = True


@pytest.mark.unit
def test_paper_run_days_uses_snapshot_span(mock_config, monkeypatch):
    monkeypatch.setattr(live_gate, "DuckDBCacheManager", FakeCache)

    days = live_gate._paper_run_days(mock_config, "primary")

    assert days == 2.5


@pytest.mark.unit
def test_validate_live_startup_allows_explicit_paper_gate_skip(mock_config):
    mock_config["live"] = {"enabled": True, "skip_paper_gate": True}

    validate_live_startup(mock_config)


@pytest.mark.unit
def test_check_api_credentials_from_env_or_config(mock_config, monkeypatch):
    mock_config["live"] = {}
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)

    ok, reason = check_api_credentials(mock_config)
    assert ok is False
    assert "BINANCE_API_KEY" in reason

    monkeypatch.setenv("BINANCE_API_KEY", "key")
    monkeypatch.setenv("BINANCE_API_SECRET", "secret")
    assert check_api_credentials(mock_config) == (True, None)
