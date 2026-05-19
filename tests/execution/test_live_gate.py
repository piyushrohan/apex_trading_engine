import pytest

from src.execution.live_gate import evaluate_paper_gate, validate_live_startup


@pytest.mark.unit
def test_paper_gate_fails_without_snapshots(mock_config):
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
def test_validate_live_startup_blocks_failed_paper_gate(mock_config):
    mock_config["live"] = {"enabled": True, "skip_paper_gate": False}
    mock_config.setdefault("paper", {})["min_days"] = 30
    with pytest.raises(RuntimeError, match="paper gate"):
        validate_live_startup(mock_config)
