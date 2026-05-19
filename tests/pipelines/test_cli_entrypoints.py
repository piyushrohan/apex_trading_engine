import asyncio
import runpy
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_base_config(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = {
        "data": {
            "target_symbol": "ETHUSDC",
            "target_interval": "3m",
            "macro_symbol": "BTCUSDC",
            "storage": {"db_path": "data_lake/apex.duckdb"},
        },
        "environment": {"initial_capital": 1000.0, "transaction_cost_pct": 0.0},
        "execution": {"max_leverage": 3, "kelly_fraction_cap": 0.3},
        "technicals": {
            "rolling_window": 10,
            "atr_period": 10,
            "macro_vol_z_period": 20,
        },
    }
    (config_dir / "base.yaml").write_text(yaml.safe_dump(config))


@pytest.mark.integration
def test_auto_retrain_cli_entrypoint_runs_in_temp_workspace(tmp_path, monkeypatch):
    """Verify the auto-retrain module entrypoint runs without touching repo state."""
    write_base_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    runpy.run_path(str(PROJECT_ROOT / "src/mlops/auto_retrain.py"), run_name="__main__")

    assert (tmp_path / "data_lake" / "models" / "registry.json").exists()


@pytest.mark.integration
def test_live_trade_cli_entrypoint_handles_keyboard_interrupt(tmp_path, monkeypatch):
    """Verify the live-trade module entrypoint invokes asyncio.run once."""
    write_base_config(tmp_path)
    (tmp_path / "configs" / "risk_profiles.yaml").write_text(
        "balanced:\n  max_leverage: 3.0\n  max_daily_drawdown: 0.05\n"
        "  kelly_fraction_cap: 0.3\n"
    )
    monkeypatch.chdir(tmp_path)
    calls = {"count": 0}

    def fake_run(coro):
        calls["count"] += 1
        coro.close()
        return None

    monkeypatch.setattr(asyncio, "run", fake_run)

    runpy.run_path(
        str(PROJECT_ROOT / "src/pipelines/live_trade.py"), run_name="__main__"
    )

    assert calls["count"] == 1
