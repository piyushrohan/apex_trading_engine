import numpy as np
import pandas as pd
import pytest

from tests.fixtures.market_data import generate_liquidity_sweep_klines


@pytest.fixture
def mock_config():
    """Deterministic configuration for all tests."""
    return {
        "data": {"target_symbol": "ETHUSDC", "target_interval": "3m"},
        "execution": {
            "operator_mode": "paper",
            "position_mode": "one_way",
            "max_leverage": 3,
            "kelly_fraction_cap": 0.3,
            "max_daily_drawdown": 0.05,
        },
        "live": {"enabled": False},
        "paper": {
            "enabled": True,
            "min_days": 7,
            "min_trades": 100,
            "min_sharpe": 1.0,
            "max_drawdown": 0.08,
        },
        "hedge": {"enabled": False},
        "risk": {"max_leverage": 3, "max_drawdown_pct": 0.10, "kelly_fraction": 0.5},
        "technicals": {
            "rolling_window": 10,
            "atr_period": 10,
            "macro_vol_z_period": 20,
        },
        "environment": {"initial_capital": 10000.0, "transaction_cost_pct": 0.0},
    }


@pytest.fixture
def mock_eth_klines() -> pd.DataFrame:
    """Deterministic dataset of ETHUSDC candles."""
    return generate_liquidity_sweep_klines()


@pytest.fixture
def mock_btc_klines() -> pd.DataFrame:
    """Mock BTC Data for relative strength calculations."""
    data = {
        "timestamp": pd.date_range("2026-05-16 12:00:00", periods=20, freq="3min"),
        "close": np.linspace(60000, 62000, 20),
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_trade_history():
    """Mock trade history for evaluator testing."""
    return [
        {"pnl": 50, "side": "LONG"},
        {"pnl": -10, "side": "SHORT"},
        {"pnl": 100, "side": "LONG"},
        {"pnl": -20, "side": "LONG"},
        {"pnl": -30, "side": "SHORT"},
    ] * 10  # 50 trades total
