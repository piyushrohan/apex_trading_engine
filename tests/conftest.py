import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def mock_config():
    """Deterministic configuration for all tests."""
    return {
        "data": {
            "target_symbol": "ETHUSDC",
            "target_interval": "3m"
        },
        "execution": {
            "mode": "MAKER_ONLY",
            "slippage_tolerance_bps": 2.0
        },
        "risk": {
            "max_leverage": 3,
            "max_drawdown_pct": 0.10,
            "kelly_fraction": 0.5
        },
        "technicals": {
            "rolling_window": 10,
            "atr_period": 10,
            "macro_vol_z_period": 20
        },
        "environment": {
            "initial_capital": 10000.0,
            "transaction_cost_pct": 0.0
        }
    }

from tests.fixtures.market_data import generate_liquidity_sweep_klines

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
    ] * 10 # 50 trades total
