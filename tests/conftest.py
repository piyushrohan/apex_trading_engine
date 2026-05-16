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

@pytest.fixture
def mock_eth_klines() -> pd.DataFrame:
    """
    Deterministic dataset of ETHUSDC candles.
    Contains artificial patterns:
    - Index 5: Massive volatility spike (Flash Crash)
    - Index 8: Liquidity Sweep (long lower wick)
    """
    data = {
        "timestamp": pd.date_range("2026-05-16 12:00:00", periods=20, freq="3min"),
        "open": np.linspace(3000, 3100, 20),
        "high": np.linspace(3010, 3110, 20),
        "low": np.linspace(2990, 3090, 20),
        "close": np.linspace(3005, 3105, 20),
        "volume": np.random.RandomState(42).uniform(100, 500, 20),
        "taker_buy_volume": np.random.RandomState(42).uniform(50, 250, 20),
    }
    df = pd.DataFrame(data)
    
    # Inject Flash Crash
    df.loc[5, 'low'] = 2800.0
    df.loc[5, 'high'] = 3050.0
    df.loc[5, 'close'] = 2850.0
    df.loc[5, 'volume'] = 5000.0
    
    # Inject Liquidity Sweep (Long Lower Wick, strong close)
    df.loc[8, 'open'] = 3020.0
    df.loc[8, 'close'] = 3030.0
    df.loc[8, 'high'] = 3035.0
    df.loc[8, 'low'] = 2950.0 # Deep sweep
    df.loc[8, 'volume'] = 2000.0
    
    return df

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
