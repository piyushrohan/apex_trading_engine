import numpy as np
import pandas as pd


def generate_flash_crash_klines() -> pd.DataFrame:
    """Generates a dataset with an artificial flash crash."""
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
    df.loc[5, "low"] = 2800.0
    df.loc[5, "high"] = 3050.0
    df.loc[5, "close"] = 2850.0
    df.loc[5, "volume"] = 5000.0
    return df


def generate_liquidity_sweep_klines() -> pd.DataFrame:
    """Generates a dataset with a deep buy-side liquidity sweep."""
    df = generate_flash_crash_klines()
    # Inject Liquidity Sweep (Long Lower Wick, strong close)
    df.loc[8, "open"] = 3020.0
    df.loc[8, "close"] = 3030.0
    df.loc[8, "high"] = 3035.0
    df.loc[8, "low"] = 2950.0  # Deep sweep
    df.loc[8, "volume"] = 2000.0
    return df


def generate_chop_compression_data() -> pd.DataFrame:
    """Generates a low volatility dataset."""
    df = pd.DataFrame({"close": [3000] * 100, "volatility_zscore": [-1.5] * 100})
    return df


def generate_trend_data() -> pd.DataFrame:
    """Generates a strong trend up dataset."""
    close_prices = [3000 + (i * 10) for i in range(100)]
    df = pd.DataFrame({"close": close_prices, "volatility_zscore": [0.5] * 100})
    return df
