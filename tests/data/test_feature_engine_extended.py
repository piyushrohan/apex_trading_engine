import numpy as np
import pandas as pd
import pytest

from src.data.feature_engine import FeatureEngine


@pytest.mark.unit
def test_feature_engine_detects_sell_liquidity_sweep(mock_config, mock_eth_klines):
    """Verify long upper wicks on high volume are tagged as sell-side sweeps."""
    engine = FeatureEngine(mock_config)
    df = mock_eth_klines.copy()
    df.loc[9, "open"] = 3030.0
    df.loc[9, "close"] = 3025.0
    df.loc[9, "high"] = 3100.0
    df.loc[9, "low"] = 3020.0
    df.loc[9, "volume"] = 2500.0

    result = engine.add_orderflow_features(df)

    assert result.loc[9, "is_sell_liquidity_sweep"]


@pytest.mark.unit
def test_feature_engine_relative_strength_and_volatility_metrics(mock_config):
    """Verify beta, spread z-score, ATR, and volatility z-score are generated."""
    engine = FeatureEngine(mock_config)
    timestamps = pd.date_range("2026-05-16 12:00:00", periods=30, freq="3min")
    target_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 3000 + (np.arange(30) ** 1.15),
            "high": 3010 + (np.arange(30) ** 1.20),
            "low": 2990 + (np.arange(30) ** 1.05),
            "close": 3005 + (np.arange(30) ** 1.18),
            "volume": np.linspace(100, 300, 30),
            "taker_buy_volume": np.linspace(60, 180, 30),
        }
    )
    macro_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": np.linspace(60_000, 61_500, 30),
        }
    )

    enriched = engine.add_relative_strength(target_df.copy(), macro_df)
    volatile = engine.add_volatility_metrics(enriched)
    processed = engine.process_all_features(target_df, macro_df)
    no_macro = engine.process_all_features(target_df, pd.DataFrame())

    assert "eth_btc_beta" in enriched.columns
    assert "eth_btc_zscore" in enriched.columns
    assert "atr" in volatile.columns
    assert "volatility_zscore" in volatile.columns
    assert not processed.empty
    assert "eth_btc_beta" not in no_macro.columns
