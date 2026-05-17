import pytest

from src.data.feature_engine import FeatureEngine


@pytest.mark.unit
def test_feature_engine_calculates_cvd(mock_config, mock_eth_klines):
    """Verify CVD mathematically matches orderflow imbalance."""
    engine = FeatureEngine(mock_config)

    df = engine.add_orderflow_features(mock_eth_klines.copy())

    assert "net_volume" in df.columns
    assert "cvd" in df.columns

    # Net volume should be taker_buy - (total - taker_buy)
    taker_sell = df.loc[0, "volume"] - df.loc[0, "taker_buy_volume"]
    expected_net = df.loc[0, "taker_buy_volume"] - taker_sell

    assert round(df.loc[0, "net_volume"], 4) == round(expected_net, 4)


@pytest.mark.unit
def test_feature_engine_liquidity_sweep_detection(mock_config, mock_eth_klines):
    """
    Data Integrity Test: Verify that abnormal candle patterns
    (long wicks + high volume) are accurately tagged as liquidity sweeps.
    """
    engine = FeatureEngine(mock_config)

    df = engine.add_orderflow_features(mock_eth_klines.copy())
    print(df.loc[8])
    assert df.loc[8, "is_buy_liquidity_sweep"]


@pytest.mark.chaos
def test_feature_engine_handles_missing_data(mock_config, mock_eth_klines):
    """
    Data Integrity Test / Chaos Test:
    Verify that the feature engine doesn't crash if `taker_buy_volume` is missing
    (e.g., Binance stream disconnect). It should fallback to approximation.
    """
    engine = FeatureEngine(mock_config)

    # Simulate a corrupted or non-premium stream that lacks taker_buy_volume
    df_missing = mock_eth_klines.drop(columns=["taker_buy_volume"])

    df_processed = engine.add_orderflow_features(df_missing)

    assert "cvd" in df_processed.columns
    # Net volume is approximated using direction of candle
    assert "net_volume" in df_processed.columns
