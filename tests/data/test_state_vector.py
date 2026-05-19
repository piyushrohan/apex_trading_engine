import pytest

from src.data.state_vector import STATE_FEATURE_COLUMNS, build_state_vector


@pytest.mark.unit
def test_state_vector_length_matches_feature_map():
    row = {
        "open": 3000.0,
        "close": 3010.0,
        "volume": 100.0,
        "net_volume": 10.0,
        "cvd": 50.0,
        "is_buy_liquidity_sweep": True,
        "is_sell_liquidity_sweep": False,
        "eth_btc_beta": 1.1,
        "eth_btc_zscore": 0.5,
        "atr": 30.0,
        "volatility_zscore": -0.2,
        "trend_slope": 0.01,
    }
    vec = build_state_vector(row)
    assert len(vec) == len(STATE_FEATURE_COLUMNS) == 10
    assert vec[3] == 1.0
    assert vec[4] == 0.0


@pytest.mark.unit
def test_state_vector_handles_missing_fields():
    vec = build_state_vector({})
    assert len(vec) == 10
    assert all(isinstance(x, float) for x in vec)
