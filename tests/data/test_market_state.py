import pandas as pd
import pytest

from src.data.cache_manager import DuckDBCacheManager
from src.data.market_state import MarketStateService


def _to_cache_df(
    klines: pd.DataFrame, symbol: str, timeframe: str = "3m"
) -> pd.DataFrame:
    df = klines.copy()
    df["symbol"] = symbol
    df["timeframe"] = timeframe
    return df[
        [
            "timestamp",
            "symbol",
            "timeframe",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]


def _short_technicals_config(mock_config):
    mock_config["technicals"]["rolling_window"] = 5
    mock_config["technicals"]["atr_period"] = 5
    mock_config["technicals"]["macro_vol_z_period"] = 5
    mock_config["technicals"]["ema_trend_long"] = 10


@pytest.mark.unit
def test_market_state_seed_from_dataframes(mock_config, mock_eth_klines):
    _short_technicals_config(mock_config)
    svc = MarketStateService(mock_config)
    snapshot = svc.seed_from_dataframes(mock_eth_klines, None)
    assert len(snapshot["state_vector"]) == 10
    assert snapshot["regime"] in (
        "STRONG_TREND_UP",
        "STRONG_TREND_DOWN",
        "CHOP_COMPRESSION",
        "VOLATILITY_EXPANSION",
        "MEAN_REVERSION",
    )
    assert snapshot["mark_price"] > 0


@pytest.mark.unit
def test_market_state_build_latest_from_cache(tmp_path, mock_config, mock_eth_klines):
    _short_technicals_config(mock_config)
    db_path = str(tmp_path / "test.duckdb")
    mock_config["data"]["storage"] = {"db_path": db_path}
    cache = DuckDBCacheManager(db_path=db_path)
    cache.insert_ohlcv(_to_cache_df(mock_eth_klines, "ETHUSDC"))
    btc = mock_eth_klines.copy()
    btc["close"] = btc["close"] * 20
    cache.insert_ohlcv(_to_cache_df(btc, "BTCUSDC"))

    svc = MarketStateService(mock_config, cache=cache)
    snapshot = svc.build_latest()
    assert snapshot is not None
    assert len(snapshot["state_vector"]) == 10
    cache.close()
