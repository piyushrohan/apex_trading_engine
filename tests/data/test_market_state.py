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


class FakeMarketCache:
    def __init__(self, eth_df=None, btc_df=None, row=None, fail_snapshot=False):
        self.eth_df = eth_df if eth_df is not None else pd.DataFrame()
        self.btc_df = btc_df if btc_df is not None else pd.DataFrame()
        self.row = row
        self.fail_snapshot = fail_snapshot
        self.closed = False
        self.insert_features_calls = []
        self.conn = self

    def load_ohlcv(self, symbol, interval):
        return self.eth_df if symbol == "ETHUSDC" else self.btc_df

    def execute(self, query, params):
        if self.fail_snapshot:
            raise RuntimeError("snapshot unavailable")
        return self

    def fetchone(self):
        return self.row

    def insert_features(self, df, feature_set_id):
        self.insert_features_calls.append((df, feature_set_id))

    def close(self):
        self.closed = True


@pytest.mark.unit
def test_market_state_empty_inputs_and_snapshot_fallbacks(mock_config, mock_eth_klines):
    svc = MarketStateService(mock_config, cache=FakeMarketCache())
    assert svc.build_latest() is None

    cache = FakeMarketCache(eth_df=_to_cache_df(mock_eth_klines, "ETHUSDC"))
    svc = MarketStateService(mock_config, cache=cache)
    svc.feature_engine.process_all_features = lambda eth, btc: pd.DataFrame()
    assert svc.build_latest() is None

    assert svc._latest_market_snapshot() == {
        "funding_rate": 0.0,
        "open_interest": 0.0,
        "mark_price": 0.0,
    }

    svc.cache = FakeMarketCache(fail_snapshot=True)
    assert svc._latest_market_snapshot() == {
        "funding_rate": 0.0,
        "open_interest": 0.0,
        "mark_price": 0.0,
    }


@pytest.mark.unit
def test_market_state_latest_snapshot_persist_exception_and_owned_close(mock_config):
    cache = FakeMarketCache(row=(0.0001, 1234.0, 3500.0))
    svc = MarketStateService(mock_config, cache=cache)

    assert svc._latest_market_snapshot() == {
        "funding_rate": 0.0001,
        "open_interest": 1234.0,
        "mark_price": 3500.0,
    }

    def fail_insert(df, feature_set_id):
        raise RuntimeError("read-only")

    svc.cache.insert_features = fail_insert
    svc._persist_latest_features(pd.Series({"timestamp": pd.Timestamp("2026-05-16")}))
    svc._owns_cache = True
    svc.close()
    assert cache.closed is True
