import pandas as pd
import pytest

from src.data.cache_manager import DuckDBCacheManager


@pytest.mark.integration
def test_insert_ticks_deduplicates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = DuckDBCacheManager(db_path=str(tmp_path / "apex.duckdb"))
    df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-05-16 12:00:00"),
                "symbol": "ETHUSDC",
                "price": 3000.0,
                "quantity": 1.0,
                "is_buyer_maker": False,
                "trade_id": 1001,
            }
        ]
    )
    assert cache.insert_ticks(df) == 1
    assert cache.insert_ticks(df) == 0
    latest = cache.get_latest_tick_timestamp("ETHUSDC")
    cache.close()
    assert pd.Timestamp(latest) == pd.Timestamp("2026-05-16 12:00:00")


@pytest.mark.integration
def test_detect_ohlcv_gaps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = DuckDBCacheManager(db_path=str(tmp_path / "apex.duckdb"))
    rows = [
        {
            "timestamp": pd.Timestamp("2026-05-16 12:00:00"),
            "symbol": "ETHUSDC",
            "timeframe": "3m",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        },
        {
            "timestamp": pd.Timestamp("2026-05-16 12:09:00"),
            "symbol": "ETHUSDC",
            "timeframe": "3m",
            "open": 2.0,
            "high": 2.0,
            "low": 2.0,
            "close": 2.0,
            "volume": 2.0,
        },
    ]
    cache.insert_ohlcv(pd.DataFrame(rows))
    gaps = cache.detect_ohlcv_gaps("ETHUSDC", "3m")
    cache.close()
    assert len(gaps) == 1
    assert gaps[0][0] == pd.Timestamp("2026-05-16 12:03:00")
    assert gaps[0][1] == pd.Timestamp("2026-05-16 12:06:00")
