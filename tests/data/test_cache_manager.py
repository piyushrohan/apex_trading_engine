import pandas as pd
import pytest

from src.data.cache_manager import DuckDBCacheManager


@pytest.mark.integration
def test_duckdb_cache_insert_load_latest_and_backup(tmp_path, monkeypatch):
    """
    Integration Test:
    Verify DuckDB cache persistence, duplicate protection, timestamp lookup,
    and parquet backup wiring.
    """
    monkeypatch.chdir(tmp_path)
    cache = DuckDBCacheManager(db_path=str(tmp_path / "apex.duckdb"))

    empty_df = pd.DataFrame()
    cache.insert_ohlcv(empty_df)

    df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-05-16 12:00:00"),
                "symbol": "ETHUSDC",
                "timeframe": "3m",
                "open": 3000.0,
                "high": 3010.0,
                "low": 2990.0,
                "close": 3005.0,
                "volume": 100.0,
            }
        ]
    )

    cache.insert_ohlcv(df)
    cache.insert_ohlcv(df)

    loaded = cache.load_ohlcv("ETHUSDC", "3m")
    latest = cache.get_latest_timestamp("ETHUSDC", "3m")
    missing = cache.get_latest_timestamp("BTCUSDC", "3m")
    filtered = cache.load_ohlcv(
        "ETHUSDC",
        "3m",
        start_time="2026-05-16 11:59:00",
        end_time="2026-05-16 12:01:00",
    )
    cache.backup_to_parquet()
    cache.close()

    assert len(loaded) == 1
    assert len(filtered) == 1
    assert latest == pd.Timestamp("2026-05-16 12:00:00")
    assert missing is None
    assert (tmp_path / "data_lake" / "ohlcv").exists()


@pytest.mark.integration
def test_duckdb_cache_logs_insert_and_backup_failures():
    """Verify cache write helpers swallow storage-layer exceptions."""

    class FailingConnection:
        def execute(self, query):
            raise RuntimeError("duckdb unavailable")

    cache = DuckDBCacheManager.__new__(DuckDBCacheManager)
    cache.conn = FailingConnection()
    df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-05-16 12:00:00"),
                "symbol": "ETHUSDC",
                "timeframe": "3m",
                "open": 3000.0,
                "high": 3010.0,
                "low": 2990.0,
                "close": 3005.0,
                "volume": 100.0,
            }
        ]
    )

    cache.insert_ohlcv(df)
    cache.backup_to_parquet()
