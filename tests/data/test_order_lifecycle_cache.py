import pandas as pd
import pytest

from src.data.cache_manager import DuckDBCacheManager


@pytest.mark.unit
def test_cache_persists_and_filters_order_lifecycle_events(tmp_path):
    cache = DuckDBCacheManager(str(tmp_path / "orders.duckdb"))
    event = {
        "timestamp": "2026-05-21T00:00:00+00:00",
        "event": "filled",
        "order_id": "o1",
        "symbol": "ETHUSDC",
        "side": "BUY",
        "quantity": 1.5,
        "price": 100.0,
        "status": "FILLED",
        "execution_mode": "paper",
        "book_id": "primary",
        "position_side": "LONG",
        "exchange_order_id": None,
        "client_order_id": "c1",
        "reason": None,
        "queue_age_ms": 25.0,
        "fill_price": 99.9,
        "mark_price_after": 100.2,
        "metadata": {"fee": 0.0},
    }

    cache.insert_order_lifecycle_event(event)
    df = cache.load_order_lifecycle_events(symbol="ETHUSDC", book_id="primary")
    cache.close()

    assert len(df) == 1
    assert pd.Timestamp(df.iloc[0]["timestamp"]).year == 2026
    assert df.iloc[0]["event"] == "filled"
    assert df.iloc[0]["metadata_json"] == '{"fee": 0.0}'
