import asyncio
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.data import ingestion_service
from src.data.ingestion_service import DataIngestionService


@pytest.fixture
def ingest_config():
    return {
        "data": {
            "target_symbol": "ETHUSDC",
            "macro_symbol": "BTCUSDC",
            "target_interval": "3m",
            "ingestion": {
                "enabled": True,
                "initial_backfill_days": 1,
                "tick_flush_size": 2,
                "funding_poll_sec": 60,
                "repair_gaps": False,
            },
            "storage": {"db_path": ":memory:"},
        }
    }


@pytest.mark.unit
def test_handle_ws_agg_trade_buffers_and_flushes(ingest_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_path = str(tmp_path / "apex.duckdb")
    ingest_config["data"]["storage"]["db_path"] = db_path

    service = DataIngestionService(ingest_config)
    payload = {
        "stream": "ethusdc@aggTrade",
        "data": {
            "s": "ETHUSDC",
            "p": "3500.5",
            "q": "0.1",
            "m": False,
            "a": 1,
            "T": 1715857200000,
        },
    }
    service.handle_ws_message(payload)
    payload["data"]["a"] = 2
    service.handle_ws_message(payload)

    count = service.cache.conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
    service.close()
    assert count == 2


@pytest.mark.asyncio
@pytest.mark.unit
async def test_sync_symbol_ohlcv_incremental(ingest_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ingest_config["data"]["storage"]["db_path"] = str(tmp_path / "apex.duckdb")

    rest = AsyncMock()
    rest.backfill_historical_data = AsyncMock(
        return_value=pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp("2026-05-16 12:00:00"),
                    "symbol": "ETHUSDC",
                    "timeframe": "3m",
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "volume": 1.0,
                }
            ]
        )
    )
    rest.fetch_premium_index = AsyncMock(
        return_value={"lastFundingRate": "0.0001", "markPrice": "3500"}
    )
    rest.fetch_open_interest = AsyncMock(return_value=1000.0)

    service = DataIngestionService(ingest_config, rest_client=rest)
    rows = await service.sync_symbol_ohlcv("ETHUSDC", "3m")
    service.close()

    assert rows == 1
    rest.backfill_historical_data.assert_awaited()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_start_and_stop_live_tasks(ingest_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ingest_config["data"]["storage"]["db_path"] = str(tmp_path / "apex.duckdb")
    ingest_config["data"]["ingestion"]["funding_poll_sec"] = 1

    service = DataIngestionService(ingest_config)
    service.poll_funding_and_oi = AsyncMock()

    async def fake_run(stop_event):
        await stop_event.wait()

    monkeypatch.setattr(
        "src.data.ingestion_service.BinanceWebSocket",
        lambda config, on_message=None: MagicMock(run_until_stopped=fake_run),
    )

    await service.start_live()
    await asyncio.sleep(0.05)
    await service.stop()
    service.close()

    service.poll_funding_and_oi.assert_awaited()


class FakeCache:
    def __init__(self, *, latest=None, gaps=None):
        self.latest = latest
        self.gaps = list(gaps or [])
        self.ohlcv_batches = []
        self.market_snapshots = []
        self.tick_batches = []
        self.closed = False

    def get_latest_timestamp(self, symbol, interval):
        return self.latest

    def insert_ohlcv(self, df):
        self.ohlcv_batches.append(df)

    def detect_ohlcv_gaps(self, symbol, interval):
        return self.gaps

    def insert_market_snapshot(self, **kwargs):
        self.market_snapshots.append(kwargs)

    def insert_ticks(self, df):
        self.tick_batches.append(df)
        return len(df)

    def close(self):
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_bootstrap_and_live_start_are_noops_when_disabled(ingest_config):
    ingest_config["data"]["ingestion"]["enabled"] = False
    service = DataIngestionService(
        ingest_config,
        rest_client=AsyncMock(),
        cache=FakeCache(),
    )

    assert service.symbols == ["ETHUSDC", "BTCUSDC"]
    assert await service.bootstrap_historical() == {}
    await service.start_live()

    assert service._ws_task is None
    assert service._funding_task is None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_bootstrap_historical_syncs_symbols_and_polls_funding(ingest_config):
    service = DataIngestionService(
        ingest_config,
        rest_client=AsyncMock(),
        cache=FakeCache(),
    )
    service.sync_symbol_ohlcv = AsyncMock(side_effect=[2, 3])
    service.poll_funding_and_oi = AsyncMock()

    totals = await service.bootstrap_historical()

    assert totals == {"ETHUSDC_3m": 2, "BTCUSDC_3m": 3}
    service.poll_funding_and_oi.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_sync_symbol_skips_when_cache_is_current(ingest_config):
    rest = AsyncMock()
    latest = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
    service = DataIngestionService(
        ingest_config,
        rest_client=rest,
        cache=FakeCache(latest=latest),
    )
    service.repair_gaps = False

    rows = await service.sync_symbol_ohlcv("ETHUSDC", "3m")

    assert rows == 0
    rest.backfill_historical_data.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_repair_ohlcv_gaps_skips_invalid_ranges_and_inserts_valid(
    ingest_config,
):
    rest = AsyncMock()
    rest.backfill_historical_data.return_value = pd.DataFrame(
        [{"timestamp": pd.Timestamp("2026-05-16"), "close": 1.0}]
    )
    cache = FakeCache(
        gaps=[
            (
                pd.Timestamp("2026-05-16 12:00:00"),
                pd.Timestamp("2026-05-16 12:00:00"),
            ),
            (
                pd.Timestamp("2026-05-16 12:03:00"),
                pd.Timestamp("2026-05-16 12:06:00"),
            ),
        ]
    )
    service = DataIngestionService(ingest_config, rest_client=rest, cache=cache)

    assert await service.repair_ohlcv_gaps("ETHUSDC", "3m") == 1
    assert len(cache.ohlcv_batches) == 1

    cache.gaps = []
    assert await service.repair_ohlcv_gaps("ETHUSDC", "3m") == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_funding_poll_loop_recovers_and_waits_for_stop(
    ingest_config, monkeypatch
):
    service = DataIngestionService(
        ingest_config,
        rest_client=AsyncMock(),
        cache=FakeCache(),
    )
    service.poll_funding_and_oi = AsyncMock(side_effect=[RuntimeError("boom"), None])
    waits = {"count": 0}

    async def fake_wait_for(awaitable, timeout):
        awaitable.close()
        waits["count"] += 1
        if waits["count"] == 1:
            raise asyncio.TimeoutError
        service._stop_event.set()
        return True

    monkeypatch.setattr(ingestion_service.asyncio, "wait_for", fake_wait_for)

    await service._funding_poll_loop()

    assert service.poll_funding_and_oi.await_count == 2


@pytest.mark.asyncio
@pytest.mark.unit
async def test_poll_funding_and_oi_uses_premium_and_cached_mark(ingest_config):
    rest = AsyncMock()
    rest.fetch_premium_index = AsyncMock(
        side_effect=[
            {"lastFundingRate": "0.0002", "markPrice": "3500.5"},
            None,
        ]
    )
    rest.fetch_open_interest = AsyncMock(side_effect=[1234.0, None])
    cache = FakeCache()
    service = DataIngestionService(ingest_config, rest_client=rest, cache=cache)
    service._last_mark["BTCUSDC"] = 71000.0

    await service.poll_funding_and_oi()

    assert cache.market_snapshots[0]["funding_rate"] == 0.0002
    assert cache.market_snapshots[0]["mark_price"] == 3500.5
    assert cache.market_snapshots[1]["funding_rate"] is None
    assert cache.market_snapshots[1]["mark_price"] == 71000.0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_ws_mark_depth_flush_getters_and_close(ingest_config):
    cache = FakeCache()
    service = DataIngestionService(ingest_config, rest_client=AsyncMock(), cache=cache)

    assert service.handle_ws_message({"stream": "ethusdc@unknown", "data": {}}) is None
    assert service.handle_ws_message(
        {"stream": "ethusdc@markPrice", "data": {"s": "ETHUSDC", "p": "3501.25"}}
    ) == {"symbol": "ETHUSDC", "mark_price": 3501.25}
    depth = {"b": [["3500", "1"]], "a": [["3502", "2"]]}
    assert (
        service.handle_ws_message({"stream": "ethusdc@depth5@100ms", "data": depth})[
            "depth"
        ]
        == depth
    )

    service._tick_buffer.append(
        {
            "timestamp": pd.Timestamp("2026-05-16", tz="UTC"),
            "symbol": "ETHUSDC",
            "price": 1.0,
            "quantity": 1.0,
            "is_buyer_maker": False,
            "trade_id": 1,
        }
    )

    assert await service.flush_ticks() == 1
    assert await service.flush_ticks() == 0
    assert service.get_last_mark_price("ETHUSDC") == 3501.25
    assert service.get_last_depth("ETHUSDC") == depth
    service.close()
    assert cache.closed is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_ingestion_main_starts_and_cleans_up(monkeypatch):
    events = []

    class FakeService:
        def __init__(self, config):
            events.append(("init", config))

        async def bootstrap_historical(self):
            events.append(("bootstrap", None))
            return {"ETHUSDC_3m": 1}

        async def start_live(self):
            events.append(("start", None))

        async def stop(self):
            events.append(("stop", None))

        def close(self):
            events.append(("close", None))

    monkeypatch.setattr("src.core.config_loader.load_config", lambda: {"data": {}})
    monkeypatch.setattr(ingestion_service, "DataIngestionService", FakeService)
    monkeypatch.setattr(ingestion_service.asyncio, "sleep", AsyncMock())

    await ingestion_service._main()

    assert [event[0] for event in events] == [
        "init",
        "bootstrap",
        "start",
        "stop",
        "close",
    ]
