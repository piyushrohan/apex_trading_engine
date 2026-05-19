import asyncio
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

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
