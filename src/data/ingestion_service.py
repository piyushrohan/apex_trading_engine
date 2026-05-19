"""
Incremental market data ingestion: REST backfill, gap repair, live WebSocket append.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.binance_rest import BinanceRESTClient
from src.data.binance_ws import BinanceWebSocket
from src.data.cache_manager import DuckDBCacheManager
from src.data.intervals import interval_to_milliseconds

logger = logging.getLogger(__name__)


class DataIngestionService:
    """Orchestrates DuckDB cache updates from Binance REST and WebSocket."""

    def __init__(
        self,
        config: Dict[str, Any],
        rest_client: Optional[BinanceRESTClient] = None,
        cache: Optional[DuckDBCacheManager] = None,
    ):
        self.config = config
        data_cfg = config.get("data", {})
        self.target_symbol = data_cfg.get("target_symbol", "ETHUSDC")
        self.macro_symbol = data_cfg.get("macro_symbol", "BTCUSDC")
        self.target_interval = data_cfg.get("target_interval", "3m")

        ingest_cfg = data_cfg.get("ingestion", {})
        self.enabled = ingest_cfg.get("enabled", True)
        self.initial_backfill_days = ingest_cfg.get("initial_backfill_days", 7)
        self.tick_flush_size = ingest_cfg.get("tick_flush_size", 100)
        self.funding_poll_sec = ingest_cfg.get("funding_poll_sec", 300)
        self.repair_gaps = ingest_cfg.get("repair_gaps", True)

        db_path = data_cfg.get("storage", {}).get(
            "db_path", "data_lake/apex_market_data.duckdb"
        )
        self.rest = rest_client or BinanceRESTClient()
        self.cache = cache or DuckDBCacheManager(db_path=db_path)

        self._tick_buffer: List[dict] = []
        self._stop_event = asyncio.Event()
        self._ws_task: Optional[asyncio.Task] = None
        self._funding_task: Optional[asyncio.Task] = None
        self._last_mark: Dict[str, float] = {}
        self._last_depth: Dict[str, dict] = {}

    @property
    def symbols(self) -> List[str]:
        return [self.target_symbol, self.macro_symbol]

    async def bootstrap_historical(self) -> Dict[str, int]:
        """Incremental REST sync for target + macro symbols."""
        if not self.enabled:
            return {}

        totals = {}
        for symbol in self.symbols:
            rows = await self.sync_symbol_ohlcv(symbol, self.target_interval)
            totals[f"{symbol}_{self.target_interval}"] = rows
        await self.poll_funding_and_oi()
        return totals

    async def sync_symbol_ohlcv(self, symbol: str, interval: str) -> int:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        step_ms = interval_to_milliseconds(interval)
        latest = self.cache.get_latest_timestamp(symbol, interval)

        if latest is None:
            start_ms = now_ms - (self.initial_backfill_days * 24 * 60 * 60 * 1000)
            logger.info(
                f"No cache for {symbol} {interval}; backfilling "
                f"{self.initial_backfill_days} days"
            )
        else:
            start_ms = int(latest.timestamp() * 1000) + step_ms
            logger.info(f"Incremental backfill {symbol} {interval} from {latest}")

        if start_ms >= now_ms:
            logger.info(f"{symbol} {interval} cache is up to date")
            inserted = 0
        else:
            df = await self.rest.backfill_historical_data(
                symbol, interval, start_ms, now_ms
            )
            self.cache.insert_ohlcv(df)
            inserted = len(df)

        if self.repair_gaps:
            inserted += await self.repair_ohlcv_gaps(symbol, interval)

        return inserted

    async def repair_ohlcv_gaps(self, symbol: str, interval: str) -> int:
        gaps = self.cache.detect_ohlcv_gaps(symbol, interval)
        if not gaps:
            return 0

        total = 0
        for gap_start, gap_end in gaps:
            start_ms = int(gap_start.timestamp() * 1000)
            end_ms = int(gap_end.timestamp() * 1000)
            if start_ms >= end_ms:
                continue
            logger.info(
                f"Repairing OHLCV gap for {symbol} {interval}: "
                f"{gap_start} -> {gap_end}"
            )
            df = await self.rest.backfill_historical_data(
                symbol, interval, start_ms, end_ms
            )
            self.cache.insert_ohlcv(df)
            total += len(df)
        return total

    async def start_live(self):
        """Start WebSocket ingestion and optional funding/OI polling."""
        if not self.enabled:
            return

        self._stop_event.clear()
        ws = BinanceWebSocket(self.config, on_message=self.handle_ws_message)
        self._ws_task = asyncio.create_task(ws.run_until_stopped(self._stop_event))
        self._funding_task = asyncio.create_task(self._funding_poll_loop())
        logger.info("Live data ingestion started")

    async def stop(self):
        self._stop_event.set()
        for task in (self._ws_task, self._funding_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await self.flush_ticks()
        logger.info("Live data ingestion stopped")

    async def _funding_poll_loop(self):
        while not self._stop_event.is_set():
            try:
                await self.poll_funding_and_oi()
            except Exception as exc:
                logger.error(f"Funding/OI poll error: {exc}")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.funding_poll_sec
                )
                break
            except asyncio.TimeoutError:
                continue

    async def poll_funding_and_oi(self):
        ts = pd.Timestamp(datetime.now(timezone.utc))
        for symbol in self.symbols:
            premium = await self.rest.fetch_premium_index(symbol)
            oi = await self.rest.fetch_open_interest(symbol)
            funding = None
            mark = self._last_mark.get(symbol)
            if premium:
                funding = float(premium.get("lastFundingRate", 0))
                mark = float(premium.get("markPrice", mark or 0))
            self.cache.insert_market_snapshot(
                symbol=symbol,
                timestamp=ts,
                funding_rate=funding,
                open_interest=oi,
                mark_price=mark,
            )

    def handle_ws_message(self, payload: dict) -> Optional[dict]:
        """Parse combined stream payloads and append to cache buffers."""
        stream = payload.get("stream", "")
        data = payload.get("data", payload)

        if "@aggTrade" in stream:
            return self._handle_agg_trade(data)
        if "@markPrice" in stream:
            return self._handle_mark_price(data)
        if "@depth" in stream:
            return self._handle_depth(stream, data)
        return None

    def _handle_agg_trade(self, data: dict) -> dict:
        symbol = data.get("s", "").upper()
        tick = {
            "timestamp": pd.to_datetime(int(data["T"]), unit="ms", utc=True),
            "symbol": symbol,
            "price": float(data["p"]),
            "quantity": float(data["q"]),
            "is_buyer_maker": bool(data.get("m", False)),
            "trade_id": int(data["a"]),
        }
        self._tick_buffer.append(tick)
        if len(self._tick_buffer) >= self.tick_flush_size:
            df = pd.DataFrame(self._tick_buffer)
            self._tick_buffer.clear()
            self.cache.insert_ticks(df)
        return tick

    def _handle_mark_price(self, data: dict) -> dict:
        symbol = data.get("s", "").upper()
        price = float(data.get("p", 0))
        self._last_mark[symbol] = price
        return {"symbol": symbol, "mark_price": price}

    def _handle_depth(self, stream: str, data: dict) -> dict:
        symbol = stream.split("@")[0].upper()
        self._last_depth[symbol] = data
        return {"symbol": symbol, "depth": data}

    async def flush_ticks(self) -> int:
        if not self._tick_buffer:
            return 0
        df = pd.DataFrame(self._tick_buffer)
        self._tick_buffer.clear()
        return self.cache.insert_ticks(df)

    def get_last_mark_price(self, symbol: str) -> Optional[float]:
        return self._last_mark.get(symbol)

    def get_last_depth(self, symbol: str) -> Optional[dict]:
        return self._last_depth.get(symbol)

    def close(self):
        self.cache.close()


async def _main():
    import logging

    from src.core.config_loader import load_config

    logging.basicConfig(level=logging.INFO)
    config = load_config()
    service = DataIngestionService(config)
    try:
        totals = await service.bootstrap_historical()
        logger.info(f"Bootstrap complete: {totals}")
        await service.start_live()
        await asyncio.sleep(10)
    finally:
        await service.stop()
        service.close()


if __name__ == "__main__":
    asyncio.run(_main())
