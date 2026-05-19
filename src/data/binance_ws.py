import asyncio
import json
import logging
from typing import Callable, Optional

import websockets

logger = logging.getLogger(__name__)


class BinanceWebSocket:
    """
    Async websocket manager for live Binance USDC-M futures market data.
    Streams: aggTrade, depth, markPrice for target + macro symbols.
    """

    def __init__(
        self,
        config: dict,
        on_message: Optional[Callable[[dict], Optional[dict]]] = None,
    ):
        self.ws_url = config["data"]["urls"]["ws_stream"]
        self.target_symbol = config["data"]["target_symbol"].lower()
        self.macro_symbol = config["data"]["macro_symbol"].lower()
        self.on_message = on_message

    def stream_names(self) -> list[str]:
        return [
            f"{self.target_symbol}@aggTrade",
            f"{self.target_symbol}@depth5@100ms",
            f"{self.target_symbol}@markPrice@1s",
            f"{self.macro_symbol}@aggTrade",
            f"{self.macro_symbol}@markPrice@1s",
        ]

    def build_stream_url(self) -> str:
        return f"{self.ws_url}?streams={'/'.join(self.stream_names())}"

    async def run_until_stopped(self, stop_event: asyncio.Event):
        """Connect with auto-reconnect until stop_event is set."""
        stream_url = self.build_stream_url()
        while not stop_event.is_set():
            try:
                logger.info(f"Connecting to Binance WS: {stream_url}")
                async with websockets.connect(stream_url) as ws:
                    logger.info("Connected successfully. Listening to USDC streams...")
                    while not stop_event.is_set():
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            data = json.loads(msg)
                            self._handle_message(data)
                        except asyncio.TimeoutError:
                            continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"WebSocket connection error: {exc}")
                await asyncio.sleep(5)

    async def connect(self):
        """Legacy single-session connect (used in tests)."""
        stop = asyncio.Event()
        task = asyncio.create_task(self.run_until_stopped(stop))

        try:
            while True:
                await asyncio.sleep(3600)
        except (asyncio.CancelledError, RuntimeError):
            stop.set()
            await task

    def _handle_message(self, data: dict) -> Optional[dict]:
        if self.on_message:
            return self.on_message(data)
        return None
