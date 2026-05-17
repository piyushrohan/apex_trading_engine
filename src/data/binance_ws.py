import json

import websockets

from src.core.logger import get_logger

logger = get_logger("BinanceWS")


class BinanceWebSocket:
    """
    Async websocket manager for live market data.
    Optimized for multi-asset (BTCUSDC/ETHUSDC) streaming.
    """

    def __init__(self, config):
        self.ws_url = config["data"]["urls"]["ws_stream"]
        self.target_symbol = config["data"]["target_symbol"].lower()
        self.macro_symbol = config["data"]["macro_symbol"].lower()

    async def connect(self):
        streams = [
            f"{self.target_symbol}@aggTrade",
            f"{self.target_symbol}@depth5@100ms",
            f"{self.macro_symbol}@aggTrade",
        ]
        stream_url = f"{self.ws_url}?streams={'/'.join(streams)}"

        logger.info(f"Connecting to Binance WS: {stream_url}")

        try:
            async with websockets.connect(stream_url) as ws:
                logger.info("Connected successfully. Listening to USDC streams...")
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    self._handle_message(data)
        except Exception as e:
            logger.error(f"WebSocket Connection Error: {e}")

    def _handle_message(self, data):
        """
        Parses live @aggTrade and @depth updates to reconstruct
        live OHLCV and Orderflow imbalance metrics.
        """
        # Pass to StreamBuffer / Feature Engine
        pass
