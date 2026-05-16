import asyncio
import json
import logging
from typing import Callable, Optional
import aiohttp

from src.data.binance_rest import BinanceRESTClient

logger = logging.getLogger(__name__)

class AccountSynchronizer:
    """
    Live account reconciliation via Binance User Data Stream.
    Listens for ACCOUNT_UPDATE and ORDER_TRADE_UPDATE to synchronize 
    internal AI state with actual Binance account state. This handles 
    manual discretionary interventions seamlessly.
    """
    
    WS_URL = "wss://fstream.binance.com/ws/"
    
    def __init__(self, rest_client: BinanceRESTClient):
        self.rest_client = rest_client
        self.listen_key: Optional[str] = None
        self.positions = {}
        self.balances = {}
        self.open_orders = {}
        
        # Callbacks for when position changes
        self.on_position_change: Optional[Callable] = None
        self.on_order_update: Optional[Callable] = None
        
        self._ws_task = None
        self._keepalive_task = None
        self._running = False

    async def start(self):
        """Starts the User Data Stream connection and keep-alive loop."""
        self.listen_key = await self.rest_client.get_listen_key()
        if not self.listen_key:
            logger.error("Could not start AccountSynchronizer. Failed to obtain listen_key.")
            return

        self._running = True
        self._ws_task = asyncio.create_task(self._websocket_loop())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        logger.info("AccountSynchronizer started.")

    async def stop(self):
        """Stops the synchronization loops and closes the listen key."""
        self._running = False
        if self._ws_task:
            self._ws_task.cancel()
        if self._keepalive_task:
            self._keepalive_task.cancel()
            
        if self.listen_key:
            await self.rest_client.close_listen_key()
            self.listen_key = None
            logger.info("AccountSynchronizer stopped.")

    async def _keepalive_loop(self):
        """Pings the REST API every 45 minutes to keep the listen_key active."""
        while self._running:
            await asyncio.sleep(45 * 60) # 45 minutes
            if self.listen_key:
                logger.info("Sending keep-alive for User Data Stream listen_key...")
                await self.rest_client.keepalive_listen_key()

    async def _websocket_loop(self):
        """Main WebSocket loop for receiving account updates."""
        ws_endpoint = f"{self.WS_URL}{self.listen_key}"
        
        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(ws_endpoint) as ws:
                        logger.info("Connected to Binance User Data Stream.")
                        
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                await self._handle_event(data)
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
                                
            except Exception as e:
                logger.error(f"User Data Stream error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)
                
            if self._running:
                # Refresh listen key on disconnect
                logger.info("Refreshing listen key before reconnect...")
                self.listen_key = await self.rest_client.get_listen_key()

    async def _handle_event(self, data: dict):
        """Routes the WS event to the appropriate handler."""
        event_type = data.get("e")
        
        if event_type == "ACCOUNT_UPDATE":
            self._handle_account_update(data)
        elif event_type == "ORDER_TRADE_UPDATE":
            self._handle_order_update(data)
        elif event_type == "listenKeyExpired":
            logger.warning("Listen key expired! Will reconnect immediately.")
            # Breaking out of the current WS connection will trigger the reconnect loop

    def _handle_account_update(self, data: dict):
        """Parses balance and position updates."""
        update = data.get("a", {})
        
        # Update Balances
        for bal in update.get("B", []):
            asset = bal["a"]
            self.balances[asset] = {
                "wallet_balance": float(bal["wb"]),
                "cross_wallet_balance": float(bal["cw"]),
                "balance_change": float(bal["bc"])
            }
            
        # Update Positions
        for pos in update.get("P", []):
            symbol = pos["s"]
            amt = float(pos["pa"])
            entry_price = float(pos["ep"])
            unrealized_pnl = float(pos["up"])
            
            self.positions[symbol] = {
                "amount": amt,
                "entry_price": entry_price,
                "unrealized_pnl": unrealized_pnl,
                "margin_type": pos["mt"]
            }
            
            logger.info(f"Position Update for {symbol}: Amount={amt}, Entry={entry_price}, UPnL={unrealized_pnl}")
            
            if self.on_position_change:
                self.on_position_change(symbol, self.positions[symbol])

    def _handle_order_update(self, data: dict):
        """Parses order execution updates."""
        order = data.get("o", {})
        symbol = order["s"]
        order_id = order["c"] # Client order ID
        status = order["X"] # NEW, FILLED, PARTIALLY_FILLED, CANCELED
        filled_qty = float(order["z"])
        price = float(order["p"])
        
        self.open_orders[order_id] = {
            "symbol": symbol,
            "status": status,
            "filled_qty": filled_qty,
            "price": price,
            "side": order["S"],
            "type": order["o"]
        }
        
        logger.info(f"Order Update [{status}] for {symbol}: ID={order_id}, Filled={filled_qty} @ {price}")
        
        if self.on_order_update:
            self.on_order_update(order_id, self.open_orders[order_id])
