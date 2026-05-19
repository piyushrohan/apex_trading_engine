import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional

import aiohttp

from src.data.binance_rest import BinanceRESTClient

logger = logging.getLogger(__name__)


def parse_position_legs(positions: list) -> Dict[str, Dict[str, Any]]:
    """
    Normalize Binance positionRisk or ACCOUNT_UPDATE legs into long/short qty.
    Hedge mode uses positionSide LONG/SHORT; one-way uses BOTH signed amount.
    """
    legs = {
        "long_qty": 0.0,
        "short_qty": 0.0,
        "entry_price_long": 0.0,
        "entry_price_short": 0.0,
        "unrealized_pnl": 0.0,
    }
    for pos in positions:
        symbol = pos.get("symbol") or pos.get("s")
        if not symbol:
            continue
        amt = float(pos.get("positionAmt", pos.get("pa", 0)))
        entry = float(pos.get("entryPrice", pos.get("ep", 0)))
        upnl = float(pos.get("unRealizedProfit", pos.get("up", 0)))
        side = (pos.get("positionSide") or pos.get("ps") or "BOTH").upper()

        legs["unrealized_pnl"] += upnl
        if side == "LONG":
            legs["long_qty"] = abs(amt)
            legs["entry_price_long"] = entry
        elif side == "SHORT":
            legs["short_qty"] = abs(amt)
            legs["entry_price_short"] = entry
        else:
            if amt >= 0:
                legs["long_qty"] = abs(amt)
                legs["entry_price_long"] = entry
            else:
                legs["short_qty"] = abs(amt)
                legs["entry_price_short"] = entry
    return legs


class AccountSynchronizer:
    """
    Live account reconciliation via Binance User Data Stream.
    Tracks LONG and SHORT legs separately in hedge mode.
    """

    WS_URL = "wss://fstream.binance.com/ws/"

    def __init__(self, rest_client: BinanceRESTClient):
        self.rest_client = rest_client
        self.listen_key: Optional[str] = None
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.balances: Dict[str, dict] = {}
        self.open_orders: Dict[str, dict] = {}

        self.on_position_change: Optional[Callable] = None
        self.on_order_update: Optional[Callable] = None

        self._ws_task = None
        self._keepalive_task = None
        self._running = False

    async def fetch_snapshot(self, symbol: str) -> Dict[str, Any]:
        """REST bootstrap of positions and USDC wallet before WS events arrive."""
        raw = await self.rest_client.get_positions(symbol)
        active = [p for p in raw if abs(float(p.get("positionAmt", 0))) > 1e-12]
        legs = parse_position_legs(active if active else raw)
        legs["symbol"] = symbol
        legs["amount"] = legs["long_qty"] - legs["short_qty"]
        self.positions[symbol] = legs
        return legs

    def get_leg_snapshot(self, symbol: str) -> Dict[str, Any]:
        return dict(self.positions.get(symbol, {}))

    async def start(self):
        self.listen_key = await self.rest_client.get_listen_key()
        if not self.listen_key:
            logger.error(
                "Could not start AccountSynchronizer. Failed to obtain listen_key."
            )
            return

        self._running = True
        self._ws_task = asyncio.create_task(self._websocket_loop())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        logger.info("AccountSynchronizer started.")

    async def stop(self):
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
        while self._running:
            await asyncio.sleep(45 * 60)
            if self.listen_key:
                logger.info("Sending keep-alive for User Data Stream listen_key...")
                await self.rest_client.keepalive_listen_key()

    async def _websocket_loop(self):
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
                            elif msg.type in (
                                aiohttp.WSMsgType.CLOSED,
                                aiohttp.WSMsgType.ERROR,
                            ):
                                break

            except Exception as e:
                logger.error(f"User Data Stream error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

            if self._running:
                logger.info("Refreshing listen key before reconnect...")
                self.listen_key = await self.rest_client.get_listen_key()

    async def _handle_event(self, data: dict):
        event_type = data.get("e")

        if event_type == "ACCOUNT_UPDATE":
            self._handle_account_update(data)
        elif event_type == "ORDER_TRADE_UPDATE":
            self._handle_order_update(data)
        elif event_type == "listenKeyExpired":
            logger.warning("Listen key expired! Will reconnect immediately.")

    def _handle_account_update(self, data: dict):
        update = data.get("a", {})

        for bal in update.get("B", []):
            asset = bal["a"]
            self.balances[asset] = {
                "wallet_balance": float(bal["wb"]),
                "cross_wallet_balance": float(bal["cw"]),
                "balance_change": float(bal["bc"]),
            }

        symbols_touched = set()
        for pos in update.get("P", []):
            symbol = pos["s"]
            amt = float(pos["pa"])
            entry_price = float(pos["ep"])
            unrealized_pnl = float(pos["up"])
            position_side = (pos.get("ps") or "BOTH").upper()

            existing = self.positions.get(symbol, {})
            long_qty = existing.get("long_qty", 0.0)
            short_qty = existing.get("short_qty", 0.0)
            entry_long = existing.get("entry_price_long", 0.0)
            entry_short = existing.get("entry_price_short", 0.0)

            if position_side == "LONG":
                long_qty = abs(amt)
                entry_long = entry_price
                if abs(amt) < 1e-12:
                    long_qty = 0.0
                    entry_long = 0.0
            elif position_side == "SHORT":
                short_qty = abs(amt)
                entry_short = entry_price
                if abs(amt) < 1e-12:
                    short_qty = 0.0
                    entry_short = 0.0
            else:
                if amt >= 0:
                    long_qty, short_qty = abs(amt), 0.0
                    entry_long, entry_short = entry_price, 0.0
                else:
                    long_qty, short_qty = 0.0, abs(amt)
                    entry_long, entry_short = 0.0, entry_price

            snapshot = {
                "long_qty": long_qty,
                "short_qty": short_qty,
                "entry_price_long": entry_long,
                "entry_price_short": entry_short,
                "unrealized_pnl": unrealized_pnl,
                "amount": long_qty - short_qty,
                "margin_type": pos.get("mt"),
                "position_side": position_side,
            }
            self.positions[symbol] = snapshot
            symbols_touched.add(symbol)

            logger.info(
                f"Position Update {symbol} [{position_side}]: "
                f"long={long_qty} short={short_qty} UPnL={unrealized_pnl}"
            )

        for symbol in symbols_touched:
            if self.on_position_change:
                self.on_position_change(symbol, self.positions[symbol])

    def _handle_order_update(self, data: dict):
        order = data.get("o", {})
        symbol = order["s"]
        order_id = order.get("i") or order.get("c")
        status = order["X"]
        filled_qty = float(order["z"])
        price = float(order["p"])

        self.open_orders[str(order_id)] = {
            "symbol": symbol,
            "status": status,
            "filled_qty": filled_qty,
            "price": price,
            "side": order["S"],
            "type": order["o"],
            "position_side": order.get("ps"),
        }

        logger.info(
            f"Order Update [{status}] for {symbol}: "
            f"ID={order_id}, Filled={filled_qty} @ {price}"
        )

        if self.on_order_update:
            self.on_order_update(str(order_id), self.open_orders[str(order_id)])
