import logging
import uuid
from typing import Any, Dict, List, Optional

from src.execution.adapters.base import ExecutionAdapter, OrderRequest, OrderResult

logger = logging.getLogger(__name__)


class PaperExecutionAdapter(ExecutionAdapter):
    """
      Simulated maker-only (GTX) execution against a virtual book.
    Multiple adapter instances can share one logical book via book_id.
    """

    def __init__(
        self,
        book_id: str = "primary",
        tick_size: float = 0.01,
        maker_fee_pct: float = 0.0,
    ):
        self.book_id = book_id
        self.tick_size = tick_size
        self.maker_fee_pct = maker_fee_pct
        self._open_orders: Dict[str, Dict[str, Any]] = {}
        self._fills: List[Dict[str, Any]] = []

    async def place_order(self, request: OrderRequest) -> OrderResult:
        if request.time_in_force != "GTX":
            return OrderResult(
                success=False,
                error="Paper adapter enforces GTX (post-only) only",
            )

        order_id = request.client_order_id or f"paper_{uuid.uuid4().hex[:12]}"
        record = {
            "orderId": order_id,
            "symbol": request.symbol,
            "side": request.side.upper(),
            "price": request.price,
            "origQty": request.quantity,
            "executedQty": 0.0,
            "status": "NEW",
            "timeInForce": "GTX",
            "positionSide": request.position_side or "BOTH",
            "book_id": self.book_id,
        }
        self._open_orders[order_id] = record
        logger.info(
            f"[PAPER:{self.book_id}] Placed {record['side']} "
            f"{request.quantity} @ {request.price} ({order_id})"
        )
        return OrderResult(success=True, order_id=order_id, status="NEW", raw=record)

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        if order_id in self._open_orders:
            self._open_orders[order_id]["status"] = "CANCELED"
            del self._open_orders[order_id]
            return True
        return False

    async def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        return [
            o
            for o in self._open_orders.values()
            if o.get("symbol") == symbol and o.get("status") == "NEW"
        ]

    async def sync_fills(self, symbol: str) -> List[Dict[str, Any]]:
        return [f for f in self._fills if f.get("symbol") == symbol]

    def try_fill_on_market(
        self,
        symbol: str,
        market_price: float,
        aggressor_side: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Maker fill simulation: BUY limit fills if market trades at or below price;
        SELL limit fills if market trades at or above price.
        """
        filled = []
        for order_id, order in list(self._open_orders.items()):
            if order["symbol"] != symbol or order["status"] != "NEW":
                continue

            side = order["side"]
            limit_price = order["price"]
            should_fill = False
            if side == "BUY" and market_price <= limit_price:
                should_fill = True
            elif side == "SELL" and market_price >= limit_price:
                should_fill = True

            if not should_fill:
                continue

            qty = order["origQty"]
            fee = qty * limit_price * self.maker_fee_pct
            fill = {
                **order,
                "status": "FILLED",
                "executedQty": qty,
                "avgPrice": limit_price,
                "fee": fee,
                "fill_price": market_price,
                "aggressor_side": aggressor_side,
            }
            self._fills.append(fill)
            del self._open_orders[order_id]
            filled.append(fill)
            logger.debug(f"[PAPER:{self.book_id}] Filled {order_id} @ {limit_price}")

        return filled

    def flatten_all_virtual_orders(self, symbol: str) -> int:
        """Cancel all resting virtual orders (kill-switch helper)."""
        to_cancel = [
            oid for oid, o in self._open_orders.items() if o.get("symbol") == symbol
        ]
        for oid in to_cancel:
            self._open_orders[oid]["status"] = "CANCELED"
            del self._open_orders[oid]
        return len(to_cancel)
