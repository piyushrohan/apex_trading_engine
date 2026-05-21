import logging
import uuid
from typing import Any, Dict, List, Optional

from src.execution.adapters.base import ExecutionAdapter, OrderRequest, OrderResult
from src.execution.order_lifecycle import OrderLifecycleRecorder

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
        lifecycle_recorder: Optional[OrderLifecycleRecorder] = None,
    ):
        self.book_id = book_id
        self.tick_size = tick_size
        self.maker_fee_pct = maker_fee_pct
        self.lifecycle_recorder = lifecycle_recorder
        self._open_orders: Dict[str, Dict[str, Any]] = {}
        self._fills: List[Dict[str, Any]] = []

    async def place_order(self, request: OrderRequest) -> OrderResult:
        if request.time_in_force != "GTX":
            return OrderResult(
                success=False,
                error="Paper adapter enforces GTX (post-only) only",
            )

        order_id = request.client_order_id or f"paper_{uuid.uuid4().hex[:12]}"
        self._record_lifecycle(
            "submitted",
            order_id=order_id,
            symbol=request.symbol,
            side=request.side.upper(),
            quantity=request.quantity,
            price=request.price,
            status="PENDING",
            position_side=request.position_side or "BOTH",
            client_order_id=request.client_order_id,
        )
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
        self._record_lifecycle(
            "open",
            order_id=order_id,
            symbol=request.symbol,
            side=record["side"],
            quantity=request.quantity,
            price=request.price,
            status="NEW",
            position_side=record["positionSide"],
            client_order_id=request.client_order_id,
        )
        return OrderResult(success=True, order_id=order_id, status="NEW", raw=record)

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        if order_id in self._open_orders:
            order = self._open_orders[order_id]
            self._open_orders[order_id]["status"] = "CANCELED"
            del self._open_orders[order_id]
            self._record_lifecycle(
                "canceled",
                order_id=order_id,
                symbol=symbol,
                side=order.get("side"),
                quantity=order.get("origQty"),
                price=order.get("price"),
                status="CANCELED",
                position_side=order.get("positionSide"),
            )
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
        available_quantity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Maker fill simulation: BUY limit fills if market trades at or below price;
        SELL limit fills if market trades at or above price.
        If available_quantity is provided, fills are capped and remaining order
        quantity stays open as PARTIALLY_FILLED.
        """
        filled = []
        remaining_liquidity = available_quantity
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

            open_qty = float(order["origQty"]) - float(order.get("executedQty", 0.0))
            if remaining_liquidity is None:
                qty = open_qty
            else:
                if remaining_liquidity <= 0:
                    break
                qty = min(open_qty, remaining_liquidity)
                remaining_liquidity -= qty
            if qty <= 0:
                continue

            fee = qty * limit_price * self.maker_fee_pct
            total_executed = float(order.get("executedQty", 0.0)) + qty
            done = total_executed >= float(order["origQty"]) - 1e-12
            fill = {
                **order,
                "status": "FILLED" if done else "PARTIALLY_FILLED",
                "executedQty": qty,
                "cumulativeExecutedQty": total_executed,
                "avgPrice": limit_price,
                "fee": fee,
                "fill_price": market_price,
                "aggressor_side": aggressor_side,
            }
            self._fills.append(fill)
            self._record_lifecycle(
                "filled" if done else "partially_filled",
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=qty,
                price=limit_price,
                status=fill["status"],
                position_side=fill.get("positionSide"),
                fill_price=market_price,
                mark_price_after=market_price,
                metadata={
                    "cumulative_executed_qty": total_executed,
                    "aggressor_side": aggressor_side,
                    "fee": fee,
                },
            )
            if done:
                del self._open_orders[order_id]
            else:
                self._open_orders[order_id]["executedQty"] = total_executed
                self._open_orders[order_id]["status"] = "NEW"
            filled.append(fill)
            logger.debug(f"[PAPER:{self.book_id}] Filled {order_id} @ {limit_price}")

        return filled

    def flatten_all_virtual_orders(self, symbol: str) -> int:
        """Cancel all resting virtual orders (kill-switch helper)."""
        to_cancel = [
            oid for oid, o in self._open_orders.items() if o.get("symbol") == symbol
        ]
        for oid in to_cancel:
            order = self._open_orders[oid]
            self._open_orders[oid]["status"] = "CANCELED"
            del self._open_orders[oid]
            self._record_lifecycle(
                "canceled",
                order_id=oid,
                symbol=symbol,
                side=order.get("side"),
                quantity=order.get("origQty"),
                price=order.get("price"),
                status="CANCELED",
                position_side=order.get("positionSide"),
                reason="flatten_all_virtual_orders",
            )
        return len(to_cancel)

    def _record_lifecycle(self, event: str, **kwargs: Any) -> None:
        if self.lifecycle_recorder is None:
            return
        self.lifecycle_recorder.record(event, **kwargs)
