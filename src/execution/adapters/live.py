import logging
from typing import Any, Dict, List

from src.data.binance_rest import BinanceRESTClient
from src.execution.adapters.base import ExecutionAdapter, OrderRequest, OrderResult
from src.execution.order_lifecycle import OrderLifecycleRecorder

logger = logging.getLogger(__name__)


class LiveExecutionAdapter(ExecutionAdapter):
    """Signed Binance USDC-M execution via REST (GTX post-only)."""

    def __init__(
        self,
        config: dict,
        rest_client: BinanceRESTClient,
        lifecycle_recorder: OrderLifecycleRecorder | None = None,
    ):
        self.config = config
        self.rest_client = rest_client
        self.lifecycle_recorder = lifecycle_recorder
        self.symbol = config.get("data", {}).get("target_symbol", "ETHUSDC")
        self.position_mode = config.get("execution", {}).get("position_mode", "one_way")

    async def place_order(self, request: OrderRequest) -> OrderResult:
        order_key = request.client_order_id or f"live_pending_{request.symbol}"
        self._record_lifecycle(
            "submitted",
            order_id=order_key,
            symbol=request.symbol,
            side=request.side.upper(),
            quantity=request.quantity,
            price=request.price,
            status="PENDING",
            position_side=request.position_side,
            client_order_id=request.client_order_id,
        )
        try:
            raw = await self.rest_client.place_order(
                symbol=request.symbol,
                side=request.side.upper(),
                quantity=request.quantity,
                price=request.price,
                timeInForce=request.time_in_force,
                orderType=request.order_type,
                positionSide=request.position_side,
            )
            if raw is None:
                self._record_lifecycle(
                    "rejected",
                    order_id=order_key,
                    symbol=request.symbol,
                    side=request.side.upper(),
                    quantity=request.quantity,
                    price=request.price,
                    status="REJECTED",
                    reason="place_order returned None",
                )
                return OrderResult(success=False, error="place_order returned None")
            order_id = str(raw.get("orderId", ""))
            self._record_lifecycle(
                "open",
                order_id=order_id or order_key,
                symbol=request.symbol,
                side=request.side.upper(),
                quantity=request.quantity,
                price=request.price,
                status=raw.get("status", "NEW"),
                position_side=request.position_side,
                exchange_order_id=order_id,
                client_order_id=request.client_order_id,
                metadata={"raw": raw},
            )
            return OrderResult(
                success=True,
                order_id=order_id,
                status=raw.get("status", "NEW"),
                filled_qty=float(raw.get("executedQty", 0)),
                raw=raw,
            )
        except Exception as exc:
            logger.error(f"Live place_order failed: {exc}")
            self._record_lifecycle(
                "rejected",
                order_id=order_key,
                symbol=request.symbol,
                side=request.side.upper(),
                quantity=request.quantity,
                price=request.price,
                status="REJECTED",
                position_side=request.position_side,
                reason=str(exc),
            )
            return OrderResult(success=False, error=str(exc))

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        cancelled = await self.rest_client.cancel_order(symbol, order_id)
        if cancelled:
            self._record_lifecycle(
                "canceled",
                order_id=order_id,
                symbol=symbol,
                status="CANCELED",
            )
        return cancelled

    async def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        return await self.rest_client.get_open_orders(symbol)

    async def sync_fills(self, symbol: str) -> List[Dict[str, Any]]:
        return await self.rest_client.get_recent_fills(symbol)

    async def flatten_all_positions(
        self,
        symbol: str,
        long_qty: float,
        short_qty: float,
    ) -> Dict[str, Any]:
        """
        Kill-switch helper: cancel resting orders then market-close both legs.
        """
        summary = {"cancelled": 0, "closed_long": False, "closed_short": False}
        summary["cancelled"] = await self.rest_client.cancel_all_open_orders(symbol)

        if long_qty > 0:
            raw = await self.rest_client.close_position_market(
                symbol,
                side="SELL",
                quantity=long_qty,
                position_side="LONG" if self.position_mode == "hedge" else None,
            )
            summary["closed_long"] = raw is not None
            if not summary["closed_long"]:
                logger.error(f"Failed to close LONG leg {long_qty} {symbol}")

        if short_qty > 0:
            raw = await self.rest_client.close_position_market(
                symbol,
                side="BUY",
                quantity=short_qty,
                position_side="SHORT" if self.position_mode == "hedge" else None,
            )
            summary["closed_short"] = raw is not None
            if not summary["closed_short"]:
                logger.error(f"Failed to close SHORT leg {short_qty} {symbol}")

        return summary

    def _record_lifecycle(self, event: str, **kwargs) -> None:
        if self.lifecycle_recorder is None:
            return
        self.lifecycle_recorder.record(event, **kwargs)
