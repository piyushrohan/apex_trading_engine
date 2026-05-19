from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OrderRequest:
    symbol: str
    side: str
    quantity: float
    price: float
    position_side: Optional[str] = None
    time_in_force: str = "GTX"
    order_type: str = "LIMIT"
    client_order_id: Optional[str] = None


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    status: str = "NEW"
    filled_qty: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class ExecutionAdapter(ABC):
    """Pluggable execution for paper (virtual) and live (exchange) books."""

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult:
        pass

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def sync_fills(self, symbol: str) -> List[Dict[str, Any]]:
        pass

    async def cancel_all_orders(self, symbol: str) -> int:
        open_orders = await self.get_open_orders(symbol)
        cancelled = 0
        for order in open_orders:
            oid = order.get("orderId") or order.get("order_id")
            if oid and await self.cancel_order(symbol, str(oid)):
                cancelled += 1
        return cancelled
