from dataclasses import dataclass
from typing import List

from src.execution.adapters.base import OrderRequest


@dataclass
class GridOrderPlan:
    orders: List[OrderRequest]


class MakerGridAdapter:
    """Build post-only two-sided grid orders for the maker_grid_hedge strategy."""

    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size

    def build_grid(
        self,
        symbol: str,
        mid_price: float,
        total_quantity: float,
        levels: int = 3,
        spacing_ticks: int = 2,
    ) -> GridOrderPlan:
        if levels <= 0 or total_quantity <= 0:
            return GridOrderPlan(orders=[])

        qty = total_quantity / levels
        orders: List[OrderRequest] = []
        for level in range(1, levels + 1):
            offset = self.tick_size * spacing_ticks * level
            orders.append(
                OrderRequest(
                    symbol=symbol,
                    side="BUY",
                    quantity=qty,
                    price=round(mid_price - offset, 8),
                    position_side="LONG",
                )
            )
            orders.append(
                OrderRequest(
                    symbol=symbol,
                    side="SELL",
                    quantity=qty,
                    price=round(mid_price + offset, 8),
                    position_side="SHORT",
                )
            )
        return GridOrderPlan(orders=orders)
