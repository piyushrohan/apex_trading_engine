from typing import Optional

from src.core.logger import get_logger
from src.data.binance_rest import BinanceRESTClient
from src.execution.slippage import SlippageManager

logger = get_logger("OrderManager")


class OrderManager:
    """
    Handles institutional-grade execution for Binance Futures.
    Optimized for ETHUSDC with 0% Maker Fees.
    Strictly uses POST-ONLY limit orders (timeInForce=GTX).
    """

    def __init__(self, config: dict, rest_client: BinanceRESTClient):
        self.config = config
        self.rest_client = rest_client
        self.symbol = config.get("data", {}).get("target_symbol", "ETHUSDC")

        exec_config = config.get("execution", {})
        self.max_leverage = exec_config.get("max_leverage", 3)
        self.chase_tolerance = exec_config.get("chase_tolerance", 3)

        self.slippage_manager = SlippageManager(
            chase_tolerance_ticks=self.chase_tolerance
        )

    async def place_maker_order(
        self, side: str, quantity: float, price: float
    ) -> Optional[dict]:
        """
        Places a POST-ONLY order. If the price would cross the spread,
        Binance will reject it, ensuring we never pay Taker fees.
        """
        logger.info(
            f"[{self.symbol}] Placing Maker-Only {side} order for {quantity} @ {price}"
        )

        try:
            response = await self.rest_client.place_order(
                symbol=self.symbol,
                side=side.upper(),
                quantity=quantity,
                price=price,
                timeInForce="GTX",
                orderType="LIMIT",
            )

            logger.info(
                f"[{self.symbol}] Maker order submitted successfully: {response}"
            )

            return response

        except Exception as e:
            logger.error(f"Failed to place Maker order: {e}")
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancels an open limit order."""
        logger.info(f"[{self.symbol}] Cancelling Order {order_id}")
        return await self.rest_client.cancel_order(self.symbol, order_id)

    async def cancel_and_replace(
        self,
        order_id: str,
        side: str,
        quantity: float,
        new_price: float,
        conviction: float,
        current_bbo: float,
    ) -> Optional[dict]:
        """
        Chasing logic: Evaluates if we should chase the market.
        If yes, cancels the existing order and places a new one closer to the BBO.
        """
        # Determine if we should chase using SlippageManager
        # Use new_price as the ideal resting price in this simplified flow.
        should_chase = self.slippage_manager.should_chase_order(
            current_bbo, new_price, side, conviction
        )

        if not should_chase:
            logger.info(f"[{self.symbol}] Abandoning order chase for {order_id}.")
            await self.cancel_order(order_id)
            return None

        logger.debug(
            f"[{self.symbol}] Chasing... C&R Order {order_id} to new price {new_price}"
        )
        await self.cancel_order(order_id)
        return await self.place_maker_order(side, quantity, new_price)
