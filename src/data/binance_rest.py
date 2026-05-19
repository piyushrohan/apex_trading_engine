import asyncio
import hashlib
import hmac
import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)


class BinanceRESTClient:
    """
    Async REST client for Binance USD-M Futures (fapi).
    Used for historical backfilling and fetching missing gaps.
    """

    BASE_URL = "https://fapi.binance.com"

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = None
        self.listen_key: Optional[str] = None

    async def _get_session(self):
        if self.session is None or self.session.closed:
            headers = {}
            if self.api_key:
                headers["X-MBX-APIKEY"] = self.api_key
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    def _sign_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_secret:
            raise ValueError("API secret required for signed Binance requests")
        payload = {**params, "timestamp": int(time.time() * 1000)}
        query = urlencode(payload)
        signature = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        payload["signature"] = signature
        return payload

    async def _signed_request(
        self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        session = await self._get_session()
        signed = self._sign_params(params or {})
        url = f"{self.BASE_URL}{endpoint}"

        for attempt in range(3):
            try:
                if method == "GET":
                    ctx = session.get(url, params=signed)
                elif method == "POST":
                    ctx = session.post(url, params=signed)
                elif method == "DELETE":
                    ctx = session.delete(url, params=signed)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                async with ctx as response:
                    if response.status in (200, 201):
                        return await response.json()
                    text = await response.text()
                    logger.error(
                        f"Signed API {method} {endpoint} failed "
                        f"{response.status}: {text}"
                    )
            except Exception as exc:
                logger.error(f"Signed request error attempt {attempt + 1}: {exc}")
            await asyncio.sleep(0.5)
        return None

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timeInForce: str = "GTX",
        orderType: str = "LIMIT",
        positionSide: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": orderType,
            "quantity": quantity,
            "price": price,
            "timeInForce": timeInForce,
        }
        if positionSide:
            params["positionSide"] = positionSide
        return await self._signed_request("POST", "/fapi/v1/order", params)

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        result = await self._signed_request(
            "DELETE",
            "/fapi/v1/order",
            {"symbol": symbol, "orderId": order_id},
        )
        return result is not None

    async def cancel_all_open_orders(self, symbol: str) -> int:
        """Cancel all open orders for a symbol."""
        result = await self._signed_request(
            "DELETE",
            "/fapi/v1/allOpenOrders",
            {"symbol": symbol},
        )
        if result is None:
            return 0
        if isinstance(result, dict) and "code" in result:
            return 0
        return 1

    async def set_hedge_mode(self, enabled: bool = True) -> bool:
        """Enable/disable Binance hedge mode for dual LONG/SHORT legs."""
        result = await self._signed_request(
            "POST",
            "/fapi/v1/positionSide/dual",
            {"dualSidePosition": "true" if enabled else "false"},
        )
        return result is not None

    async def get_position_mode(self) -> Optional[Dict[str, Any]]:
        """Return Binance dual-side position mode state."""
        result = await self._signed_request("GET", "/fapi/v1/positionSide/dual")
        return result if isinstance(result, dict) else None

    async def set_leverage(
        self, symbol: str, leverage: int
    ) -> Optional[Dict[str, Any]]:
        """Set futures leverage for the configured symbol."""
        return await self._signed_request(
            "POST",
            "/fapi/v1/leverage",
            {"symbol": symbol, "leverage": int(leverage)},
        )

    async def get_usdc_balance(self) -> Optional[Dict[str, Any]]:
        """Fetch USDC wallet balance for account-equity reconciliation."""
        result = await self._signed_request("GET", "/fapi/v2/balance")
        if not isinstance(result, list):
            return None
        for row in result:
            if row.get("asset") == "USDC":
                return row
        return None

    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch position risk snapshot (dual-leg in hedge mode)."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        result = await self._signed_request("GET", "/fapi/v2/positionRisk", params)
        if not isinstance(result, list):
            return []
        if symbol:
            return [p for p in result if p.get("symbol") == symbol]
        return result

    async def close_position_market(
        self,
        symbol: str,
        side: str,
        quantity: float,
        position_side: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Reduce-only market order to flatten a leg.
        side: SELL to close long, BUY to close short.
        """
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": quantity,
            "reduceOnly": "true",
        }
        if position_side:
            params["positionSide"] = position_side
        return await self._signed_request("POST", "/fapi/v1/order", params)

    async def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        result = await self._signed_request(
            "GET", "/fapi/v1/openOrders", {"symbol": symbol}
        )
        return result if isinstance(result, list) else []

    async def get_recent_fills(
        self, symbol: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        result = await self._signed_request(
            "GET",
            "/fapi/v1/userTrades",
            {"symbol": symbol, "limit": limit},
        )
        return result if isinstance(result, list) else []

    async def _public_get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        try:
            async with session.get(url, params=params or {}) as response:
                if response.status == 200:
                    return await response.json()
                logger.error(
                    f"Public API GET {endpoint} failed {response.status}: "
                    f"{await response.text()}"
                )
        except Exception as exc:
            logger.error(f"Public GET {endpoint} error: {exc}")
        return None

    async def fetch_premium_index(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Mark price and last funding rate for a symbol."""
        data = await self._public_get("/fapi/v1/premiumIndex", {"symbol": symbol})
        return data if isinstance(data, dict) else None

    async def fetch_open_interest(self, symbol: str) -> Optional[float]:
        data = await self._public_get("/fapi/v1/openInterest", {"symbol": symbol})
        if isinstance(data, dict) and "openInterest" in data:
            return float(data["openInterest"])
        return None

    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
        limit: int = 1500,
    ) -> List[list]:
        """
        Fetches historical klines (OHLCV) from Binance Futures.
        Times must be in milliseconds.
        """
        session = await self._get_session()
        endpoint = f"{self.BASE_URL}/fapi/v1/klines"

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit,
        }

        retries = 3
        for attempt in range(retries):
            try:
                async with session.get(endpoint, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    elif response.status == 429:
                        logger.warning("Rate limit hit fetching klines. Backing off.")
                        await asyncio.sleep(int(response.headers.get("Retry-After", 5)))
                    else:
                        text = await response.text()
                        logger.error(f"Binance API Error {response.status}: {text}")
                        await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Network error on attempt {attempt+1}: {e}")
                await asyncio.sleep(2)

        return []

    async def backfill_historical_data(
        self, symbol: str, interval: str, start_time_ms: int, end_time_ms: int
    ) -> pd.DataFrame:
        """
        Paginates through Binance API to fetch all klines between start and end time.
        Returns a formatted pandas DataFrame ready for DuckDB insertion.
        """
        all_klines = []
        current_start = start_time_ms

        logger.info(
            f"Starting historical backfill for {symbol} {interval} "
            f"from {start_time_ms} to {end_time_ms}"
        )

        while current_start < end_time_ms:
            klines = await self.fetch_klines(
                symbol, interval, current_start, end_time_ms
            )

            if not klines:
                break

            all_klines.extend(klines)

            # The last kline's open time + 1ms becomes the new start time
            last_timestamp = klines[-1][0]
            if last_timestamp == current_start:
                break  # Prevent infinite loops if API gets stuck

            current_start = last_timestamp + 1

            # Respect rate limits; klines has variable request weight.
            await asyncio.sleep(0.1)

        if not all_klines:
            return pd.DataFrame()

        # Format to DataFrame
        columns = [
            "timestamp_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ]

        df = pd.DataFrame(all_klines, columns=columns)

        # Cast and clean
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
        df["symbol"] = symbol
        df["timeframe"] = interval

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        # Select columns matching our schema
        final_df = df[
            [
                "timestamp",
                "symbol",
                "timeframe",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]

        logger.info(
            f"Successfully fetched {len(final_df)} historical klines for {symbol}."
        )
        return final_df

    async def get_listen_key(self) -> Optional[str]:
        """Creates a new listen key for the User Data Stream."""
        session = await self._get_session()
        endpoint = f"{self.BASE_URL}/fapi/v1/listenKey"

        async with session.post(endpoint) as response:
            if response.status == 200:
                data = await response.json()
                self.listen_key = data.get("listenKey")
                return self.listen_key
            else:
                logger.error(f"Failed to get listen key: {await response.text()}")
                return None

    async def keepalive_listen_key(self):
        """Keeps the listen key alive. Call every 30-60 minutes."""
        session = await self._get_session()
        endpoint = f"{self.BASE_URL}/fapi/v1/listenKey"
        params = {"listenKey": self.listen_key} if self.listen_key else None

        try:
            ctx = session.put(endpoint, params=params)
        except TypeError:
            ctx = session.put(endpoint)
        async with ctx as response:
            if response.status != 200:
                logger.error(
                    f"Failed to keep-alive listen key: {await response.text()}"
                )

    async def close_listen_key(self):
        """Closes the current listen key."""
        session = await self._get_session()
        endpoint = f"{self.BASE_URL}/fapi/v1/listenKey"
        params = {"listenKey": self.listen_key} if self.listen_key else None

        try:
            ctx = session.delete(endpoint, params=params)
        except TypeError:
            ctx = session.delete(endpoint)
        async with ctx as response:
            if response.status != 200:
                logger.error(f"Failed to close listen key: {await response.text()}")
            else:
                self.listen_key = None
