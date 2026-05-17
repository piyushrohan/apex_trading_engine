import asyncio
import logging
from typing import List, Optional

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
                return data.get("listenKey")
            else:
                logger.error(f"Failed to get listen key: {await response.text()}")
                return None

    async def keepalive_listen_key(self):
        """Keeps the listen key alive. Call every 30-60 minutes."""
        session = await self._get_session()
        endpoint = f"{self.BASE_URL}/fapi/v1/listenKey"

        async with session.put(endpoint) as response:
            if response.status != 200:
                logger.error(
                    f"Failed to keep-alive listen key: {await response.text()}"
                )

    async def close_listen_key(self):
        """Closes the current listen key."""
        session = await self._get_session()
        endpoint = f"{self.BASE_URL}/fapi/v1/listenKey"

        async with session.delete(endpoint) as response:
            if response.status != 200:
                logger.error(f"Failed to close listen key: {await response.text()}")
