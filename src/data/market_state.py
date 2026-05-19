"""Assemble features, regime, and model inputs from the local data lake."""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from src.data.cache_manager import DuckDBCacheManager
from src.data.feature_engine import FeatureEngine
from src.data.state_vector import build_state_vector
from src.models.regime_detector import RegimeDetector

logger = logging.getLogger(__name__)


class MarketStateService:
    """Builds the latest inference snapshot from cached market data."""

    def __init__(self, config: dict, cache: Optional[DuckDBCacheManager] = None):
        self.config = config
        data = config.get("data", {})
        self.target_symbol = data.get("target_symbol", "ETHUSDC")
        self.macro_symbol = data.get("macro_symbol", "BTCUSDC")
        self.interval = data.get("target_interval", "3m")
        self.lookback_bars = data.get("market_state", {}).get("lookback_bars", 200)

        db_path = data.get("storage", {}).get(
            "db_path", "data_lake/apex_market_data.duckdb"
        )
        self._owns_cache = cache is None
        self.cache = cache or DuckDBCacheManager(db_path=db_path)
        self.feature_engine = FeatureEngine(config)
        self.regime_detector = RegimeDetector(config)

    def build_latest(self) -> Optional[Dict[str, Any]]:
        eth = self.cache.load_ohlcv(self.target_symbol, self.interval)
        if eth.empty:
            logger.warning(
                f"No OHLCV in cache for {self.target_symbol} {self.interval}; "
                "cannot build market state"
            )
            return None

        eth = eth.tail(self.lookback_bars).copy()
        btc = self.cache.load_ohlcv(self.macro_symbol, self.interval)
        if not btc.empty:
            btc = btc.tail(self.lookback_bars)

        features = self.feature_engine.process_all_features(eth, btc)
        if features.empty:
            return None

        regime_df = self.regime_detector.detect(features)
        row = regime_df.iloc[-1]
        state_vector = build_state_vector(row)

        return {
            "state_vector": state_vector,
            "regime": str(row.get("regime_str", "MEAN_REVERSION")),
            "mark_price": float(row["close"]),
            "row": row,
            "eth_btc_zscore": float(row.get("eth_btc_zscore", 0.0)),
            "volatility_zscore": float(row.get("volatility_zscore", 0.0)),
            "trend_slope": float(row.get("trend_slope", 0.0)),
            "is_buy_liquidity_sweep": bool(row.get("is_buy_liquidity_sweep", False)),
            "is_sell_liquidity_sweep": bool(row.get("is_sell_liquidity_sweep", False)),
            "cvd": float(row.get("cvd", 0.0)),
        }

    def seed_from_dataframes(
        self, eth_df: pd.DataFrame, btc_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Test helper: build state directly from in-memory klines."""
        features = self.feature_engine.process_all_features(eth_df.copy(), btc_df)
        regime_df = self.regime_detector.detect(features)
        row = regime_df.iloc[-1]
        return {
            "state_vector": build_state_vector(row),
            "regime": str(row.get("regime_str", "MEAN_REVERSION")),
            "mark_price": float(row["close"]),
            "row": row,
            "eth_btc_zscore": float(row.get("eth_btc_zscore", 0.0)),
            "volatility_zscore": float(row.get("volatility_zscore", 0.0)),
            "trend_slope": float(row.get("trend_slope", 0.0)),
            "is_buy_liquidity_sweep": bool(row.get("is_buy_liquidity_sweep", False)),
            "is_sell_liquidity_sweep": bool(row.get("is_sell_liquidity_sweep", False)),
            "cvd": float(row.get("cvd", 0.0)),
        }

    def close(self):
        if self._owns_cache:
            self.cache.close()
