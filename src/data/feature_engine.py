import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Live streaming and historical feature generation engine.
    Calculates orderflow imbalances, Cumulative Volume Delta (CVD),
    and relative strength between ETH and BTC.
    """

    def __init__(self, config: dict):
        self.config = config
        self.tech_config = config.get("technicals", {})

        self.rolling_window = self.tech_config.get("rolling_window", 120)
        self.atr_period = self.tech_config.get("atr_period", 10)
        self.macro_vol_z_period = self.tech_config.get("macro_vol_z_period", 144)

    def add_orderflow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates microstructure and orderflow features.
        Requires tick-level or very granular OHLCV data with taker buy/sell volume.
        If using standard OHLCV, approximates CVD based on price action vs volume.
        """
        # Approximated CVD if exact taker_buy_volume is missing
        # Assigns volume to buyers if close > open, sellers if close < open
        if "taker_buy_volume" not in df.columns:
            price_delta = df["close"] - df["open"]
            direction = np.sign(price_delta)
            direction = direction.replace(0, 1)  # Assume flat is a buy for accumulation

            # Simple approximation
            df["net_volume"] = df["volume"] * direction
        else:
            taker_sell_volume = df["volume"] - df["taker_buy_volume"]
            df["net_volume"] = df["taker_buy_volume"] - taker_sell_volume

        df["cvd"] = df["net_volume"].cumsum()

        # Liquidity Sweep detection (tails)
        body = abs(df["close"] - df["open"])
        upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
        lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

        volume_baseline = df["volume"].rolling(20, min_periods=1).mean()

        df["is_buy_liquidity_sweep"] = (lower_wick > (body * 2)) & (
            df["volume"] > volume_baseline
        )
        df["is_sell_liquidity_sweep"] = (upper_wick > (body * 2)) & (
            df["volume"] > volume_baseline
        )

        return df

    def add_relative_strength(
        self, target_df: pd.DataFrame, macro_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculates ETH relative strength against BTC.
        Requires synchronized dataframes on the timestamp.
        """
        # Merge on timestamp
        merged = pd.merge(
            target_df,
            macro_df[["timestamp", "close"]],
            on="timestamp",
            how="left",
            suffixes=("", "_btc"),
        )

        # Calculate Rolling Beta
        target_returns = merged["close"].pct_change()
        macro_returns = merged["close_btc"].pct_change()

        cov = target_returns.rolling(window=self.rolling_window).cov(macro_returns)
        var = macro_returns.rolling(window=self.rolling_window).var()
        merged["eth_btc_beta"] = cov / var

        # Calculate Spread Z-Score
        spread = merged["close"] / merged["close_btc"]
        spread_mean = spread.rolling(self.rolling_window).mean()
        spread_std = spread.rolling(self.rolling_window).std()

        merged["eth_btc_zscore"] = (spread - spread_mean) / spread_std

        return merged

    def add_volatility_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates ATR and Volatility Expansion metrics."""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)

        df["atr"] = true_range.rolling(self.atr_period).mean()

        # Volatility Z-Score (detects expansions and compressions)
        vol_mean = df["atr"].rolling(self.macro_vol_z_period).mean()
        vol_std = df["atr"].rolling(self.macro_vol_z_period).std()
        df["volatility_zscore"] = (df["atr"] - vol_mean) / vol_std

        return df

    def process_all_features(
        self, target_df: pd.DataFrame, macro_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Runs the entire feature generation pipeline."""
        df = target_df.copy()
        df = self.add_orderflow_features(df)
        if macro_df is not None and not macro_df.empty:
            df = self.add_relative_strength(df, macro_df)
        df = self.add_volatility_metrics(df)

        # Drop NaN rows due to rolling windows
        df.dropna(inplace=True)
        return df
