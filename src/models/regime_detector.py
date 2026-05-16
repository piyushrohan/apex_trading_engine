import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RegimeDetector:
    """
    Classifies the market into distinct regimes based on volatility, 
    trend, and orderflow. This allows the Meta-Controller to select 
    the appropriate specialized model or adjust risk dynamically.
    """
    
    REGIMES = [
        "STRONG_TREND_UP",
        "STRONG_TREND_DOWN",
        "CHOP_COMPRESSION",
        "VOLATILITY_EXPANSION",
        "MEAN_REVERSION"
    ]
    
    def __init__(self, config: dict):
        self.config = config
        self.trend_short = config.get("technicals", {}).get("ema_trend_short", 10)
        self.trend_long = config.get("technicals", {}).get("ema_trend_long", 50)
        
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a DataFrame enriched by FeatureEngine and appends 
        regime classifications.
        """
        # Calculate Trend
        df['ema_short'] = df['close'].ewm(span=self.trend_short, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=self.trend_long, adjust=False).mean()
        
        df['trend_slope'] = (df['ema_short'] - df['ema_long']) / df['ema_long']
        
        # Determine Regimes
        # Note: We assign numerical IDs and string tags.
        conditions = [
            (df['volatility_zscore'] < -1.0) & (abs(df['trend_slope']) < 0.001), # Chop/Compression
            (df['volatility_zscore'] > 1.5), # Volatility Expansion
            (df['trend_slope'] > 0.002) & (df['volatility_zscore'] > 0.0), # Strong Trend Up
            (df['trend_slope'] < -0.002) & (df['volatility_zscore'] > 0.0), # Strong Trend Down
        ]
        
        choices = [
            "CHOP_COMPRESSION",
            "VOLATILITY_EXPANSION",
            "STRONG_TREND_UP",
            "STRONG_TREND_DOWN"
        ]
        
        df['regime_str'] = np.select(conditions, choices, default="MEAN_REVERSION")
        df['regime_id'] = df['regime_str'].apply(lambda x: self.REGIMES.index(x) if x in self.REGIMES else 4)
        
        return df
