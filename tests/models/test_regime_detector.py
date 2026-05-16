import pytest
import pandas as pd
from src.models.regime_detector import RegimeDetector

@pytest.mark.unit
def test_regime_detector_chop_compression(mock_config):
    """Test that low volatility and zero trend correctly tags CHOP_COMPRESSION."""
    detector = RegimeDetector(mock_config)
    
    # Create a DataFrame that should definitely be CHOP
    df = pd.DataFrame({
        "close": [3000] * 100, # perfectly flat price
        "volatility_zscore": [-1.5] * 100 # deeply negative volatility z-score
    })
    
    result = detector.detect(df)
    
    assert result['regime_str'].iloc[-1] == "CHOP_COMPRESSION"
    assert result['regime_id'].iloc[-1] == detector.REGIMES.index("CHOP_COMPRESSION")

@pytest.mark.unit
def test_regime_detector_volatility_expansion(mock_config):
    """Test that high volatility z-score triggers VOLATILITY_EXPANSION regardless of trend."""
    detector = RegimeDetector(mock_config)
    
    df = pd.DataFrame({
        "close": [3000, 3010, 3020, 3500], # sudden spike
        "volatility_zscore": [0.0, 0.1, 0.2, 2.5] # explodes
    })
    
    result = detector.detect(df)
    
    assert result['regime_str'].iloc[-1] == "VOLATILITY_EXPANSION"

@pytest.mark.unit
def test_regime_detector_strong_trend_up(mock_config):
    """Test that positive trend slope and normal volatility yields STRONG_TREND_UP."""
    detector = RegimeDetector(mock_config)
    
    # Need to give it enough periods for the EMA short and long to diverge positively
    close_prices = [3000 + (i * 10) for i in range(100)] # steady 10 point climb per bar
    df = pd.DataFrame({
        "close": close_prices,
        "volatility_zscore": [0.5] * 100 # Positive but not explosive
    })
    
    result = detector.detect(df)
    
    assert result['regime_str'].iloc[-1] == "STRONG_TREND_UP"
    assert result['trend_slope'].iloc[-1] > 0.002
