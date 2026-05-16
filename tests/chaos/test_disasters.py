import pytest
from src.execution.risk_engine import RiskEngine
from src.data.feature_engine import FeatureEngine
from src.models.regime_detector import RegimeDetector

def test_risk_flash_crash_survival(mock_config, mock_eth_klines):
    """
    Risk Catastrophe Test:
    Verify that an extreme flash crash in the data lake does not 
    break the mathematical feature engine and immediately forces 
    the regime detector into Volatility Expansion mode to protect capital.
    """
    feature_engine = FeatureEngine(mock_config)
    regime_detector = RegimeDetector(mock_config)
    
    # Process the data up to index 5 (where the flash crash occurs)
    df_crash = mock_eth_klines.iloc[:6].copy()
    
    df_features = feature_engine.add_volatility_metrics(df_crash)
    df_regime = regime_detector.detect(df_features)
    
    # Verify the massive true range spike forces a high volatility z-score
    # and triggers VOLATILITY_EXPANSION regime.
    assert df_regime['volatility_zscore'].iloc[-1] > 1.5
    assert df_regime['regime_str'].iloc[-1] == "VOLATILITY_EXPANSION"

def test_chaos_invalid_payload(mock_config):
    """
    Chaos Test:
    Simulate a corrupted AI output generating an invalid leverage 
    request (e.g., trying to short with 100x leverage on a 3x account).
    RiskEngine MUST reject it.
    """
    engine = RiskEngine(mock_config)
    
    # AI goes rogue and requests 100x leverage
    approved = engine.approve_order("SELL", 100.0, current_exposure=0.0)
    
    # RiskEngine caps it at 3x
    assert approved == mock_config['risk']['max_leverage']
