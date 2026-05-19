import pytest

from src.observability.metrics import ApexMetrics


@pytest.mark.unit
def test_metrics_wrapper_times_callable():
    metrics = ApexMetrics()
    result = metrics.time_inference("paper", lambda a, b: a + b, 2, 3)
    assert result == 5
    metrics.set_pnl("paper", "primary", "primary", 12.0)
    metrics.set_ws_health("paper", True)
