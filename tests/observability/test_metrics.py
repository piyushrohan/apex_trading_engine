import pytest

from src.observability.metrics import ApexMetrics


@pytest.mark.unit
def test_metrics_wrapper_times_callable():
    metrics = ApexMetrics()
    result = metrics.time_inference("paper", lambda a, b: a + b, 2, 3)
    assert result == 5
    metrics.set_pnl("paper", "primary", "primary", 12.0)
    metrics.set_ws_health("paper", True)


@pytest.mark.unit
def test_metrics_instances_are_isolated():
    first = ApexMetrics()
    second = ApexMetrics()

    first.set_pnl("paper", "primary", "primary", 12.0)
    second.set_pnl("paper", "shadow", "candidate", -3.5)


@pytest.mark.unit
def test_metrics_start_server_is_idempotent_and_records_fill_rate(monkeypatch):
    import src.observability.metrics as metrics_module

    started = []
    monkeypatch.setattr(
        metrics_module,
        "start_http_server",
        lambda port, **kwargs: started.append((port, kwargs)),
    )
    metrics = ApexMetrics()

    metrics.start_server(port=0)
    metrics.start_server(port=0)
    metrics.set_paper_fill_rate("paper", "primary", 0.75)

    assert len(started) == 1
