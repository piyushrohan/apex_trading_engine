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


@pytest.mark.unit
def test_metrics_prometheus_branch_and_port_conflict(monkeypatch):
    import src.observability.metrics as metrics_module

    class FakeMetric:
        def __init__(self, *args, **kwargs):
            self.calls = []

        def labels(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self

        def set(self, value):
            self.calls.append(("set", value))

        def observe(self, value):
            self.calls.append(("observe", value))

    class FakeRegistry:
        pass

    started = []
    monkeypatch.setattr(metrics_module, "Gauge", FakeMetric)
    monkeypatch.setattr(metrics_module, "Histogram", FakeMetric)
    monkeypatch.setattr(metrics_module, "CollectorRegistry", FakeRegistry)
    monkeypatch.setattr(
        metrics_module,
        "start_http_server",
        lambda port, **kwargs: started.append((port, kwargs)),
    )

    metrics = ApexMetrics()
    metrics.start_server(port=9912)
    metrics.start_server(port=9912)
    metrics.set_pnl("paper", "primary", "primary", 1.5)
    metrics.set_ws_health("paper", False)
    metrics.set_paper_fill_rate("paper", "primary", 0.25)

    assert len(started) == 1
    assert "registry" in started[0][1]

    def raise_oserror(port, **kwargs):
        raise OSError("address already in use")

    monkeypatch.setattr(metrics_module, "start_http_server", raise_oserror)
    conflicted = ApexMetrics(registry=FakeRegistry())
    conflicted.start_server(port=9912)

    assert conflicted._server_started is True


@pytest.mark.unit
def test_metrics_noop_fallback_branch(monkeypatch):
    import src.observability.metrics as metrics_module

    started = []
    monkeypatch.setattr(metrics_module, "Gauge", None)
    monkeypatch.setattr(metrics_module, "Histogram", None)
    monkeypatch.setattr(metrics_module, "CollectorRegistry", None)
    monkeypatch.setattr(
        metrics_module,
        "start_http_server",
        lambda port, **kwargs: started.append((port, kwargs)),
    )

    metrics = ApexMetrics()
    metrics.set_pnl("paper", "primary", "primary", 1.0)
    metrics.set_ws_health("paper", True)
    metrics.set_paper_fill_rate("paper", "primary", 0.5)
    assert metrics.time_inference("paper", lambda: "ok") == "ok"

    metrics.start_server(port=9913)

    assert started == [(9913, {})]
