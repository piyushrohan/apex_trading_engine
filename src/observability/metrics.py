import time
from typing import Any, Callable


class _NoopMetric:
    def labels(self, *args: Any, **kwargs: Any) -> "_NoopMetric":
        return self

    def set(self, value: float) -> None:
        return None

    def observe(self, value: float) -> None:
        return None


try:
    from prometheus_client import Gauge, Histogram, start_http_server
except Exception:  # pragma: no cover - exercised only without optional dependency
    Gauge = Histogram = None

    def start_http_server(port: int) -> None:
        return None


class ApexMetrics:
    """Thin wrapper around Prometheus metrics with a safe no-op fallback."""

    def __init__(self):
        self._server_started = False
        if Gauge is None or Histogram is None:
            self.paper_live_pnl = _NoopMetric()
            self.inference_latency = _NoopMetric()
            self.ws_connected = _NoopMetric()
            self.paper_fill_rate = _NoopMetric()
            return

        self.paper_live_pnl = Gauge(
            "apex_pnl_usdc",
            "Current PnL in USDC by operator mode and book.",
            ["mode", "book_role", "book_id"],
        )
        self.inference_latency = Histogram(
            "apex_inference_latency_seconds",
            "Model inference latency by operator mode.",
            ["mode"],
        )
        self.ws_connected = Gauge(
            "apex_ws_connected",
            "Market data websocket/ingestion health, 1 for connected/enabled.",
            ["mode"],
        )
        self.paper_fill_rate = Gauge(
            "apex_paper_fill_rate",
            "Paper maker fill rate by operator mode and book.",
            ["mode", "book_id"],
        )

    def start_server(self, port: int = 9108) -> None:
        if self._server_started:
            return
        try:
            start_http_server(port)
            self._server_started = True
        except OSError:
            self._server_started = True

    def set_pnl(self, mode: str, book_role: str, book_id: str, pnl: float) -> None:
        self.paper_live_pnl.labels(mode, book_role, book_id).set(float(pnl))

    def set_ws_health(self, mode: str, connected: bool) -> None:
        self.ws_connected.labels(mode).set(1.0 if connected else 0.0)

    def set_paper_fill_rate(self, mode: str, book_id: str, fill_rate: float) -> None:
        self.paper_fill_rate.labels(mode, book_id).set(float(fill_rate))

    def time_inference(self, mode: str, fn: Callable[..., Any], *args: Any) -> Any:
        started = time.perf_counter()
        try:
            return fn(*args)
        finally:
            self.inference_latency.labels(mode).observe(time.perf_counter() - started)


APEX_METRICS = ApexMetrics()
