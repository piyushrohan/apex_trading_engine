"""Static and optional live smoke checks for the operator frontend/API contract."""

import argparse
import http.client
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urljoin, urlparse

from src.reports.ops_common import (
    add_finding,
    should_exit_nonzero,
    status_from_findings,
    utc_now,
    write_report,
)

EXPECTED_GET_ENDPOINTS = (
    "/status",
    "/explain/latest",
    "/portfolio",
    "/metrics/paper",
    "/reports/hedge",
    "/live/gate",
    "/history/decisions",
    "/history/equity",
    "/history/market",
    "/models",
    "/models/lifecycle",
    "/models/promotion/status",
    "/ops/readiness",
    "/logs/runtime",
    "/audit",
    "/control/state",
)
SMOKE_URLS = (
    "/status",
    "/portfolio",
    "/metrics/paper",
    "/reports/hedge",
    "/live/gate",
    "/history/decisions?limit=1",
    "/history/equity?limit=1",
    "/history/market?limit=1",
    "/models",
    "/models/lifecycle?limit=1",
    "/models/promotion/status",
    "/ops/readiness",
    "/logs/runtime?limit=1",
    "/audit?limit=1",
    "/control/state",
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _api_routes(api_server_path: Path) -> Dict[str, set[str]]:
    if not api_server_path.exists():
        return {}
    text = _read_text(api_server_path)
    routes: Dict[str, set[str]] = {}
    for method, path in re.findall(r'@app\.(get|post|websocket)\("([^"]+)"\)', text):
        routes.setdefault(method, set()).add(path)
    return routes


def _route_present(routes: Dict[str, set[str]], method: str, path: str) -> bool:
    if path in routes.get(method, set()):
        return True
    if path.startswith("/control/") and "/control/{command}" in routes.get(
        method, set()
    ):
        return True
    return False


def _live_get_json(api_base_url: str, path: str, timeout: float) -> Dict[str, Any]:
    url = urljoin(api_base_url.rstrip("/") + "/", path.lstrip("/"))
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return {
            "url": url,
            "ok": False,
            "status_code": None,
            "error": "unsupported_url_scheme",
        }
    connection_cls = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_cls(parsed.netloc, timeout=timeout)
    request_path = parsed.path or "/"
    if parsed.query:
        request_path = f"{request_path}?{parsed.query}"
    try:
        connection.request("GET", request_path, headers={"Accept": "application/json"})
        response = connection.getresponse()
        body = response.read().decode("utf-8", errors="replace")
        status_code = response.status
    except OSError as exc:
        return {
            "url": url,
            "ok": False,
            "status_code": None,
            "error": str(exc),
        }
    finally:
        connection.close()

    try:
        payload = json.loads(body or "{}")
    except json.JSONDecodeError:
        return {
            "url": url,
            "ok": False,
            "status_code": status_code,
            "error": "response_not_json",
        }
    return {
        "url": url,
        "ok": 200 <= status_code < 300,
        "status_code": status_code,
        "keys": sorted(payload.keys()) if isinstance(payload, dict) else None,
    }


def generate_frontend_api_contract_smoke(
    *,
    frontend_dir: str = "frontend",
    api_server_path: str = "src/api/server.py",
    api_base_url: str = "http://127.0.0.1:8080",
    live_api: bool = False,
    timeout: float = 2.0,
    strict: bool = False,
) -> Dict[str, Any]:
    """Validate that the browser terminal still matches the FastAPI surface."""
    findings: list[Dict[str, Any]] = []
    frontend = Path(frontend_dir)
    app_path = frontend / "app.js"
    index_path = frontend / "index.html"
    api_path = Path(api_server_path)
    app_text = _read_text(app_path) if app_path.exists() else ""
    index_text = _read_text(index_path) if index_path.exists() else ""
    routes = _api_routes(api_path)
    live_results: list[Dict[str, Any]] = []

    if not app_path.exists():
        add_finding(
            findings,
            "error",
            "frontend_app_missing",
            f"Frontend app file is missing at {app_path}.",
        )
    if not index_path.exists():
        add_finding(
            findings,
            "error",
            "frontend_index_missing",
            f"Frontend index file is missing at {index_path}.",
        )
    if not api_path.exists():
        add_finding(
            findings,
            "error",
            "api_server_missing",
            f"API server file is missing at {api_path}.",
        )

    for endpoint in EXPECTED_GET_ENDPOINTS:
        if endpoint not in app_text:
            add_finding(
                findings,
                "error" if strict else "warning",
                "frontend_endpoint_missing",
                "Frontend no longer references an expected API endpoint.",
                endpoint=endpoint,
            )
        if routes and not _route_present(routes, "get", endpoint):
            add_finding(
                findings,
                "error",
                "api_endpoint_missing",
                "FastAPI server no longer exposes an endpoint used by frontend.",
                endpoint=endpoint,
            )

    if "/control/${command}" not in app_text:
        add_finding(
            findings,
            "error",
            "frontend_control_post_missing",
            "Frontend no longer posts confirmed operator commands.",
        )
    if routes and not _route_present(routes, "post", "/control/{command}"):
        add_finding(
            findings,
            "error",
            "api_control_post_missing",
            "FastAPI server no longer exposes the control command endpoint.",
        )
    if "/ws/status" not in app_text:
        add_finding(
            findings,
            "warning",
            "frontend_websocket_missing",
            "Frontend no longer subscribes to the status websocket.",
        )
    if routes and not _route_present(routes, "websocket", "/ws/status"):
        add_finding(
            findings,
            "error",
            "api_websocket_missing",
            "FastAPI server no longer exposes the status websocket.",
        )
    if '.get("api")' not in app_text:
        add_finding(
            findings,
            "warning",
            "frontend_api_override_missing",
            "Frontend should keep the ?api= override for local smoke testing.",
        )
    if 'id="root"' not in index_text:
        add_finding(
            findings,
            "error",
            "frontend_root_missing",
            "Frontend index does not expose the React root element.",
        )

    if live_api:
        for path in SMOKE_URLS:
            result = _live_get_json(api_base_url, path, timeout)
            live_results.append(result)
            if not result["ok"]:
                add_finding(
                    findings,
                    "error",
                    "live_api_endpoint_failed",
                    "Live API endpoint failed the JSON smoke test.",
                    path=path,
                    status_code=result.get("status_code"),
                    error=result.get("error"),
                )

    summary = {
        "frontend_dir": frontend_dir,
        "api_server_path": api_server_path,
        "expected_get_endpoints": len(EXPECTED_GET_ENDPOINTS),
        "api_get_routes": len(routes.get("get", set())),
        "api_post_routes": len(routes.get("post", set())),
        "api_websocket_routes": len(routes.get("websocket", set())),
        "live_api": live_api,
        "live_checks": len(live_results),
    }
    return {
        "title": "APEX Frontend/API Contract Smoke Test",
        "generated_at": utc_now().isoformat(),
        "status": status_from_findings(findings),
        "summary": summary,
        "routes": {method: sorted(paths) for method, paths in routes.items()},
        "live_results": live_results,
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="APEX frontend/API contract smoke")
    parser.add_argument("--frontend-dir", default="frontend")
    parser.add_argument("--api-server-path", default="src/api/server.py")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--live-api", action="store_true")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = generate_frontend_api_contract_smoke(
        frontend_dir=args.frontend_dir,
        api_server_path=args.api_server_path,
        api_base_url=args.api_base_url,
        live_api=args.live_api,
        timeout=args.timeout,
        strict=args.strict,
    )
    rendered = write_report(report, output=args.output, fmt=args.format)
    if not args.output:
        print(rendered)
    if should_exit_nonzero(report, fail_on_warning=args.fail_on_warning):
        sys.exit(1)


if __name__ == "__main__":
    main()
