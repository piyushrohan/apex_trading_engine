import json
import sys
from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd
import pytest

from src.data.cache_manager import DuckDBCacheManager
from src.mlops.experiment_tracker import ExperimentTracker
from src.mlops.registry import ModelRegistry
from src.models.gbm_agent import GBMAgent
from src.reports import (
    data_freshness_check,
    experiment_ledger_auditor,
    frontend_api_contract_smoke,
    model_governance_report,
    ops_common,
    paper_health_watchdog,
    shadow_sanity_monitor,
)
from src.reports.data_freshness_check import generate_data_freshness_report
from src.reports.experiment_ledger_auditor import generate_experiment_ledger_audit
from src.reports.frontend_api_contract_smoke import generate_frontend_api_contract_smoke
from src.reports.model_governance_report import generate_model_governance_report
from src.reports.paper_health_watchdog import generate_paper_health_report
from src.reports.shadow_sanity_monitor import generate_shadow_sanity_report


def _iso(minutes_ago: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


@pytest.mark.unit
def test_ops_common_helpers_cover_markdown_files_and_exit_semantics(tmp_path):
    invalid_json = tmp_path / "invalid.json"
    jsonl = tmp_path / "rows.jsonl"
    output = tmp_path / "reports" / "report.md"
    invalid_json.write_text("{bad-json", encoding="utf-8")
    jsonl.write_text('\n{"ok": true}\n{bad-json\n', encoding="utf-8")
    payload = {
        "title": "Ops",
        "status": "warn",
        "generated_at": _iso(),
        "summary": {"x": 1},
        "findings": [
            {
                "severity": "warning",
                "code": "pipe",
                "message": "contains | pipe",
            }
        ],
    }

    assert ops_common.parse_timestamp(None) is None
    assert ops_common.parse_timestamp("not-a-date") is None
    assert ops_common.age_minutes("not-a-date") is None
    assert ops_common.read_json_file(tmp_path / "missing.json") is None
    assert ops_common.read_json_file(invalid_json) is None
    assert ops_common.read_jsonl(tmp_path / "missing.jsonl") == []
    assert ops_common.read_jsonl(jsonl)[1]["_invalid_json"] == "{bad-json"
    assert ops_common.status_from_findings([]) == "pass"
    assert ops_common.should_exit_nonzero({"status": "fail"}) is True
    assert ops_common.should_exit_nonzero(payload, fail_on_warning=True) is True

    rendered = ops_common.write_report(payload, output=str(output), fmt="markdown")

    assert "# Ops" in rendered
    assert "contains \\| pipe" in output.read_text(encoding="utf-8")


@pytest.mark.unit
def test_paper_health_watchdog_passes_with_fresh_runtime_journal_and_snapshots(
    tmp_path, mock_config
):
    db_path = str(tmp_path / "paper.duckdb")
    journal = tmp_path / "trade_journal.jsonl"
    runtime = tmp_path / "runtime_status.json"
    mock_config["data"]["storage"] = {"db_path": db_path}
    mock_config["explainability"] = {"journal_path": str(journal)}
    cache = DuckDBCacheManager(db_path=db_path)
    try:
        cache.insert_paper_equity_snapshot(
            "primary", 1000.0, 0.1, 0.0, 3000.0, "MEAN_REVERSION"
        )
    finally:
        cache.close()
    journal.write_text(
        json.dumps(
            {
                "timestamp": _iso(),
                "execution": {"mode": "paper"},
                "book": {"role": "primary"},
                "decision": "LONG",
            }
        )
        + "\n"
        + json.dumps(
            {
                "timestamp": _iso(),
                "execution": {"mode": "paper"},
                "book": {"role": "primary"},
                "event": "paper_fill",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runtime.write_text(
        json.dumps(
            {
                "operator_mode": "paper",
                "updated_at": _iso(),
                "kill_switch_active": False,
                "last_explanation": {"decision": "LONG"},
                "portfolio": {"equity": 1000.0},
            }
        ),
        encoding="utf-8",
    )

    report = generate_paper_health_report(
        mock_config, runtime_status_path=str(runtime), journal_path=str(journal)
    )

    assert report["status"] == "pass"
    assert report["summary"]["snapshots"] == 1
    assert report["summary"]["fill_rate"] == 1.0


@pytest.mark.unit
def test_paper_health_watchdog_strict_flags_missing_and_stale_inputs(
    tmp_path, mock_config
):
    journal = tmp_path / "trade_journal.jsonl"
    mock_config["data"]["storage"] = {"db_path": str(tmp_path / "missing.duckdb")}
    journal.write_text(
        json.dumps(
            {
                "timestamp": _iso(120),
                "execution": {"mode": "paper"},
                "book": {"role": "primary"},
                "decision": "SHORT",
            }
        )
        + "\n{bad-json\n",
        encoding="utf-8",
    )

    report = generate_paper_health_report(
        mock_config,
        runtime_status_path=str(tmp_path / "missing_status.json"),
        journal_path=str(journal),
        strict=True,
        max_decision_age_minutes=30,
    )

    codes = {finding["code"] for finding in report["findings"]}
    assert report["status"] == "fail"
    assert "runtime_status_missing" in codes
    assert "paper_decision_stale" in codes
    assert "paper_equity_snapshots_missing" in codes
    assert "journal_invalid_rows" in codes


@pytest.mark.unit
def test_paper_health_watchdog_flags_bad_runtime_state(tmp_path, mock_config):
    runtime = tmp_path / "runtime_status.json"
    mock_config["data"]["storage"] = {"db_path": str(tmp_path / "missing.duckdb")}
    runtime.write_text(
        json.dumps(
            {
                "operator_mode": "live",
                "kill_switch_active": True,
            }
        ),
        encoding="utf-8",
    )

    report = generate_paper_health_report(
        mock_config,
        runtime_status_path=str(runtime),
        journal_path=str(tmp_path / "missing_journal.jsonl"),
        strict=True,
    )

    codes = {finding["code"] for finding in report["findings"]}
    assert report["status"] == "fail"
    assert "operator_not_paper" in codes
    assert "runtime_timestamp_missing" in codes
    assert "kill_switch_active" in codes
    assert "last_explanation_missing" in codes
    assert "portfolio_missing" in codes
    assert "paper_decisions_missing" in codes


@pytest.mark.unit
def test_shadow_sanity_monitor_passes_for_virtual_shadow_evidence(
    tmp_path, mock_config
):
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))
    model_path = registry.register_model("shadow-v1", "GBM", {"sharpe": 1.1})
    GBMAgent(mock_config).save(model_path)
    registry.write_model_manifest("shadow-v1", data_snapshot_id="snapshot-1")
    registry.promote_to_shadow("shadow-v1")
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        json.dumps(
            {
                "timestamp": _iso(),
                "book": {"role": "shadow", "id": "shadow_shadow-v1"},
                "model_id": "shadow-v1",
                "action": 2,
                "approved_fraction": 0.2,
                "hedge": {
                    "enabled": True,
                    "selected": "protective_hedge",
                    "candidates": {"protective_hedge": 0.8},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    mock_config["shadow"] = {
        "enabled": True,
        "decision_log_path": str(decisions),
    }

    report = generate_shadow_sanity_report(
        mock_config, registry=registry, decision_path=str(decisions), strict=True
    )

    assert report["status"] == "pass"
    assert report["summary"]["active_shadow"] == "shadow-v1"
    assert report["summary"]["shadow_decision_rows"] == 1


@pytest.mark.unit
def test_shadow_sanity_monitor_flags_missing_active_and_bad_book_tags(
    tmp_path, mock_config
):
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        json.dumps(
            {
                "timestamp": _iso(90),
                "book": {"role": "primary", "id": "shadow_bad"},
                "model_id": "bad",
                "hedge": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )
    mock_config["shadow"] = {"enabled": False, "decision_log_path": str(decisions)}

    report = generate_shadow_sanity_report(
        mock_config,
        registry=registry,
        decision_path=str(decisions),
        strict=True,
        max_decision_age_minutes=30,
    )

    codes = {finding["code"] for finding in report["findings"]}
    assert report["status"] == "fail"
    assert "active_shadow_missing" in codes
    assert "shadow_decision_stale" in codes
    assert "shadow_book_tagging_invalid" in codes
    assert "hedge_candidate_scores_missing" in codes


@pytest.mark.unit
def test_shadow_sanity_monitor_flags_registry_and_artifact_inconsistency(
    tmp_path, mock_config
):
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))
    registry.registry_data["active_shadow"] = "missing-model"
    registry._save_registry()
    mock_config["shadow"] = {"enabled": True}

    missing = generate_shadow_sanity_report(mock_config, registry=registry, strict=True)
    assert missing["status"] == "fail"
    assert {finding["code"] for finding in missing["findings"]} >= {
        "active_shadow_registry_missing",
        "shadow_decisions_missing",
    }

    model_path = registry.register_model("candidate-v1", "GBM", {"sharpe": 0.1})
    registry.registry_data["active_shadow"] = "candidate-v1"
    registry._save_registry()
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        json.dumps(
            {
                "timestamp": _iso(),
                "book": {"role": "shadow", "id": "shadow_candidate-v1"},
                "model_id": "other-model",
            }
        )
        + "\n{bad-json\n",
        encoding="utf-8",
    )

    inconsistent = generate_shadow_sanity_report(
        mock_config, registry=registry, decision_path=str(decisions), strict=True
    )

    codes = {finding["code"] for finding in inconsistent["findings"]}
    assert model_path.endswith("candidate-v1")
    assert inconsistent["status"] == "fail"
    assert "active_shadow_status_unexpected" in codes
    assert "active_shadow_artifact_missing" in codes
    assert "active_shadow_manifest_missing" in codes
    assert "decision_log_invalid_rows" in codes
    assert "active_shadow_decisions_missing" in codes


@pytest.mark.unit
def test_model_governance_report_summarizes_registry_runs_and_promotion(
    tmp_path, mock_config
):
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))
    prod_path = registry.register_model(
        "prod-v1", "GBM", {"sharpe": 0.4}, status="SHADOW"
    )
    GBMAgent(mock_config).save(prod_path)
    registry.write_model_manifest("prod-v1", data_snapshot_id="prod-snapshot")
    registry.approve_for_prod("prod-v1")
    registry.promote_to_prod("prod-v1")

    shadow_path = registry.register_model("shadow-v1", "GBM", {"sharpe": 1.0})
    GBMAgent(mock_config).save(shadow_path)
    registry.write_model_manifest("shadow-v1", data_snapshot_id="shadow-snapshot")
    registry.promote_to_shadow("shadow-v1")

    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        "\n".join(
            json.dumps(
                {
                    "timestamp": _iso(),
                    "book": {"role": "shadow", "id": "shadow_shadow-v1"},
                    "model_id": "shadow-v1",
                    "action": 2,
                    "approved_fraction": 0.2,
                    "equity": equity,
                }
            )
            for equity in (1000.0, 1010.0, 1005.0)
        ),
        encoding="utf-8",
    )
    tracker = ExperimentTracker(str(tmp_path / "experiments.jsonl"))
    started = tracker.start_run("nightly_retrain")
    tracker.complete_run(started["run_id"], "PASSED", model_id="shadow-v1")
    mock_config["shadow"] = {"decision_log_path": str(decisions)}
    mock_config["promotion"] = {"min_shadow_trades": 50}

    report = generate_model_governance_report(
        mock_config, registry=registry, tracker=tracker, decision_path=str(decisions)
    )

    assert report["production_readiness"]["ready"] is True
    assert report["summary"]["active_prod"] == "prod-v1"
    assert report["summary"]["active_shadow"] == "shadow-v1"
    assert report["summary"]["latest_run_status"] == "PASSED"
    assert report["promotion_decision"]["reason"] == "insufficient_shadow_trades"
    assert report["shadow_metrics"]["book_id"] == "shadow_shadow-v1"


@pytest.mark.unit
def test_model_governance_report_flags_empty_registry(tmp_path, mock_config):
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))
    tracker = ExperimentTracker(str(tmp_path / "missing" / "experiments.jsonl"))

    report = generate_model_governance_report(
        mock_config, registry=registry, tracker=tracker
    )

    codes = {finding["code"] for finding in report["findings"]}
    assert report["status"] == "warn"
    assert report["summary"]["recommendation"] == "train_and_promote_prod_candidate"
    assert "active_prod_missing" in codes
    assert "active_shadow_missing" in codes
    assert "experiment_runs_missing" in codes


@pytest.mark.unit
def test_model_governance_report_promote_and_readiness_recommendations(
    tmp_path, mock_config
):
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))
    prod_path = registry.register_model("prod-v1", "GBM", {"sharpe": 0.1})
    GBMAgent(mock_config).save(prod_path)
    registry.registry_data["active_prod"] = "prod-v1"
    registry.registry_data["models"]["prod-v1"]["status"] = "PROD"
    registry._save_registry()

    tracker = ExperimentTracker(str(tmp_path / "experiments.jsonl"))
    started = tracker.start_run("nightly_retrain")
    tracker.complete_run(started["run_id"], "FAILED")

    not_ready = generate_model_governance_report(
        mock_config, registry=registry, tracker=tracker
    )

    assert not_ready["summary"]["recommendation"] == "fix_active_prod_readiness"
    assert "active_prod_not_ready" in {
        finding["code"] for finding in not_ready["findings"]
    }
    assert "latest_run_not_successful" in {
        finding["code"] for finding in not_ready["findings"]
    }

    registry.write_model_manifest("prod-v1", data_snapshot_id="prod-snapshot")
    shadow_path = registry.register_model("shadow-v1", "GBM", {"sharpe": 3.0})
    GBMAgent(mock_config).save(shadow_path)
    registry.write_model_manifest("shadow-v1", data_snapshot_id="shadow-snapshot")
    registry.promote_to_shadow("shadow-v1")
    decisions = tmp_path / "promote_decisions.jsonl"
    decisions.write_text(
        "\n".join(
            json.dumps(
                {
                    "timestamp": _iso(),
                    "book": {"role": "shadow", "id": "shadow-v1"},
                    "model_id": "shadow-v1",
                    "action": 2,
                    "approved_fraction": 0.2,
                    "equity": equity,
                }
            )
            for equity in (1000.0, 1015.0, 1008.0, 1035.0)
        ),
        encoding="utf-8",
    )
    mock_config["promotion"] = {
        "min_shadow_trades": 1,
        "min_sharpe_delta": -1.0,
        "max_shadow_drawdown": 1.0,
    }

    promoted = generate_model_governance_report(
        mock_config,
        registry=registry,
        tracker=tracker,
        decision_path=str(decisions),
    )

    assert promoted["promotion_decision"]["action"] == "promote"
    assert promoted["summary"]["recommendation"] == "review_shadow_for_prod_promotion"
    assert "shadow_ready_for_review" in {
        finding["code"] for finding in promoted["findings"]
    }


@pytest.mark.unit
def test_experiment_ledger_auditor_passes_for_governed_candidate_run(
    tmp_path, mock_config
):
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))
    model_path = registry.register_model("candidate-v1", "GBM", {"sharpe": 1.8})
    GBMAgent(mock_config).save(model_path)
    registry.write_model_manifest("candidate-v1", data_snapshot_id="snapshot-1")
    tracker = ExperimentTracker(str(tmp_path / "experiments.jsonl"))
    started = tracker.start_run("candidate_retrain")
    run_id = started["run_id"]
    for step in ("data_snapshot", "train", "oos_backtest", "stress"):
        tracker.log_step(run_id, step, "PASSED")
    tracker.complete_run(run_id, "COMPLETED", model_id="candidate-v1")

    report = generate_experiment_ledger_audit(
        mock_config, tracker=tracker, registry=registry, strict=True
    )

    assert report["status"] == "pass"
    assert report["summary"]["runs"] == 1
    assert report["summary"]["latest_run_status"] == "COMPLETED"


@pytest.mark.unit
def test_experiment_ledger_auditor_flags_malformed_and_stale_runs(
    tmp_path, mock_config
):
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))
    ledger = tmp_path / "experiments.jsonl"
    ledger.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "run_started",
                        "run_id": "run-stale",
                        "run_type": "candidate_retrain",
                        "status": "RUNNING",
                        "timestamp": _iso(500),
                    }
                ),
                json.dumps(
                    {
                        "event": "step",
                        "run_id": "run-stale",
                        "step": "train",
                        "status": "FAILED",
                        "timestamp": _iso(490),
                    }
                ),
                json.dumps(
                    {
                        "event": "run_completed",
                        "run_id": "run-missing-start",
                        "status": "COMPLETED",
                        "model_id": "ghost-model",
                        "timestamp": _iso(),
                    }
                ),
                "{bad-json",
            ]
        ),
        encoding="utf-8",
    )
    tracker = ExperimentTracker(str(ledger))

    report = generate_experiment_ledger_audit(
        mock_config,
        tracker=tracker,
        registry=registry,
        max_running_age_minutes=30,
        strict=True,
    )

    codes = {finding["code"] for finding in report["findings"]}
    assert report["status"] == "fail"
    assert "ledger_invalid_rows" in codes
    assert "stale_running_run" in codes
    assert "run_failed_steps" in codes
    assert "run_start_missing" in codes
    assert "run_model_missing_registry" in codes


@pytest.mark.unit
def test_frontend_api_contract_smoke_matches_current_browser_terminal():
    report = generate_frontend_api_contract_smoke()

    assert report["status"] == "pass"
    assert report["summary"]["expected_get_endpoints"] >= 10
    assert "/ws/status" in report["routes"]["websocket"]


@pytest.mark.unit
def test_frontend_api_contract_smoke_flags_missing_controls(tmp_path):
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "app.js").write_text(
        'const apiBase = "http://localhost:8080"; fetch(`${apiBase}/status`);',
        encoding="utf-8",
    )
    (frontend / "index.html").write_text("<html></html>", encoding="utf-8")
    api = tmp_path / "server.py"
    api.write_text('@app.get("/status")\ndef status(): pass\n', encoding="utf-8")

    report = generate_frontend_api_contract_smoke(
        frontend_dir=str(frontend),
        api_server_path=str(api),
        strict=True,
    )

    codes = {finding["code"] for finding in report["findings"]}
    assert report["status"] == "fail"
    assert "frontend_endpoint_missing" in codes
    assert "api_endpoint_missing" in codes
    assert "frontend_control_post_missing" in codes
    assert "frontend_root_missing" in codes


@pytest.mark.unit
def test_frontend_api_contract_smoke_live_helpers_and_cli_failure(
    tmp_path, monkeypatch, capsys
):
    assert frontend_api_contract_smoke._api_routes(tmp_path / "missing.py") == {}
    assert frontend_api_contract_smoke._route_present(
        {"post": {"/control/{command}"}}, "post", "/control/kill-switch"
    )
    assert (
        frontend_api_contract_smoke._live_get_json("ftp://host", "/status", 1)["error"]
        == "unsupported_url_scheme"
    )

    class ErrorConnection:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            raise OSError("connection refused")

        def close(self):
            pass

    monkeypatch.setattr(
        frontend_api_contract_smoke.http.client, "HTTPConnection", ErrorConnection
    )
    refused = frontend_api_contract_smoke._live_get_json(
        "http://127.0.0.1:8080", "/status", 1
    )
    assert refused["ok"] is False
    assert refused["error"] == "connection refused"

    class Response:
        def __init__(self, status, body):
            self.status = status
            self._body = body

        def read(self):
            return self._body

    class JsonConnection:
        status = 200
        body = b'{"status": "ok"}'

        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return Response(self.status, self.body)

        def close(self):
            pass

    monkeypatch.setattr(
        frontend_api_contract_smoke.http.client, "HTTPConnection", JsonConnection
    )
    live = frontend_api_contract_smoke._live_get_json(
        "http://127.0.0.1:8080", "/status", 1
    )
    assert live["ok"] is True
    assert live["keys"] == ["status"]

    JsonConnection.status = 503
    JsonConnection.body = b"not-json"
    non_json = frontend_api_contract_smoke._live_get_json(
        "http://127.0.0.1:8080", "/status", 1
    )
    assert non_json["error"] == "response_not_json"
    assert non_json["status_code"] == 503

    monkeypatch.setattr(
        frontend_api_contract_smoke,
        "_live_get_json",
        lambda *args, **kwargs: {
            "url": "http://127.0.0.1:8080/status",
            "ok": False,
            "status_code": 500,
            "error": "down",
        },
    )
    missing = generate_frontend_api_contract_smoke(
        frontend_dir=str(tmp_path / "no_frontend"),
        api_server_path=str(tmp_path / "missing_server.py"),
        live_api=True,
        strict=True,
    )
    missing_codes = {finding["code"] for finding in missing["findings"]}
    assert missing["summary"]["live_checks"] == len(
        frontend_api_contract_smoke.SMOKE_URLS
    )
    assert "frontend_app_missing" in missing_codes
    assert "frontend_index_missing" in missing_codes
    assert "api_server_missing" in missing_codes
    assert "live_api_endpoint_failed" in missing_codes

    monkeypatch.setattr(
        frontend_api_contract_smoke,
        "generate_frontend_api_contract_smoke",
        lambda *args, **kwargs: {
            "title": "Frontend",
            "status": "warn",
            "summary": {},
            "findings": [{"severity": "warning", "code": "x", "message": "warn"}],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["frontend_api_contract_smoke", "--fail-on-warning"],
    )
    with pytest.raises(SystemExit):
        frontend_api_contract_smoke.main()
    assert '"title": "Frontend"' in capsys.readouterr().out


@pytest.mark.unit
def test_data_freshness_report_passes_for_fresh_complete_duckdb(tmp_path, mock_config):
    db_path = str(tmp_path / "fresh.duckdb")
    now = pd.Timestamp.now(tz="UTC").floor("s").tz_localize(None)
    timestamps = pd.date_range(now - pd.Timedelta(minutes=6), periods=3, freq="3min")
    cache = DuckDBCacheManager(db_path=db_path)
    try:
        base = pd.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": "ETHUSDC",
                "timeframe": "3m",
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [10.0, 11.0, 12.0],
            }
        )
        cache.insert_ohlcv(base)
        cache.insert_ticks(
            pd.DataFrame(
                {
                    "timestamp": [now],
                    "symbol": ["ETHUSDC"],
                    "price": [102.5],
                    "quantity": [1.0],
                    "is_buyer_maker": [False],
                    "trade_id": [1],
                }
            )
        )
        cache.insert_features(
            base.assign(features=[{"r": 1}, {"r": 2}, {"r": 3}])[
                ["timestamp", "symbol", "timeframe", "features"]
            ]
        )
        cache.insert_market_snapshot("ETHUSDC", now, 0.001, 10_000.0, 102.5)
        cache.insert_paper_equity_snapshot(
            "primary", 1000.0, 0.1, 0.0, 102.5, "TREND", timestamp=now
        )
    finally:
        cache.close()
    mock_config["data"]["storage"] = {"db_path": db_path}

    report = generate_data_freshness_report(
        mock_config,
        now=now.to_pydatetime(),
        max_ohlcv_age_minutes=10,
        max_tick_age_minutes=10,
        max_market_age_minutes=10,
        max_feature_age_minutes=10,
        max_equity_age_minutes=10,
        strict=True,
    )

    assert report["status"] == "pass"
    assert report["summary"]["tables"]["ohlcv"] == 3
    assert report["summary"]["ohlcv_gap_count"] == 0


@pytest.mark.unit
def test_data_freshness_report_flags_missing_db_and_corrupt_features(
    tmp_path, mock_config
):
    missing = generate_data_freshness_report(
        mock_config,
        db_path=str(tmp_path / "missing.duckdb"),
        strict=True,
    )
    assert missing["status"] == "fail"
    assert {finding["code"] for finding in missing["findings"]} == {"duckdb_missing"}

    db_path = str(tmp_path / "stale.duckdb")
    old = (pd.Timestamp.now(tz="UTC").floor("s") - pd.Timedelta(hours=3)).tz_localize(
        None
    )
    cache = DuckDBCacheManager(db_path=db_path)
    try:
        cache.insert_ohlcv(
            pd.DataFrame(
                {
                    "timestamp": [old, old + pd.Timedelta(minutes=9)],
                    "symbol": ["ETHUSDC", "ETHUSDC"],
                    "timeframe": ["3m", "3m"],
                    "open": [100.0, 101.0],
                    "high": [101.0, 102.0],
                    "low": [99.0, 100.0],
                    "close": [100.5, 101.5],
                    "volume": [10.0, 11.0],
                }
            )
        )
        cache.conn.execute(
            """
            INSERT INTO features
            VALUES (?, ?, ?, ?, ?)
            """,
            [old, "ETHUSDC", "3m", "default", "{bad-json"],
        )
    finally:
        cache.close()
    mock_config["data"]["storage"] = {"db_path": db_path}

    stale = generate_data_freshness_report(
        mock_config,
        now=pd.Timestamp.now(tz="UTC").to_pydatetime(),
        max_ohlcv_age_minutes=5,
        max_feature_age_minutes=5,
        strict=True,
    )

    codes = {finding["code"] for finding in stale["findings"]}
    assert stale["status"] == "fail"
    assert "ohlcv_stale" in codes
    assert "ohlcv_gaps_detected" in codes
    assert "feature_json_invalid" in codes


@pytest.mark.unit
def test_data_freshness_report_flags_integrity_edges_and_cli_failure(
    tmp_path, mock_config, monkeypatch, capsys
):
    partial_db = tmp_path / "partial.duckdb"
    conn = duckdb.connect(str(partial_db))
    try:
        conn.execute(
            """
            CREATE TABLE ohlcv (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                timeframe VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE
            )
            """
        )
    finally:
        conn.close()

    partial = generate_data_freshness_report(
        mock_config,
        db_path=str(partial_db),
        strict=True,
    )
    partial_codes = {finding["code"] for finding in partial["findings"]}
    assert "duckdb_table_missing" in partial_codes
    assert "duckdb_table_empty" in partial_codes
    assert "ohlcv_timestamp_missing" in partial_codes

    db_path = tmp_path / "integrity.duckdb"
    conn = duckdb.connect(str(db_path))
    now = pd.Timestamp.now(tz="UTC").floor("s").tz_localize(None)
    try:
        conn.execute(
            """
            CREATE TABLE ohlcv (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                timeframe VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE ticks (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                price DOUBLE,
                quantity DOUBLE,
                is_buyer_maker BOOLEAN,
                trade_id BIGINT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE features (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                timeframe VARCHAR,
                feature_set_id VARCHAR,
                features_json VARCHAR
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE market_snapshots (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                funding_rate DOUBLE,
                open_interest DOUBLE,
                mark_price DOUBLE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE paper_equity_snapshots (
                timestamp TIMESTAMP,
                book_id VARCHAR,
                equity DOUBLE,
                long_qty DOUBLE,
                short_qty DOUBLE,
                mark_price DOUBLE,
                regime VARCHAR
            )
            """
        )
        conn.executemany(
            "INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (now, "ETHUSDC", "3m", 100, 101, 99, 100, 10),
                (now, "ETHUSDC", "3m", None, 101, 99, 100, -1),
            ],
        )
        conn.executemany(
            "INSERT INTO ticks VALUES (?, ?, ?, ?, ?, ?)",
            [
                (now, "ETHUSDC", 100, 1, False, 7),
                (now, "ETHUSDC", 101, 1, True, 7),
            ],
        )
        conn.executemany(
            "INSERT INTO features VALUES (?, ?, ?, ?, ?)",
            [
                (now, "ETHUSDC", "3m", "default", "{bad-json"),
                (now, "ETHUSDC", "3m", "default", '{"ok": true}'),
            ],
        )
        conn.executemany(
            "INSERT INTO market_snapshots VALUES (?, ?, ?, ?, ?)",
            [
                (now, "ETHUSDC", 0.0, 1000, 100),
                (now, "ETHUSDC", 0.0, 1000, 100),
            ],
        )
        conn.executemany(
            "INSERT INTO paper_equity_snapshots VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (now, "primary", 1000, 0.1, 0.0, 100, "TREND"),
                (now, "primary", 1001, 0.1, 0.0, 100, "TREND"),
            ],
        )
    finally:
        conn.close()

    integrity = generate_data_freshness_report(
        mock_config,
        db_path=str(db_path),
        now=now.to_pydatetime(),
        max_ohlcv_age_minutes=10,
        max_tick_age_minutes=10,
        max_market_age_minutes=10,
        max_feature_age_minutes=10,
        max_equity_age_minutes=10,
        strict=True,
    )
    codes = {finding["code"] for finding in integrity["findings"]}
    assert "ohlcv_duplicate_keys" in codes
    assert "ohlcv_null_required_fields" in codes
    assert "ohlcv_negative_volume" in codes
    assert "tick_duplicate_keys" in codes
    assert "feature_duplicate_keys" in codes
    assert "feature_json_invalid" in codes
    assert "market_snapshot_duplicate_keys" in codes
    assert "paper_equity_duplicate_keys" in codes

    unreadable_db = tmp_path / "unreadable.duckdb"
    unreadable_db.touch()
    monkeypatch.setattr(
        data_freshness_check.duckdb,
        "connect",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    unreadable = generate_data_freshness_report(
        mock_config,
        db_path=str(unreadable_db),
        strict=True,
    )
    assert "duckdb_unreadable" in {
        finding["code"] for finding in unreadable["findings"]
    }

    monkeypatch.setattr(data_freshness_check, "load_config", lambda path: mock_config)
    monkeypatch.setattr(
        data_freshness_check,
        "generate_data_freshness_report",
        lambda *args, **kwargs: {
            "title": "Data",
            "status": "fail",
            "summary": {},
            "findings": [{"severity": "error", "code": "x", "message": "fail"}],
        },
    )
    monkeypatch.setattr(sys, "argv", ["data_freshness_check"])
    with pytest.raises(SystemExit):
        data_freshness_check.main()
    assert '"title": "Data"' in capsys.readouterr().out


@pytest.mark.unit
def test_experiment_ledger_auditor_flags_registry_and_ledger_edges(
    tmp_path, mock_config
):
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))
    model_path = registry.register_model("candidate-v1", "GBM", {"sharpe": 0.5})
    assert experiment_ledger_auditor._artifact_exists({}) is False
    misc_dir = tmp_path / "custom_artifact"
    misc_dir.mkdir()
    (misc_dir / "artifact.bin").write_text("x", encoding="utf-8")
    assert experiment_ledger_auditor._artifact_exists(
        {"artifact_path": str(misc_dir), "type": "CUSTOM"}
    )

    ledger = tmp_path / "experiments.jsonl"
    future = datetime(2026, 5, 21, 10, 0, tzinfo=timezone.utc).isoformat()
    past = datetime(2026, 5, 21, 9, 0, tzinfo=timezone.utc).isoformat()
    ledger.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "run_started",
                        "run_id": "run-dup",
                        "run_type": "candidate_retrain",
                        "status": "RUNNING",
                        "timestamp": future,
                    }
                ),
                json.dumps(
                    {
                        "event": "run_started",
                        "run_id": "run-dup",
                        "run_type": "candidate_retrain",
                        "status": "RUNNING",
                        "timestamp": future,
                    }
                ),
                json.dumps(
                    {
                        "event": "run_completed",
                        "run_id": "run-dup",
                        "status": "MYSTERY",
                        "timestamp": past,
                    }
                ),
                json.dumps(
                    {
                        "event": "run_completed",
                        "run_id": "run-dup",
                        "status": "MYSTERY",
                        "timestamp": past,
                    }
                ),
                json.dumps(
                    {
                        "event": "run_started",
                        "run_id": "run-fail",
                        "run_type": "nightly",
                        "status": "RUNNING",
                        "timestamp": past,
                    }
                ),
                json.dumps(
                    {
                        "event": "run_completed",
                        "run_id": "run-fail",
                        "status": "FAILED",
                        "timestamp": future,
                    }
                ),
                json.dumps(
                    {
                        "event": "run_started",
                        "run_id": "run-model",
                        "run_type": "candidate_retrain",
                        "status": "RUNNING",
                        "timestamp": past,
                    }
                ),
                json.dumps(
                    {
                        "event": "step",
                        "run_id": "run-model",
                        "step": "train",
                        "status": "PASSED",
                        "timestamp": past,
                    }
                ),
                json.dumps(
                    {
                        "event": "run_completed",
                        "run_id": "run-model",
                        "status": "COMPLETED",
                        "model_id": "candidate-v1",
                        "timestamp": future,
                    }
                ),
                json.dumps({"event": "surprise", "run_id": "run-unknown"}),
                json.dumps({"event": "step", "step": "orphan"}),
            ]
        ),
        encoding="utf-8",
    )
    tracker = ExperimentTracker(str(ledger))

    report = generate_experiment_ledger_audit(
        mock_config,
        tracker=tracker,
        registry=registry,
        strict=True,
        now=datetime(2026, 5, 21, 10, 30, tzinfo=timezone.utc),
    )

    codes = {finding["code"] for finding in report["findings"]}
    assert model_path.endswith("candidate-v1")
    assert "duplicate_run_start" in codes
    assert "duplicate_run_completion" in codes
    assert "completion_before_start" in codes
    assert "unknown_run_status" in codes
    assert "run_unsuccessful" in codes
    assert "candidate_retrain_steps_missing" in codes
    assert "run_model_manifest_missing" in codes
    assert "run_model_artifact_missing" in codes
    assert "run_model_snapshot_missing" in codes
    assert "ledger_unknown_events" in codes
    assert "ledger_events_without_run_id" in codes

    empty_ledger = tmp_path / "empty.jsonl"
    empty_ledger.write_text("", encoding="utf-8")
    empty = generate_experiment_ledger_audit(
        mock_config,
        tracker=ExperimentTracker(str(empty_ledger)),
        registry=registry,
        ledger_path=str(empty_ledger),
    )
    assert "experiment_runs_missing" in {
        finding["code"] for finding in empty["findings"]
    }

    missing = generate_experiment_ledger_audit(
        mock_config,
        tracker=ExperimentTracker(str(tmp_path / "missing.jsonl")),
        registry=registry,
        ledger_path=str(tmp_path / "missing.jsonl"),
        strict=True,
    )
    assert "ledger_missing" in {finding["code"] for finding in missing["findings"]}


@pytest.mark.unit
def test_operational_report_clis_write_outputs(tmp_path, monkeypatch, mock_config):
    paper_out = tmp_path / "paper.md"
    shadow_out = tmp_path / "shadow.json"
    governance_out = tmp_path / "governance.json"
    ledger_out = tmp_path / "ledger.json"
    frontend_out = tmp_path / "frontend.json"
    data_out = tmp_path / "data.json"

    monkeypatch.setattr(paper_health_watchdog, "load_config", lambda path: mock_config)
    monkeypatch.setattr(
        paper_health_watchdog,
        "generate_paper_health_report",
        lambda *args, **kwargs: {
            "title": "Paper",
            "status": "pass",
            "summary": {},
            "findings": [],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paper_health_watchdog",
            "--config",
            "config.yaml",
            "--format",
            "markdown",
            "--output",
            str(paper_out),
        ],
    )
    paper_health_watchdog.main()

    monkeypatch.setattr(shadow_sanity_monitor, "load_config", lambda path: mock_config)
    monkeypatch.setattr(
        shadow_sanity_monitor,
        "generate_shadow_sanity_report",
        lambda *args, **kwargs: {
            "title": "Shadow",
            "status": "pass",
            "summary": {},
            "findings": [],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["shadow_sanity_monitor", "--output", str(shadow_out)],
    )
    shadow_sanity_monitor.main()

    monkeypatch.setattr(
        model_governance_report, "load_config", lambda path: mock_config
    )
    monkeypatch.setattr(
        model_governance_report,
        "generate_model_governance_report",
        lambda *args, **kwargs: {
            "title": "Governance",
            "status": "pass",
            "summary": {},
            "findings": [],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["model_governance_report", "--output", str(governance_out)],
    )
    model_governance_report.main()

    monkeypatch.setattr(
        experiment_ledger_auditor, "load_config", lambda path: mock_config
    )
    monkeypatch.setattr(
        experiment_ledger_auditor,
        "generate_experiment_ledger_audit",
        lambda *args, **kwargs: {
            "title": "Ledger",
            "status": "pass",
            "summary": {},
            "findings": [],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["experiment_ledger_auditor", "--output", str(ledger_out)],
    )
    experiment_ledger_auditor.main()

    monkeypatch.setattr(
        frontend_api_contract_smoke,
        "generate_frontend_api_contract_smoke",
        lambda *args, **kwargs: {
            "title": "Frontend",
            "status": "pass",
            "summary": {},
            "findings": [],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["frontend_api_contract_smoke", "--output", str(frontend_out)],
    )
    frontend_api_contract_smoke.main()

    monkeypatch.setattr(data_freshness_check, "load_config", lambda path: mock_config)
    monkeypatch.setattr(
        data_freshness_check,
        "generate_data_freshness_report",
        lambda *args, **kwargs: {
            "title": "Data",
            "status": "pass",
            "summary": {},
            "findings": [],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["data_freshness_check", "--output", str(data_out)],
    )
    data_freshness_check.main()

    assert "# Paper" in paper_out.read_text(encoding="utf-8")
    assert json.loads(shadow_out.read_text(encoding="utf-8"))["title"] == "Shadow"
    assert (
        json.loads(governance_out.read_text(encoding="utf-8"))["title"] == "Governance"
    )
    assert json.loads(ledger_out.read_text(encoding="utf-8"))["title"] == "Ledger"
    assert json.loads(frontend_out.read_text(encoding="utf-8"))["title"] == "Frontend"
    assert json.loads(data_out.read_text(encoding="utf-8"))["title"] == "Data"
