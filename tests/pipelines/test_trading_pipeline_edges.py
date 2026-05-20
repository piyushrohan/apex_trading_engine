import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.execution.adapters.base import OrderResult
from src.execution.adapters.live import LiveExecutionAdapter
from src.execution.adapters.paper import PaperExecutionAdapter
from src.execution.portfolio import PortfolioService
from src.pipelines import trading_pipeline
from src.pipelines.trading_pipeline import TradingPipeline
from src.strategies.hedge.base import HedgeContext, HedgeProposal


class FakeMetrics:
    def __init__(self):
        self.started = []
        self.ws_health = []
        self.pnl = []
        self.fill_rates = []

    def start_server(self, port):
        self.started.append(port)

    def set_ws_health(self, mode, enabled):
        self.ws_health.append((mode, enabled))

    def set_pnl(self, mode, role, book_id, pnl):
        self.pnl.append((mode, role, book_id, pnl))

    def set_paper_fill_rate(self, mode, book_id, fill_rate):
        self.fill_rates.append((mode, book_id, fill_rate))

    def time_inference(self, mode, fn, *args):
        return fn(*args)


class FakeRisk:
    def __init__(self, approvals=None):
        self.approvals = list(approvals or [0.2])
        self.is_kill_switch_active = False
        self.equity_updates = []
        self.orders = []

    def calculate_kelly_size(self, win_rate, win_loss_ratio, conviction):
        return 0.25

    def approve_order(self, side, fraction, **kwargs):
        self.orders.append((side, fraction, kwargs))
        if self.approvals:
            return self.approvals.pop(0)
        return fraction

    def update_equity(self, equity):
        self.equity_updates.append(equity)


class FakeAdapter:
    def __init__(self, success=True):
        self.place_order = AsyncMock(
            return_value=OrderResult(success=success, order_id="order-1")
        )
        self.cancel_all_orders = AsyncMock(return_value=2)


class FakeStatus:
    regime = "MEAN_REVERSION"

    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


def build_shell_pipeline(tmp_path, *, mode="paper", position_mode="hedge"):
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.config = {
        "data": {
            "target_symbol": "ETHUSDC",
            "ingestion": {"enabled": True},
            "storage": {"db_path": str(tmp_path / "apex.duckdb")},
        },
        "execution": {"operator_mode": mode, "position_mode": position_mode},
        "environment": {"initial_capital": 1000.0},
        "shadow": {"decision_log_path": str(tmp_path / "decisions.jsonl")},
    }
    pipeline.operator_mode = mode
    pipeline.symbol = "ETHUSDC"
    pipeline.position_mode = position_mode
    pipeline.portfolio = PortfolioService(position_mode=position_mode)
    pipeline.primary_book = pipeline.portfolio.get_or_create_book(
        book_id="primary",
        role="primary",
        model_id="prod-v1",
        symbol="ETHUSDC",
        initial_equity=1000.0,
    )
    pipeline.risk_engine = FakeRisk()
    pipeline.execution_adapter = FakeAdapter()
    pipeline.ingestion = MagicMock()
    pipeline.ingestion.get_last_mark_price.return_value = 3500.0
    pipeline.ingestion.cache.insert_paper_equity_snapshot = MagicMock()
    pipeline.ingestion.flush_ticks = AsyncMock()
    pipeline.ingestion.stop = AsyncMock()
    pipeline.ingestion.close = MagicMock()
    pipeline.market_state = MagicMock()
    pipeline.market_state.close = MagicMock()
    pipeline.account_sync = None
    pipeline.rest_client = MagicMock()
    pipeline.rest_client.close = AsyncMock()
    pipeline.explainability = MagicMock()
    pipeline.explainability.decode_portfolio_state.return_value = {
        "event": "portfolio_sync"
    }
    pipeline.explainability._log_to_journal = MagicMock()
    pipeline._status_store = FakeStatus()
    pipeline._running = False
    pipeline._last_approved_fraction = 0.0
    pipeline._last_control_command_id = None
    return pipeline


@pytest.mark.unit
def test_build_rest_client_uses_env_config_and_encrypted_fallback(monkeypatch):
    captured = []

    class FakeREST:
        def __init__(self, api_key=None, api_secret=None):
            captured.append((api_key, api_secret))

    class FakeSecurity:
        def get_api_credentials(self):
            return "vault-key", "vault-secret"

    monkeypatch.setattr(trading_pipeline, "BinanceRESTClient", FakeREST)
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    monkeypatch.setattr(trading_pipeline, "SecurityManager", lambda: FakeSecurity())

    TradingPipeline._build_rest_client({"live": {}})
    TradingPipeline._build_rest_client(
        {"live": {"api_key": "cfg-key", "api_secret": "cfg-secret"}}
    )
    monkeypatch.setenv("BINANCE_API_KEY", "env-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "env-secret")
    TradingPipeline._build_rest_client({"live": {}})

    assert captured == [
        ("vault-key", "vault-secret"),
        ("cfg-key", "cfg-secret"),
        ("env-key", "env-secret"),
    ]


@pytest.mark.unit
def test_load_active_prod_model_success_missing_artifact_and_unregistered():
    pipeline = TradingPipeline.__new__(TradingPipeline)
    pipeline.registry = MagicMock()
    pipeline.registry.registry_data = {"models": {}, "active_prod": None}
    pipeline.meta_controller = MagicMock()
    assert pipeline._load_active_prod_model() == "unregistered"

    pipeline.registry.registry_data = {
        "active_prod": "prod-v1",
        "models": {"prod-v1": {"type": "GBM"}},
    }
    pipeline.registry.get_model_path.return_value = "/models/prod-v1"
    assert pipeline._load_active_prod_model() == "prod-v1"
    pipeline.meta_controller.load_model_artifact.assert_called_once_with(
        "GBM", "/models/prod-v1"
    )

    pipeline.meta_controller.load_model_artifact.side_effect = FileNotFoundError
    assert pipeline._load_active_prod_model() == "prod-v1"


@pytest.mark.unit
def test_start_metrics_server_honors_enabled_flag(monkeypatch):
    metrics = FakeMetrics()
    monkeypatch.setattr(trading_pipeline, "APEX_METRICS", metrics)
    pipeline = TradingPipeline.__new__(TradingPipeline)

    pipeline.config = {"observability": {"metrics": {"enabled": False}}}
    pipeline._start_metrics_server()
    pipeline.config = {"observability": {"metrics": {"enabled": True, "port": 9911}}}
    pipeline._start_metrics_server()

    assert metrics.started == [9911]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_prepare_live_exchange_validates_hedge_mode_and_leverage(tmp_path):
    pipeline = build_shell_pipeline(tmp_path, mode="live", position_mode="hedge")
    pipeline.config["live"] = {"leverage": 3}
    pipeline.rest_client = SimpleNamespace(
        set_hedge_mode=AsyncMock(return_value=False),
        set_leverage=AsyncMock(return_value={"leverage": 3}),
    )
    with pytest.raises(RuntimeError, match="hedge mode"):
        await pipeline._prepare_live_exchange()

    pipeline.rest_client.set_hedge_mode = AsyncMock(return_value=True)
    pipeline.rest_client.set_leverage = AsyncMock(return_value=None)
    with pytest.raises(RuntimeError, match="set leverage"):
        await pipeline._prepare_live_exchange()

    pipeline.rest_client.set_leverage = AsyncMock(return_value={"leverage": 3})
    await pipeline._prepare_live_exchange()
    pipeline.rest_client.set_leverage.assert_awaited_once_with("ETHUSDC", 3)


@pytest.mark.unit
def test_validate_startup_blocks_live_when_credentials_fail(monkeypatch, tmp_path):
    pipeline = build_shell_pipeline(tmp_path, mode="live")
    monkeypatch.setattr(trading_pipeline, "validate_live_startup", lambda config: None)
    monkeypatch.setattr(
        trading_pipeline,
        "check_api_credentials",
        lambda config: (False, "missing credentials"),
    )

    with pytest.raises(RuntimeError, match="missing credentials"):
        pipeline._validate_startup()

    pipeline.operator_mode = "paper"
    pipeline._validate_startup()


@pytest.mark.unit
def test_validate_startup_blocks_live_without_ready_prod_model(monkeypatch, tmp_path):
    pipeline = build_shell_pipeline(tmp_path, mode="live")
    pipeline.primary_book.model_id = "unregistered"
    pipeline.registry = SimpleNamespace(
        production_readiness=lambda model_id=None: {
            "ready": False,
            "blockers": ["no_active_prod_model"],
        }
    )
    pipeline._model_artifact_loaded = False
    monkeypatch.setattr(trading_pipeline, "validate_live_startup", lambda config: None)
    monkeypatch.setattr(
        trading_pipeline,
        "check_api_credentials",
        lambda config: (True, None),
    )

    with pytest.raises(RuntimeError, match="no approved production model"):
        pipeline._validate_startup()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_live_account_bootstrap_and_snapshot_application(tmp_path):
    pipeline = build_shell_pipeline(tmp_path, mode="live")
    pipeline.account_sync = SimpleNamespace(
        balances={"USDC": {"wallet_balance": "1200"}},
        fetch_snapshot=AsyncMock(
            return_value={
                "long_qty": 0.0,
                "short_qty": 0.0,
                "amount": -0.4,
                "entry_price_short": 3490.0,
                "unrealized_pnl": 12.0,
            }
        ),
    )
    pipeline.ingestion.get_last_mark_price.return_value = None

    pipeline._apply_account_snapshot("BTCUSDC", {"amount": 1}, 100.0)
    await pipeline._bootstrap_live_account()

    assert pipeline.primary_book.short_qty == 0.4
    assert pipeline.primary_book.initial_equity == 1200.0
    assert pipeline.risk_engine.equity_updates == [1212.0]
    assert pipeline._status_store.updates[-1]["last_explanation"] == {
        "event": "portfolio_sync"
    }


@pytest.mark.unit
def test_publish_status_and_bandit_decision_logging(tmp_path, monkeypatch):
    metrics = FakeMetrics()
    monkeypatch.setattr(trading_pipeline, "APEX_METRICS", metrics)
    pipeline = build_shell_pipeline(tmp_path)
    hedge_ctx = HedgeContext(
        symbol="ETHUSDC",
        regime="MEAN_REVERSION",
        mark_price=3500.0,
        feature_vector=[0.1],
        ppo_action_probs=[0.1, 0.2, 0.7],
        gbm_action_probs=[0.7, 0.2, 0.1],
        primary_action=2,
        primary_size_fraction=0.2,
        volatility_zscore=0.3,
        funding_rate=0.0001,
    )

    pipeline._publish_status(regime="MEAN_REVERSION", mark_price=3500.0)
    pipeline._append_hedge_bandit_decision(
        action=2,
        conviction=0.9,
        regime="MEAN_REVERSION",
        hedge_payload={"enabled": False},
        hedge_ctx=hedge_ctx,
    )
    pipeline._append_hedge_bandit_decision(
        action=2,
        conviction=0.9,
        regime="MEAN_REVERSION",
        hedge_payload={"enabled": True, "selected": "protective_hedge"},
        hedge_ctx=hedge_ctx,
    )

    row = json.loads((tmp_path / "decisions.jsonl").read_text(encoding="utf-8"))
    assert metrics.ws_health == [("paper", True)]
    assert metrics.pnl[-1][1:3] == ("primary", "primary")
    assert row["hedge"]["selected"] == "protective_hedge"
    assert row["bandit_context"]["primary_size_fraction"] == 0.2


@pytest.mark.asyncio
@pytest.mark.unit
async def test_operator_controls_pause_kill_clear_and_flatten(tmp_path, monkeypatch):
    control_path = tmp_path / "operator_controls.json"
    monkeypatch.setenv("APEX_CONTROL_STATE_PATH", str(control_path))
    pipeline = build_shell_pipeline(tmp_path, position_mode="hedge")

    def write_control(command, state=None, payload=None):
        body = {
            "paused": False,
            "kill_switch_requested": False,
            "flatten_requested_at": None,
            "last_command": {
                "timestamp": f"2026-05-20T00:00:0{write_control.counter}+00:00",
                "command": command,
                "reason": "test",
                "payload": payload or {},
            },
        }
        write_control.counter += 1
        body.update(state or {})
        control_path.write_text(json.dumps(body), encoding="utf-8")

    write_control.counter = 1

    write_control("pause", {"paused": True})
    assert await pipeline._apply_operator_controls(3500.0) is True

    write_control("kill-switch", {"kill_switch_requested": True})
    assert await pipeline._apply_operator_controls(3500.0) is False
    assert pipeline.risk_engine.is_kill_switch_active is True

    write_control("clear-kill-switch")
    assert await pipeline._apply_operator_controls(3500.0) is False
    assert pipeline.risk_engine.is_kill_switch_active is False

    paper = PaperExecutionAdapter(book_id="primary")
    pipeline.execution_adapter = paper
    pipeline.primary_book.apply_fill("BUY", 0.2, 3500.0, "LONG")
    write_control("flatten", {"flatten_requested_at": "2026-05-20T00:00:04+00:00"})

    assert await pipeline._apply_operator_controls(3490.0) is False
    assert pipeline.primary_book.long_qty == 0.0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_operator_control_applies_risk_profile_and_logs_mode_request(
    tmp_path, monkeypatch
):
    control_path = tmp_path / "operator_controls.json"
    monkeypatch.setenv("APEX_CONTROL_STATE_PATH", str(control_path))
    pipeline = build_shell_pipeline(tmp_path, mode="paper")
    pipeline.risk_engine = trading_pipeline.RiskEngine(pipeline.config)

    control_path.write_text(
        json.dumps(
            {
                "last_command": {
                    "timestamp": "2026-05-20T00:00:00+00:00",
                    "command": "set-risk-profile",
                    "payload": {"profile": "aggressive"},
                }
            }
        ),
        encoding="utf-8",
    )
    await pipeline._apply_operator_controls(3500.0)

    assert pipeline.config["risk"]["profile"] == "aggressive"
    assert pipeline.risk_engine.max_leverage == 5.0

    control_path.write_text(
        json.dumps(
            {
                "last_command": {
                    "timestamp": "2026-05-20T00:00:01+00:00",
                    "command": "set-mode",
                    "payload": {"mode": "live"},
                }
            }
        ),
        encoding="utf-8",
    )

    assert await pipeline._apply_operator_controls(3500.0) is False


@pytest.mark.asyncio
@pytest.mark.unit
async def test_execute_signal_handles_zero_failure_and_hedge_position_side(tmp_path):
    pipeline = build_shell_pipeline(tmp_path, position_mode="hedge")
    assert await pipeline._execute_signal(2, 0.9, {}, 3500.0, 0.0) == 0.0

    pipeline.execution_adapter = FakeAdapter(success=False)
    assert (
        await pipeline._execute_signal(0, 0.9, {"primary_reasons": []}, 3500.0, 0.2)
        == 0.0
    )
    sell_request = pipeline.execution_adapter.place_order.await_args[0][0]
    assert sell_request.side == "SELL"
    assert sell_request.position_side == "SHORT"

    pipeline.execution_adapter = FakeAdapter(success=True)
    assert (
        await pipeline._execute_signal(2, 0.9, {"primary_reasons": []}, 3500.0, 0.2)
        == 0.2
    )
    buy_request = pipeline.execution_adapter.place_order.await_args[0][0]
    assert buy_request.side == "BUY"
    assert buy_request.position_side == "LONG"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_execute_hedge_places_long_short_and_grid_orders(tmp_path, monkeypatch):
    pipeline = build_shell_pipeline(tmp_path, position_mode="hedge")
    pipeline.risk_engine = FakeRisk(approvals=[0.1, 0.2])
    await pipeline._execute_hedge(
        HedgeProposal("protective_hedge", long_delta_qty=0.1, short_delta_qty=0.2),
        3500.0,
    )

    placed = pipeline.execution_adapter.place_order.await_args_list
    assert [call.args[0].side for call in placed] == ["BUY", "SELL"]
    assert [call.args[0].position_side for call in placed] == ["LONG", "SHORT"]

    pipeline.execution_adapter = FakeAdapter()
    pipeline.risk_engine = FakeRisk(approvals=[0.0])
    await pipeline._execute_hedge(
        HedgeProposal("protective_hedge", long_delta_qty=0.1), 3500.0
    )
    pipeline.execution_adapter.place_order.assert_not_awaited()

    pipeline.execution_adapter = FakeAdapter()
    pipeline.risk_engine = FakeRisk(approvals=[0.0])
    await pipeline._execute_hedge(
        HedgeProposal("protective_hedge", short_delta_qty=0.1), 3500.0
    )
    pipeline.execution_adapter.place_order.assert_not_awaited()

    class FakeGridAdapter:
        def build_grid(self, **kwargs):
            return SimpleNamespace(
                orders=[
                    SimpleNamespace(symbol="ETHUSDC", side="BUY"),
                    SimpleNamespace(symbol="ETHUSDC", side="SELL"),
                ]
            )

    monkeypatch.setattr(trading_pipeline, "MakerGridAdapter", FakeGridAdapter)
    pipeline.execution_adapter = FakeAdapter()
    pipeline.risk_engine = FakeRisk(approvals=[0.3])
    await pipeline._execute_hedge(
        HedgeProposal("maker_grid_hedge", long_delta_qty=0.3),
        3500.0,
    )
    assert pipeline.execution_adapter.place_order.await_count == 2

    pipeline.execution_adapter = FakeAdapter()
    pipeline.risk_engine = FakeRisk(approvals=[0.0])
    await pipeline._execute_hedge(
        HedgeProposal("maker_grid_hedge", long_delta_qty=0.3),
        3500.0,
    )
    pipeline.execution_adapter.place_order.assert_not_awaited()


@pytest.mark.unit
def test_simulate_paper_fills_updates_book_and_metrics(tmp_path, monkeypatch):
    metrics = FakeMetrics()
    monkeypatch.setattr(trading_pipeline, "APEX_METRICS", metrics)
    pipeline = build_shell_pipeline(tmp_path, mode="paper", position_mode="one_way")
    adapter = PaperExecutionAdapter(book_id="primary")
    pipeline.execution_adapter = adapter
    pipeline.explainability._log_to_journal = MagicMock()

    adapter._open_orders["order-1"] = {
        "orderId": "order-1",
        "symbol": "ETHUSDC",
        "side": "BUY",
        "price": 3500.0,
        "origQty": 0.2,
        "executedQty": 0.0,
        "status": "NEW",
        "positionSide": "BOTH",
    }

    pipeline._simulate_paper_fills(3499.0)

    assert pipeline.primary_book.long_qty == 0.2
    assert pipeline.explainability._log_to_journal.called
    assert metrics.fill_rates == [("paper", "primary", 1.0)]

    pipeline.operator_mode = "live"
    pipeline._simulate_paper_fills(3499.0)
    pipeline.operator_mode = "paper"
    pipeline.execution_adapter = FakeAdapter()
    pipeline._simulate_paper_fills(3499.0)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_kill_switch_handles_paper_live_and_generic_adapters(tmp_path):
    pipeline = build_shell_pipeline(tmp_path, position_mode="hedge")
    paper = PaperExecutionAdapter(book_id="primary")
    paper._open_orders["order-1"] = {"symbol": "ETHUSDC", "status": "NEW"}
    pipeline.execution_adapter = paper
    pipeline.primary_book.apply_fill("BUY", 0.2, 3500.0, "LONG")

    await pipeline._handle_kill_switch(3490.0)

    assert paper._open_orders == {}
    assert pipeline.primary_book.long_qty == 0.0

    rest = AsyncMock()
    rest.cancel_all_open_orders.return_value = 1
    rest.close_position_market.return_value = {"orderId": 1}
    pipeline.execution_adapter = LiveExecutionAdapter(
        {"execution": {"position_mode": "hedge"}}, rest
    )
    pipeline.primary_book.apply_fill("BUY", 0.3, 3500.0, "LONG")
    pipeline.primary_book.apply_fill("SELL", 0.1, 3500.0, "SHORT")
    await pipeline._handle_kill_switch(3490.0)
    rest.close_position_market.assert_any_await(
        "ETHUSDC", side="SELL", quantity=0.3, position_side="LONG"
    )
    rest.close_position_market.assert_any_await(
        "ETHUSDC", side="BUY", quantity=0.1, position_side="SHORT"
    )

    pipeline.execution_adapter = FakeAdapter()
    await pipeline._handle_kill_switch(3490.0)
    pipeline.execution_adapter.cancel_all_orders.assert_awaited_once_with("ETHUSDC")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_stop_closes_enabled_components(tmp_path):
    pipeline = build_shell_pipeline(tmp_path, mode="live")
    pipeline.account_sync = SimpleNamespace(stop=AsyncMock())

    await pipeline.stop()

    pipeline.ingestion.stop.assert_awaited_once()
    pipeline.account_sync.stop.assert_awaited_once()
    pipeline.rest_client.close.assert_awaited_once()
    pipeline.ingestion.close.assert_called_once()
    pipeline.market_state.close.assert_called_once()


@pytest.mark.unit
def test_live_model_gate_bypass_and_readiness_unavailable(tmp_path):
    pipeline = build_shell_pipeline(tmp_path, mode="live")

    pipeline.config["models"] = {"allow_unregistered_live": True}
    pipeline._validate_live_model_gate()

    pipeline.config["models"] = {}
    pipeline.registry = object()
    pipeline._validate_live_model_gate()

    readiness = {"ready": True, "blockers": []}
    pipeline.registry = SimpleNamespace(
        production_readiness=lambda model_id=None: readiness
    )
    pipeline._model_artifact_loaded = False
    pipeline._validate_live_model_gate()

    assert readiness["blockers"] == ["missing_model_artifact"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_operator_control_edge_paths(tmp_path, monkeypatch):
    control_path = tmp_path / "operator_controls.json"
    monkeypatch.setenv("APEX_CONTROL_STATE_PATH", str(control_path))
    pipeline = build_shell_pipeline(tmp_path)

    await pipeline._bootstrap_live_account()
    assert pipeline._wallet_equity_from_sync() == 0.0

    control_path.write_text("{not-json", encoding="utf-8")
    assert pipeline._load_operator_controls() == {"error": "control state unreadable"}
    assert pipeline._control_command_id({"last_command": {"command": "pause"}}) is None

    pipeline._apply_operator_risk_profile(None)

    def raise_profile(config, profile):
        raise ValueError("unknown profile")

    monkeypatch.setattr(trading_pipeline, "apply_risk_profile", raise_profile)
    pipeline._apply_operator_risk_profile("does-not-exist")

    control_path.write_text(
        json.dumps(
            {
                "paused": False,
                "kill_switch_requested": False,
                "last_command": {
                    "timestamp": "2026-05-20T00:00:00+00:00",
                    "command": "flatten",
                    "payload": {},
                },
            }
        ),
        encoding="utf-8",
    )

    assert await pipeline._apply_operator_controls(None) is False


def _loop_snapshot():
    return {
        "state_vector": [0.1] * 10,
        "regime": "MEAN_REVERSION",
        "mark_price": 3500.0,
        "eth_btc_zscore": 0.2,
        "volatility_zscore": 0.3,
        "trend_slope": 0.01,
        "is_buy_liquidity_sweep": False,
        "is_sell_liquidity_sweep": False,
        "funding_rate": 0.0,
        "cvd": 1.0,
        "spread_bps": 0.5,
    }


@pytest.mark.asyncio
@pytest.mark.unit
async def test_trading_loop_empty_snapshot_pause_execute_hedge_and_kill_paths(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(trading_pipeline, "APEX_METRICS", FakeMetrics())

    empty_pipeline = build_shell_pipeline(tmp_path)
    empty_pipeline._running = True
    empty_pipeline.config["data"]["loop_interval_sec"] = 0

    def stop_after_empty_snapshot():
        empty_pipeline._running = False
        return None

    empty_pipeline.market_state.build_latest.side_effect = stop_after_empty_snapshot
    await empty_pipeline._trading_loop()

    paused_pipeline = build_shell_pipeline(tmp_path)
    paused_pipeline._running = True
    paused_pipeline.config["data"]["loop_interval_sec"] = 0
    paused_pipeline.market_state.build_latest.return_value = _loop_snapshot()

    async def pause_once(mark_price):
        paused_pipeline._running = False
        return True

    paused_pipeline._apply_operator_controls = AsyncMock(side_effect=pause_once)
    paused_pipeline._publish_status = MagicMock()
    await paused_pipeline._trading_loop()
    paused_pipeline._publish_status.assert_called_once()

    hedge_pipeline = build_shell_pipeline(tmp_path)
    hedge_pipeline._running = True
    hedge_pipeline.config["data"]["loop_interval_sec"] = 0
    hedge_pipeline.config["data"]["max_ticks"] = 1
    hedge_pipeline.market_state.build_latest.return_value = _loop_snapshot()
    hedge_pipeline._apply_operator_controls = AsyncMock(return_value=False)
    hedge_pipeline.meta_controller = SimpleNamespace(
        get_dual_inference=lambda state, regime: (
            2,
            0.9,
            {"action_probs": [0.1, 0.1, 0.8]},
            [0.1, 0.1, 0.8],
            [0.2, 0.2, 0.6],
        )
    )
    hedge_pipeline.explainability.decode_decision.return_value = {
        "primary_reasons": ["trend aligned"],
        "risk_factors": [],
    }
    hedge_pipeline.hedge_orchestrator = SimpleNamespace(
        evaluate=lambda ctx: (
            HedgeProposal("protective_hedge", short_delta_qty=0.05),
            {"enabled": True, "selected": "protective_hedge"},
        )
    )
    hedge_pipeline.shadow_runner = SimpleNamespace(run_tick=AsyncMock())
    hedge_pipeline.risk_engine = FakeRisk(approvals=[0.2, 0.1])
    await hedge_pipeline._trading_loop()

    assert hedge_pipeline._last_approved_fraction == 0.2
    assert hedge_pipeline.execution_adapter.place_order.await_count == 2
    hedge_pipeline.shadow_runner.run_tick.assert_awaited_once()

    kill_pipeline = build_shell_pipeline(tmp_path)
    kill_pipeline._running = True
    kill_pipeline.config["data"]["loop_interval_sec"] = 0
    kill_pipeline.market_state.build_latest.return_value = _loop_snapshot()
    kill_pipeline._apply_operator_controls = AsyncMock(return_value=False)
    kill_pipeline.meta_controller = SimpleNamespace(
        get_dual_inference=lambda state, regime: (
            1,
            0.2,
            {"action_probs": [0.1, 0.8, 0.1]},
            [0.1, 0.8, 0.1],
            [0.2, 0.6, 0.2],
        )
    )
    kill_pipeline.explainability.decode_decision.return_value = {
        "primary_reasons": [],
        "risk_factors": [],
    }
    kill_pipeline.hedge_orchestrator = SimpleNamespace(
        evaluate=lambda ctx: (None, {"enabled": False})
    )
    kill_pipeline.shadow_runner = SimpleNamespace(run_tick=AsyncMock())
    kill_pipeline.risk_engine.is_kill_switch_active = True

    async def stop_after_kill(mark_price):
        kill_pipeline._running = False

    kill_pipeline._handle_kill_switch = AsyncMock(side_effect=stop_after_kill)
    await kill_pipeline._trading_loop()

    kill_pipeline._handle_kill_switch.assert_awaited_once_with(3500.0)
