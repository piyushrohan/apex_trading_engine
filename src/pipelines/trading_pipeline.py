import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.api.status_store import get_status_store
from src.core.config_loader import apply_risk_profile
from src.core.security import SecurityManager
from src.data.binance_rest import BinanceRESTClient
from src.data.ingestion_service import DataIngestionService
from src.data.market_state import MarketStateService
from src.execution.adapters.base import OrderRequest
from src.execution.factory import create_execution_adapter, get_operator_mode
from src.execution.grid_adapter import MakerGridAdapter
from src.execution.live_gate import check_api_credentials, validate_live_startup
from src.execution.portfolio import PortfolioService
from src.execution.position_sync import AccountSynchronizer
from src.execution.risk_engine import RiskEngine
from src.mlops.explainability import ExplainabilityEngine
from src.mlops.registry import ModelRegistry
from src.mlops.shadow_lane import ShadowLaneRunner
from src.models.meta_controller import MetaController
from src.observability.metrics import APEX_METRICS
from src.strategies.hedge.base import HedgeContext, HedgeProposal
from src.strategies.hedge.registry import build_hedge_orchestrator

logger = logging.getLogger(__name__)


class TradingPipeline:
    """
    Shared autonomous loop for operator paper and live modes.
    ingest -> features -> regime -> model -> hedge -> explain -> risk -> execute
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.operator_mode = get_operator_mode(config)
        self.symbol = config.get("data", {}).get("target_symbol", "ETHUSDC")
        self.position_mode = config.get("execution", {}).get("position_mode", "one_way")

        self.rest_client = self._build_rest_client(config)
        self.ingestion = DataIngestionService(config, rest_client=self.rest_client)
        self.market_state = MarketStateService(config, cache=self.ingestion.cache)
        self.risk_engine = RiskEngine(config)
        self.registry = ModelRegistry()
        self.meta_controller = MetaController(config)
        self._model_artifact_loaded = False
        self._model_readiness: Dict[str, Any] = {}
        model_id = self._load_active_prod_model()
        self.explainability = ExplainabilityEngine(config)
        self.hedge_orchestrator = build_hedge_orchestrator(config)

        self.portfolio = PortfolioService(position_mode=self.position_mode)

        initial_equity = config.get("environment", {}).get("initial_capital", 1000.0)
        self.primary_book = self.portfolio.get_or_create_book(
            book_id="primary",
            role="primary",
            model_id=model_id,
            symbol=self.symbol,
            initial_equity=initial_equity,
        )

        self.execution_adapter = create_execution_adapter(
            config, self.rest_client, book_id="primary"
        )
        self.shadow_runner = ShadowLaneRunner(
            config=config,
            registry=self.registry,
            portfolio=self.portfolio,
            symbol=self.symbol,
            operator_mode=self.operator_mode,
        )
        self.account_sync: Optional[AccountSynchronizer] = None
        if self.operator_mode == "live":
            self.account_sync = AccountSynchronizer(self.rest_client)

        self._running = False
        self._last_approved_fraction = 0.0
        self._last_control_command_id: Optional[str] = None
        self._status_store = get_status_store()
        self._status_store.update(
            operator_mode=self.operator_mode,
            symbol=self.symbol,
            model_id=model_id,
            ingestion_enabled=self.config.get("data", {})
            .get("ingestion", {})
            .get("enabled", True),
            hedge_enabled=self.config.get("hedge", {}).get("enabled", False),
        )

    async def start(self):
        mode_label = self.operator_mode.upper()
        logger.info(f"Initializing APEX TradingPipeline [{mode_label}]...")
        self._validate_startup()
        self._start_metrics_server()
        self._running = True

        if self.operator_mode == "live":
            await self._prepare_live_exchange()

        if self.account_sync:
            await self.account_sync.start()
            self.account_sync.on_position_change = self._on_account_position_update
            await self._bootstrap_live_account()

        if self.config.get("data", {}).get("ingestion", {}).get("enabled", True):
            await self.ingestion.bootstrap_historical()
            await self.ingestion.start_live()

        await self._trading_loop()

    @staticmethod
    def _build_rest_client(config: Dict[str, Any]) -> BinanceRESTClient:
        api_key = os.getenv("BINANCE_API_KEY") or config.get("live", {}).get("api_key")
        api_secret = os.getenv("BINANCE_API_SECRET") or config.get("live", {}).get(
            "api_secret"
        )
        if not (api_key and api_secret):
            try:
                api_key, api_secret = SecurityManager().get_api_credentials()
            except Exception as exc:
                logger.debug("Encrypted API credentials unavailable: %s", exc)
        return BinanceRESTClient(api_key=api_key, api_secret=api_secret)

    def _load_active_prod_model(self) -> str:
        registry_data = getattr(self.registry, "registry_data", {}) or {}
        active = registry_data.get("active_prod")
        models = registry_data.get("models", {})
        if not active or active not in models:
            self._model_readiness = self._registry_production_readiness(active)
            return "unregistered"

        meta = models[active]
        model_path = self.registry.get_model_path(active)
        self._model_readiness = self._registry_production_readiness(active)
        try:
            self.meta_controller.load_model_artifact(
                meta.get("type", "GBM"), model_path
            )
            self._model_artifact_loaded = True
            logger.info("Loaded active production model %s from %s", active, model_path)
        except FileNotFoundError:
            self._model_artifact_loaded = False
            logger.warning(
                "Active production model %s has no artifact at %s; using defaults",
                active,
                model_path,
            )
        return active

    def _registry_production_readiness(
        self, model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        readiness_fn = getattr(self.registry, "production_readiness", None)
        if readiness_fn:
            return readiness_fn(model_id)
        return {
            "model_id": model_id,
            "ready": False,
            "blockers": ["production_readiness_unavailable"],
        }

    def _start_metrics_server(self):
        metrics_cfg = self.config.get("observability", {}).get("metrics", {})
        if not metrics_cfg.get("enabled", True):
            return
        APEX_METRICS.start_server(int(metrics_cfg.get("port", 9108)))

    async def _prepare_live_exchange(self):
        live_cfg = self.config.get("live", {})
        if self.position_mode == "hedge":
            set_hedge_mode = getattr(self.rest_client, "set_hedge_mode", None)
            ok = await set_hedge_mode(True) if set_hedge_mode else True
            if not ok:
                raise RuntimeError("Live startup blocked: failed to enable hedge mode")
        leverage = int(
            live_cfg.get(
                "leverage", self.config.get("execution", {}).get("max_leverage", 1)
            )
        )
        if leverage > 0:
            set_leverage = getattr(self.rest_client, "set_leverage", None)
            result = await set_leverage(self.symbol, leverage) if set_leverage else {}
            if result is None:
                raise RuntimeError("Live startup blocked: failed to set leverage")

    def _validate_startup(self):
        if self.operator_mode != "live":
            return
        self._validate_live_model_gate()
        validate_live_startup(self.config)
        ok, err = check_api_credentials(self.config)
        if not ok:
            raise RuntimeError(f"Live operator mode blocked: {err}")

    def _validate_live_model_gate(self) -> None:
        if self.config.get("models", {}).get("allow_unregistered_live", False):
            logger.warning("Live model gate bypassed by models.allow_unregistered_live")
            return
        registry = getattr(self, "registry", None)
        if registry is None or not hasattr(registry, "production_readiness"):
            logger.warning("Live model gate skipped: registry readiness unavailable")
            return
        readiness = registry.production_readiness(
            self.primary_book.model_id
            if self.primary_book.model_id != "unregistered"
            else None
        )
        if (
            not getattr(self, "_model_artifact_loaded", False)
            and "missing_model_artifact" not in readiness["blockers"]
        ):
            readiness["blockers"].append("missing_model_artifact")
        if not readiness.get("ready"):
            blockers = ", ".join(readiness.get("blockers", []))
            raise RuntimeError(
                "Live operator mode blocked: no approved production model "
                f"ready for inference ({blockers})"
            )

    async def _bootstrap_live_account(self):
        if not self.account_sync:
            return
        snapshot = await self.account_sync.fetch_snapshot(self.symbol)
        mark = (
            self.ingestion.get_last_mark_price(self.symbol)
            or snapshot.get("entry_price_long")
            or snapshot.get("entry_price_short")
            or 0.0
        )
        self._apply_account_snapshot(self.symbol, snapshot, mark)

    def _apply_account_snapshot(
        self, symbol: str, position_data: dict, mark_price: float
    ):
        if symbol != self.symbol:
            return
        long_qty = float(position_data.get("long_qty", 0.0))
        short_qty = float(position_data.get("short_qty", 0.0))
        if long_qty == 0.0 and short_qty == 0.0:
            amount = float(position_data.get("amount", 0.0))
            if amount >= 0:
                long_qty = abs(amount)
            else:
                short_qty = abs(amount)

        self.primary_book.sync_legs(
            long_qty=long_qty,
            short_qty=short_qty,
            entry_long=float(position_data.get("entry_price_long", 0.0)),
            entry_short=float(position_data.get("entry_price_short", 0.0)),
            mark_price=mark_price if mark_price > 0 else None,
        )
        wallet = self._wallet_equity_from_sync()
        equity = wallet if wallet > 0 else self.primary_book.equity
        equity += float(position_data.get("unrealized_pnl", 0.0))
        self.primary_book.initial_equity = max(wallet, self.primary_book.initial_equity)
        self.risk_engine.update_equity(equity)

        sync_explanation = self.explainability.decode_portfolio_state(
            symbol=symbol,
            operator_mode=self.operator_mode,
            book={
                "long_qty": self.primary_book.long_qty,
                "short_qty": self.primary_book.short_qty,
                "equity": self.primary_book.equity,
            },
            regime=self._status_store.regime or "UNKNOWN",
            mark_price=mark_price,
            source="account_sync",
        )
        self.explainability._log_to_journal(sync_explanation)
        self._publish_status(
            mark_price=mark_price if mark_price > 0 else None,
            last_explanation=sync_explanation,
        )

    def _wallet_equity_from_sync(self) -> float:
        if not self.account_sync:
            return 0.0
        usdc = self.account_sync.balances.get("USDC", {})
        return float(usdc.get("wallet_balance", 0.0) or 0.0)

    def _on_account_position_update(self, symbol: str, position_data: dict):
        logger.info(
            f"Account sync update for {symbol}: "
            f"long={position_data.get('long_qty')} "
            f"short={position_data.get('short_qty')}"
        )
        mark = self.ingestion.get_last_mark_price(symbol) or 0.0
        self._apply_account_snapshot(symbol, position_data, mark)

    @staticmethod
    def _operator_control_state_path() -> Path:
        return Path(
            os.getenv("APEX_CONTROL_STATE_PATH", "data_lake/operator_controls.json")
        )

    def _load_operator_controls(self) -> Dict[str, Any]:
        path = self._operator_control_state_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Operator control state unreadable: %s", exc)
            return {"error": "control state unreadable"}

    @staticmethod
    def _control_command_id(controls: Dict[str, Any]) -> Optional[str]:
        command = controls.get("last_command") or {}
        name = command.get("command")
        timestamp = command.get("timestamp")
        if not name or not timestamp:
            return None
        return f"{timestamp}:{name}"

    def _apply_operator_risk_profile(self, profile: Optional[str]) -> None:
        if not profile:
            return
        try:
            current_equity = getattr(
                self.risk_engine, "current_equity", self.primary_book.equity
            )
            high_water_mark = getattr(
                self.risk_engine, "high_water_mark", current_equity
            )
            kill_switch = getattr(self.risk_engine, "is_kill_switch_active", False)
            self.config = apply_risk_profile(self.config, profile)
            refreshed = RiskEngine(self.config)
            refreshed.current_equity = current_equity
            refreshed.high_water_mark = high_water_mark
            refreshed.is_kill_switch_active = kill_switch
            self.risk_engine = refreshed
            logger.warning("Operator risk profile applied: %s", profile)
        except Exception as exc:
            logger.error("Rejected operator risk profile %s: %s", profile, exc)

    async def _apply_operator_controls(self, mark_price: Optional[float]) -> bool:
        controls = self._load_operator_controls()
        if not controls or "error" in controls:
            return False

        command_id = self._control_command_id(controls)
        command = controls.get("last_command") or {}
        if command_id and command_id != self._last_control_command_id:
            self._last_control_command_id = command_id
            name = command.get("command")
            payload = command.get("payload") or {}
            if name == "kill-switch":
                self.risk_engine.is_kill_switch_active = True
            elif name == "clear-kill-switch":
                self.risk_engine.is_kill_switch_active = False
            elif name == "flatten":
                if mark_price and mark_price > 0:
                    await self._handle_kill_switch(mark_price)
                else:
                    logger.warning("Flatten request deferred: no mark price available")
            elif name == "set-risk-profile":
                self._apply_operator_risk_profile(payload.get("profile"))
            elif name == "set-mode" and payload.get("mode") != self.operator_mode:
                logger.warning(
                    "Operator mode change to %s requires a controlled restart",
                    payload.get("mode"),
                )

        if controls.get("kill_switch_requested"):
            self.risk_engine.is_kill_switch_active = True
        if controls.get("paused"):
            logger.warning("Operator pause active - skipping trading tick.")
            return True
        return False

    async def _trading_loop(self):
        logger.info(f"Trading loop active for {self.symbol} ({self.operator_mode})")
        loop_interval = self.config.get("data", {}).get("loop_interval_sec", 3.0)
        max_ticks = self.config.get("data", {}).get("max_ticks")
        processed_ticks = 0

        while self._running:
            try:
                await asyncio.sleep(loop_interval)
                await self.ingestion.flush_ticks()

                snapshot = self.market_state.build_latest()
                if not snapshot:
                    logger.warning("Skipping tick — no market state (empty cache?)")
                    continue

                state_vector = snapshot["state_vector"]
                current_regime = snapshot["regime"]
                current_bbo = (
                    self.ingestion.get_last_mark_price(self.symbol)
                    or snapshot["mark_price"]
                )
                if await self._apply_operator_controls(current_bbo):
                    self._publish_status(
                        regime=current_regime,
                        mark_price=current_bbo,
                    )
                    continue

                (
                    action,
                    conviction,
                    context,
                    ppo_probs,
                    gbm_probs,
                ) = APEX_METRICS.time_inference(
                    self.operator_mode,
                    self.meta_controller.get_dual_inference,
                    state_vector,
                    current_regime,
                )

                explanation = self.explainability.decode_decision(
                    action,
                    conviction,
                    context,
                    write_journal=False,
                    portfolio={
                        "long_qty": self.primary_book.long_qty,
                        "short_qty": self.primary_book.short_qty,
                    },
                    market_snapshot=snapshot,
                    kill_switch=self.risk_engine.is_kill_switch_active,
                )
                risk_factors = explanation.get("risk_factors", [])

                approved_fraction = 0.0
                if action != 1:
                    approved_fraction = self._compute_approved_fraction(
                        action, conviction, current_bbo
                    )

                hedge_ctx = HedgeContext(
                    symbol=self.symbol,
                    regime=current_regime,
                    mark_price=current_bbo,
                    feature_vector=state_vector,
                    ppo_action_probs=ppo_probs,
                    gbm_action_probs=gbm_probs,
                    risk_factors=risk_factors,
                    primary_long_qty=self.primary_book.long_qty,
                    primary_short_qty=self.primary_book.short_qty,
                    primary_action=action,
                    primary_size_fraction=approved_fraction,
                    eth_btc_zscore=snapshot.get("eth_btc_zscore", 0.0),
                    volatility_zscore=snapshot.get("volatility_zscore", 0.0),
                    trend_slope=snapshot.get("trend_slope", 0.0),
                    is_buy_liquidity_sweep=snapshot.get(
                        "is_buy_liquidity_sweep", False
                    ),
                    is_sell_liquidity_sweep=snapshot.get(
                        "is_sell_liquidity_sweep", False
                    ),
                    funding_rate=snapshot.get("funding_rate", 0.0),
                    extra={
                        "cvd": snapshot.get("cvd", 0.0),
                        "spread_bps": snapshot.get("spread_bps", 0.0),
                    },
                )
                hedge_proposal, hedge_payload = self.hedge_orchestrator.evaluate(
                    hedge_ctx
                )
                self._append_hedge_bandit_decision(
                    action=action,
                    conviction=conviction,
                    regime=current_regime,
                    hedge_payload=hedge_payload,
                    hedge_ctx=hedge_ctx,
                )

                explanation = self._enrich_journal_entry(explanation, hedge_payload)
                self.explainability._log_to_journal(explanation)
                self._publish_status(
                    regime=current_regime,
                    mark_price=current_bbo,
                    last_explanation=explanation,
                )

                self._persist_paper_snapshot(current_bbo, current_regime)
                await self.shadow_runner.run_tick(
                    snapshot=snapshot,
                    mark_price=current_bbo,
                    hedge_payload=hedge_payload,
                )

                if self.risk_engine.is_kill_switch_active:
                    await self._handle_kill_switch(current_bbo)
                    continue

                if action != 1 and approved_fraction > 0:
                    placed = await self._execute_signal(
                        action,
                        conviction,
                        explanation,
                        current_bbo,
                        approved_fraction,
                    )
                    self._last_approved_fraction = placed
                    if hedge_proposal and placed > 0:
                        await self._execute_hedge(hedge_proposal, current_bbo)

                self._simulate_paper_fills(current_bbo)
                processed_ticks += 1
                if max_ticks is not None and processed_ticks >= int(max_ticks):
                    self._running = False

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(1)

    def _persist_paper_snapshot(self, mark_price: float, regime: str):
        if self.operator_mode != "paper":
            return
        self.primary_book.update_equity_mark(mark_price)
        self.ingestion.cache.insert_paper_equity_snapshot(
            book_id=self.primary_book.book_id,
            equity=self.primary_book.equity,
            long_qty=self.primary_book.long_qty,
            short_qty=self.primary_book.short_qty,
            mark_price=mark_price,
            regime=regime,
        )

    def _publish_status(
        self,
        *,
        regime: Optional[str] = None,
        mark_price: Optional[float] = None,
        last_explanation: Optional[Dict[str, Any]] = None,
    ):
        book = self.primary_book
        self._status_store.update(
            operator_mode=self.operator_mode,
            symbol=self.symbol,
            regime=regime,
            mark_price=mark_price,
            kill_switch_active=self.risk_engine.is_kill_switch_active,
            model_id=book.model_id,
            last_explanation=last_explanation,
            portfolio={
                "book_id": book.book_id,
                "role": book.role,
                "long_qty": book.long_qty,
                "short_qty": book.short_qty,
                "equity": book.equity,
                "net_qty": book.net_qty,
                "gross_qty": book.gross_qty,
            },
        )
        APEX_METRICS.set_ws_health(
            self.operator_mode,
            self.config.get("data", {}).get("ingestion", {}).get("enabled", True),
        )
        if mark_price is not None:
            APEX_METRICS.set_pnl(
                self.operator_mode,
                book.role,
                book.book_id,
                book.equity - book.initial_equity,
            )

    def _enrich_journal_entry(
        self, explanation: Dict[str, Any], hedge_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        explanation["execution"] = {"mode": self.operator_mode}
        explanation["book"] = {
            "role": self.primary_book.role,
            "id": self.primary_book.book_id,
        }
        explanation["model_id"] = self.primary_book.model_id
        explanation["hedge"] = hedge_payload
        return explanation

    def _append_hedge_bandit_decision(
        self,
        *,
        action: int,
        conviction: float,
        regime: str,
        hedge_payload: Dict[str, Any],
        hedge_ctx: HedgeContext,
    ) -> None:
        if not hedge_payload.get("enabled"):
            return
        decision_path = Path(
            self.config.get("shadow", {}).get(
                "decision_log_path",
                "data_lake/hedge_bandit/training/decisions.jsonl",
            )
        )
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution": {"mode": self.operator_mode},
            "book": {"role": self.primary_book.role, "id": self.primary_book.book_id},
            "model_id": self.primary_book.model_id,
            "action": action,
            "conviction": conviction,
            "regime": regime,
            "bandit_context": {
                "volatility_zscore": hedge_ctx.volatility_zscore,
                "funding_rate": hedge_ctx.funding_rate,
                "primary_action": hedge_ctx.primary_action,
                "ppo_action_probs": hedge_ctx.ppo_action_probs,
                "gbm_action_probs": hedge_ctx.gbm_action_probs,
                "primary_size_fraction": hedge_ctx.primary_size_fraction,
            },
            "hedge": hedge_payload,
            "equity": self.primary_book.equity,
            "pnl": self.primary_book.equity - self.primary_book.initial_equity,
        }
        with open(decision_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _compute_approved_fraction(
        self, action: int, conviction: float, current_bbo: float
    ) -> float:
        side = "BUY" if action == 2 else "SELL"
        win_rate, win_loss_ratio = 0.55, 1.2
        exposure = self.portfolio.current_exposure_fraction(
            self.primary_book.book_id, current_bbo
        )
        kelly_fraction = self.risk_engine.calculate_kelly_size(
            win_rate, win_loss_ratio, conviction
        )
        return self.risk_engine.approve_order(
            side,
            kelly_fraction,
            current_exposure=exposure,
            long_qty=self.primary_book.long_qty,
            short_qty=self.primary_book.short_qty,
            equity=self.primary_book.equity,
            mark_price=current_bbo,
            is_hedge_leg=False,
        )

    async def _execute_signal(
        self,
        action: int,
        conviction: float,
        explanation: Dict[str, Any],
        current_bbo: float,
        approved_fraction: float,
    ) -> float:
        side = "BUY" if action == 2 else "SELL"
        if approved_fraction <= 0:
            return 0.0

        quantity = max(
            (self.primary_book.equity * approved_fraction) / current_bbo, 0.001
        )
        price = current_bbo - 0.01 if side == "BUY" else current_bbo + 0.01
        position_side = None
        if self.position_mode == "hedge":
            position_side = "LONG" if side == "BUY" else "SHORT"

        logger.info(
            f"[{self.operator_mode.upper()}] Executing {side} "
            f"Reason: {explanation.get('primary_reasons')}"
        )

        result = await self.execution_adapter.place_order(
            OrderRequest(
                symbol=self.symbol,
                side=side,
                quantity=quantity,
                price=price,
                position_side=position_side,
            )
        )

        if not result.success:
            return 0.0
        return approved_fraction

    async def _execute_hedge(self, proposal: HedgeProposal, mark_price: float):
        """Place hedge leg adjustments (paper or live) from selected hedge strategy."""
        if proposal.strategy_name == "maker_grid_hedge":
            await self._execute_grid_hedge(proposal, mark_price)
            return

        equity = max(self.primary_book.equity, 1.0)
        if proposal.long_delta_qty > 0:
            fraction = proposal.long_delta_qty
            approved = self.risk_engine.approve_order(
                "BUY",
                fraction,
                current_exposure=self.portfolio.gross_exposure_fraction(
                    self.primary_book.book_id, mark_price
                ),
                long_qty=self.primary_book.long_qty,
                short_qty=self.primary_book.short_qty,
                equity=equity,
                mark_price=mark_price,
                is_hedge_leg=True,
            )
            if approved <= 0:
                return
            qty = max(approved * equity / mark_price, 0.001)
            await self.execution_adapter.place_order(
                OrderRequest(
                    symbol=self.symbol,
                    side="BUY",
                    quantity=qty,
                    price=mark_price - 0.01,
                    position_side="LONG" if self.position_mode == "hedge" else None,
                )
            )
        if proposal.short_delta_qty > 0:
            fraction = proposal.short_delta_qty
            approved = self.risk_engine.approve_order(
                "SELL",
                fraction,
                current_exposure=self.portfolio.gross_exposure_fraction(
                    self.primary_book.book_id, mark_price
                ),
                long_qty=self.primary_book.long_qty,
                short_qty=self.primary_book.short_qty,
                equity=equity,
                mark_price=mark_price,
                is_hedge_leg=True,
            )
            if approved <= 0:
                return
            qty = max(approved * equity / mark_price, 0.001)
            await self.execution_adapter.place_order(
                OrderRequest(
                    symbol=self.symbol,
                    side="SELL",
                    quantity=qty,
                    price=mark_price + 0.01,
                    position_side="SHORT" if self.position_mode == "hedge" else None,
                )
            )

    async def _execute_grid_hedge(self, proposal: HedgeProposal, mark_price: float):
        """Route maker-grid hedge selections through the grid order planner."""
        grid_cfg = (
            self.config.get("hedge", {})
            .get("strategies", {})
            .get("maker_grid_hedge", {})
        )
        equity = max(self.primary_book.equity, 1.0)
        fraction = max(proposal.long_delta_qty, proposal.short_delta_qty)
        approved = self.risk_engine.approve_order(
            "BUY",
            fraction,
            current_exposure=self.portfolio.gross_exposure_fraction(
                self.primary_book.book_id, mark_price
            ),
            long_qty=self.primary_book.long_qty,
            short_qty=self.primary_book.short_qty,
            equity=equity,
            mark_price=mark_price,
            is_hedge_leg=True,
        )
        if approved <= 0:
            return
        total_qty = max(approved * equity / mark_price, 0.001)
        plan = MakerGridAdapter().build_grid(
            symbol=self.symbol,
            mid_price=mark_price,
            total_quantity=total_qty,
            levels=int(grid_cfg.get("grid_levels", 3)),
            spacing_ticks=int(grid_cfg.get("grid_spacing_ticks", 2)),
        )
        for order in plan.orders:
            await self.execution_adapter.place_order(order)

    def _simulate_paper_fills(self, mark_price: float):
        if self.operator_mode != "paper":
            return
        from src.execution.adapters.paper import PaperExecutionAdapter

        if not isinstance(self.execution_adapter, PaperExecutionAdapter):
            return
        fills = self.execution_adapter.try_fill_on_market(self.symbol, mark_price)
        for fill in fills:
            self.primary_book.apply_fill(
                fill["side"],
                fill["executedQty"],
                fill["avgPrice"],
                fill.get("positionSide"),
            )
            self.explainability._log_to_journal(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "schema_version": 2,
                    "event": "paper_fill",
                    "execution": {"mode": self.operator_mode},
                    "book": {
                        "role": self.primary_book.role,
                        "id": self.primary_book.book_id,
                    },
                    "model_id": self.primary_book.model_id,
                    "symbol": self.symbol,
                    "side": fill["side"],
                    "positionSide": fill.get("positionSide"),
                    "executedQty": fill["executedQty"],
                    "avgPrice": fill["avgPrice"],
                    "fee": fill.get("fee", 0.0),
                }
            )
        total_orders = len(self.execution_adapter._fills) + len(
            self.execution_adapter._open_orders
        )
        if total_orders:
            APEX_METRICS.set_paper_fill_rate(
                self.operator_mode,
                self.primary_book.book_id,
                len(self.execution_adapter._fills) / total_orders,
            )

    async def _handle_kill_switch(self, mark_price: float):
        logger.critical("Kill switch active — cancel all orders and flatten both legs.")
        from src.execution.adapters.live import LiveExecutionAdapter
        from src.execution.adapters.paper import PaperExecutionAdapter

        if isinstance(self.execution_adapter, PaperExecutionAdapter):
            self.execution_adapter.flatten_all_virtual_orders(self.symbol)
            self.primary_book.flatten_all(mark_price)
        elif isinstance(self.execution_adapter, LiveExecutionAdapter):
            await self.execution_adapter.flatten_all_positions(
                self.symbol,
                self.primary_book.long_qty,
                self.primary_book.short_qty,
            )
            self.primary_book.flatten_all(mark_price)
        else:
            await self.execution_adapter.cancel_all_orders(self.symbol)

    async def stop(self):
        self._running = False
        if self.config.get("data", {}).get("ingestion", {}).get("enabled", True):
            await self.ingestion.stop()
        if self.account_sync:
            await self.account_sync.stop()
        await self.rest_client.close()
        self.ingestion.close()
        self.market_state.close()
        logger.info("TradingPipeline stopped gracefully.")
