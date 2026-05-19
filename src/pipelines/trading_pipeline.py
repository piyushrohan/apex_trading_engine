import asyncio
import logging
import os
from typing import Any, Dict, Optional

from src.api.status_store import get_status_store
from src.data.binance_rest import BinanceRESTClient
from src.data.ingestion_service import DataIngestionService
from src.data.market_state import MarketStateService
from src.execution.adapters.base import OrderRequest
from src.execution.factory import create_execution_adapter, get_operator_mode
from src.execution.live_gate import check_api_credentials, validate_live_startup
from src.execution.portfolio import PortfolioService
from src.execution.position_sync import AccountSynchronizer
from src.execution.risk_engine import RiskEngine
from src.mlops.explainability import ExplainabilityEngine
from src.mlops.registry import ModelRegistry
from src.models.meta_controller import MetaController
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
        self.explainability = ExplainabilityEngine(config)
        self.hedge_orchestrator = build_hedge_orchestrator(config)

        self.portfolio = PortfolioService(position_mode=self.position_mode)
        prod_path = self.registry.get_prod_model_path()
        model_id = "unregistered"
        if prod_path:
            model_id = prod_path.rstrip("/").split("/")[-1]

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
        self.account_sync: Optional[AccountSynchronizer] = None
        if self.operator_mode == "live":
            self.account_sync = AccountSynchronizer(self.rest_client)

        self._running = False
        self._last_approved_fraction = 0.0
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
        self._running = True

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
        return BinanceRESTClient(api_key=api_key, api_secret=api_secret)

    def _validate_startup(self):
        if self.operator_mode != "live":
            return
        validate_live_startup(self.config)
        ok, err = check_api_credentials(self.config)
        if not ok:
            raise RuntimeError(f"Live operator mode blocked: {err}")

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

    async def _trading_loop(self):
        logger.info(f"Trading loop active for {self.symbol} ({self.operator_mode})")
        loop_interval = self.config.get("data", {}).get("loop_interval_sec", 3.0)

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

                (
                    action,
                    conviction,
                    context,
                    ppo_probs,
                    gbm_probs,
                ) = self.meta_controller.get_dual_inference(
                    state_vector, current_regime
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
                    funding_rate=0.0,
                )
                hedge_proposal, hedge_payload = self.hedge_orchestrator.evaluate(
                    hedge_ctx
                )

                explanation = self._enrich_journal_entry(explanation, hedge_payload)
                self.explainability._log_to_journal(explanation)
                self._publish_status(
                    regime=current_regime,
                    mark_price=current_bbo,
                    last_explanation=explanation,
                )

                self._persist_paper_snapshot(current_bbo, current_regime)

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
