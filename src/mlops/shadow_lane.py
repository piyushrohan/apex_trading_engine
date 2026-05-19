import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.execution.adapters.base import OrderRequest
from src.execution.adapters.paper import PaperExecutionAdapter
from src.execution.portfolio import PortfolioService
from src.execution.risk_engine import RiskEngine
from src.mlops.registry import ModelRegistry
from src.models.meta_controller import MetaController
from src.strategies.hedge.base import HedgeContext, HedgeProposal
from src.strategies.hedge.registry import build_hedge_orchestrator

logger = logging.getLogger(__name__)


class ShadowLaneRunner:
    """
    Runs MLOps candidate models in virtual books alongside the primary pipeline.

    Shadow lanes are never live exchange order paths; they always use the shared
    PaperExecutionAdapter with book.role=shadow.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        registry: ModelRegistry,
        portfolio: PortfolioService,
        symbol: str,
        operator_mode: str,
    ):
        self.config = config
        self.registry = registry
        self.portfolio = portfolio
        self.symbol = symbol
        self.operator_mode = operator_mode
        self.enabled = config.get("shadow", {}).get("enabled", False)
        self.max_parallel = config.get("shadow", {}).get("max_parallel_candidates", 2)
        self.initial_equity = config.get("environment", {}).get(
            "initial_capital", 1000.0
        )
        self.decision_path = Path(
            config.get("shadow", {}).get(
                "decision_log_path",
                "data_lake/hedge_bandit/training/decisions.jsonl",
            )
        )
        self.lanes: Dict[str, Dict[str, Any]] = {}
        if self.enabled:
            self.refresh_candidates()

    def refresh_candidates(self) -> List[str]:
        """Discover active shadow/evaluating registry candidates."""
        candidates = self._candidate_model_ids()
        self.lanes = {}
        for model_id in candidates[: self.max_parallel]:
            meta = self.registry.registry_data["models"][model_id]
            book_id = f"shadow_{model_id}"
            book = self.portfolio.get_or_create_book(
                book_id=book_id,
                role="shadow",
                model_id=model_id,
                symbol=self.symbol,
                initial_equity=self.initial_equity,
            )
            controller = MetaController(self.config)
            model_path = self.registry.get_model_path(model_id)
            try:
                controller.load_model_artifact(meta.get("type", "GBM"), model_path)
            except FileNotFoundError:
                logger.warning("Shadow model artifact missing for %s", model_id)
            adapter = PaperExecutionAdapter(book_id=book_id)
            self.lanes[model_id] = {
                "book": book,
                "controller": controller,
                "adapter": adapter,
                "risk": RiskEngine(self.config),
                "hedge": build_hedge_orchestrator(self.config),
            }
        return list(self.lanes.keys())

    def _candidate_model_ids(self) -> List[str]:
        models = self.registry.registry_data.get("models", {})
        ids: List[str] = []
        active_shadow = self.registry.registry_data.get("active_shadow")
        if active_shadow in models:
            ids.append(active_shadow)
        if self.config.get("shadow", {}).get("auto_register", True):
            ids.extend(
                model_id
                for model_id, meta in models.items()
                if meta.get("status") == "EVALUATING" and model_id not in ids
            )
        return ids

    async def run_tick(
        self,
        snapshot: Dict[str, Any],
        mark_price: float,
        hedge_payload: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        if not self.enabled or not self.lanes:
            return []

        decisions = []
        state_vector = snapshot["state_vector"]
        regime = snapshot["regime"]
        for model_id, lane in self.lanes.items():
            action, conviction, context, _, _ = lane["controller"].get_dual_inference(
                state_vector, regime
            )
            ppo_probs = list(context.get("ppo_action_probs", []))
            gbm_probs = list(context.get("gbm_action_probs", []))
            approved_fraction = 0.0
            if action != 1:
                approved_fraction = self._approved_fraction(
                    lane, action, conviction, mark_price
                )
                if approved_fraction > 0:
                    await self._place_shadow_order(
                        lane, action, approved_fraction, mark_price
                    )
            lane_hedge_payload = hedge_payload or {"enabled": False}
            bandit_context = {
                "volatility_zscore": snapshot.get("volatility_zscore", 0.0),
                "funding_rate": snapshot.get("funding_rate", 0.0),
                "primary_action": action,
                "ppo_action_probs": ppo_probs,
                "gbm_action_probs": gbm_probs,
                "primary_size_fraction": approved_fraction,
            }
            if self.config.get("hedge", {}).get("enabled", False):
                proposal, lane_hedge_payload = lane["hedge"].evaluate(
                    self._hedge_context(
                        snapshot=snapshot,
                        book=lane["book"],
                        action=action,
                        approved_fraction=approved_fraction,
                        ppo_probs=ppo_probs,
                        gbm_probs=gbm_probs,
                        mark_price=mark_price,
                    )
                )
                if proposal and approved_fraction > 0:
                    await self._place_shadow_hedge(lane, proposal, mark_price)
            fills = lane["adapter"].try_fill_on_market(self.symbol, mark_price)
            for fill in fills:
                lane["book"].apply_fill(
                    fill["side"],
                    fill["executedQty"],
                    fill["avgPrice"],
                    fill.get("positionSide"),
                )
            lane["book"].update_equity_mark(mark_price)
            decision = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execution": {"mode": self.operator_mode},
                "book": {"role": "shadow", "id": lane["book"].book_id},
                "model_id": model_id,
                "action": action,
                "conviction": conviction,
                "regime": regime,
                "bandit_context": bandit_context,
                "approved_fraction": approved_fraction,
                "hedge": lane_hedge_payload,
                "action_probs": context.get("action_probs", []),
                "equity": lane["book"].equity,
                "pnl": lane["book"].equity - lane["book"].initial_equity,
            }
            decisions.append(decision)
            self._append_decision(decision)
        return decisions

    def _approved_fraction(
        self, lane: Dict[str, Any], action: int, conviction: float, mark_price: float
    ) -> float:
        side = "BUY" if action == 2 else "SELL"
        kelly = lane["risk"].calculate_kelly_size(0.55, 1.2, conviction)
        exposure = self.portfolio.current_exposure_fraction(
            lane["book"].book_id, mark_price
        )
        return lane["risk"].approve_order(
            side,
            kelly,
            current_exposure=exposure,
            long_qty=lane["book"].long_qty,
            short_qty=lane["book"].short_qty,
            equity=lane["book"].equity,
            mark_price=mark_price,
            is_hedge_leg=False,
        )

    async def _place_shadow_order(
        self, lane: Dict[str, Any], action: int, fraction: float, mark_price: float
    ):
        side = "BUY" if action == 2 else "SELL"
        qty = max((lane["book"].equity * fraction) / mark_price, 0.001)
        price = mark_price - 0.01 if side == "BUY" else mark_price + 0.01
        position_side = None
        if self.portfolio.position_mode == "hedge":
            position_side = "LONG" if side == "BUY" else "SHORT"
        await lane["adapter"].place_order(
            OrderRequest(
                symbol=self.symbol,
                side=side,
                quantity=qty,
                price=price,
                position_side=position_side,
            )
        )

    async def _place_shadow_hedge(
        self, lane: Dict[str, Any], proposal: HedgeProposal, mark_price: float
    ):
        equity = max(lane["book"].equity, 1.0)
        for side, fraction, position_side in (
            ("BUY", proposal.long_delta_qty, "LONG"),
            ("SELL", proposal.short_delta_qty, "SHORT"),
        ):
            if fraction <= 0:
                continue
            approved = lane["risk"].approve_order(
                side,
                fraction,
                current_exposure=self.portfolio.gross_exposure_fraction(
                    lane["book"].book_id, mark_price
                ),
                long_qty=lane["book"].long_qty,
                short_qty=lane["book"].short_qty,
                equity=equity,
                mark_price=mark_price,
                is_hedge_leg=True,
            )
            if approved <= 0:
                continue
            qty = max(approved * equity / mark_price, 0.001)
            await lane["adapter"].place_order(
                OrderRequest(
                    symbol=self.symbol,
                    side=side,
                    quantity=qty,
                    price=mark_price - 0.01 if side == "BUY" else mark_price + 0.01,
                    position_side=position_side
                    if self.portfolio.position_mode == "hedge"
                    else None,
                )
            )

    def _hedge_context(
        self,
        *,
        snapshot: Dict[str, Any],
        book,
        action: int,
        approved_fraction: float,
        ppo_probs: List[float],
        gbm_probs: List[float],
        mark_price: float,
    ) -> HedgeContext:
        return HedgeContext(
            symbol=self.symbol,
            regime=snapshot["regime"],
            mark_price=mark_price,
            feature_vector=snapshot["state_vector"],
            ppo_action_probs=ppo_probs,
            gbm_action_probs=gbm_probs,
            risk_factors=[],
            primary_long_qty=book.long_qty,
            primary_short_qty=book.short_qty,
            primary_action=action,
            primary_size_fraction=approved_fraction,
            eth_btc_zscore=snapshot.get("eth_btc_zscore", 0.0),
            volatility_zscore=snapshot.get("volatility_zscore", 0.0),
            trend_slope=snapshot.get("trend_slope", 0.0),
            is_buy_liquidity_sweep=snapshot.get("is_buy_liquidity_sweep", False),
            is_sell_liquidity_sweep=snapshot.get("is_sell_liquidity_sweep", False),
            funding_rate=snapshot.get("funding_rate", 0.0),
            extra={
                "cvd": snapshot.get("cvd", 0.0),
                "spread_bps": snapshot.get("spread_bps", 0.0),
            },
        )

    def _append_decision(self, decision: Dict[str, Any]):
        self.decision_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.decision_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(decision) + "\n")
