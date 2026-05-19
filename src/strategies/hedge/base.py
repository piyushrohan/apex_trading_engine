from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol


@dataclass
class HedgeContext:
    """Market and portfolio snapshot for hedge strategy scoring."""

    symbol: str
    regime: str
    mark_price: float
    feature_vector: List[float]
    ppo_action_probs: List[float]
    gbm_action_probs: List[float]
    risk_factors: List[str] = field(default_factory=list)
    primary_long_qty: float = 0.0
    primary_short_qty: float = 0.0
    primary_action: int = 1
    primary_size_fraction: float = 0.0
    eth_btc_zscore: float = 0.0
    volatility_zscore: float = 0.0
    trend_slope: float = 0.0
    funding_rate: float = 0.0
    is_buy_liquidity_sweep: bool = False
    is_sell_liquidity_sweep: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HedgeProposal:
    """Target leg adjustments proposed by a hedge strategy."""

    strategy_name: str
    long_delta_qty: float = 0.0
    short_delta_qty: float = 0.0
    intent: str = ""
    score: float = 0.0


class HedgeStrategy(Protocol):
    name: str

    def score(self, ctx: HedgeContext) -> float:
        ...

    def propose(self, ctx: HedgeContext) -> HedgeProposal:
        ...
