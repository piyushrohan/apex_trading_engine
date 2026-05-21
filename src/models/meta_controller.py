import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from src.models.gbm_agent import GBMAgent
from src.models.ppo_agent import PPOAgent

logger = logging.getLogger(__name__)


class MetaController:
    """
    The orchestrator of the Intelligence Layer.
    Uses the Regime classification to route inference to the model
    that historically performs best in that specific market environment.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ppo_agent = PPOAgent(state_dim=10, config=config)  # Assuming 10 features
        self.gbm_agent = GBMAgent(config=config)

        # Mapping of Regime ID to Preferred Model
        # e.g., GBM handles chop better, PPO handles trends better
        self.regime_model_map = {
            "STRONG_TREND_UP": "PPO",
            "STRONG_TREND_DOWN": "PPO",
            "CHOP_COMPRESSION": "GBM",
            "VOLATILITY_EXPANSION": "PPO",
            "MEAN_REVERSION": "GBM",
        }

        logger.info("Initialized Meta-Controller with PPO and GBM Agents.")

    def load_model_artifact(self, model_type: str, model_path: str):
        """Load a specialist model artifact into the routed controller."""
        normalized = model_type.upper()
        if normalized == "PPO":
            self.ppo_agent.load(model_path)
        elif normalized in ("GBM", "LIGHTGBM"):
            self._assert_gbm_artifact_safe(model_path)
            self.gbm_agent.load(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        return self

    def _assert_gbm_artifact_safe(self, model_path: str) -> None:
        """
        Preflight native GBM artifacts in a child process before in-process load.

        On macOS, some LightGBM artifacts can segfault when restored after Torch
        has initialized. A subprocess lets the runtime quarantine that artifact
        instead of crashing the paper/live operator process.
        """
        cfg = self.config.get("models", {}).get("gbm", {})
        if not cfg.get("combined_runtime_preflight_enabled", True):
            return

        path = Path(model_path)
        artifact = path if path.is_file() else path / "gbm_model.pkl"
        if not artifact.exists():
            return

        timeout = float(cfg.get("artifact_preflight_timeout", 5.0))
        code = (
            "import sys\n"
            "from src.models.ppo_agent import PPOAgent\n"
            "from src.models.gbm_agent import GBMAgent\n"
            "try:\n"
            "    cfg = {'models': {'gbm': "
            "{'combined_runtime_preflight_enabled': False}}}\n"
            "    PPOAgent(state_dim=10, config=cfg)\n"
            "    GBMAgent(cfg).load(sys.argv[1])\n"
            "except Exception as exc:\n"
            "    print(f'{type(exc).__name__}: {exc}', file=sys.stderr)\n"
            "    sys.exit(2)\n"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-X", "faulthandler", "-c", code, str(path)],
                cwd=str(Path.cwd()),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"GBM artifact preflight timed out after {timeout:.1f}s: {artifact}"
            ) from exc

        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown_error"
            raise RuntimeError(
                "GBM artifact is unsafe in combined PPO/GBM runtime: "
                f"{detail[-600:]}"
            )

    def get_action(
        self, state_vector: list, regime_str: str
    ) -> Tuple[int, float, Dict[str, Any]]:
        """
        Determines the active model based on regime, executes inference,
        and returns the action and explainability payload.
        """
        preferred_model_name = self.regime_model_map.get(regime_str, "PPO")

        if preferred_model_name == "PPO":
            action, conviction, context = self.ppo_agent.act(state_vector)
        else:
            action, conviction, context = self.gbm_agent.act(state_vector)

        logger.debug(
            f"MetaController routed to {preferred_model_name} for "
            f"regime {regime_str}. Action: {action}, "
            f"Conviction: {conviction:.2f}"
        )

        # Enrich context for the Explainability Engine
        context["active_regime"] = regime_str
        context["selected_by_meta"] = preferred_model_name

        return action, conviction, context

    def get_dual_inference(
        self, state_vector: list, regime_str: str
    ) -> Tuple[int, float, Dict[str, Any], list, list]:
        """
        Run both agents; route primary action by regime preference.
        Returns: action, conviction, context, ppo_probs, gbm_probs
        """
        ppo_action, ppo_conv, ppo_ctx = self.ppo_agent.act(state_vector)
        gbm_action, gbm_conv, gbm_ctx = self.gbm_agent.act(state_vector)
        ppo_probs = list(ppo_ctx.get("action_probs", []))
        gbm_probs = list(gbm_ctx.get("action_probs", []))

        preferred = self.regime_model_map.get(regime_str, "PPO")
        if preferred == "PPO":
            action, conviction, context = ppo_action, ppo_conv, ppo_ctx
        else:
            action, conviction, context = gbm_action, gbm_conv, gbm_ctx

        context["active_regime"] = regime_str
        context["selected_by_meta"] = preferred
        context["ppo_action_probs"] = ppo_probs
        context["gbm_action_probs"] = gbm_probs
        return action, conviction, context, ppo_probs, gbm_probs
