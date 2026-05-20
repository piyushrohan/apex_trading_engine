import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

MODEL_STATUSES = {
    "CANDIDATE",
    "EVALUATING",
    "SHADOW",
    "APPROVED",
    "PROD",
    "REJECTED",
    "ROLLED_BACK",
    "ARCHIVED",
}


class ModelRegistry:
    """
    Manages model versioning, artifact storage, and deployment states (SHADOW vs PROD).
    Ensures safe promotion and rollback capabilities.
    """

    def __init__(self, registry_dir: str = "data_lake/models"):
        self.registry_dir = registry_dir
        self.registry_file = os.path.join(registry_dir, "registry.json")
        self._ensure_directories()
        self.registry_data = self._load_registry()

    def _ensure_directories(self):
        os.makedirs(self.registry_dir, exist_ok=True)
        if not os.path.exists(self.registry_file):
            with open(self.registry_file, "w") as f:
                json.dump(
                    {
                        "schema_version": 2,
                        "models": {},
                        "active_prod": None,
                        "active_shadow": None,
                        "previous_prod": None,
                        "events": [],
                    },
                    f,
                )

    def _load_registry(self) -> dict:
        with open(self.registry_file, "r") as f:
            data = json.load(f)
        data.setdefault("schema_version", 2)
        data.setdefault("models", {})
        data.setdefault("active_prod", None)
        data.setdefault("active_shadow", None)
        data.setdefault("previous_prod", None)
        data.setdefault("events", [])
        return data

    def _save_registry(self):
        with open(self.registry_file, "w") as f:
            json.dump(self.registry_data, f, indent=4)

    def get_model_path(self, model_id: str) -> str:
        return os.path.join(self.registry_dir, model_id)

    def register_model(
        self,
        model_id: str,
        model_type: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        status: str = "CANDIDATE",
    ) -> str:
        """
        Registers a newly trained model.
        Returns the path where the model artifact should be saved.
        """
        self._validate_status(status)
        timestamp = datetime.now(timezone.utc).isoformat()
        model_path = self.get_model_path(model_id)

        self.registry_data["models"][model_id] = {
            "type": model_type,
            "created_at": timestamp,
            "status": status,
            "metrics": metrics,
            "artifact_path": model_path,
            "metadata": metadata or {},
            "lifecycle": [
                {
                    "timestamp": timestamp,
                    "event": "registered",
                    "status": status,
                    "actor": "auto_retrain",
                    "reason": "candidate_registered",
                }
            ],
        }
        self._record_event(
            model_id,
            "registered",
            status=status,
            actor="auto_retrain",
            reason="candidate_registered",
        )
        self._save_registry()

        os.makedirs(model_path, exist_ok=True)
        logger.info(f"Registered new model {model_id} ({model_type})")
        return model_path

    def write_model_manifest(
        self,
        model_id: str,
        *,
        data_snapshot_id: Optional[str] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        allow_overwrite: bool = False,
    ) -> str:
        """Persist a reproducibility manifest next to the model artifacts."""
        if model_id not in self.registry_data["models"]:
            raise ValueError(f"Model {model_id} not found in registry.")
        model_path = self.get_model_path(model_id)
        os.makedirs(model_path, exist_ok=True)
        manifest_path = os.path.join(model_path, "manifest.json")
        if os.path.exists(manifest_path) and not allow_overwrite:
            raise FileExistsError(
                f"Manifest for {model_id} already exists; manifests are immutable."
            )
        payload = {
            "model_id": model_id,
            "model_type": self.registry_data["models"][model_id].get("type"),
            "created_at": self.registry_data["models"][model_id].get("created_at"),
            "git_hash": self._current_git_hash(),
            "data_snapshot_id": data_snapshot_id,
            "hyperparams": hyperparams or {},
            "metrics": metrics
            or self.registry_data["models"][model_id].get("metrics", {}),
        }
        with open(manifest_path, "w") as f:
            json.dump(payload, f, indent=4)
        self.registry_data["models"][model_id]["manifest_path"] = manifest_path
        metadata = self.registry_data["models"][model_id].setdefault("metadata", {})
        metadata["git_hash"] = payload["git_hash"]
        metadata["data_snapshot_id"] = data_snapshot_id
        self._record_model_lifecycle(
            model_id,
            "manifest_written",
            actor="auto_retrain",
            reason="reproducibility_manifest",
            metadata={"manifest_path": manifest_path},
        )
        self._save_registry()
        return manifest_path

    @staticmethod
    def _current_git_hash() -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        return result.stdout.strip() or None

    def update_model_metrics(self, model_id: str, metrics: Dict[str, float]):
        if model_id not in self.registry_data["models"]:
            raise ValueError(f"Model {model_id} not found in registry.")
        self.registry_data["models"][model_id]["metrics"] = metrics
        self._save_registry()

    def set_model_status(
        self,
        model_id: str,
        status: str,
        *,
        actor: str = "system",
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self._validate_status(status)
        if model_id not in self.registry_data["models"]:
            raise ValueError(f"Model {model_id} not found in registry.")
        self.registry_data["models"][model_id]["status"] = status
        self._record_model_lifecycle(
            model_id,
            f"status_{status.lower()}",
            actor=actor,
            reason=reason,
            metadata=metadata,
        )
        self._save_registry()

    def promote_to_shadow(self, model_id: str):
        """Moves a model to SHADOW status for live paper-trading."""
        if model_id not in self.registry_data["models"]:
            raise ValueError(f"Model {model_id} not found in registry.")

        old_shadow = self.registry_data.get("active_shadow")
        if old_shadow and old_shadow in self.registry_data["models"]:
            self.set_model_status(
                old_shadow,
                "ARCHIVED",
                actor="registry",
                reason="replaced_by_new_shadow",
            )

        self.registry_data["models"][model_id]["status"] = "SHADOW"
        self.registry_data["active_shadow"] = model_id
        self._record_model_lifecycle(
            model_id,
            "promoted_to_shadow",
            actor="promotion_gate",
            reason="offline_safety_passed",
        )
        self._save_registry()
        logger.info(f"Model {model_id} promoted to SHADOW.")

    def approve_for_prod(
        self,
        model_id: str,
        *,
        reviewer: str = "human",
        reason: str = "promotion_gate_passed",
    ) -> None:
        if model_id not in self.registry_data["models"]:
            raise ValueError(f"Model {model_id} not found in registry.")
        current = self.registry_data["models"][model_id].get("status")
        if current not in {"SHADOW", "APPROVED"}:
            raise ValueError(
                f"Model {model_id} must be SHADOW before approval; got {current}."
            )
        self.set_model_status(
            model_id,
            "APPROVED",
            actor=reviewer,
            reason=reason,
        )

    def promote_to_prod(
        self,
        model_id: str,
        *,
        reviewer: str = "promotion_service",
        reason: str = "approved_for_production",
    ):
        """Moves a model to PROD status for live capital execution."""
        if model_id not in self.registry_data["models"]:
            raise ValueError(f"Model {model_id} not found in registry.")
        current = self.registry_data["models"][model_id].get("status")
        if current != "APPROVED":
            raise ValueError(
                f"Model {model_id} must be APPROVED before PROD promotion; got "
                f"{current}."
            )

        old_prod = self.registry_data.get("active_prod")
        if old_prod and old_prod in self.registry_data["models"]:
            self.registry_data["models"][old_prod]["status"] = "ARCHIVED"
            self._record_model_lifecycle(
                old_prod,
                "archived",
                actor="registry",
                reason=f"replaced_by_{model_id}",
            )
            self.registry_data["previous_prod"] = old_prod

        self.registry_data["models"][model_id]["status"] = "PROD"
        self.registry_data["active_prod"] = model_id
        self._record_model_lifecycle(
            model_id,
            "promoted_to_prod",
            actor=reviewer,
            reason=reason,
        )
        self._save_registry()
        logger.critical(f"*** Model {model_id} promoted to PRODUCTION! ***")

    def rollback_prod(self) -> Optional[str]:
        """Restore the previous production model if one is available."""
        previous = self.registry_data.get("previous_prod")
        active = self.registry_data.get("active_prod")
        if not previous or previous not in self.registry_data["models"]:
            return None
        if active and active in self.registry_data["models"]:
            self.registry_data["models"][active]["status"] = "ROLLED_BACK"
            self._record_model_lifecycle(
                active,
                "rolled_back",
                actor="promotion_service",
                reason="live_breach",
            )
        self.registry_data["models"][previous]["status"] = "PROD"
        self.registry_data["active_prod"] = previous
        self.registry_data["previous_prod"] = active
        self._record_model_lifecycle(
            previous,
            "restored_to_prod",
            actor="promotion_service",
            reason="rollback_restore",
        )
        self._save_registry()
        return previous

    def get_prod_model_path(self) -> Optional[str]:
        """Returns the directory path of the active PROD model."""
        active = self.registry_data.get("active_prod")
        if active:
            return self.get_model_path(active)
        return None

    def production_readiness(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Return deterministic blockers for allowing a model into live inference."""
        selected = model_id or self.registry_data.get("active_prod")
        blockers = []
        if not selected:
            blockers.append("no_active_prod_model")
            return {
                "model_id": None,
                "ready": False,
                "status": None,
                "manifest_exists": False,
                "artifact_exists": False,
                "blockers": blockers,
            }

        model = self.registry_data.get("models", {}).get(selected)
        if not model:
            blockers.append("active_prod_missing_registry_entry")
            return {
                "model_id": selected,
                "ready": False,
                "status": None,
                "manifest_exists": False,
                "artifact_exists": False,
                "blockers": blockers,
            }

        status = model.get("status")
        if status != "PROD":
            blockers.append(f"status_not_prod:{status}")

        manifest_path = model.get("manifest_path")
        manifest_exists = bool(manifest_path and os.path.exists(manifest_path))
        if not manifest_exists:
            blockers.append("missing_manifest")

        artifact_exists = self._artifact_exists(selected, model)
        if not artifact_exists:
            blockers.append("missing_model_artifact")

        metadata = model.get("metadata", {})
        if not metadata.get("data_snapshot_id"):
            blockers.append("missing_data_snapshot_id")
        if not metadata.get("git_hash"):
            blockers.append("missing_git_hash")

        return {
            "model_id": selected,
            "ready": not blockers,
            "status": status,
            "manifest_exists": manifest_exists,
            "artifact_exists": artifact_exists,
            "blockers": blockers,
            "metadata": metadata,
        }

    def _artifact_exists(self, model_id: str, model: Dict[str, Any]) -> bool:
        model_path = model.get("artifact_path") or self.get_model_path(model_id)
        if not model_path or not os.path.isdir(model_path):
            return False
        expected = {
            "PPO": "ppo_actor_critic.pt",
            "GBM": "gbm_model.pkl",
            "LIGHTGBM": "gbm_model.pkl",
        }.get(str(model.get("type", "")).upper())
        if expected:
            return os.path.exists(os.path.join(model_path, expected))
        return any(os.scandir(model_path))

    @staticmethod
    def _validate_status(status: str) -> None:
        if status not in MODEL_STATUSES:
            raise ValueError(f"Invalid model status {status}.")

    def _record_model_lifecycle(
        self,
        model_id: str,
        event: str,
        *,
        actor: str,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        status = self.registry_data["models"][model_id].get("status")
        self._record_event(
            model_id,
            event,
            status=status,
            actor=actor,
            reason=reason,
            metadata=metadata,
        )
        self.registry_data["models"][model_id].setdefault("lifecycle", []).append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "status": status,
                "actor": actor,
                "reason": reason,
                "metadata": metadata or {},
            }
        )

    def _record_event(
        self,
        model_id: str,
        event: str,
        *,
        status: str,
        actor: str,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.registry_data.setdefault("events", []).append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_id": model_id,
                "event": event,
                "status": status,
                "actor": actor,
                "reason": reason,
                "metadata": metadata or {},
            }
        )
