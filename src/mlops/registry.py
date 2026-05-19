import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


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
                json.dump({"models": {}, "active_prod": None, "active_shadow": None}, f)

    def _load_registry(self) -> dict:
        with open(self.registry_file, "r") as f:
            return json.load(f)

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
    ) -> str:
        """
        Registers a newly trained model.
        Returns the path where the model artifact should be saved.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        model_path = self.get_model_path(model_id)

        self.registry_data["models"][model_id] = {
            "type": model_type,
            "created_at": timestamp,
            "status": "EVALUATING",
            "metrics": metrics,
            "artifact_path": model_path,
            "metadata": metadata or {},
        }
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
    ) -> str:
        """Persist a reproducibility manifest next to the model artifacts."""
        if model_id not in self.registry_data["models"]:
            raise ValueError(f"Model {model_id} not found in registry.")
        model_path = self.get_model_path(model_id)
        os.makedirs(model_path, exist_ok=True)
        manifest_path = os.path.join(model_path, "manifest.json")
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

    def set_model_status(self, model_id: str, status: str):
        if model_id not in self.registry_data["models"]:
            raise ValueError(f"Model {model_id} not found in registry.")
        self.registry_data["models"][model_id]["status"] = status
        self._save_registry()

    def promote_to_shadow(self, model_id: str):
        """Moves a model to SHADOW status for live paper-trading."""
        if model_id not in self.registry_data["models"]:
            raise ValueError(f"Model {model_id} not found in registry.")

        old_shadow = self.registry_data.get("active_shadow")
        if old_shadow and old_shadow in self.registry_data["models"]:
            self.registry_data["models"][old_shadow]["status"] = "ARCHIVED"

        self.registry_data["models"][model_id]["status"] = "SHADOW"
        self.registry_data["active_shadow"] = model_id
        self._save_registry()
        logger.info(f"Model {model_id} promoted to SHADOW.")

    def promote_to_prod(self, model_id: str):
        """Moves a model to PROD status for live capital execution."""
        if model_id not in self.registry_data["models"]:
            raise ValueError(f"Model {model_id} not found in registry.")

        old_prod = self.registry_data.get("active_prod")
        if old_prod and old_prod in self.registry_data["models"]:
            self.registry_data["models"][old_prod]["status"] = "ARCHIVED"
            self.registry_data["previous_prod"] = old_prod

        self.registry_data["models"][model_id]["status"] = "PROD"
        self.registry_data["active_prod"] = model_id
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
        self.registry_data["models"][previous]["status"] = "PROD"
        self.registry_data["active_prod"] = previous
        self.registry_data["previous_prod"] = active
        self._save_registry()
        return previous

    def get_prod_model_path(self) -> Optional[str]:
        """Returns the directory path of the active PROD model."""
        active = self.registry_data.get("active_prod")
        if active:
            return self.get_model_path(active)
        return None
