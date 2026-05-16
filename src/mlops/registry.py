import json
import os
import shutil
import logging
from typing import Dict, Any, Optional
from datetime import datetime

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
            with open(self.registry_file, 'w') as f:
                json.dump({"models": {}, "active_prod": None, "active_shadow": None}, f)

    def _load_registry(self) -> dict:
        with open(self.registry_file, 'r') as f:
            return json.load(f)

    def _save_registry(self):
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry_data, f, indent=4)

    def register_model(self, model_id: str, model_type: str, metrics: Dict[str, float]) -> str:
        """
        Registers a newly trained model.
        Returns the path where the model artifact should be saved.
        """
        timestamp = datetime.utcnow().isoformat()
        
        self.registry_data["models"][model_id] = {
            "type": model_type,
            "created_at": timestamp,
            "status": "EVALUATING",
            "metrics": metrics
        }
        self._save_registry()
        
        model_path = os.path.join(self.registry_dir, model_id)
        os.makedirs(model_path, exist_ok=True)
        logger.info(f"Registered new model {model_id} ({model_type})")
        return model_path

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
            
        self.registry_data["models"][model_id]["status"] = "PROD"
        self.registry_data["active_prod"] = model_id
        self._save_registry()
        logger.critical(f"*** Model {model_id} promoted to PRODUCTION! ***")

    def get_prod_model_path(self) -> Optional[str]:
        """Returns the directory path of the active PROD model."""
        active = self.registry_data.get("active_prod")
        if active:
            return os.path.join(self.registry_dir, active)
        return None
