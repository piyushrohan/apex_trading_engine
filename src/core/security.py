import json
import logging
import os
from pathlib import Path

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class SecurityManager:
    """
    Institutional grade security manager for API keys.
    API keys should never be stored in plain text.
    This class uses Fernet (symmetric encryption) to encrypt API keys on disk.
    The master key must be provided via environment variable: APEX_MASTER_KEY.
    """

    def __init__(
        self,
        key_path: str = "configs/.keys.enc",
        key_file_path: str | None = None,
    ):
        resolved_path = key_file_path or key_path
        self.key_path = Path(resolved_path)
        self.master_key = os.getenv("APEX_MASTER_KEY")
        self.cipher = None
        if not self.master_key:
            # During first run or setup, we might need to generate a master key
            logger.warning(
                "APEX_MASTER_KEY environment variable not found. "
                "Encryption/Decryption will fail unless generating a new key."
            )
        else:
            try:
                self.cipher = Fernet(self.master_key.encode())
            except Exception as e:
                logger.error(
                    f"Failed to initialize cipher with provided APEX_MASTER_KEY: {e}"
                )
                self.cipher = None

    @staticmethod
    def generate_master_key() -> str:
        """Generates a new master key for secure environment storage."""
        return Fernet.generate_key().decode()

    def encrypt_and_save_keys(
        self, api_key: str | dict[str, str], api_secret: str | None = None
    ):
        """Encrypts Binance API key and secret and saves to disk."""
        if not self.master_key or not self.cipher:
            raise ValueError("Master key not initialized. Set APEX_MASTER_KEY.")

        if isinstance(api_key, dict):
            api_secret = api_key["api_secret"]
            api_key = api_key["api_key"]

        if api_secret is None:
            raise ValueError("API secret is required.")

        payload = json.dumps({"api_key": api_key, "api_secret": api_secret}).encode()

        encrypted_data = self.cipher.encrypt(payload)

        # Ensure directory exists
        self.key_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.key_path, "wb") as f:
            f.write(encrypted_data)

        # Secure the file permissions
        os.chmod(self.key_path, 0o600)
        logger.info(f"API keys encrypted and saved successfully to {self.key_path}")

    def load_and_decrypt_keys(self) -> dict:
        """Loads and decrypts API keys from disk."""
        if not self.master_key or not self.cipher:
            raise ValueError("Master key not initialized. Set APEX_MASTER_KEY.")

        if not self.key_path.exists():
            raise FileNotFoundError(
                f"Encrypted key file not found at {self.key_path}. Run setup first."
            )

        with open(self.key_path, "rb") as f:
            encrypted_data = f.read()

        decrypted_data = self.cipher.decrypt(encrypted_data)
        keys = json.loads(decrypted_data.decode())
        return keys

    def get_api_credentials(self) -> tuple[str, str]:
        """
        Retrieves API credentials.
        Returns: (api_key, api_secret)
        """
        # Allow fallback to environment variables for CI/CD or docker
        env_api_key = os.getenv("BINANCE_API_KEY")
        env_api_secret = os.getenv("BINANCE_API_SECRET")

        if env_api_key and env_api_secret:
            logger.info("Loaded Binance API credentials from environment variables.")
            return env_api_key, env_api_secret

        # Attempt to load from encrypted file
        keys = self.load_and_decrypt_keys()
        logger.info("Loaded Binance API credentials from encrypted keystore.")
        return keys["api_key"], keys["api_secret"]
