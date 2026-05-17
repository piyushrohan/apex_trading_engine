import pytest
from cryptography.fernet import Fernet

from src.core.security import SecurityManager


@pytest.mark.unit
def test_security_manager_generates_valid_master_key():
    """Verify generated master keys can initialize a Fernet cipher."""
    generated_key = SecurityManager.generate_master_key()

    assert Fernet(generated_key.encode())


@pytest.mark.unit
def test_security_manager_requires_master_key(monkeypatch, tmp_path):
    """Verify encrypted operations fail when no master key is configured."""
    monkeypatch.delenv("APEX_MASTER_KEY", raising=False)
    manager = SecurityManager(key_file_path=str(tmp_path / "api_keys.enc"))

    with pytest.raises(ValueError, match="Master key"):
        manager.encrypt_and_save_keys("key", "secret")


@pytest.mark.unit
def test_security_manager_rejects_invalid_master_key(monkeypatch, tmp_path):
    """Verify malformed master keys disable the cipher and block writes."""
    monkeypatch.setenv("APEX_MASTER_KEY", "not-a-fernet-key")
    manager = SecurityManager(key_file_path=str(tmp_path / "api_keys.enc"))

    with pytest.raises(ValueError, match="Master key"):
        manager.encrypt_and_save_keys("key", "secret")


@pytest.mark.unit
def test_security_manager_requires_api_secret(monkeypatch, tmp_path):
    """Verify callers cannot persist partial credentials."""
    monkeypatch.setenv("APEX_MASTER_KEY", Fernet.generate_key().decode())
    manager = SecurityManager(key_file_path=str(tmp_path / "api_keys.enc"))

    with pytest.raises(ValueError, match="API secret"):
        manager.encrypt_and_save_keys("key")


@pytest.mark.unit
def test_security_manager_load_raises_for_missing_key_file(monkeypatch, tmp_path):
    """Verify decrypting fails clearly if no encrypted key file exists."""
    monkeypatch.setenv("APEX_MASTER_KEY", Fernet.generate_key().decode())
    manager = SecurityManager(key_file_path=str(tmp_path / "missing.enc"))

    with pytest.raises(FileNotFoundError):
        manager.load_and_decrypt_keys()


@pytest.mark.unit
def test_security_manager_load_requires_master_key(monkeypatch, tmp_path):
    """Verify decrypting also requires an initialized master key."""
    monkeypatch.delenv("APEX_MASTER_KEY", raising=False)
    manager = SecurityManager(key_file_path=str(tmp_path / "api_keys.enc"))

    with pytest.raises(ValueError, match="Master key"):
        manager.load_and_decrypt_keys()


@pytest.mark.unit
def test_security_manager_prefers_environment_credentials(monkeypatch, tmp_path):
    """Verify live credentials can be sourced directly from the environment."""
    monkeypatch.setenv("APEX_MASTER_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("BINANCE_API_KEY", "env-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "env-secret")
    manager = SecurityManager(key_file_path=str(tmp_path / "api_keys.enc"))

    assert manager.get_api_credentials() == ("env-key", "env-secret")

    monkeypatch.delenv("BINANCE_API_KEY")
    monkeypatch.delenv("BINANCE_API_SECRET")


@pytest.mark.unit
def test_security_manager_loads_credentials_from_keystore(monkeypatch, tmp_path):
    """Verify credentials fall back to the encrypted keystore."""
    monkeypatch.setenv("APEX_MASTER_KEY", Fernet.generate_key().decode())
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    manager = SecurityManager(key_file_path=str(tmp_path / "api_keys.enc"))
    manager.encrypt_and_save_keys("stored-key", "stored-secret")

    assert manager.get_api_credentials() == ("stored-key", "stored-secret")
