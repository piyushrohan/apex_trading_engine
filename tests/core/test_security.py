import pytest
import os
from cryptography.fernet import Fernet
from src.core.security import SecurityManager

@pytest.mark.unit
def test_security_manager_encryption_cycle(tmp_path):
    """
    Test that the SecurityManager successfully encrypts and decrypts 
    sensitive API credentials using a generated master key.
    """
    # Generate a master key and set it in the environment
    master_key = Fernet.generate_key().decode()
    os.environ['APEX_MASTER_KEY'] = master_key
    
    # Use temporary directory for the encrypted keys file
    test_key_file = str(tmp_path / "api_keys.enc")
    
    manager = SecurityManager(key_file_path=test_key_file)
    
    # Test Payload
    payload = {
        "api_key": "BINANCE_TEST_API_KEY_123",
        "api_secret": "BINANCE_TEST_API_SECRET_456"
    }
    
    # Encrypt and save
    manager.encrypt_and_save_keys(payload)
    
    assert os.path.exists(test_key_file)
    
    # Load and decrypt
    decrypted_payload = manager.load_and_decrypt_keys()
    
    assert decrypted_payload["api_key"] == "BINANCE_TEST_API_KEY_123"
    assert decrypted_payload["api_secret"] == "BINANCE_TEST_API_SECRET_456"
    
    # Cleanup
    del os.environ['APEX_MASTER_KEY']
