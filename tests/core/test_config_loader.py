import pytest
import yaml

from src.core.config_loader import load_config


@pytest.mark.unit
def test_load_config_reads_yaml_file(tmp_path):
    """Verify config loader parses a valid YAML file."""
    config_path = tmp_path / "base.yaml"
    config_data = {"data": {"target_symbol": "ETHUSDC"}, "risk": {"max_leverage": 3}}
    config_path.write_text(yaml.safe_dump(config_data))

    loaded = load_config(str(config_path))

    assert loaded == config_data


@pytest.mark.unit
def test_load_config_raises_for_missing_file(tmp_path):
    """Verify config loader fails loudly when config path is invalid."""
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "missing.yaml"))
