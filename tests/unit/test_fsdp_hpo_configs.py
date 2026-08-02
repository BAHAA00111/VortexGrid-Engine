from pathlib import Path
from typing import Any, Dict

import pytest
import yaml


@pytest.mark.unit
def test_fsdp_configs_parsing() -> None:
    fsdp_dir = Path("configs/fsdp")
    expected_files = ["full_shard.yaml", "hybrid_shard.yaml"]

    for filename in expected_files:
        filepath = fsdp_dir / filename
        assert filepath.exists(), f"Missing expected FSDP config: {filepath}"

        with open(filepath, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        assert isinstance(raw_data, dict), f"Config {filepath} must parse into a dict"
        data: Dict[str, Any] = raw_data

        assert "fsdp_config" in data
        fsdp_config = data.get("fsdp_config")
        assert isinstance(fsdp_config, dict), "fsdp_config must be a dictionary"

        assert "sharding_strategy" in fsdp_config
        assert fsdp_config["sharding_strategy"] in ["FULL_SHARD", "HYBRID_SHARD"]
        assert "mixed_precision" in fsdp_config


@pytest.mark.unit
def test_optuna_hpo_config_parsing() -> None:
    hpo_file = Path("configs/hpo/optuna_config.yaml")
    assert hpo_file.exists(), f"Missing Optuna HPO config file: {hpo_file}"

    with open(hpo_file, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    assert isinstance(raw_data, dict), "Optuna HPO config must parse into a dict"
    data: Dict[str, Any] = raw_data

    assert "optuna_study" in data
    assert "sampler" in data
    assert "pruner" in data
    assert "search_space" in data

    search_space = data.get("search_space")
    assert isinstance(search_space, dict), "search_space must be a dictionary"
    assert "learning_rate" in search_space
