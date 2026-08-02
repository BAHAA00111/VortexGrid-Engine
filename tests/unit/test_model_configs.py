from pathlib import Path
from typing import Any, Dict

import pytest
import yaml


@pytest.mark.unit
def test_model_configs_parsing() -> None:
    config_dir = Path("configs/model-configs")
    expected_files = ["llama_7b.yaml", "mistral_7b.yaml"]

    for filename in expected_files:
        filepath = config_dir / filename
        assert filepath.exists(), f"Missing expected model config: {filepath}"

        with open(filepath, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        assert isinstance(raw_data, dict), f"Config {filepath} must parse into a dict"
        data: Dict[str, Any] = raw_data

        assert "model_config" in data
        cfg = data.get("model_config")
        assert isinstance(cfg, dict), "model_config must be a dictionary"

        # Verify key architecture specs
        assert isinstance(cfg.get("vocab_size"), int)
        assert isinstance(cfg.get("hidden_size"), int)
        assert isinstance(cfg.get("num_hidden_layers"), int)
        assert isinstance(cfg.get("num_attention_heads"), int)
        assert isinstance(cfg.get("num_key_value_heads"), int)
        assert isinstance(cfg.get("intermediate_size"), int)
        assert isinstance(cfg.get("max_position_embeddings"), int)


@pytest.mark.unit
def test_mistral_gqa_ratio() -> None:
    filepath = Path("configs/model-configs/mistral_7b.yaml")
    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    assert isinstance(raw_data, dict)
    cfg: Dict[str, Any] = raw_data.get("model_config", {})

    # Mistral uses GQA: 32 attention heads / 8 KV heads = 4
    attn_heads = int(cfg.get("num_attention_heads", 0))
    kv_heads = int(cfg.get("num_key_value_heads", 0))

    assert attn_heads % kv_heads == 0
    assert attn_heads // kv_heads == 4