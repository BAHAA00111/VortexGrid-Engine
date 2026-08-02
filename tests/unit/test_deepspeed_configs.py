import json
from pathlib import Path
import pytest


@pytest.mark.unit
def test_deepspeed_configs_valid_json():
    config_dir = Path("configs/deepspeed")
    expected_files = ["zero1.json", "zero2.json", "zero3_offload.json"]

    for filename in expected_files:
        filepath = config_dir / filename
        assert filepath.exists(), f"Missing config file: {filepath}"

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "zero_optimization" in data
        assert "stage" in data["zero_optimization"]

    assert data["zero_optimization"]["stage"] in [1, 2, 3]