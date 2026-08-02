import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from vortexgrid.checkpointing.async_sharded_saver import (
    AsyncShardedSaver,
    CheckpointMetadata,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.unit
def test_checkpoint_metadata_serialization():
    meta = CheckpointMetadata(
        step=100,
        timestamp=123456789.0,
        world_size=1,
        sharding_strategy="FULL_SHARD",
        precision="bfloat16",
    )
    json_str = meta.to_json()
    restored = CheckpointMetadata.from_json(json_str)

    assert restored.step == 100
    assert restored.sharding_strategy == "FULL_SHARD"
    assert restored.precision == "bfloat16"


@pytest.mark.unit
def test_async_sharded_saver_sync_save(tmp_path: Path):
    saver = AsyncShardedSaver(base_dir=tmp_path)
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_dir = saver.save_checkpoint(
        step=10,
        model=model,
        optimizer=optimizer,
        async_save=False,
    )

    assert ckpt_dir.exists()
    assert (ckpt_dir / "metadata.json").exists()

    with open(ckpt_dir / "metadata.json", "r") as f:
        meta_data = json.load(f)
        assert meta_data["step"] == 10

    saver.shutdown()


@pytest.mark.unit
def test_async_sharded_saver_async_save(tmp_path: Path):
    saver = AsyncShardedSaver(base_dir=tmp_path)
    model = SimpleModel()

    ckpt_dir = saver.save_checkpoint(
        step=20,
        model=model,
        async_save=True,
    )

    saver.wait_until_idle()
    assert ckpt_dir.exists()
    assert (ckpt_dir / "metadata.json").exists()

    saver.shutdown()
