from pathlib import Path
import pytest
import torch
import torch.nn as nn
from vortexgrid.checkpointing.async_sharded_saver import AsyncShardedSaver
from vortexgrid.checkpointing.state_loader import StateLoader


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        return self.fc(x)


@pytest.mark.unit
def test_inspect_checkpoint_metadata(tmp_path: Path):
    saver = AsyncShardedSaver(base_dir=tmp_path)
    loader = StateLoader()
    model = DummyModel()

    ckpt_dir = saver.save_checkpoint(
        step=42,
        model=model,
        async_save=False,
    )

    metadata = loader.inspect_checkpoint(ckpt_dir)
    assert metadata.step == 42
    assert metadata.world_size == 1

    saver.shutdown()


@pytest.mark.unit
def test_save_and_load_state_cycle(tmp_path: Path):
    saver = AsyncShardedSaver(base_dir=tmp_path)
    loader = StateLoader()

    model1 = DummyModel()
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)

    # Perform a dummy forward + backward + optimizer step to populate momentum states
    x = torch.randn(2, 8)
    loss = model1(x).sum()
    loss.backward()
    optimizer1.step()

    # Save checkpoint
    ckpt_dir = saver.save_checkpoint(
        step=100,
        model=model1,
        optimizer=optimizer1,
        extra_state={"epoch": 5, "loss": 0.123},
        async_save=False,
    )

    # Instantiate fresh model and optimizer targets
    model2 = DummyModel()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

    # Verify parameters differ prior to loading
    assert not torch.allclose(model1.fc.weight, model2.fc.weight)

    # Restore state
    extra_state = loader.load_checkpoint(
        checkpoint_dir=ckpt_dir,
        model=model2,
        optimizer=optimizer2,
    )

    # Verify model weights match exactly after restore
    assert torch.allclose(model1.fc.weight, model2.fc.weight)
    assert extra_state.get("epoch") == 5
    assert extra_state.get("loss") == 0.123

    saver.shutdown()