from pathlib import Path
import pytest
import torch
import torch.nn as nn

from vortexgrid.tracker import TensorBoardTracker, WandbTracker


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.mark.unit
def test_tensorboard_tracker_logging(tmp_path: Path):
    tracker = TensorBoardTracker(
        log_dir=tmp_path,
        project_name="test_proj",
        run_name="test_run",
    )

    # Log metrics
    tracker.log_metrics({"train/loss": 0.5, "train/lr": 1e-4}, step=1)

    # Test Gradient logging
    model = SimpleModel()
    x = torch.randn(2, 10)
    out = model(x).sum()
    out.backward()

    norms = tracker.log_gradient_norms(model, step=1)
    assert "gradients/total_norm" in norms
    assert norms["gradients/total_norm"] >= 0.0

    tracker.close()


@pytest.mark.unit
def test_wandb_tracker_disabled_fallback():
    # Verify WandB tracker runs safely without crashing when disabled/uninitialized
    tracker = WandbTracker(enabled=False)
    tracker.log_metrics({"train/loss": 0.5}, step=1)

    model = SimpleModel()
    norms = tracker.log_gradient_norms(model, step=1)
    assert norms == {}
    tracker.close()
