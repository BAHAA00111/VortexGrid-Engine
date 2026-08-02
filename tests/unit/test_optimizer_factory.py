import pytest
import torch
import torch.nn as nn
from vortexgrid.engine.optimizer_factory import (
    OptimizerConfig,
    build_grad_scaler,
    build_lr_scheduler,
    build_optimizer,
    separate_weight_decay_params,
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.norm = nn.LayerNorm(10)

    def forward(self, x):
        return self.norm(self.fc(x))


@pytest.mark.unit
def test_separate_weight_decay_params():
    model = DummyModel()
    groups = separate_weight_decay_params(model, weight_decay=0.1)

    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 0.1
    assert groups[1]["weight_decay"] == 0.0

    # Ensure 2D weight in decay group and 1D bias/norm in no-decay group
    decay_params = set(groups[0]["params"])
    no_decay_params = set(groups[1]["params"])

    assert model.fc.weight in decay_params
    assert model.fc.bias in no_decay_params
    assert model.norm.weight in no_decay_params


@pytest.mark.unit
def test_build_optimizer_and_scheduler():
    model = DummyModel()
    cfg = OptimizerConfig(
        lr=1e-3,
        warmup_steps=10,
        max_steps=100,
        schedule_type="cosine",
        fused=False,
    )

    optimizer = build_optimizer(model, cfg)
    # Check base LR on optimizer prior to scheduler attachment
    assert optimizer.param_groups[0]["lr"] == 1e-3

    scheduler = build_lr_scheduler(optimizer, cfg)

    # Initial warmup step (step 0 starts at 0.0)
    assert scheduler.get_last_lr()[0] == 0.0

    # Step to mid-warmup
    for _ in range(5):
        optimizer.step()
        scheduler.step()

    mid_warmup_lr = scheduler.get_last_lr()[0]
    assert 0.0 < mid_warmup_lr < 1e-3


@pytest.mark.unit
def test_build_grad_scaler():
    fp16_cfg = OptimizerConfig(precision_dtype="float16")
    scaler = build_grad_scaler(fp16_cfg, enabled=False)
    assert scaler.is_enabled() is False

    bf16_cfg = OptimizerConfig(precision_dtype="bfloat16")
    bf16_scaler = build_grad_scaler(bf16_cfg)
    assert bf16_scaler.is_enabled() is False
