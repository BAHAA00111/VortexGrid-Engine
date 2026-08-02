import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from vortexgrid.engine.fsdp_wrapper import (
    FSDPConfig,
    get_mixed_precision_policy,
    get_sharding_strategy,
    wrap_model_fsdp,
)
from vortexgrid.models import ModelConfig, TransformerBlock


@pytest.fixture(autouse=True)
def cleanup_dist_group():
    """Teardown active process group between unit test runs."""
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


class DummyModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layer1 = TransformerBlock(config)
        self.layer2 = TransformerBlock(config)

    def forward(self, x, cos, sin):
        h = self.layer1(x, cos, sin)
        return self.layer2(h, cos, sin)


@pytest.mark.unit
def test_sharding_strategy_mapping():
    assert get_sharding_strategy("FULL_SHARD") == ShardingStrategy.FULL_SHARD
    assert get_sharding_strategy("SHARD_GRAD_OP") == ShardingStrategy.SHARD_GRAD_OP
    with pytest.raises(ValueError):
        get_sharding_strategy("INVALID_STRATEGY")


@pytest.mark.unit
def test_mixed_precision_policy():
    policy = get_mixed_precision_policy("bfloat16")
    assert policy.param_dtype == torch.bfloat16
    assert policy.reduce_dtype == torch.float32

    fp32_policy = get_mixed_precision_policy("float32")
    assert fp32_policy.param_dtype == torch.float32


@pytest.mark.unit
def test_fsdp_wrapper_fallback_cpu():
    config = ModelConfig(dim=128, n_heads=4, n_layers=2)
    model = DummyModel(config)
    fsdp_cfg = FSDPConfig(sharding_strategy="NO_SHARD", mixed_precision_dtype="float32")

    wrapped_model = wrap_model_fsdp(model, fsdp_config=fsdp_cfg, device_id=None)
    assert isinstance(wrapped_model, FSDP)
