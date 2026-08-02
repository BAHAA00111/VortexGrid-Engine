import pytest
import torch
from vortexgrid.models import (
    FusedCrossEntropyLoss,
    ModelConfig,
    RMSNorm,
    RotaryEmbedding,
    TransformerBlock,
)


@pytest.fixture
def small_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=1000,
        dim=256,
        n_layers=2,
        n_heads=8,
        n_kv_heads=4,
        max_seq_len=512,
        use_gradient_checkpointing=True,
    )


@pytest.mark.unit
def test_rmsnorm_forward():
    norm = RMSNorm(dim=256)
    x = torch.randn(2, 16, 256)
    out = norm(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


@pytest.mark.unit
def test_rotary_embeddings(small_config):
    rope = RotaryEmbedding(
        dim=small_config.dim // small_config.n_heads,
        max_seq_len=small_config.max_seq_len,
    )
    x = torch.randn(2, 16, small_config.dim)
    cos, sin = rope(x, seq_len=16)
    assert cos.shape == (16, 32)
    assert sin.shape == (16, 32)


@pytest.mark.unit
def test_transformer_block_forward_and_recomputation(small_config):
    block = TransformerBlock(small_config)
    block.train()

    rope = RotaryEmbedding(
        dim=small_config.dim // small_config.n_heads,
        max_seq_len=small_config.max_seq_len,
    )

    x = torch.randn(2, 16, small_config.dim, requires_grad=True)
    cos, sin = rope(x, seq_len=16)

    out = block(x, cos, sin)
    assert out.shape == x.shape

    # Test backward pass gradient propagation through recomputed checkpoint
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


@pytest.mark.unit
def test_fused_cross_entropy_loss():
    loss_fn = FusedCrossEntropyLoss(ignore_index=-100)
    logits = torch.randn(2, 16, 1000, requires_grad=True)
    targets = torch.randint(0, 1000, (2, 16))

    loss = loss_fn(logits, targets)
    assert loss.dim() == 0  # Scalar loss tensor
    assert not torch.isnan(loss)

    loss.backward()
    assert logits.grad is not None