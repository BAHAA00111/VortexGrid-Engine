import pytest
import torch

from vortexgrid.models import FusedCrossEntropyLoss, VocabParallelCrossEntropyLoss


@pytest.mark.unit
def test_fused_cross_entropy_basic():
    loss_fn = FusedCrossEntropyLoss(ignore_index=-100)
    logits = torch.randn(4, 16, 1000, requires_grad=True)  # [Batch, Seq, Vocab]
    targets = torch.randint(0, 1000, (4, 16))

    loss = loss_fn(logits, targets)
    loss.backward()

    assert loss.dim() == 0  # Scalar loss
    assert loss.item() > 0.0
    assert logits.grad is not None


@pytest.mark.unit
def test_fused_cross_entropy_with_z_loss():
    loss_fn = FusedCrossEntropyLoss(ignore_index=-100, z_loss_coeff=1e-4)
    logits = torch.randn(2, 8, 500, requires_grad=True)
    targets = torch.randint(0, 500, (2, 8))

    # Mask some targets
    targets[0, :2] = -100

    loss = loss_fn(logits, targets)
    loss.backward()

    assert not torch.isnan(loss)
    assert loss.item() > 0.0


@pytest.mark.unit
def test_vocab_parallel_cross_entropy():
    loss_fn = VocabParallelCrossEntropyLoss()
    logits = torch.randn(2, 4, 256)
    targets = torch.randint(0, 256, (2, 4))

    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0
