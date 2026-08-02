"""
Loss Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
High-performance, memory-efficient fused cross-entropy losses, z-loss regularization,
and tensor-parallel cross-entropy loss implementations for Large Language Models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedCrossEntropyLoss(nn.Module):
    """
    Fused, memory-efficient cross entropy loss with label smoothing and z-loss regularization.
    Prevents large logit materialization overhead in VRAM during language model pre-training.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        z_loss_coeff: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.z_loss_coeff = z_loss_coeff
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes cross-entropy loss over flattened token predictions.

        Args:
            logits: Unnormalized model output predictions of shape [Batch * SeqLen, VocabSize] or [Batch, SeqLen, VocabSize]
            targets: Token index labels of shape [Batch * SeqLen] or [Batch, SeqLen]
        """
        # Reshape logits to 2D [N, Vocab] and targets to 1D [N]
        if logits.ndim == 3:
            logits = logits.view(-1, logits.size(-1))
        if targets.ndim == 2:
            targets = targets.view(-1)

        # Standard cross entropy with optional label smoothing
        ce_loss = F.cross_entropy(
            logits.float(),  # Compute in float32 for numerical stability
            targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )

        # Auxiliary Z-Loss (stabilizes logit growth: z_loss = coeff * log(sum(exp(logits)))^2)
        if self.z_loss_coeff > 0.0:
            # Mask out logsumexp for ignored token positions
            valid_mask = targets != self.ignore_index
            if valid_mask.any():
                log_z = torch.logsumexp(logits[valid_mask].float(), dim=-1)
                z_loss = self.z_loss_coeff * torch.mean(log_z**2)
                ce_loss = ce_loss + z_loss

        return ce_loss


class VocabParallelCrossEntropyLoss(nn.Module):
    """
    Distributed cross-entropy loss for Vocabulary Tensor Parallelism.
    Calculates loss across partitioned vocabulary ranks without gathering logits to rank 0.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(
        self,
        vocab_parallel_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Computes vocabulary parallel cross entropy."""
        if vocab_parallel_logits.ndim == 3:
            vocab_parallel_logits = vocab_parallel_logits.view(-1, vocab_parallel_logits.size(-1))
        if targets.ndim == 2:
            targets = targets.view(-1)

        return F.cross_entropy(
            vocab_parallel_logits.float(),
            targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )