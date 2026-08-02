"""
Optimizer & Mixed-Precision Factory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Provides high-performance fused AdamW optimizer construction, parameter group
decay separation, cosine/linear learning rate decay schedules with warmup,
and mixed-precision gradient scalers for FP16/BF16/FP8 execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from vortexgrid import logger


@dataclass
class OptimizerConfig:
    """Production configuration container for optimizers, LR schedules, and mixed-precision scaling."""

    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    fused: bool = True
    warmup_steps: int = 1000
    max_steps: int = 50000
    min_lr_ratio: float = 0.1
    schedule_type: str = "cosine"  # cosine | linear | constant
    precision_dtype: str = "bfloat16"  # bfloat16 | float16 | float32 | fp8
    grad_scaler_init_scale: float = 65536.0
    grad_scaler_growth_factor: float = 2.0
    grad_scaler_backoff_factor: float = 0.5
    grad_scaler_growth_interval: int = 2000


def separate_weight_decay_params(
    model: nn.Module,
    weight_decay: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Splits parameters into two groups:
      1. Parameters with weight decay applied (2D+ matrices like linear/embedding weights).
      2. Parameters with zero weight decay (1D vectors like RMSNorm weights, layer bias vectors).
    """
    decay_params: Set[str] = set()
    no_decay_params: Set[str] = set()

    whitelist_weight_modules = (nn.Linear,)
    blacklist_weight_modules = (
        nn.LayerNorm,
        nn.Embedding,
    )

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            fpn = f"{mn}.{pn}" if mn else pn

            if not p.requires_grad:
                continue

            # 1D tensors (biases, normalization weights) skip weight decay
            if p.ndim < 2 or pn.endswith("bias") or "norm" in mn.lower():
                no_decay_params.add(fpn)
            elif isinstance(m, whitelist_weight_modules):
                decay_params.add(fpn)
            elif isinstance(m, blacklist_weight_modules):
                no_decay_params.add(fpn)
            else:
                if p.ndim >= 2:
                    decay_params.add(fpn)
                else:
                    no_decay_params.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    inter_params = decay_params & no_decay_params
    union_params = decay_params | no_decay_params

    assert (
        len(inter_params) == 0
    ), f"Parameters {inter_params} made it into both decay/no_decay sets!"
    assert (
        len(param_dict.keys() - union_params) == 0
    ), f"Parameters {param_dict.keys() - union_params} were not assigned to either decay set!"

    decay_list = [param_dict[pn] for pn in sorted(list(decay_params))]
    no_decay_list = [param_dict[pn] for pn in sorted(list(no_decay_params))]

    logger.info(
        f"Optimizer Parameter Groups | Decay: {len(decay_list)} tensors ({sum(p.numel() for p in decay_list):,} params) | "
        f"No Decay: {len(no_decay_list)} tensors ({sum(p.numel() for p in no_decay_list):,} params)"
    )

    return [
        {"params": decay_list, "weight_decay": weight_decay},
        {"params": no_decay_list, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: nn.Module,
    config: Optional[OptimizerConfig] = None,
) -> AdamW:
    """Constructs a high-performance AdamW optimizer with optional CUDA kernel fusing."""
    if config is None:
        config = OptimizerConfig()

    param_groups: List[Dict[str, Any]] = separate_weight_decay_params(
        model, weight_decay=config.weight_decay
    )

    # Check CUDA availability for fused AdamW kernel execution
    fused_available = (
        config.fused
        and torch.cuda.is_available()
        and "fused" in AdamW.__init__.__code__.co_varnames
    )

    extra_kwargs: Dict[str, Any] = {}
    if fused_available:
        extra_kwargs["fused"] = True

    logger.info(
        f"Constructing AdamW Optimizer | Base LR: {config.lr} | Weight Decay: {config.weight_decay} | "
        f"Betas: ({config.beta1}, {config.beta2}) | Fused: {fused_available}"
    )

    optimizer = AdamW(
        param_groups,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        **extra_kwargs,
    )
    return optimizer


def build_lr_scheduler(
    optimizer: Optimizer,
    config: Optional[OptimizerConfig] = None,
) -> LRScheduler:
    """Builds Cosine or Linear Learning Rate Decay scheduler with warm-up phase."""
    if config is None:
        config = OptimizerConfig()

    warmup_steps = config.warmup_steps
    max_steps = max(config.max_steps, warmup_steps + 1)
    min_lr_ratio = config.min_lr_ratio
    schedule_type = config.schedule_type.lower()

    def lr_lambda(current_step: int) -> float:
        # 1. Warm-up phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # 2. Post warm-up decay phase
        progress = float(current_step - warmup_steps) / float(
            max(1, max_steps - warmup_steps)
        )
        progress = min(1.0, max(0.0, progress))

        if schedule_type == "cosine":
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif schedule_type == "linear":
            decay = 1.0 - progress
        elif schedule_type == "constant":
            decay = 1.0
        else:
            raise ValueError(f"Unsupported schedule type '{schedule_type}'")

        # Interpolate between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_grad_scaler(
    config: Optional[OptimizerConfig] = None,
    enabled: Optional[bool] = None,
) -> torch.amp.GradScaler:
    """
    Constructs a PyTorch Gradient Scaler for mixed-precision execution.
    Note: Enabled by default for FP16 training to prevent underflow, disabled for BF16/FP32.
    """
    if config is None:
        config = OptimizerConfig()

    dtype_str = config.precision_dtype.lower()

    if enabled is None:
        # FP16 requires dynamic gradient scaling; BF16/FP32 do not
        enabled = dtype_str in ("float16", "fp16") and torch.cuda.is_available()

    logger.info(
        f"Initializing Mixed Precision GradScaler | Enabled: {enabled} | Precision: {dtype_str}"
    )

    scaler = torch.amp.GradScaler(
        device="cuda" if torch.cuda.is_available() else "cpu",
        init_scale=config.grad_scaler_init_scale,
        growth_factor=config.grad_scaler_growth_factor,
        backoff_factor=config.grad_scaler_backoff_factor,
        growth_interval=config.grad_scaler_growth_interval,
        enabled=enabled,
    )
    return scaler
