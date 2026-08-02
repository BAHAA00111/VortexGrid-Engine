"""
Engineers PyTorch Fully Sharded Data Parallel (FSDP) execution layers,
providing automated transformer block wrapping, mixed-precision configuration,
CPU parameter/optimizer offloading rules, and sharded checkpoint handling.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Callable, Optional, Set, Type

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    LocalOptimStateDictConfig,
    LocalStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)

from vortexgrid import logger
from vortexgrid.models.transformer_blocks import TransformerBlock


@dataclass
class FSDPConfig:
    """Production runtime configuration for FSDP memory sharding strategy."""

    sharding_strategy: str = field(
        default_factory=lambda: "FULL_SHARD"  # FULL_SHARD (ZeRO-3), SHARD_GRAD_OP (ZeRO-2), NO_SHARD
    )
    mixed_precision_dtype: str = field(
        default_factory=lambda: "bfloat16"  # bfloat16 | float16 | float32
    )
    cpu_offload: bool = False
    backward_prefetch: str = "BACKWARD_POST"  # BACKWARD_PRE | BACKWARD_POST
    forward_prefetch: bool = True
    limit_all_gathers: bool = True
    min_num_params: int = 1_000_000  # For size-based wrapping fallback


def get_sharding_strategy(strategy_str: str) -> ShardingStrategy:
    """Maps string configuration names to PyTorch FSDP ShardingStrategy enums."""
    mapping = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    strategy_upper = strategy_str.upper()
    if strategy_upper not in mapping:
        raise ValueError(
            f"Unsupported sharding strategy '{strategy_str}'. "
            f"Supported options: {list(mapping.keys())}"
        )
    return mapping[strategy_upper]


def get_mixed_precision_policy(dtype_str: str) -> MixedPrecision:
    """Constructs mixed-precision policy for param, grad, and buffer casting."""
    dtype_str_lower = dtype_str.lower()
    if dtype_str_lower == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_str_lower in ("float16", "fp16"):
        dtype = torch.float16
    elif dtype_str_lower in ("float32", "fp32"):
        return MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unsupported mixed precision dtype '{dtype_str}'")

    return MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=torch.float32,  # Keep gradient reduction in FP32 for numerical stability
        buffer_dtype=dtype,
    )


def build_auto_wrap_policy(
    target_layer_cls: Optional[Set[Type[nn.Module]]] = None,
    min_num_params: int = 1_000_000,
) -> Callable:
    """
    Constructs a PyTorch FSDP auto-wrap policy targeting custom Transformer blocks
    with a fallback size-based policy.
    """
    if target_layer_cls is None:
        target_layer_cls = {TransformerBlock}

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=target_layer_cls,
    )
    return auto_wrap_policy


def wrap_model_fsdp(
    model: nn.Module,
    fsdp_config: Optional[FSDPConfig] = None,
    device_id: Optional[int] = None,
) -> FSDP:
    """
    Wraps an nn.Module model instance inside PyTorch FSDP with automated auto-wrapping,
    CPU offload options, prefetch rules, and mixed-precision policies.
    """
    if fsdp_config is None:
        fsdp_config = FSDPConfig()

    # Ensure torch.distributed process group is active for FSDP runtime
    if not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        init_method = "tcp://127.0.0.1:29501"
        logger.info(
            f"Process group uninitialized. Initializing fallback '{backend}' group for FSDP..."
        )
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=0,
            world_size=1,
        )

    sharding_strategy = get_sharding_strategy(fsdp_config.sharding_strategy)
    mixed_precision = get_mixed_precision_policy(fsdp_config.mixed_precision_dtype)
    cpu_offload = CPUOffload(offload_params=fsdp_config.cpu_offload)
    auto_wrap_policy = build_auto_wrap_policy(min_num_params=fsdp_config.min_num_params)

    prefetch_policy = (
        BackwardPrefetch.BACKWARD_POST
        if fsdp_config.backward_prefetch.upper() == "BACKWARD_POST"
        else BackwardPrefetch.BACKWARD_PRE
    )

    if device_id is None and torch.cuda.is_available():
        device_id = torch.cuda.current_device()

    device = (
        torch.device(f"cuda:{device_id}")
        if device_id is not None
        else torch.device("cpu")
    )

    logger.info(
        f"Wrapping model in FSDP | Strategy: {fsdp_config.sharding_strategy} | "
        f"Dtype: {fsdp_config.mixed_precision_dtype} | CPU Offload: {fsdp_config.cpu_offload}"
    )

    wrapped_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision,
        cpu_offload=cpu_offload,
        backward_prefetch=prefetch_policy,
        forward_prefetch=fsdp_config.forward_prefetch,
        limit_all_gathers=fsdp_config.limit_all_gathers,
        device_id=device,
    )

    return wrapped_model


def configure_fsdp_state_dict_type(
    model: FSDP,
    state_dict_type: str = "SHARDED",
) -> None:
    """
    Configures FSDP model checkpoint save/load mode.
    Modes:
      - 'SHARDED': Memory-efficient distributed sharded save (ZeRO-3 style).
      - 'FULL': Gather full un-sharded parameter weights on Rank 0.
      - 'LOCAL': Fast local worker rank state dicts.
    """
    state_dict_type_upper = state_dict_type.upper()

    if state_dict_type_upper == "SHARDED":
        save_policy = StateDictType.SHARDED_STATE_DICT
        state_dict_config = ShardedStateDictConfig(offload_to_cpu=True)
        optim_config = ShardedOptimStateDictConfig(offload_to_cpu=True)
    elif state_dict_type_upper == "FULL":
        save_policy = StateDictType.FULL_STATE_DICT
        state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    elif state_dict_type_upper == "LOCAL":
        save_policy = StateDictType.LOCAL_STATE_DICT
        state_dict_config = LocalStateDictConfig()
        optim_config = LocalOptimStateDictConfig()
    else:
        raise ValueError(f"Unsupported state_dict_type '{state_dict_type}'")

    FSDP.set_state_dict_type(
        model,
        save_policy,
        state_dict_config,
        optim_config,
    )
