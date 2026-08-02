"""
Megatron-Style Tensor Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Provides ColumnParallelLinear and RowParallelLinear layers with distributed autograd
communication primitives (All-Reduce, All-Gather, Reduce-Scatter) for intra-node
model parallel slicing.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from vortexgrid import logger


# -----------------------------------------------------------------------------
# Autograd Communication Primitives for Tensor Parallelism
# -----------------------------------------------------------------------------

class _CopyToTensorParallelRegion(torch.autograd.Function):
    """Passes tensor forward unchanged; All-Reduces gradients in backward pass."""

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
        setattr(ctx, "group", group)
        return input_

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        group: Optional[dist.ProcessGroup] = getattr(ctx, "group", None)
        if group is not None and dist.is_initialized() and dist.get_world_size(group) > 1:
            dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=group)
        return grad_output, None


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    """All-Reduces tensor forward; passes gradients unchanged in backward pass."""

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
        setattr(ctx, "group", group)
        if group is not None and dist.is_initialized() and dist.get_world_size(group) > 1:
            output = input_.clone()
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
            return output
        return input_

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output, None


class _ScatterToTensorParallelRegion(torch.autograd.Function):
    """Scatters tensor along last dimension forward; All-Gathers in backward pass."""

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
        setattr(ctx, "group", group)
        if group is None or not dist.is_initialized() or dist.get_world_size(group) <= 1:
            return input_

        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        last_dim = input_.dim() - 1
        dim_size = input_.size(last_dim) // world_size

        return input_.narrow(last_dim, rank * dim_size, dim_size).clone()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        group: Optional[dist.ProcessGroup] = getattr(ctx, "group", None)
        if group is None or not dist.is_initialized() or dist.get_world_size(group) <= 1:
            return grad_output, None

        world_size = dist.get_world_size(group)
        last_dim = grad_output.dim() - 1

        tensor_list = [torch.empty_like(grad_output) for _ in range(world_size)]
        dist.all_gather(tensor_list, grad_output, group=group)
        return torch.cat(tensor_list, dim=last_dim), None


class _GatherFromTensorParallelRegion(torch.autograd.Function):
    """All-Gathers tensor along last dimension forward; Scatters in backward pass."""

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, group: Optional[dist.ProcessGroup]) -> torch.Tensor:
        setattr(ctx, "group", group)
        if group is None or not dist.is_initialized() or dist.get_world_size(group) <= 1:
            return input_

        world_size = dist.get_world_size(group)
        last_dim = input_.dim() - 1
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        dist.all_gather(tensor_list, input_, group=group)
        return torch.cat(tensor_list, dim=last_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        group: Optional[dist.ProcessGroup] = getattr(ctx, "group", None)
        if group is None or not dist.is_initialized() or dist.get_world_size(group) <= 1:
            return grad_output, None

        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        last_dim = grad_output.dim() - 1
        dim_size = grad_output.size(last_dim) // world_size

        return grad_output.narrow(last_dim, rank * dim_size, dim_size).clone(), None

def copy_to_tensor_parallel_region(input_: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    return _CopyToTensorParallelRegion.apply(input_, group)  


def reduce_from_tensor_parallel_region(input_: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    return _ReduceFromTensorParallelRegion.apply(input_, group)  


def scatter_to_tensor_parallel_region(input_: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    return _ScatterToTensorParallelRegion.apply(input_, group)  


def gather_from_tensor_parallel_region(input_: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    return _GatherFromTensorParallelRegion.apply(input_, group)  


# -----------------------------------------------------------------------------
# Megatron Column and Row Parallel Linear Modules
# -----------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    """
    Linear layer sliced along output columns (Megatron-style).
    
    Splits out_features across tensor_parallel_world_size ranks.
    Input: [..., in_features]
    Output: [..., out_features / tp_world_size] (or [..., out_features] if gather_output=True)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        process_group: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.process_group = process_group

        world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
        if out_features % world_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by tensor_parallel_world_size ({world_size})"
            )

        self.output_size_per_partition = out_features // world_size

        self.weight = nn.Parameter(
            torch.empty((self.output_size_per_partition, in_features), device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.output_size_per_partition, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Duplicate input gradient flow to all TP ranks
        input_parallel = copy_to_tensor_parallel_region(x, self.process_group)

        # Local linear transformation
        output_parallel = F.linear(input_parallel, self.weight, self.bias)

        if self.gather_output:
            # All-Gather outputs across TP ranks along the last dimension
            output = gather_from_tensor_parallel_region(output_parallel, self.process_group)
        else:
            output = output_parallel

        return output


class RowParallelLinear(nn.Module):
    """
    Linear layer sliced along input rows (Megatron-style).
    
    Splits in_features across tensor_parallel_world_size ranks.
    Input: [..., in_features / tp_world_size] (or [..., in_features] if input_is_parallel=False)
    Output: [..., out_features]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        process_group: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.process_group = process_group

        world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
        if in_features % world_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by tensor_parallel_world_size ({world_size})"
            )

        self.input_size_per_partition = in_features // world_size

        self.weight = nn.Parameter(
            torch.empty((out_features, self.input_size_per_partition), device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            input_parallel = scatter_to_tensor_parallel_region(x, self.process_group)
        else:
            input_parallel = x

        # Local linear transformation without bias addition before all-reduce
        output_parallel = F.linear(input_parallel, self.weight, None)

        # All-Reduce partial outputs across TP ranks
        output_ = reduce_from_tensor_parallel_region(output_parallel, self.process_group)

        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_

        return output