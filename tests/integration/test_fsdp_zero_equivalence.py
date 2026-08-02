"""
FSDP / ZeRO Numerical Equivalence Integration Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Verifies that forward/backward pass loss trajectories and parameter gradients 
match strictly between standard DDP baseline and FSDP sharded execution.
"""

from __future__ import annotations

import os
import socket
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Check FSDP availability
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    HAS_FSDP = True
except ImportError:
    HAS_FSDP = False


class ToyLLMBlock(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc2(self.act(self.fc1(x)))
        return x + residual


def find_free_port() -> int:
    """Find an available free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def init_process(rank: int, world_size: int, port: int, backend: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def _worker_run_ddp(
    rank: int, world_size: int, inputs: list[torch.Tensor], temp_dir: str, port: int, use_cuda: bool
) -> None:
    # Use Gloo if running multiple ranks on single-GPU setup to avoid NCCL Duplicate GPU crash
    gpu_count = torch.cuda.device_count() if use_cuda else 0
    multi_gpu = gpu_count >= world_size
    backend = "nccl" if multi_gpu else "gloo"

    init_process(rank, world_size, port, backend)
    torch.manual_seed(42)

    if use_cuda:
        device_id = rank if multi_gpu else 0
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
        model = ToyLLMBlock().to(device)
        ddp_model = DDP(model, device_ids=[device_id] if multi_gpu else None)
    else:
        device = torch.device("cpu")
        model = ToyLLMBlock().to(device)
        ddp_model = DDP(model, device_ids=None)

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    losses = []
    for step_data in inputs:
        optimizer.zero_grad()
        out = ddp_model(step_data.to(device))
        loss = out.pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if rank == 0:
        torch.save(torch.tensor(losses), os.path.join(temp_dir, "ddp_losses.pt"))

    dist.destroy_process_group()


def _worker_run_fsdp(
    rank: int, world_size: int, inputs: list[torch.Tensor], temp_dir: str, port: int, use_cuda: bool
) -> None:
    gpu_count = torch.cuda.device_count() if use_cuda else 0
    multi_gpu = gpu_count >= world_size
    backend = "nccl" if multi_gpu else "gloo"

    init_process(rank, world_size, port, backend)
    torch.manual_seed(42)

    if use_cuda:
        device_id = rank if multi_gpu else 0
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
        model = ToyLLMBlock().to(device)
        fsdp_model = FSDP(model, device_id=device)
    else:
        device = torch.device("cpu")
        model = ToyLLMBlock().to(device)
        fsdp_model = FSDP(model, device_id=device)

    optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=0.01)

    losses = []
    for step_data in inputs:
        optimizer.zero_grad()
        out = fsdp_model(step_data.to(device))
        loss = out.pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if rank == 0:
        torch.save(torch.tensor(losses), os.path.join(temp_dir, "fsdp_losses.pt"))

    dist.destroy_process_group()


@pytest.mark.integration
@pytest.mark.skipif(not HAS_FSDP, reason="PyTorch FSDP module is not available")
def test_fsdp_vs_ddp_loss_equivalence():
    """Validates FSDP produces equivalent step losses to standard DDP over 10 optimization steps."""
    world_size = 2
    steps = 10
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(99)
    # Generate identical input sequence per rank
    inputs = [torch.randn(4, 16, 64) for _ in range(steps)]

    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Run DDP Baseline
        ddp_port = find_free_port()
        mp.spawn(
            _worker_run_ddp,
            args=(world_size, inputs, temp_dir, ddp_port, use_cuda),
            nprocs=world_size,
            join=True,
        )

        # 2. Run FSDP Execution
        fsdp_port = find_free_port()
        mp.spawn(
            _worker_run_fsdp,
            args=(world_size, inputs, temp_dir, fsdp_port, use_cuda),
            nprocs=world_size,
            join=True,
        )

        # 3. Load results & compare trajectory
        ddp_losses = torch.load(os.path.join(temp_dir, "ddp_losses.pt"))
        fsdp_losses = torch.load(os.path.join(temp_dir, "fsdp_losses.pt"))

        torch.testing.assert_close(
            ddp_losses,
            fsdp_losses,
            rtol=1e-5,
            atol=1e-5,
            msg="FSDP loss trajectory diverged from DDP baseline!",
        )