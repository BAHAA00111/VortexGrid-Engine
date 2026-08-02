"""
VortexGrid :: Fault Recovery Integration Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Validates state reconstruction and mathematical determinism across
simulated process node failures and checkpoint recovery.
"""

from __future__ import annotations

import os
import socket
import tempfile
from typing import TypedDict

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.optim import AdamW


class TrainingResult(TypedDict):
    losses: torch.Tensor
    state_dict: dict[str, torch.Tensor]


class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim: int = 128, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


def find_free_port() -> int:
    """Find an available free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def setup_dist(rank: int, world_size: int, port: int):
    """Initialize process group using a dynamic free port."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


def run_continuous_training(
    rank: int,
    world_size: int,
    total_steps: int,
    data_inputs: list[torch.Tensor],
    temp_dir: str,
    port: int,
) -> None:
    """Runs uninterrupted training for `total_steps` and dumps result to disk."""
    setup_dist(rank, world_size, port)
    torch.manual_seed(42 + rank)

    model = SimpleTransformerBlock()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    losses = []
    for step in range(total_steps):
        optimizer.zero_grad()
        x = data_inputs[step]
        out = model(x)
        loss = out.pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if rank == 0:
        res: TrainingResult = {
            "losses": torch.tensor(losses),
            "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
        }
        torch.save(res, os.path.join(temp_dir, "continuous_res.pt"))

    cleanup_dist()


def run_interrupted_and_resumed_training(
    rank: int,
    world_size: int,
    interrupt_step: int,
    total_steps: int,
    data_inputs: list[torch.Tensor],
    ckpt_path: str,
    temp_dir: str,
    port1: int,
    port2: int,
) -> None:
    """Runs training up to `interrupt_step`, saves checkpoint, terminates, restarts, and finishes."""
    # Phase 1: Train up to interrupt_step
    setup_dist(rank, world_size, port1)
    torch.manual_seed(42 + rank)

    model = SimpleTransformerBlock()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    phase1_losses = []
    for step in range(interrupt_step):
        optimizer.zero_grad()
        x = data_inputs[step]
        out = model(x)
        loss = out.pow(2).mean()
        loss.backward()
        optimizer.step()
        phase1_losses.append(loss.item())

    # Save Checkpoint at interrupt_step
    if rank == 0:
        checkpoint = {
            "step": interrupt_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        torch.save(checkpoint, ckpt_path)

    cleanup_dist()

    # Phase 2: Simulate Failure & Resume from Checkpoint using a new fresh port
    setup_dist(rank, world_size, port2)
    model_resumed = SimpleTransformerBlock()
    optimizer_resumed = AdamW(model_resumed.parameters(), lr=1e-3, weight_decay=0.01)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_resumed.load_state_dict(checkpoint["model_state"])
    optimizer_resumed.load_state_dict(checkpoint["optimizer_state"])
    start_step = checkpoint["step"]

    phase2_losses = []
    for step in range(start_step, total_steps):
        optimizer_resumed.zero_grad()
        x = data_inputs[step]
        out = model_resumed(x)
        loss = out.pow(2).mean()
        loss.backward()
        optimizer_resumed.step()
        phase2_losses.append(loss.item())

    if rank == 0:
        all_losses = phase1_losses + phase2_losses
        res: TrainingResult = {
            "losses": torch.tensor(all_losses),
            "state_dict": {
                k: v.cpu().clone() for k, v in model_resumed.state_dict().items()
            },
        }
        torch.save(res, os.path.join(temp_dir, "resumed_res.pt"))

    cleanup_dist()


@pytest.mark.integration
def test_checkpoint_fault_recovery_equivalence():
    """Verify uninterrupted training vs interrupt-and-resumed training produces identical outputs."""
    world_size = 2
    total_steps = 20
    interrupt_step = 10

    # Generate deterministic input sequence
    torch.manual_seed(1337)
    data_inputs = [torch.randn(2, 16, 128) for _ in range(total_steps)]

    with tempfile.TemporaryDirectory() as temp_dir:
        ckpt_path = os.path.join(temp_dir, "fault_recovery.pt")

        # 1. Continuous Run (Port allocation 1)
        port_continuous = find_free_port()
        mp.spawn(
            run_continuous_training,
            args=(world_size, total_steps, data_inputs, temp_dir, port_continuous),
            nprocs=world_size,
            join=True,
        )

        # 2. Resumed Run (Port allocations 2 & 3 to prevent socket lingering locks)
        port_resumed_p1 = find_free_port()
        port_resumed_p2 = find_free_port()
        mp.spawn(
            run_interrupted_and_resumed_training,
            args=(
                world_size,
                interrupt_step,
                total_steps,
                data_inputs,
                ckpt_path,
                temp_dir,
                port_resumed_p1,
                port_resumed_p2,
            ),
            nprocs=world_size,
            join=True,
        )

        # 3. Load results & Assert Equivalence
        continuous_res: TrainingResult = torch.load(
            os.path.join(temp_dir, "continuous_res.pt")
        )
        resumed_res: TrainingResult = torch.load(
            os.path.join(temp_dir, "resumed_res.pt")
        )

        # Loss trajectory check
        torch.testing.assert_close(
            continuous_res["losses"], resumed_res["losses"], rtol=1e-5, atol=1e-5
        )

        # Final parameters check
        for param_key in continuous_res["state_dict"]:
            torch.testing.assert_close(
                continuous_res["state_dict"][param_key],
                resumed_res["state_dict"][param_key],
                rtol=1e-5,
                atol=1e-5,
            )
