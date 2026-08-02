"""
Distributed Training Launcher Entrypoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Handles multi-node / multi-GPU training setup using PyTorch FSDP / DeepSpeed wrappers.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import torch
import torch.distributed as dist
import yaml

from vortexgrid.models import ModelConfig, TransformerBlock


def setup_distributed() -> tuple[int, int, int]:
    """Initializes PyTorch Distributed process group from environment variables."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0

    if not dist.is_initialized() and world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    return rank, world_size, local_rank


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Distributed LLM Training")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model YAML config"
    )
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()

    # Load Model Configuration
    with open(args.config, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    assert isinstance(raw_config, dict), "Config file must be a YAML dictionary"
    model_cfg_dict: Dict[str, Any] = raw_config.get("model_config", {})

    n_heads = int(
        model_cfg_dict.get("n_heads", model_cfg_dict.get("num_attention_heads", 32))
    )
    n_kv_heads = int(model_cfg_dict.get("n_kv_heads", n_heads))

    cfg = ModelConfig(
        vocab_size=int(model_cfg_dict.get("vocab_size", 32000)),
        dim=int(model_cfg_dict.get("dim", model_cfg_dict.get("hidden_size", 4096))),
        n_layers=int(
            model_cfg_dict.get("n_layers", model_cfg_dict.get("num_hidden_layers", 32))
        ),
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=int(model_cfg_dict.get("max_seq_len", 2048)),
    )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = TransformerBlock(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    head_dim = cfg.dim // cfg.n_heads
    # 2D shape: (seq_len, head_dim)
    cos = torch.ones((512, head_dim), device=device)
    sin = torch.zeros((512, head_dim), device=device)

    if rank == 0:
        print(f"[VortexGrid] Training started on {world_size} rank(s)...")

    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad()
        inputs = torch.randn(2, 512, cfg.dim, device=device, requires_grad=True)
        targets = torch.randn(2, 512, cfg.dim, device=device)

        outputs = model(inputs, cos, sin)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        if rank == 0 and step % 10 == 0:
            print(f"Step {step}/{args.max_steps} | Loss: {loss.item():.4f}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
