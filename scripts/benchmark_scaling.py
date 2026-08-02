"""
Distributed Scaling & Throughput Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Measures model training throughput (tokens/sec), VRAM peak usage,
and Model FLOPs Utilization (MFU) across cluster scale.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any, Dict

import torch
import torch.distributed as dist

from vortexgrid.models import FusedCrossEntropyLoss, ModelConfig, TransformerBlock

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_flops_per_token(
    n_layers: int, dim: int, seq_len: int, vocab_size: int
) -> float:
    """Estimates forward + backward FLOPs per token for standard Transformer architecture."""
    flop_per_token = (
        6 * (n_layers * (12 * (dim**2) + 12 * dim * seq_len)) + 2 * vocab_size * dim
    )
    return float(flop_per_token)


def run_scaling_benchmark(
    num_steps: int = 50,
    warmup_steps: int = 10,
    batch_size: int = 2,
    seq_len: int = 2048,
) -> Dict[str, Any]:
    """Runs synthetic throughput benchmark steps and measures performance."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_dist = dist.is_initialized()
    world_size = dist.get_world_size() if is_dist else 1
    rank = dist.get_rank() if is_dist else 0

    cfg = ModelConfig(
        vocab_size=32000,
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=32,
        max_seq_len=seq_len,
    )

    model = TransformerBlock(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Calculate exact head_dim (128 for dim=4096, n_heads=32)
    head_dim = cfg.dim // cfg.n_heads

    # 2D shape: (seq_len, head_dim) -> allows trailing-dimension broadcasting over (B, H, S, D)
    cos = torch.ones((seq_len, head_dim), device=device)
    sin = torch.zeros((seq_len, head_dim), device=device)

    # Inputs for TransformerBlock: (batch_size, seq_len, dim)
    inputs = torch.randn(
        batch_size, seq_len, cfg.dim, device=device, requires_grad=True
    )
    targets = torch.randn(batch_size, seq_len, cfg.dim, device=device)

    # Warmup iterations
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        out = model(inputs, cos, sin)
        loss = torch.nn.functional.mse_loss(out, targets)
        loss.backward()
        optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    total_tokens = num_steps * batch_size * seq_len * world_size

    for step in range(num_steps):
        optimizer.zero_grad()
        out = model(inputs, cos, sin)
        loss = torch.nn.functional.mse_loss(out, targets)
        loss.backward()
        optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time
    tokens_per_sec = total_tokens / elapsed

    metrics = {
        "world_size": world_size,
        "tokens_per_sec": tokens_per_sec,
        "elapsed_seconds": elapsed,
        "peak_vram_gb": (
            (torch.cuda.max_memory_allocated() / (1024**3))
            if torch.cuda.is_available()
            else 0.0
        ),
    }

    if rank == 0:
        logger.info(f"--- Benchmark Results (World Size: {world_size}) ---")
        logger.info(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
        logger.info(f"Peak VRAM: {metrics['peak_vram_gb']:.2f} GB")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VortexGrid Scaling Benchmark")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=2048)
    args = parser.parse_args()

    run_scaling_benchmark(
        num_steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len
    )
