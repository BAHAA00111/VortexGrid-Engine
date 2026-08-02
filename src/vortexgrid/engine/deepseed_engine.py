from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from vortexgrid import logger


@dataclass(frozen=True)
class DistributedConfig:
    """Immutable runtime configuration container for distributed process groups."""

    backend: str = field(
        default_factory=lambda: os.getenv("VORTEXGRID_DIST_BACKEND", "nccl").lower()
    )
    init_method: str = field(default_factory=lambda: os.getenv("MASTER_ADDR_PORT_INIT", "env://"))
    timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("VORTEXGRID_DIST_TIMEOUT", "1800"))
    )
    find_unused_parameters: bool = field(
        default_factory=lambda: (
            os.getenv("VORTEXGRID_FIND_UNUSED_PARAMS", "0").lower() in ("1", "true", "yes")
        )
    )
    nccl_buffsize: int = field(default_factory=lambda: int(os.getenv("NCCL_BUFFSIZE", "4194304")))


class DistributedContext:
    """
    Production process group lifecycle manager. Handles NCCL initializations, rank binding,
    barrier synchronizations, and graceful teardown across multi-node/multi-GPU execution.
    """

    _instance: Optional[DistributedContext] = None
    _initialized: bool = False

    def __init__(self, config: Optional[DistributedConfig] = None) -> None:
        self.config: DistributedConfig = config or DistributedConfig()
        self.rank: int = 0
        self.local_rank: int = 0
        self.world_size: int = 1
        self.local_world_size: int = 1
        self.node_id: int = 0
        self.is_master: bool = True
        self.device: torch.device = torch.device("cpu")
        self.process_group: Optional[dist.ProcessGroup] = None

    @classmethod
    def get_instance(cls) -> DistributedContext:
        """Returns the active global DistributedContext singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self) -> DistributedContext:
        """
        Discovers environment variables (Torchrun / Slurm / Ray), initializes the PyTorch
        distributed process group, binds the active CUDA device, and runs a diagnostic barrier.
        """
        if DistributedContext._initialized:
            logger.warning("DistributedContext is already initialized. Skipping re-initialization.")
            return self

        # ----------------------------------------------------------------------
        # 1. Environment Topology Discovery
        # ----------------------------------------------------------------------
        self.rank = int(os.getenv("RANK", os.getenv("SLURM_PROCID", "0")))
        self.local_rank = int(os.getenv("LOCAL_RANK", os.getenv("SLURM_LOCALID", "0")))
        self.world_size = int(os.getenv("WORLD_SIZE", os.getenv("SLURM_NTASKS", "1")))
        self.local_world_size = int(
            os.getenv("LOCAL_WORLD_SIZE", os.getenv("SLURM_NTASKS_PER_NODE", "1"))
        )
        self.node_id = self.rank // max(1, self.local_world_size)
        self.is_master = self.rank == 0

        # Set master network defaults if omitted
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            logger.info("MASTER_ADDR not set. Defaulted to 127.0.0.1")

        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
            logger.info("MASTER_PORT not set. Defaulted to 29500")

        # ----------------------------------------------------------------------
        # 2. Single-GPU vs Multi-GPU Isolation Setup
        # ----------------------------------------------------------------------
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            if self.local_rank >= available_gpus:
                raise RuntimeError(
                    f"LOCAL_RANK ({self.local_rank}) exceeds available CUDA GPUs ({available_gpus})."
                )

            # Explicitly lock this process to its isolated local CUDA index
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")

            # Flush existing allocator caching
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu")
            logger.warning("CUDA runtime is unavailable. DistributedContext falling back to CPU.")

        # ----------------------------------------------------------------------
        # 3. PyTorch Process Group Construction
        # ----------------------------------------------------------------------
        if self.world_size > 1:
            if not dist.is_initialized():
                logger.info(
                    f"Initializing Process Group [Rank {self.rank}/{self.world_size - 1}] "
                    f"[Local Rank {self.local_rank}] on {socket.gethostname()} -> {self.device}"
                )

                # Optimizations for NCCL networking
                if self.config.backend == "nccl":
                    os.environ["NCCL_BUFFSIZE"] = str(self.config.nccl_buffsize)

                dist.init_process_group(
                    backend=self.config.backend,
                    init_method=self.config.init_method,
                    world_size=self.world_size,
                    rank=self.rank,
                    timeout=timedelta(seconds=self.config.timeout_seconds),
                )
                self.process_group = dist.group.WORLD

            # Sanity barrier test
            self.barrier()

        DistributedContext._initialized = True
        DistributedContext._instance = self

        if self.is_master:
            logger.info(
                f"Distributed Context Online | Backend: {self.config.backend.upper()} | "
                f"World Size: {self.world_size} | Local World Size: {self.local_world_size}"
            )

        return self

    def barrier(self) -> None:
        """Blocks execution until all ranks in the world process group enter this point."""
        if dist.is_initialized() and self.world_size > 1:
            if self.device.type == "cuda":
                dist.barrier(device_ids=[self.local_rank])
            else:
                dist.barrier()

    def destroy(self) -> None:
        """Cleans up and gracefully tears down active distributed process groups."""
        if dist.is_initialized():
            if self.is_master:
                logger.info("Destroying PyTorch distributed process groups...")
            self.barrier()
            dist.destroy_process_group()
            DistributedContext._initialized = False
            DistributedContext._instance = None

    def get_summary(self) -> Dict[str, Any]:
        """Provides runtime telemetry metadata for logging or dashboard monitoring."""
        return {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_id": self.node_id,
            "is_master": self.is_master,
            "device": str(self.device),
            "backend": self.config.backend,
            "cuda_available": torch.cuda.is_available(),
            "allocated_vram_mb": (
                torch.cuda.memory_allocated(self.local_rank) / (1024**2)
                if torch.cuda.is_available()
                else 0.0
            ),
        }


def get_context() -> DistributedContext:
    """Helper to fetch or instantiate the default global context."""
    ctx = DistributedContext.get_instance()
    if not ctx._initialized:
        ctx.initialize()
    return ctx


def is_master() -> bool:
    """Returns True if current execution context is Rank 0 / Master Process."""
    return get_context().is_master


def get_rank() -> int:
    """Returns the global rank index of current process."""
    return get_context().rank


def get_world_size() -> int:
    """Returns total world size across all nodes."""
    return get_context().world_size
