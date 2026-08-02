"""
Elastic Fault Tolerance & Resilience Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Provides failure hooks, CUDA OOM recovery handlers, heartbeat background monitoring,
and automated distributed process group reconstruction for zero-downtime training.
"""

from __future__ import annotations

import gc
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer

from vortexgrid import logger
from vortexgrid.checkpointing.async_sharded_saver import AsyncShardedSaver
from vortexgrid.checkpointing.state_loader import StateLoader


@dataclass
class FaultToleranceConfig:
    """Configuration container for distributed resilience and recovery."""

    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    heartbeat_interval_seconds: float = 10.0
    auto_recover_oom: bool = True
    clear_cuda_cache_on_oom: bool = True
    recovery_checkpoint_dir: str = "checkpoints"


class HeartbeatMonitor:
    """Monitors distributed process group health via periodic barrier keep-alives."""

    def __init__(
        self,
        interval_seconds: float = 10.0,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.process_group = process_group
        self.last_heartbeat = time.time()

    def ping(self) -> bool:
        """Executes a non-blocking or swift barrier ping across process ranks."""
        if not dist.is_initialized():
            return True

        current_time = time.time()
        if current_time - self.last_heartbeat >= self.interval_seconds:
            try:
                # Issue a fast barrier check to ensure no rank has silently hung or dropped
                dist.barrier(group=self.process_group)
                self.last_heartbeat = current_time
                return True
            except Exception as e:
                logger.error(
                    f"Heartbeat failure detected on rank {dist.get_rank()}: {str(e)}"
                )
                return False
        return True


class ElasticFaultHandler:
    """
    Elastic fault recovery manager. Wraps step execution to intercept CUDA OOM
    and network connection failures, re-initializes process groups, and restores model state.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        config: Optional[FaultToleranceConfig] = None,
        saver: Optional[AsyncShardedSaver] = None,
        loader: Optional[StateLoader] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config or FaultToleranceConfig()
        self.saver = saver or AsyncShardedSaver(
            base_dir=self.config.recovery_checkpoint_dir
        )
        self.loader = loader or StateLoader()
        self.heartbeat = HeartbeatMonitor(
            interval_seconds=self.config.heartbeat_interval_seconds
        )
        self.retry_count = 0

    def handle_cuda_oom(self) -> None:
        """Cleans up CUDA allocator state and garbage collects memory upon OOM."""
        logger.warning("CUDA Out-Of-Memory (OOM) detected! Commencing memory purge...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            logger.info("CUDA cache cleared and memory stats reset successfully.")

    def reconstruct_process_group(self) -> bool:
        """Safely tears down and reconstructs PyTorch distributed process group upon network failure."""
        logger.warning("Reconstructing PyTorch Distributed Process Group...")
        try:
            if dist.is_initialized():
                dist.destroy_process_group()

            # Attempt process group re-initialization using existing environment settings
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            logger.info("Successfully reconstructed distributed process group.")
            return True
        except Exception as e:
            logger.critical(
                f"Failed to reconstruct distributed process group: {str(e)}"
            )
            return False

    def recover_from_latest_checkpoint(
        self,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Scans for and restores from the latest available DCP checkpoint."""
        target_dir = checkpoint_dir or self.config.recovery_checkpoint_dir
        path = os.path.abspath(target_dir)

        if not os.path.exists(path):
            logger.error(
                f"Recovery failed: Checkpoint directory '{path}' does not exist."
            )
            return {}

        subdirs = [
            os.path.join(path, d)
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d)) and d.startswith("checkpoint_step_")
        ]

        if not subdirs:
            logger.warning(
                f"No valid step checkpoints found in '{path}' for auto-recovery."
            )
            return {}

        # Sort subdirectories by step number
        subdirs.sort(key=lambda x: int(x.split("_step_")[-1]))
        latest_ckpt = subdirs[-1]

        logger.info(f"Auto-recovering state from latest checkpoint: '{latest_ckpt}'...")
        return self.loader.load_checkpoint(
            checkpoint_dir=latest_ckpt,
            model=self.model,
            optimizer=self.optimizer,
        )

    def execute_with_fault_tolerance(
        self,
        step_fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[bool, Any]:
        """
        Executes a training step function within an elastic fault-tolerant wrapper.

        Automatically catches Torch CUDA OOM exceptions and PyTorch Distributed RuntimeErrors.
        """
        while self.retry_count <= self.config.max_retries:
            try:
                # 1. Heartbeat health ping
                if not self.heartbeat.ping():
                    raise RuntimeError("Distributed rank heartbeat failure detected.")

                # 2. Execute inner step
                output = step_fn(*args, **kwargs)

                # Reset consecutive retry count upon successful step execution
                self.retry_count = 0
                return True, output

            except torch.cuda.OutOfMemoryError as oom_err:
                logger.error(f"Step failed with CUDA OOM: {str(oom_err)}")
                if not self.config.auto_recover_oom:
                    raise oom_err

                self.handle_cuda_oom()
                self.retry_count += 1
                time.sleep(self.config.retry_delay_seconds)

            except (RuntimeError, dist.DistError) as dist_err:
                err_msg = str(dist_err).lower()
                is_dist_failure = any(
                    k in err_msg
                    for k in [
                        "nccl",
                        "gloo",
                        "connection",
                        "socket",
                        "heartbeat",
                        "work",
                    ]
                )

                if is_dist_failure:
                    logger.error(
                        f"Distributed network failure encountered: {str(dist_err)}"
                    )
                    self.retry_count += 1

                    if self.retry_count > self.config.max_retries:
                        logger.critical(
                            "Exceeded maximum fault recovery retries. Aborting execution."
                        )
                        raise dist_err

                    # Reconstruct communications & load checkpoint state
                    if self.reconstruct_process_group():
                        self.recover_from_latest_checkpoint()

                    time.sleep(self.config.retry_delay_seconds)
                else:
                    # Unhandled general RuntimeError
                    raise dist_err

            except Exception as unhandled_e:
                logger.critical(
                    f"Unhandled non-recoverable error in step execution: {str(unhandled_e)}"
                )
                raise unhandled_e

        return False, None
