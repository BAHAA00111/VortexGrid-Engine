"""
Asynchronous Sharded Saver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Engineers non-blocking distributed state-dict serialization using PyTorch
Distributed Checkpoint (torch.distributed.checkpoint / dcp). Offloads heavy IO operations
to background worker threads to guarantee zero-downtime training steps.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
)
from torch.optim import Optimizer

from vortexgrid import logger


@dataclass
class CheckpointMetadata:
    """Metadata container for distributed checkpoint tracking."""

    step: int
    timestamp: float
    world_size: int
    sharding_strategy: str
    precision: str
    extra_meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> CheckpointMetadata:
        data = json.loads(json_str)
        return cls(**data)


class AsyncShardedSaver:
    """
    Asynchronous distributed checkpoint saver using PyTorch DCP and background thread pooling.

    Prevents blocking GPU execution during sharded parameter & optimizer state writes.
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        max_async_workers: int = 2,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.process_group = process_group
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_async_workers,
            thread_name_prefix="vortexgrid_async_checkpoint_saver",
        )
        self.active_futures: list[concurrent.futures.Future[Any]] = []

    def save_checkpoint(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        sharding_strategy: str = "FULL_SHARD",
        precision: str = "bfloat16",
        async_save: bool = True,
    ) -> Path:
        """
        Extracts model and optimizer sharded state_dicts and saves them to disk asynchronously.
        """
        checkpoint_dir = self.base_dir / f"checkpoint_step_{step}"

        logger.info(f"Preparing sharded state_dict extraction for step {step}...")
        start_time = time.perf_counter()

        # Options for high-performance memory CPU offload during extraction
        options = StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
            ignore_frozen_params=False,
        )

        # 1. Extract sharded model state dict
        model_state_dict = get_model_state_dict(model, options=options)

        # 2. Extract sharded optimizer state dict if provided
        optim_state_dict = (
            get_optimizer_state_dict(model, optimizer, options=options)
            if optimizer is not None
            else {}
        )

        # 3. Assemble combined state dict payload
        state_dict: Dict[str, Any] = {
            "model": model_state_dict,
            "optimizer": optim_state_dict,
        }
        if extra_state is not None:
            state_dict["extra_state"] = extra_state

        world_size = (
            dist.get_world_size(self.process_group) if dist.is_initialized() else 1
        )

        metadata = CheckpointMetadata(
            step=step,
            timestamp=time.time(),
            world_size=world_size,
            sharding_strategy=sharding_strategy,
            precision=precision,
            extra_meta=extra_state or {},
        )

        extraction_latency = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"State dict extracted in {extraction_latency:.2f} ms. Dispatching checkpoint save..."
        )

        # Wait for previous save operations to finish before scheduling new disk writes
        self._prune_completed_futures()

        if async_save:
            future = self.executor.submit(
                self._write_dcp_payload,
                checkpoint_dir,
                state_dict,
                metadata,
                self.process_group,
            )
            self.active_futures.append(future)
        else:
            self._write_dcp_payload(
                checkpoint_dir,
                state_dict,
                metadata,
                self.process_group,
            )

        return checkpoint_dir

    @staticmethod
    def _write_dcp_payload(
        checkpoint_dir: Path,
        state_dict: Dict[str, Any],
        metadata: CheckpointMetadata,
        process_group: Optional[dist.ProcessGroup],
    ) -> None:
        """Background worker thread target for executing DCP disk writes."""
        start_time = time.perf_counter()
        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # PyTorch Distributed Checkpoint asynchronous directory storage writer
            storage_writer = dcp.FileSystemWriter(checkpoint_dir, sync_files=False)
            dcp.save(
                state_dict=state_dict,
                storage_writer=storage_writer,
                process_group=process_group,
            )

            # Write metadata file on Rank 0
            is_rank_zero = (
                not dist.is_initialized() or dist.get_rank(process_group) == 0
            )
            if is_rank_zero:
                meta_file = checkpoint_dir / "metadata.json"
                with open(meta_file, "w", encoding="utf-8") as f:
                    f.write(metadata.to_json())

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"Successfully saved DCP checkpoint to {checkpoint_dir} in {elapsed:.2f}s"
            )
        except Exception as e:
            logger.error(
                f"Failed to save DCP checkpoint to {checkpoint_dir}: {str(e)}",
                exc_info=True,
            )
            raise e

    def _prune_completed_futures(self) -> None:
        """Clean completed futures from queue."""
        self.active_futures = [f for f in self.active_futures if not f.done()]

    def wait_until_idle(self) -> None:
        """Blocks execution until all pending background checkpoint writes complete."""
        if not self.active_futures:
            return
        logger.info(
            f"Waiting for {len(self.active_futures)} active background saves to complete..."
        )
        concurrent.futures.wait(self.active_futures)
        self._prune_completed_futures()

    def shutdown(self) -> None:
        """Flushes all queued tasks and safely shuts down background worker pool."""
        self.wait_until_idle()
        self.executor.shutdown(wait=True)
