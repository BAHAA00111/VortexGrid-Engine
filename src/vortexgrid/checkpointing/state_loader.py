"""
State Recovery Loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Engineers rank-agnostic, elastic state restoration using PyTorch Distributed
Checkpoint (dcp). Automatically handles dynamic world-size scaling and re-sharding
of model and optimizer states during cluster fault recovery or elastic restarts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.optim import Optimizer

from vortexgrid import logger
from vortexgrid.checkpointing.async_sharded_saver import CheckpointMetadata


class StateLoader:
    """
    Elastic, rank-agnostic checkpoint loader for restoring sharded models
    and optimizers across arbitrary GPU world-size topologies.
    """

    def __init__(self, process_group: Optional[dist.ProcessGroup] = None) -> None:
        self.process_group = process_group

    def inspect_checkpoint(self, checkpoint_dir: Union[str, Path]) -> CheckpointMetadata:
        """Inspects checkpoint metadata without loading heavy tensor parameters."""
        ckpt_path = Path(checkpoint_dir)
        meta_file = ckpt_path / "metadata.json"

        if not meta_file.exists():
            raise FileNotFoundError(
                f"Checkpoint metadata file not found at '{meta_file}'. Ensure directory is a valid DCP checkpoint."
            )

        with open(meta_file, "r", encoding="utf-8") as f:
            meta_json = f.read()

        return CheckpointMetadata.from_json(meta_json)

    def load_checkpoint(
        self,
        checkpoint_dir: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Loads and re-shards model and optimizer state_dicts from disk.
        
        Supports elastic world-size restarts (e.g., trained on N GPUs, restored on M GPUs).
        """
        ckpt_path = Path(checkpoint_dir)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint directory '{ckpt_path}' does not exist.")

        metadata = self.inspect_checkpoint(ckpt_path)
        current_world_size = (
            dist.get_world_size(self.process_group) if dist.is_initialized() else 1
        )

        if metadata.world_size != current_world_size:
            logger.warning(
                f"Elastic World-Size Scale Detected | Saved World Size: {metadata.world_size} -> "
                f"Current World Size: {current_world_size}. Re-sharding parameters..."
            )

        options = StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
            strict=strict,
        )

        # 1. Prepare destination target state_dicts matching current model/optimizer layout
        model_state_dict = get_model_state_dict(model, options=options)
        optim_state_dict = (
            get_optimizer_state_dict(model, optimizer, options=options)
            if optimizer is not None
            else {}
        )

        state_dict: Dict[str, Any] = {
            "model": model_state_dict,
            "optimizer": optim_state_dict,
        }

        # 2. Execute rank-agnostic parallel DCP load and re-sharding
        logger.info(f"Loading DCP state dict from '{ckpt_path}'...")
        storage_reader = dcp.FileSystemReader(ckpt_path)
        dcp.load(
            state_dict=state_dict,
            storage_reader=storage_reader,
            process_group=self.process_group,
        )

        # 3. Apply re-sharded weights back into model parameters
        set_model_state_dict(
            model=model,
            model_state_dict=state_dict["model"],
            options=options,
        )

        # 4. Apply optimizer states if provided
        if optimizer is not None and "optimizer" in state_dict and state_dict["optimizer"]:
            set_optimizer_state_dict(
                model=model,
                optimizers=optimizer,
                optim_state_dict=state_dict["optimizer"],
                options=options,
            )

        logger.info(
            f"Successfully restored state from step {metadata.step} "
            f"(Saved world size: {metadata.world_size} | Restored world size: {current_world_size})"
        )

        return metadata.extra_meta