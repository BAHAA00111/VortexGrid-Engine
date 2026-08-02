"""
PyTorch TensorBoard Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Exposes local file event writers for TensorBoard loss, perplexity, and gradient logging.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist

from vortexgrid import logger
from vortexgrid.tracker.base_tracker import BaseTracker

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class TensorBoardTracker(BaseTracker):
    """Local PyTorch TensorBoard SummaryWriter logger."""

    def __init__(
        self,
        log_dir: Union[str, Path] = "runs/tensorboard",
        project_name: str = "vortexgrid-llm",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled and HAS_TENSORBOARD
        self.is_rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        self.writer: Optional[SummaryWriter] = None

        if not HAS_TENSORBOARD and enabled:
            logger.warning(
                "torch.utils.tensorboard module not found. TensorBoard tracker disabled."
            )

        if self.enabled and self.is_rank_zero:
            log_path = Path(log_dir) / project_name / (run_name or "default_run")
            log_path.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_path))
            logger.info(f"Initialized TensorBoard Tracker at path '{log_path}'")

    def init_run(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """No-op for TensorBoard as initialization occurs in constructor."""
        pass

    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: int,
    ) -> None:
        """Writes scalar metrics and computes perplexity dynamically."""
        if not self.enabled or self.writer is None:
            return

        for tag, val in metrics.items():
            self.writer.add_scalar(tag, float(val), global_step=step)

        # Compute dynamic perplexities
        if "train/loss" in metrics and "train/perplexity" not in metrics:
            loss_val = float(metrics["train/loss"])
            try:
                ppl = math.exp(min(loss_val, 100))
            except (OverflowError, ValueError):
                ppl = float("inf")
            self.writer.add_scalar("train/perplexity", ppl, global_step=step)

    def log_gradient_norms(
        self,
        model: torch.nn.Module,
        step: int,
        norm_type: float = 2.0,
    ) -> Dict[str, float]:
        """Calculates and writes gradient norms to TensorBoard."""
        if not self.enabled or self.writer is None:
            return {}

        grad_norms: Dict[str, float] = {}
        total_sq_norm = 0.0

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(norm_type).item()
                self.writer.add_scalar(
                    f"gradients/layer_norm/{name}", param_norm, global_step=step
                )
                total_sq_norm += param_norm**norm_type

        total_grad_norm = total_sq_norm ** (1.0 / norm_type)
        self.writer.add_scalar(
            "gradients/total_norm", total_grad_norm, global_step=step
        )
        grad_norms["gradients/total_norm"] = round(total_grad_norm, 4)

        return grad_norms

    def close(self) -> None:
        """Flushes and closes TensorBoard event writer."""
        if self.enabled and self.writer is not None:
            self.writer.flush()
            self.writer.close()
            logger.info("Closed TensorBoard tracker writer.")
