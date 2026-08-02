"""
Weights & Biases (WandB) Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Captures deep learning science metrics, loss/perplexity curves, and gradient dynamics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist

from vortexgrid import logger
from vortexgrid.tracker.base_tracker import BaseTracker

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class WandbTracker(BaseTracker):
    """Weights & Biases tracking engine with distributed multi-rank safety."""

    def __init__(
        self,
        project_name: str = "vortexgrid-llm",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        rank_zero_only: bool = True,
    ) -> None:
        self.enabled = enabled and HAS_WANDB
        self.rank_zero_only = rank_zero_only
        self.is_rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        self.run = None

        if not HAS_WANDB and enabled:
            logger.warning(
                "wandb module not found in python environment. WandB tracker disabled."
            )

        if self.enabled and (self.is_rank_zero or not self.rank_zero_only):
            self.init_run(project_name=project_name, run_name=run_name, config=config)

    def init_run(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Starts WandB run session on rank 0."""
        if not self.enabled:
            return

        try:
            self.run = wandb.init(
                project=project_name,
                name=run_name,
                config=config,
                reinit=True,
            )
            logger.info(
                f"Initialized WandB Tracker run: '{run_name}' in project '{project_name}'"
            )
        except Exception as e:
            logger.error(f"Failed to initialize WandB run: {str(e)}")
            self.enabled = False

    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: int,
    ) -> None:
        """Logs metrics, auto-computing perplexity if training loss is present."""
        if not self.enabled or self.run is None:
            return

        payload = dict(metrics)

        # Compute Perplexity = exp(loss) if loss is supplied and perplexity is missing
        if "train/loss" in payload and "train/perplexity" not in payload:
            loss_val = payload["train/loss"]
            try:
                payload["train/perplexity"] = math.exp(min(loss_val, 100))
            except (OverflowError, ValueError):
                payload["train/perplexity"] = float("inf")

        if "val/loss" in payload and "val/perplexity" not in payload:
            loss_val = payload["val/loss"]
            try:
                payload["val/perplexity"] = math.exp(min(loss_val, 100))
            except (OverflowError, ValueError):
                payload["val/perplexity"] = float("inf")

        wandb.log(payload, step=step)

    def log_gradient_norms(
        self,
        model: torch.nn.Module,
        step: int,
        norm_type: float = 2.0,
    ) -> Dict[str, float]:
        """Calculates global and per-layer parameter gradient norms."""
        if not self.enabled or self.run is None:
            return {}

        grad_norms: Dict[str, float] = {}
        total_sq_norm = 0.0

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(norm_type).item()
                grad_norms[f"gradients/layer_norm/{name}"] = round(param_norm, 4)
                total_sq_norm += param_norm**norm_type

        total_grad_norm = total_sq_norm ** (1.0 / norm_type)
        grad_norms["gradients/total_norm"] = round(total_grad_norm, 4)

        wandb.log(grad_norms, step=step)
        return grad_norms

    def close(self) -> None:
        """Closes WandB run."""
        if self.enabled and self.run is not None:
            wandb.finish()
            logger.info("Closed WandB tracker run.")
