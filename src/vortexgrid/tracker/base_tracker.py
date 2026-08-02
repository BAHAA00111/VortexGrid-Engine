"""
Abstract Base Experiment Tracker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Defines standard interface for experiment tracking backends (WandB, TensorBoard, etc.).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import torch


class BaseTracker(ABC):
    """Abstract interface defining required metrics and gradient tracking methods."""

    @abstractmethod
    def init_run(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes tracking run session and registers global hyperparameter configuration."""
        pass

    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: int,
    ) -> None:
        """Logs scalar metrics dictionary (loss, perplexity, throughput, TFLOPS)."""
        pass

    @abstractmethod
    def log_gradient_norms(
        self,
        model: torch.nn.Module,
        step: int,
        norm_type: float = 2.0,
    ) -> Dict[str, float]:
        """Calculates and logs total gradient norm and per-layer gradient norms."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Flushes buffers and closes experiment tracking session cleanly."""
        pass