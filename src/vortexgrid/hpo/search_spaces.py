"""
Search Space Definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Defines hyperparameter search spaces and dynamic trial sampling rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import optuna


@dataclass
class HyperparameterSearchSpace:
    """Configurable search space definition for Optuna trials."""

    lr_min: float = 1e-5
    lr_max: float = 1e-2
    lr_log: bool = True
    warmup_steps_min: int = 50
    warmup_steps_max: int = 1000
    warmup_steps_step: int = 50
    weight_decay_min: float = 0.001
    weight_decay_max: float = 0.2
    batch_size_choices: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])

    def sample_trial_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Samples hyperparameter configuration from Optuna trial suggested ranges."""
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                self.lr_min,
                self.lr_max,
                log=self.lr_log,
            ),
            "warmup_steps": trial.suggest_int(
                "warmup_steps",
                self.warmup_steps_min,
                self.warmup_steps_max,
                step=self.warmup_steps_step,
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay",
                self.weight_decay_min,
                self.weight_decay_max,
                log=True,
            ),
            "micro_batch_size": trial.suggest_categorical(
                "micro_batch_size",
                self.batch_size_choices,
            ),
        }
