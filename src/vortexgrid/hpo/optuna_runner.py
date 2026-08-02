"""
Distributed Optuna Study Runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Executes parallel Bayesian optimization sweeps over hyperparameter search spaces
with automated trial pruning and trial result persistence.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from vortexgrid import logger
from vortexgrid.hpo.search_spaces import HyperparameterSearchSpace


class OptunaRunner:
    """
    Distributed HPO trial orchestrator and Optuna Study lifecycle controller.
    """

    def __init__(
        self,
        study_name: str = "vortexgrid_hpo_sweep",
        storage_url: Optional[str] = None,
        direction: str = "minimize",
        n_startup_trials: int = 5,
        pruning_warmup_steps: int = 10,
        search_space: Optional[HyperparameterSearchSpace] = None,
    ) -> None:
        self.study_name = study_name
        self.storage_url = storage_url
        self.direction = direction
        self.search_space = search_space or HyperparameterSearchSpace()

        # Define TPE Sampler and Early-Stopping Median Pruner
        self.sampler = TPESampler(n_startup_trials=n_startup_trials, seed=42)
        self.pruner = MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=pruning_warmup_steps,
            interval_steps=1,
        )

        # Create or load distributed Optuna Study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_url,
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=True,
        )

        logger.info(
            f"Initialized Optuna Study '{self.study_name}' "
            f"[direction={self.direction}, storage={self.storage_url or 'in-memory'}]"
        )

    def run_optimization(
        self,
        objective_fn: Callable[[optuna.Trial, Dict[str, Any]], float],
        n_trials: int = 20,
        timeout_seconds: Optional[float] = None,
    ) -> optuna.Study:
        """
        Executes optimization loop across suggested trials.
        """

        def wrapped_objective(trial: optuna.Trial) -> float:
            config = self.search_space.sample_trial_config(trial)
            logger.info(f"Starting HPO Trial #{trial.number} with parameters: {config}")

            try:
                score = objective_fn(trial, config)
                logger.info(
                    f"Completed HPO Trial #{trial.number} with final metric: {score:.6f}"
                )
                return score
            except optuna.TrialPruned as e:
                logger.info(f"Trial #{trial.number} was pruned by Optuna.")
                raise e
            except Exception as e:
                logger.error(f"Trial #{trial.number} failed with error: {str(e)}")
                raise e

        self.study.optimize(
            wrapped_objective,
            n_trials=n_trials,
            timeout=timeout_seconds,
            catch=(Exception,),
        )

        if len(self.study.trials) > 0:
            logger.info(
                f"Optuna Optimization Complete. Best Trial #{self.study.best_trial.number} "
                f"with score {self.study.best_value:.6f}"
            )

        return self.study

    def get_best_hyperparameters(self) -> Dict[str, Any]:
        """Returns optimal hyperparameter parameters dictionary from finished study."""
        if len(self.study.trials) == 0:
            return {}
        return dict(self.study.best_params)
