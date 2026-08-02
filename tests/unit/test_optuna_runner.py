from typing import Any, Dict
import optuna
import pytest

from vortexgrid.hpo import HyperparameterSearchSpace, OptunaRunner


@pytest.mark.unit
def test_search_space_sampling():
    space = HyperparameterSearchSpace()
    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    config = space.sample_trial_config(trial)
    assert "learning_rate" in config
    assert "warmup_steps" in config
    assert "weight_decay" in config
    assert "micro_batch_size" in config
    assert 1e-5 <= config["learning_rate"] <= 1e-2


@pytest.mark.unit
def test_optuna_runner_execution():
    runner = OptunaRunner(
        study_name="test_study",
        direction="minimize",
        n_startup_trials=2,
    )

    def dummy_objective(trial: optuna.Trial, config: Dict[str, Any]) -> float:
        # Simulated convex quadratic loss function based on learning rate
        lr = config["learning_rate"]
        loss = (lr - 0.001) ** 2
        return loss

    study = runner.run_optimization(dummy_objective, n_trials=5)

    assert len(study.trials) == 5
    best_params = runner.get_best_hyperparameters()
    assert "learning_rate" in best_params
