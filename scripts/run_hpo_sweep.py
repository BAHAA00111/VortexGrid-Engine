"""
Distributed Optuna Hyperparameter Optimization Sweep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Executes automated HPO trials based on configs/hpo/optuna_config.yaml specs.
"""

from __future__ import annotations

import argparse
import logging
import math
from typing import Any, Dict

import optuna
import torch
from torch.optim.lr_scheduler import LambdaLR
import yaml

from vortexgrid.models import ModelConfig, TransformerBlock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Linear warmup followed by Cosine Annealing learning rate schedule."""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def objective(trial: optuna.Trial, search_space: Dict[str, Any], max_eval_steps: int = 100) -> float:
    """Optuna objective evaluation trial function with full LR scheduler and loss tracking."""
    # 1. Sample Hyperparameters
    lr = trial.suggest_float(
        "learning_rate",
        float(search_space["learning_rate"]["low"]),
        float(search_space["learning_rate"]["high"]),
        log=bool(search_space["learning_rate"].get("log", True)),
    )
    weight_decay = trial.suggest_float(
        "weight_decay",
        float(search_space["weight_decay"]["low"]),
        float(search_space["weight_decay"]["high"]),
        log=bool(search_space["weight_decay"].get("log", True)),
    )
    warmup_steps = trial.suggest_int(
        "warmup_steps",
        int(search_space["warmup_steps"]["low"]),
        int(search_space["warmup_steps"]["high"]),
        step=int(search_space["warmup_steps"].get("step", 5)),
    )
    beta2 = trial.suggest_float(
        "beta2",
        float(search_space.get("beta2", {}).get("low", 0.95)),
        float(search_space.get("beta2", {}).get("high", 0.999)),
    )
    adam_eps = trial.suggest_float(
        "adam_eps",
        float(search_space.get("adam_eps", {}).get("low", 1e-8)),
        float(search_space.get("adam_eps", {}).get("high", 1e-5)),
        log=bool(search_space.get("adam_eps", {}).get("log", True)),
    )

    # 2. Build Model & Setup Device
    cfg = ModelConfig(
        vocab_size=32000,
        dim=512,            # Reduced model size for ultra-fast HPO evaluation
        n_layers=4,
        n_heads=8,
        n_kv_heads=8,
        max_seq_len=256,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerBlock(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        betas=(0.9, beta2), 
        eps=adam_eps, 
        weight_decay=weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_eval_steps)

    head_dim = cfg.dim // cfg.n_heads
    cos = torch.ones((256, head_dim), device=device)
    sin = torch.zeros((256, head_dim), device=device)

    # 3. Execution Loop over Horizon
    running_loss = 0.0
    for step in range(1, max_eval_steps + 1):
        optimizer.zero_grad()
        inputs = torch.randn(2, 256, cfg.dim, device=device, requires_grad=True)
        targets = torch.randn(2, 256, cfg.dim, device=device)

        out = model(inputs, cos, sin)
        loss = torch.nn.functional.mse_loss(out, targets)
        
        # Check for numeric divergence (NaN / Inf)
        loss_val = float(loss.item())
        if math.isnan(loss_val) or math.isinf(loss_val):
            logger.warning(f"Trial {trial.number} diverged at step {step} with loss={loss_val}")
            raise optuna.TrialPruned()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss_val

        # Report to Optuna Pruner
        trial.report(loss_val, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return running_loss / max_eval_steps


def run_hpo(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    assert isinstance(raw_config, dict)
    study_cfg: Dict[str, Any] = raw_config.get("optuna_study", {})
    search_space: Dict[str, Any] = raw_config.get("search_space", {})

    pruner_cfg = raw_config.get("pruner", {})
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(pruner_cfg.get("n_startup_trials", 5)),
        n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 25)),
        interval_steps=int(pruner_cfg.get("interval_steps", 5)),
    )

    study = optuna.create_study(
        study_name=study_cfg.get("study_name", "vortexgrid_hpo"),
        direction=study_cfg.get("direction", "minimize"),
        storage=study_cfg.get("storage_url", "sqlite:///vortexgrid_optuna.db"),
        pruner=pruner,
        load_if_exists=True,
    )

    execution_cfg: Dict[str, Any] = raw_config.get("sweep_execution", {})
    n_trials = int(execution_cfg.get("n_trials", 50))

    logger.info(f"Starting Optuna HPO sweep: {study.study_name} with {n_trials} trials...")
    study.optimize(lambda trial: objective(trial, search_space, max_eval_steps=100), n_trials=n_trials)

    logger.info("--- HPO Sweep Finished ---")
    logger.info(f"Best Trial Parameters: {study.best_params}")
    logger.info(f"Best Trial Value: {study.best_value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VortexGrid Optuna HPO Sweep")
    parser.add_argument("--config", type=str, default="configs/hpo/optuna_config.yaml")
    args = parser.parse_args()

    run_hpo(args.config)