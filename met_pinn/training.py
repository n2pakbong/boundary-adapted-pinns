"""A small, restartable strong-form runner for the OU boundary-factor benchmark."""

from __future__ import annotations

import json
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpointing import atomic_torch_save, restore_rng_state, rng_state, write_json
from .models import BoundaryFactorPINN, MLP
from .problems import OUDiskProblem


@dataclass(frozen=True)
class OUBoundaryFactorConfig:
    seed: int
    device: str
    dtype: str
    problem: dict[str, float]
    model: dict[str, Any]
    training: dict[str, Any]

    @classmethod
    def from_file(cls, path: Path) -> "OUBoundaryFactorConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def _dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError("dtype must be 'float64' or 'float32'")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(config: OUBoundaryFactorConfig, problem: OUDiskProblem) -> BoundaryFactorPINN:
    base = MLP(
        input_dim=2,
        width=int(config.model["width"]),
        depth=int(config.model["depth"]),
        activation=str(config.model.get("activation", "tanh")),
    )
    for module in base.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain=float(config.model.get("gain", 1.0)))
            torch.nn.init.zeros_(module.bias)
    return BoundaryFactorPINN(
        base,
        problem.signed_distance_factor,
        exponent=float(config.model.get("boundary_exponent", 1.0)),
    )


def _checkpoint_payload(
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_loss: float,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_loss": best_loss,
        "rng_state": rng_state(),
    }


def run_ou_boundary_factor(
    config: OUBoundaryFactorConfig,
    run_dir: Path,
    *,
    resume: bool = False,
) -> None:
    """Run or resume an OU strong-form experiment.

    `last.pt` is a complete restart checkpoint and is written every configured
    number of epochs. `best.pt`, `config.json`, `metadata.json`, and
    `metrics.jsonl` make a completed Colab run self-describing.
    """

    requested_device = torch.device(config.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The configuration requests CUDA but no CUDA device is available.")
    dtype = _dtype(config.dtype)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if not resume:
        _seed_everything(config.seed)
        write_json(config_path, {
            "seed": config.seed,
            "device": config.device,
            "dtype": config.dtype,
            "problem": config.problem,
            "model": config.model,
            "training": config.training,
        })
        write_json(run_dir / "metadata.json", {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(requested_device),
        })
    elif not config_path.exists():
        raise FileNotFoundError(f"Cannot resume: {config_path} is missing.")

    problem = OUDiskProblem(**config.problem)
    model = _build_model(config, problem).to(device=requested_device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.training["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=float(config.training.get("lr_gamma", 1.0)),
    )

    start_epoch, best_loss = 1, float("inf")
    last_path = run_dir / "last.pt"
    if resume:
        if not last_path.exists():
            raise FileNotFoundError(f"Cannot resume: {last_path} is missing.")
        checkpoint = torch.load(last_path, map_location=requested_device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_loss"])

    epochs = int(config.training["epochs"])
    collocation_points = int(config.training["collocation_points"])
    checkpoint_every = int(config.training.get("checkpoint_every", 100))
    log_every = int(config.training.get("log_every", 25))
    metrics_path = run_dir / "metrics.jsonl"
    metrics_mode = "a" if resume else "w"
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            x = problem.sample_interior(
                collocation_points, device=requested_device, dtype=dtype
            )
            residual = problem.residual(model, x)
            loss = residual.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_value = float(loss.detach().cpu())
            record = {
                "epoch": epoch,
                "pde_loss": loss_value,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_file.flush()

            if loss_value < best_loss:
                best_loss = loss_value
                atomic_torch_save(
                    _checkpoint_payload(
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        best_loss=best_loss,
                    ),
                    run_dir / "best.pt",
                )
            if epoch % checkpoint_every == 0 or epoch == epochs:
                atomic_torch_save(
                    _checkpoint_payload(
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        best_loss=best_loss,
                    ),
                    last_path,
                )
