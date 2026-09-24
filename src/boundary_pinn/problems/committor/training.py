"""Training runs for the periodic two-well committor example."""

from __future__ import annotations

import json
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...checkpointing import atomic_torch_save, restore_rng_state, rng_state, write_json
from ...models import LiftedBoundaryFactorPINN, MLP, PeriodicMLP
from .periodic_two_well import PeriodicTwoWellCommittor

@dataclass(frozen=True)
class PeriodicCommittorConfig:
    """Configuration for a periodic committor comparison run."""

    seed: int
    device: str
    dtype: str
    method: str
    problem: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]

    @classmethod
    def from_file(cls, path: Path) -> "PeriodicCommittorConfig":
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


def _cpu_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy parameters for an evaluation checkpoint without GPU storage."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _initialize_mlp(model_config: dict[str, Any], *, input_dim: int = 2) -> MLP:
    base = MLP(
        input_dim=input_dim,
        width=int(model_config["width"]),
        depth=int(model_config["depth"]),
        activation=str(model_config.get("activation", "tanh")),
    )
    for module in base.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain=float(model_config.get("gain", 1.0)))
            torch.nn.init.zeros_(module.bias)
    return base


def _build_periodic_committor_model(
    config: PeriodicCommittorConfig, problem: PeriodicTwoWellCommittor
) -> torch.nn.Module:
    base = PeriodicMLP(_initialize_mlp(config.model, input_dim=4))
    if config.method == "boundary_adapted":
        return LiftedBoundaryFactorPINN(base, problem.lifting, problem.boundary_adapted_factor)
    if config.method == "naive_product":
        return LiftedBoundaryFactorPINN(base, problem.lifting, problem.naive_product_factor)
    if config.method == "penalty":
        return base
    raise ValueError("method must be 'boundary_adapted', 'naive_product', or 'penalty'")


def run_periodic_committor(
    config: PeriodicCommittorConfig,
    run_dir: Path,
    *,
    resume: bool = False,
) -> None:
    """Run or resume one periodic committor method comparison.

    Exact-boundary methods minimize only the interior strong-form residual.
    The ``penalty`` baseline uses the same interior loss plus sampled boundary
    penalties.  Every run writes a complete restart checkpoint independently
    of whether it improves the objective.
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
            "method": config.method,
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

    problem = PeriodicTwoWellCommittor(**config.problem)
    model = _build_periodic_committor_model(config, problem).to(
        device=requested_device, dtype=dtype
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.training["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(config.training.get("lr_gamma", 1.0))
    )

    start_epoch, best_loss, best_epoch = 1, float("inf"), 0
    best_model_state: dict[str, torch.Tensor] | None = None
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
        best_checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        best_model_state = best_checkpoint["model_state"]
        best_epoch = int(best_checkpoint["epoch"])

    epochs = int(config.training["epochs"])
    collocation_points = int(config.training["collocation_points"])
    boundary_points = int(config.training.get("boundary_points", collocation_points // 4))
    boundary_weight = float(config.training.get("boundary_weight", 100.0))
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
            pde_loss = residual.square().mean()
            boundary_loss = torch.zeros((), device=requested_device, dtype=dtype)
            if config.method == "penalty":
                source_boundary = problem.sample_boundary(
                    "source", boundary_points, device=requested_device, dtype=dtype
                )
                target_boundary = problem.sample_boundary(
                    "target", boundary_points, device=requested_device, dtype=dtype
                )
                boundary_loss = (
                    model(source_boundary).square().mean()
                    + (model(target_boundary) - 1.0).square().mean()
                )
            loss = pde_loss + boundary_weight * boundary_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_value = float(loss.detach().cpu())
            record = {
                "epoch": epoch,
                "loss": loss_value,
                "pde_loss": float(pde_loss.detach().cpu()),
                "boundary_loss": float(boundary_loss.detach().cpu()),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_file.flush()
                # Stream the same fields saved in metrics.jsonl for live run progress (every log_every epochs).
                print(
                    f"{config.method} | epoch {epoch}/{epochs} "
                    f"objective={loss_value:.6g} | PDE loss={record['pde_loss']:.6g} | "
                    f"boundary loss={record['boundary_loss']:.6g}",
                    flush=True,
                )

            if loss_value < best_loss:
                best_loss = loss_value
                best_epoch = epoch
                best_model_state = _cpu_model_state(model)
            checkpoint_due = epoch == 1 or epoch % checkpoint_every == 0 or epoch == epochs
            if checkpoint_due:
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
                if best_model_state is not None:
                    atomic_torch_save(
                        {
                            "epoch": best_epoch,
                            "model_state": best_model_state,
                            "best_loss": best_loss,
                            "checkpoint_kind": "evaluation_best",
                        },
                        run_dir / "best.pt",
                    )
