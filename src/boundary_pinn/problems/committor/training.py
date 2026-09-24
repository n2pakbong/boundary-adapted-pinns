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
from torch import nn

from ...checkpointing import atomic_torch_save, restore_rng_state, rng_state, write_json
from ...models import LiftedBoundaryFactorPINN, MLP, PeriodicMLP
from .paper_methods import (
    CommittorMonteCarloData,
    estimate_domain_area,
    generate_committor_mc_data,
    nitsche_committor_losses,
)
from .periodic_two_well import PeriodicTwoWellCommittor

PAPER_METHODS = (
    "simple_pinn", "boundary_adapted", "over_enforced", "under_enforced",
    "variational_nitsche",
)
LEGACY_METHODS = ("naive_product", "penalty")

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
    loss_scales: nn.ParameterDict | None = None,
) -> dict[str, Any]:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_loss": best_loss,
        "rng_state": rng_state(),
    }
    if loss_scales is not None:
        payload["loss_scales_state"] = loss_scales.state_dict()
    return payload


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
    if config.method in ("simple_pinn", "penalty"):
        return base
    if config.method == "boundary_adapted":
        return LiftedBoundaryFactorPINN(base, problem.lifting, problem.boundary_adapted_factor)
    if config.method == "naive_product":
        return LiftedBoundaryFactorPINN(base, problem.lifting, problem.naive_product_factor)
    if config.method in ("over_enforced", "under_enforced"):
        exponent = 1.5 if config.method == "over_enforced" else 0.5

        def powered_factor(x: torch.Tensor) -> torch.Tensor:
            return problem.boundary_adapted_factor(x).clamp_min(0.0).pow(exponent)

        return LiftedBoundaryFactorPINN(base, problem.lifting, powered_factor)
    if config.method == "variational_nitsche":
        return base
    raise ValueError(f"unknown method {config.method!r}; supported methods: {(*PAPER_METHODS, *LEGACY_METHODS)}")


def _save_mc_data(data: CommittorMonteCarloData, destination: Path) -> None:
    np.savez_compressed(
        destination,
        anchors=data.anchors.cpu().numpy(),
        values=data.values.cpu().numpy(),
        standard_errors=data.standard_errors.cpu().numpy(),
        resolved_paths=data.resolved_paths.cpu().numpy(),
        total_paths=data.total_paths.cpu().numpy(),
    )
    write_json(destination.with_suffix(".json"), data.metadata)


def _load_or_generate_mc_data(
    config: PeriodicCommittorConfig,
    problem: PeriodicTwoWellCommittor,
    run_dir: Path,
    supplied: CommittorMonteCarloData | None,
) -> CommittorMonteCarloData | None:
    settings = config.training.get("monte_carlo")
    if settings is None or supplied is not None:
        return supplied
    path = run_dir / "monte_carlo_labels.npz"
    if path.is_file():
        with np.load(path) as saved:
            return CommittorMonteCarloData(
                anchors=torch.as_tensor(saved["anchors"], dtype=torch.float64),
                values=torch.as_tensor(saved["values"], dtype=torch.float64),
                standard_errors=torch.as_tensor(saved["standard_errors"], dtype=torch.float64),
                resolved_paths=torch.as_tensor(saved["resolved_paths"], dtype=torch.int64),
                total_paths=torch.as_tensor(saved["total_paths"], dtype=torch.int64),
                metadata=json.loads(path.with_suffix(".json").read_text(encoding="utf-8")),
            )
    data = generate_committor_mc_data(problem, **settings)
    _save_mc_data(data, path)
    return data


def run_periodic_committor(
    config: PeriodicCommittorConfig,
    run_dir: Path,
    *,
    resume: bool = False,
    mc_data: CommittorMonteCarloData | None = None,
) -> dict[str, Any]:
    """Run or resume one periodic committor method comparison.

    New configurations with a Monte Carlo section use the five named paper
    formulations, including pointwise committor labels.  Configurations from
    the earlier three-method exploratory run retain their original objectives
    when that section is absent, so existing checkpoints remain resumable.
    """

    requested_device = torch.device(config.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The configuration requests CUDA but no CUDA device is available.")
    dtype = _dtype(config.dtype)
    run_dir.mkdir(parents=True, exist_ok=True)
    problem = PeriodicTwoWellCommittor(**config.problem)
    mc_data = _load_or_generate_mc_data(config, problem, run_dir, mc_data)
    uses_paper_objective = config.training.get("monte_carlo") is not None
    if uses_paper_objective:
        if config.method not in PAPER_METHODS:
            raise ValueError("Monte Carlo objective requires one of the five paper methods")
        if mc_data is None:
            raise RuntimeError("Monte Carlo data could not be loaded or generated")
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
            "objective_version": "paper_methods_v1" if uses_paper_objective else "legacy_pilot_v1",
        })
    elif not config_path.exists():
        raise FileNotFoundError(f"Cannot resume: {config_path} is missing.")

    model = _build_periodic_committor_model(config, problem).to(
        device=requested_device, dtype=dtype
    )
    scale_names = {
        "simple_pinn": ("pde_loss", "data_loss", "boundary_loss"),
        "boundary_adapted": ("pde_loss", "data_loss"),
        "over_enforced": ("pde_loss", "data_loss"),
        "under_enforced": ("pde_loss", "data_loss"),
        "variational_nitsche": ("nitsche_weak_loss", "nitsche_penalty_loss", "data_loss"),
    }
    loss_scales = None
    if uses_paper_objective:
        loss_scales = nn.ParameterDict({
            name: nn.Parameter(torch.zeros((), device=requested_device, dtype=dtype))
            for name in scale_names[config.method]
        })
    parameters = list(model.parameters())
    if loss_scales is not None:
        parameters.extend(loss_scales.parameters())
    optimizer = torch.optim.Adam(parameters, lr=float(config.training["learning_rate"]))
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
        if loss_scales is not None and "loss_scales_state" in checkpoint:
            loss_scales.load_state_dict(checkpoint["loss_scales_state"])
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
    mc_x = mc_y = None
    if mc_data is not None:
        mc_x = mc_data.anchors.to(device=requested_device, dtype=dtype)
        mc_y = mc_data.values.to(device=requested_device, dtype=dtype)
    area = (
        estimate_domain_area(problem, device=requested_device, dtype=dtype)
        if uses_paper_objective and config.method == "variational_nitsche"
        else 0.0
    )
    final_record: dict[str, Any] | None = None
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            losses: dict[str, torch.Tensor] = {}
            diagnostics: dict[str, torch.Tensor] = {}
            if uses_paper_objective and config.method == "variational_nitsche":
                weak = config.training["nitsche"]
                volume_points = problem.sample_interior(
                    int(weak["volume_points"]), device=requested_device, dtype=dtype
                )
                test_centers = problem.sample_interior(
                    int(weak["test_functions"]), device=requested_device, dtype=dtype
                )
                _, weak_losses = nitsche_committor_losses(
                    model,
                    problem,
                    volume_points=volume_points,
                    test_centers=test_centers,
                    boundary_points=int(weak["boundary_points"]),
                    domain_area=area,
                    length_scale=float(weak["length_scale"]),
                    gamma=float(weak["gamma"]),
                    h=float(weak["h"]),
                )
                losses["nitsche_weak_loss"] = weak_losses["nitsche_weak_loss"]
                losses["nitsche_penalty_loss"] = weak_losses["nitsche_penalty_loss"]
                diagnostics.update({
                    name: value for name, value in weak_losses.items() if name.endswith("_mse")
                })
                losses["data_loss"] = (model(mc_x) - mc_y).square().mean()
            else:
                x = problem.sample_interior(
                    collocation_points, device=requested_device, dtype=dtype
                )
                residual = problem.residual(model, x)
                pde_loss = residual.square().mean()
                if uses_paper_objective:
                    losses["pde_loss"] = pde_loss
                    losses["data_loss"] = (model(mc_x) - mc_y).square().mean()
                    if config.method == "simple_pinn":
                        source_boundary = problem.sample_boundary(
                            "source", boundary_points, device=requested_device, dtype=dtype
                        )
                        target_boundary = problem.sample_boundary(
                            "target", boundary_points, device=requested_device, dtype=dtype
                        )
                        losses["boundary_loss"] = (
                            model(source_boundary).square().mean()
                            + (model(target_boundary) - 1.0).square().mean()
                        )
                else:
                    boundary_loss = torch.zeros((), device=requested_device, dtype=dtype)
                    if config.method in ("penalty", "simple_pinn"):
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
                    losses["pde_loss"] = pde_loss
                    losses["boundary_loss"] = boundary_loss

            if loss_scales is not None:
                weighted = {
                    f"weighted_{name}": torch.exp(-2.0 * loss_scales[name]) * value
                    for name, value in losses.items()
                }
                objective = sum(weighted.values()) + sum(loss_scales.values())
            else:
                objective = losses["pde_loss"] + boundary_weight * losses["boundary_loss"]
                weighted = {
                    "weighted_pde_loss": losses["pde_loss"],
                    "weighted_boundary_loss": boundary_weight * losses["boundary_loss"],
                }
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            scheduler.step()

            loss_value = float(objective.detach().cpu())
            record: dict[str, Any] = {
                "epoch": epoch,
                "loss": loss_value,
                "objective": loss_value,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{name: float(value.detach().cpu()) for name, value in losses.items()},
                **{name: float(value.detach().cpu()) for name, value in weighted.items()},
                **{name: float(value.detach().cpu()) for name, value in diagnostics.items()},
            }
            if loss_scales is not None:
                record["adaptive_loss_weights"] = {
                    name: float(torch.exp(-2.0 * value.detach()).cpu())
                    for name, value in loss_scales.items()
                }
            if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_file.flush()
                fields = " | ".join(
                    f"{name}={record[name]:.4g}" for name in record
                    if name == "objective" or name in losses or name in diagnostics
                )
                print(f"{config.method} | epoch {epoch}/{epochs} | {fields}", flush=True)

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
                        loss_scales=loss_scales,
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
            final_record = record

    if final_record is None:
        metric_lines = [line for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
        final_record = json.loads(metric_lines[-1]) if metric_lines else {}
    final_metrics = {
        "method": config.method,
        "final_epoch": final_record.get("epoch"),
        "best_epoch": best_epoch,
        "best_training_objective": best_loss,
        "final_training_objective": final_record.get("objective", final_record.get("loss")),
        "raw_loss_components": {
            key: value for key, value in final_record.items()
            if key.endswith("_loss") or key.endswith("_mse")
        },
        "weighted_loss_components": {
            key: value for key, value in final_record.items() if key.startswith("weighted_")
        },
        "adaptive_loss_weights": final_record.get("adaptive_loss_weights", {}),
        "learning_rate": final_record.get("learning_rate"),
        "monte_carlo_labels": mc_data.metadata if uses_paper_objective and mc_data is not None else None,
    }
    write_json(run_dir / "final_training_metrics.json", final_metrics)
    return final_metrics
