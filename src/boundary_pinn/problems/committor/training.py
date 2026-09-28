"""Training runs for the periodic two-well committor example."""

from __future__ import annotations

import hashlib
import inspect
import json
import types
import platform
import random
import time
from collections.abc import Callable
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
    sample_boundary_geometry,
)
from .periodic_two_well import (
    InteriorSampler,
    PeriodicTwoWellCommittor,
    sample_interior_points,
)

PAPER_METHODS = (
    "simple_pinn", "boundary_adapted", "over_enforced", "under_enforced",
    "variational_nitsche",
)
LEGACY_METHODS = ("naive_product", "penalty")

ModelFactory = Callable[[], torch.nn.Module]
OptimizerFactory = Callable[[list[torch.nn.Parameter]], torch.optim.Optimizer]
SchedulerFactory = Callable[[torch.optim.Optimizer], Any]


def _callable_name(function: Callable[..., Any] | None, default: str) -> str:
    if function is None:
        return default
    return f"{function.__module__}.{function.__qualname__}"

def _callable_fingerprint(callback: Callable[..., Any] | None) -> str | None:
    """Hash callback code and referenced scalar globals for reproducible resume checks."""
    if callback is None:
        return None
    if isinstance(callback, type):
        members = list(vars(callback).items())
    elif hasattr(callback, "__code__"):
        members = [("", callback)]
    else:
        members = [("__call__", getattr(type(callback), "__call__", callback))]

    code_objects = []
    dependencies: dict[str, str] = {}
    for name, member in members:
        function = getattr(member, "__func__", member)
        code = getattr(function, "__code__", None)
        if code is None:
            continue

        def describe(code_object: types.CodeType) -> dict[str, Any]:
            return {
                "bytecode": code_object.co_code.hex(),
                "names": code_object.co_names,
                "variables": code_object.co_varnames,
                "constants": [
                    describe(value) if isinstance(value, types.CodeType) else repr(value)
                    for value in code_object.co_consts
                ],
            }

        code_objects.append((name, describe(code)))
        namespace = getattr(function, "__globals__", {})
        for global_name in code.co_names:
            value = namespace.get(global_name)
            if isinstance(value, (str, int, float, bool, tuple, list, dict, type(None))):
                dependencies[f"{name}:{global_name}"] = repr(value)
        for cell_index, cell in enumerate(getattr(function, "__closure__", None) or ()):
            value = cell.cell_contents
            if isinstance(value, (str, int, float, bool, tuple, list, dict, type(None))):
                dependencies[f"{name}:closure:{cell_index}"] = repr(value)

    try:
        source = inspect.getsource(callback).encode("utf-8")
    except (OSError, TypeError):
        if not code_objects:
            return None
        source = json.dumps(code_objects, sort_keys=True).encode("utf-8")
    defaults = repr((
        getattr(callback, "__defaults__", None),
        getattr(callback, "__kwdefaults__", None),
    )).encode("utf-8")
    dependency_data = json.dumps(dependencies, sort_keys=True).encode("utf-8")
    return hashlib.sha256(source + defaults + dependency_data).hexdigest()


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
        activation_kwargs=model_config.get("activation_kwargs"),
    )
    initialization = str(model_config.get("initialization", "xavier_normal")).lower()
    gain = float(model_config.get("gain", 1.0))
    initializers = {
        "xavier_normal": torch.nn.init.xavier_normal_,
        "xavier_uniform": torch.nn.init.xavier_uniform_,
    }
    if initialization not in initializers:
        raise ValueError(
            f"Unknown initialization {initialization!r}; "
            f"choose one of: {', '.join(initializers)}."
        )
    for module in base.modules():
        if isinstance(module, torch.nn.Linear):
            initializers[initialization](module.weight, gain=gain)
            torch.nn.init.zeros_(module.bias)
    return base


class _ScalarOutput(nn.Module):
    """Adapt a one-value-per-row PyTorch model to the scalar PINN interface."""

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = self.network(x)
        if not isinstance(values, torch.Tensor):
            raise TypeError("the model must return a torch.Tensor")
        if values.numel() != x.shape[0]:
            raise ValueError(
                "the model must return exactly one scalar for each input row"
            )
        return values.reshape(x.shape[0])


def _build_periodic_committor_model(
    config: PeriodicCommittorConfig,
    problem: PeriodicTwoWellCommittor,
    model_factory: ModelFactory | None = None,
) -> torch.nn.Module:
    """Build a built-in periodic MLP or a user-defined PyTorch model."""
    if model_factory is None:
        architecture = str(config.model.get("architecture", "periodic_mlp")).lower()
        if architecture == "periodic_mlp":
            frequencies = (1.0,)
        elif architecture == "periodic_fourier_mlp":
            frequencies = tuple(
                float(value) for value in config.model.get("frequencies", (1, 2, 4))
            )
        else:
            raise ValueError(
                f"Unknown committor architecture {architecture!r}; choose "
                "'periodic_mlp' or 'periodic_fourier_mlp'."
            )
        base = PeriodicMLP(
            _initialize_mlp(config.model, input_dim=4 * len(frequencies)),
            frequencies=frequencies,
        )
    else:
        base = model_factory()
        if not isinstance(base, torch.nn.Module):
            raise TypeError("model_factory must return a torch.nn.Module")
        # Boundary wrappers expect one scalar per input point. Accept the
        # common [batch, 1] output shape from user-written modules as well.
        base = _ScalarOutput(base)

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


def _build_optimizer(
    parameters: list[torch.nn.Parameter], training: dict[str, Any]
) -> torch.optim.Optimizer:
    """Build an explicitly configured Adam-family optimizer."""
    specification = training.get("optimizer", {})
    if isinstance(specification, str):
        name, options = specification.lower(), {}
    else:
        options = dict(specification)
        name = str(options.pop("name", "adam")).lower()
    allowed = {"betas", "eps", "weight_decay", "amsgrad"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unsupported optimizer options: {sorted(unknown)}")
    if "betas" in options:
        options["betas"] = tuple(float(value) for value in options["betas"])
    options["eps"] = float(options.get("eps", 1.0e-8))
    options["weight_decay"] = float(options.get("weight_decay", 0.0))
    options["amsgrad"] = bool(options.get("amsgrad", False))
    learning_rate = float(training["learning_rate"])
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, **options)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, **options)
    raise ValueError("optimizer.name must be 'adam' or 'adamw'")


def _build_scheduler(
    optimizer: torch.optim.Optimizer, training: dict[str, Any]
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build the configured per-epoch learning-rate schedule."""
    specification = training.get("scheduler")
    if specification is None:
        name = "exponential"
        options: dict[str, Any] = {"gamma": float(training.get("lr_gamma", 1.0))}
    elif isinstance(specification, str):
        name, options = specification.lower(), {}
    else:
        options = dict(specification)
        name = str(options.pop("name", "exponential")).lower()
    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)
    if name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=float(options.get("gamma", training.get("lr_gamma", 1.0)))
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(options["step_size_epochs"]),
            gamma=float(options["gamma"]),
        )
    raise ValueError("scheduler.name must be 'constant', 'exponential', or 'step'")


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _boundary_mse(
    model: torch.nn.Module,
    problem: PeriodicTwoWellCommittor,
    which: str,
    target_value: float,
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Monte Carlo boundary MSE with arclength quadrature weights."""
    points, _, weights = sample_boundary_geometry(
        problem, which, n, device=device, dtype=dtype
    )
    squared_error = (model(points) - target_value).square()
    return (weights * squared_error).sum() / weights.sum()


def _truncate_metrics_to_epoch(metrics_path: Path, checkpoint_epoch: int) -> None:
    """Discard log rows newer than the checkpoint before continuing a run."""
    if not metrics_path.is_file():
        return
    kept: list[str] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(record.get("epoch", -1)) <= checkpoint_epoch:
            kept.append(line)
    metrics_path.write_text(
        "".join(f"{line}\n" for line in kept),
        encoding="utf-8",
    )


def run_periodic_committor(
    config: PeriodicCommittorConfig,
    run_dir: Path,
    *,
    resume: bool = False,
    mc_data: CommittorMonteCarloData | None = None,
    model_factory: ModelFactory | None = None,
    interior_sampler: InteriorSampler | None = None,
    optimizer_factory: OptimizerFactory | None = None,
    scheduler_factory: SchedulerFactory | None = None,
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
    uses_paper_objective = config.training.get("monte_carlo") is not None
    if uses_paper_objective and config.method not in PAPER_METHODS:
        raise ValueError("Monte Carlo objective requires one of the five paper methods")
    objective_version = (
        "paper_methods_v2_nitsche_split"
        if uses_paper_objective and config.method == "variational_nitsche"
        else "paper_methods_v1" if uses_paper_objective else "legacy_pilot_v1"
    )
    metadata_path = run_dir / "metadata.json"
    runtime_hooks = {
        "model": _callable_name(model_factory, "repository model builder"),
        "interior_sampler": _callable_name(
            interior_sampler, "uniform rejection on the punctured torus"
        ),
        "optimizer": _callable_name(optimizer_factory, "repository optimizer builder"),
        "scheduler": _callable_name(scheduler_factory, "repository scheduler builder"),
    }
    runtime_hook_fingerprints = {
        "model": _callable_fingerprint(model_factory),
        "interior_sampler": _callable_fingerprint(interior_sampler),
        "optimizer": _callable_fingerprint(optimizer_factory),
        "scheduler": _callable_fingerprint(scheduler_factory),
    }
    if resume and metadata_path.exists():
        prior_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        prior_version = prior_metadata.get("objective_version")
        if prior_version != objective_version:
            raise RuntimeError(
                "This checkpoint uses objective version "
                f"{prior_version!r}, but the selected method now uses {objective_version!r}. "
                "Start a new run folder; the old checkpoint was not resumed."
            )
        if prior_metadata.get("runtime_hooks", runtime_hooks) != runtime_hooks:
            raise RuntimeError(
                "The model, sampler, optimizer, or scheduler used for this resume "
                "does not match the saved run. Restore the original notebook code "
                "or start a new output folder."
            )
        prior_fingerprints = prior_metadata.get("runtime_hook_fingerprints")
        if (
            prior_fingerprints is not None
            and prior_fingerprints != runtime_hook_fingerprints
        ):
            raise RuntimeError(
                "The model, sampler, optimizer, or scheduler code changed since the "
                "checkpoint was saved. Restore the original code or start a new run folder."
            )
    mc_data = _load_or_generate_mc_data(config, problem, run_dir, mc_data)
    if uses_paper_objective and mc_data is None:
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
            "objective_version": objective_version,
            "runtime_hooks": runtime_hooks,
            "runtime_hook_fingerprints": runtime_hook_fingerprints,
        })
    elif not config_path.exists():
        raise FileNotFoundError(f"Cannot resume: {config_path} is missing.")

    model = _build_periodic_committor_model(config, problem, model_factory).to(
        device=requested_device, dtype=dtype
    )
    scale_names = {
        "simple_pinn": ("pde_loss", "data_loss", "boundary_loss"),
        "boundary_adapted": ("pde_loss", "data_loss"),
        "over_enforced": ("pde_loss", "data_loss"),
        "under_enforced": ("pde_loss", "data_loss"),
        "variational_nitsche": ("pde_loss", "boundary_loss", "data_loss"),
    }
    loss_scales = None
    if uses_paper_objective:
        initial_log_scale = float(
            config.training.get("adaptive_loss_log_scale_init", 0.0)
        )
        if not np.isfinite(initial_log_scale):
            raise ValueError("training.adaptive_loss_log_scale_init must be finite")
        loss_scales = nn.ParameterDict({
            name: nn.Parameter(torch.full(
                (), initial_log_scale, device=requested_device, dtype=dtype
            ))
            for name in scale_names[config.method]
        })
    parameters = list(model.parameters())
    if loss_scales is not None:
        parameters.extend(loss_scales.parameters())
    optimizer = (
        optimizer_factory(parameters)
        if optimizer_factory is not None
        else _build_optimizer(parameters, config.training)
    )
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer_factory must return a torch.optim.Optimizer")
    scheduler = (
        scheduler_factory(optimizer)
        if scheduler_factory is not None
        else _build_scheduler(optimizer, config.training)
    )
    if not all(hasattr(scheduler, name) for name in ("step", "state_dict", "load_state_dict")):
        raise TypeError(
            "scheduler_factory must return an object with step, state_dict, and load_state_dict"
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
    log_interval_seconds = float(config.training.get("log_interval_seconds", 30.0))
    if log_interval_seconds <= 0.0:
        raise ValueError("training.log_interval_seconds must be positive")
    # Read old configs for already-created runs; new presets use wall-clock reporting.
    legacy_log_every = (
        int(config.training["log_every"])
        if "log_every" in config.training and "log_interval_seconds" not in config.training
        else None
    )
    metrics_path = run_dir / "metrics.jsonl"
    if resume:
        _truncate_metrics_to_epoch(metrics_path, start_epoch - 1)
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
    training_started = time.monotonic()
    next_report_at = training_started
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            losses: dict[str, torch.Tensor] = {}
            diagnostics: dict[str, torch.Tensor] = {}
            if uses_paper_objective and config.method == "variational_nitsche":
                weak = config.training["nitsche"]
                volume_points = sample_interior_points(
                    problem,
                    int(weak["volume_points"]),
                    device=requested_device,
                    dtype=dtype,
                    sampler=interior_sampler,
                )
                test_centers = sample_interior_points(
                    problem,
                    int(weak["test_functions"]),
                    device=requested_device,
                    dtype=dtype,
                    sampler=interior_sampler,
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
                losses["pde_loss"] = weak_losses["pde_loss"]
                losses["boundary_loss"] = weak_losses["boundary_loss"]
                diagnostics.update({
                    name: value for name, value in weak_losses.items() if name.endswith("_mse")
                })
                losses["data_loss"] = (model(mc_x) - mc_y).square().mean()
            else:
                x = sample_interior_points(
                    problem,
                    collocation_points,
                    device=requested_device,
                    dtype=dtype,
                    sampler=interior_sampler,
                )
                residual = problem.residual(model, x)
                pde_loss = residual.square().mean()
                if uses_paper_objective:
                    losses["pde_loss"] = pde_loss
                    losses["data_loss"] = (model(mc_x) - mc_y).square().mean()
                    if config.method == "simple_pinn":
                        losses["boundary_loss"] = (
                            _boundary_mse(
                                model, problem, "source", 0.0, boundary_points,
                                device=requested_device, dtype=dtype,
                            )
                            + _boundary_mse(
                                model, problem, "target", 1.0, boundary_points,
                                device=requested_device, dtype=dtype,
                            )
                        )
                else:
                    boundary_loss = torch.zeros((), device=requested_device, dtype=dtype)
                    if config.method in ("penalty", "simple_pinn"):
                        boundary_loss = (
                            _boundary_mse(
                                model, problem, "source", 0.0, boundary_points,
                                device=requested_device, dtype=dtype,
                            )
                            + _boundary_mse(
                                model, problem, "target", 1.0, boundary_points,
                                device=requested_device, dtype=dtype,
                            )
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
            now = time.monotonic()
            elapsed = now - training_started
            completed_epochs = epoch - start_epoch + 1
            rate = completed_epochs / max(elapsed, 1.0e-12)
            remaining_seconds = max(0, epochs - epoch) / max(rate, 1.0e-12)
            record["session_elapsed_seconds"] = elapsed
            record["session_epochs_per_second"] = rate
            record["estimated_remaining_seconds"] = remaining_seconds

            if legacy_log_every is not None:
                report_due = (
                    epoch == start_epoch or epoch % legacy_log_every == 0 or epoch == epochs
                )
            else:
                report_due = epoch == start_epoch or epoch == epochs or now >= next_report_at

            # Keep every epoch for a full-resolution loss history. The time
            # interval controls console output only, not the saved metrics.
            metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
            if report_due:
                metrics_file.flush()
                fields = " | ".join(
                    f"{name}={record[name]:.4g}" for name in record
                    if name == "objective" or name in losses or name in diagnostics
                )
                print(
                    f"{config.method} | epoch {epoch:,}/{epochs:,} | "
                    f"elapsed {_format_duration(elapsed)} | "
                    f"ETA {_format_duration(remaining_seconds)} | {fields}",
                    flush=True,
                )
                next_report_at = now + log_interval_seconds

            if loss_value < best_loss:
                best_loss = loss_value
                best_epoch = epoch
                best_model_state = _cpu_model_state(model)
            checkpoint_due = epoch == 1 or epoch % checkpoint_every == 0 or epoch == epochs
            if checkpoint_due:
                # Persist progress through each restartable checkpoint.
                metrics_file.flush()
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
