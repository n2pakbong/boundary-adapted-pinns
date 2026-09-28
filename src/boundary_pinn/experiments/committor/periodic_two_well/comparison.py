"""One-command periodic committor comparison with durable artifacts and plots."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ....checkpointing import write_json
from ....models import _activation
from ....problems.committor import PeriodicTwoWellCommittor
from ....problems.committor.reference import (
    PeriodicCommittorReference,
    periodic_reference_errors,
    periodic_reference_grid_errors,
    solve_periodic_committor_fd,
)
from ....problems.committor.paper_methods import (
    CommittorMonteCarloData,
    generate_committor_mc_data,
    sample_boundary_geometry,
)
from ....problems.committor.periodic_two_well import InteriorSampler
from ....problems.committor.training import (
    _callable_fingerprint,
    PAPER_METHODS,
    PeriodicCommittorConfig,
    _build_periodic_committor_model,
    _dtype,
    ModelFactory,
    OptimizerFactory,
    run_periodic_committor,
    SchedulerFactory,
)


PREPARATION_SCHEMA_VERSION = 1


def _git_revision() -> str | None:
    """Return the checkout revision when this package is running from a Git clone."""
    repository_root = Path(__file__).resolve().parents[5]
    if not (repository_root / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _hook_name(function: Callable[..., Any] | None, default: str) -> str:
    if function is None:
        return default
    return f"{function.__module__}.{function.__qualname__}"


def _preparation_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """Settings that determine the saved reference and Monte Carlo labels."""
    return {
        "problem": config["problem"],
        "reference_grid_sizes": config["reference_grid_sizes"],
        "monte_carlo": config["training"].get("monte_carlo"),
    }


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _saved_final_training_metrics(run_directory: Path) -> dict[str, Any]:
    """Read final raw losses, distinguishing absent terms in older pilot runs."""
    path = run_directory / "final_training_metrics.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    records = _read_metrics(run_directory / "metrics.jsonl")
    if not records:
        return {"available": False}
    last = records[-1]
    components = {
        key: value for key, value in last.items()
        if key.endswith("_loss") or key.endswith("_mse")
    }
    return {
        "available": True,
        "final_epoch": last.get("epoch"),
        "final_training_objective": last.get("objective", last.get("loss")),
        "raw_loss_components": components,
        "data_loss": None,
        "data_term_used": False,
        "note": "This saved run predates Monte Carlo solution-value regularization.",
    }


def _evaluate(
    model: torch.nn.Module,
    reference: PeriodicCommittorReference,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> np.ndarray:
    """Evaluate a network on a reference grid without retaining derivatives."""

    points = reference.grid.reshape(-1, 2)
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            batch = torch.as_tensor(points[start : start + batch_size], device=device, dtype=dtype)
            values.append(model(batch).detach().cpu().numpy())
    return np.concatenate(values).reshape(reference.values.shape)


def _boundary_rmse(
    model: torch.nn.Module,
    problem: PeriodicTwoWellCommittor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    points_per_boundary: int = 512,
) -> dict[str, float]:
    """Evaluate Dirichlet errors on the two actual level-set boundaries."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    model.eval()
    with torch.no_grad():
        source, _, source_weights = sample_boundary_geometry(
            problem, "source", points_per_boundary, device=torch.device("cpu"),
            dtype=dtype, generator=generator,
        )
        target, _, target_weights = sample_boundary_geometry(
            problem, "target", points_per_boundary, device=torch.device("cpu"),
            dtype=dtype, generator=generator,
        )
        source = source.to(device)
        target = target.to(device)
        source_weights = source_weights.to(device)
        target_weights = target_weights.to(device)
        source_error = model(source)
        target_error = model(target) - 1.0
        return {
            "source_boundary_rmse": float(
                ((source_weights * source_error.square()).sum() / source_weights.sum()).sqrt().cpu()
            ),
            "target_boundary_rmse": float(
                ((target_weights * target_error.square()).sum() / target_weights.sum()).sqrt().cpu()
            ),
        }


def _plot_method(
    method: str,
    prediction: np.ndarray,
    reference: PeriodicCommittorReference,
    destination: Path,
) -> None:
    error = prediction - reference.values
    error_limit = max(float(np.max(np.abs(error))), 1.0e-12)
    panels = ((prediction, "PINN committor", "viridis", 0.0, 1.0),
              (reference.values, "finite-difference reference", "viridis", 0.0, 1.0),
              (error, "PINN minus reference", "coolwarm", -error_limit, error_limit))
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, (field, title, colourmap, low, high) in zip(axes, panels):
        image = axis.imshow(field.T, origin="lower", extent=(0, 1, 0, 1), cmap=colourmap, vmin=low, vmax=high)
        axis.contour(reference.source.T.astype(float), levels=[0.5], colors="white", linewidths=0.8,
                     origin="lower", extent=(0, 1, 0, 1))
        axis.contour(reference.target.T.astype(float), levels=[0.5], colors="black", linewidths=0.8,
                     origin="lower", extent=(0, 1, 0, 1))
        axis.set(title=title, xlabel="$x_1$", ylabel="$x_2$", aspect="equal")
        figure.colorbar(image, ax=axis, shrink=0.82)
    figure.suptitle(method.replace("_", " "))
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _plot_monte_carlo_labels(
    problem: PeriodicTwoWellCommittor,
    data: CommittorMonteCarloData,
    destination: Path,
) -> None:
    """Plot the sampled anchor locations and their noisy committor labels."""
    anchors = data.anchors.detach().cpu().numpy()
    values = data.values.detach().cpu().numpy()
    errors = data.standard_errors.detach().cpu().numpy()
    grid_axis = np.linspace(0.0, 1.0, 257)
    grid_x, grid_y = np.meshgrid(grid_axis, grid_axis, indexing="ij")
    grid_points = torch.as_tensor(
        np.stack((grid_x.ravel(), grid_y.ravel()), axis=-1), dtype=torch.float64
    )
    source_level = problem.defining_function(
        grid_points, problem.source_center
    ).detach().cpu().numpy().reshape(grid_x.shape)
    target_level = problem.defining_function(
        grid_points, problem.target_center
    ).detach().cpu().numpy().reshape(grid_x.shape)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    scatter = axes[0].scatter(
        anchors[:, 0], anchors[:, 1], c=values, cmap="viridis",
        vmin=0.0, vmax=1.0, s=55, edgecolors="black", linewidths=0.45,
    )
    axes[0].contour(
        grid_x, grid_y, source_level, levels=[0.0],
        colors=["tab:blue"], linewidths=1.4,
    )
    axes[0].contour(
        grid_x, grid_y, target_level, levels=[0.0],
        colors=["tab:red"], linewidths=1.4,
    )
    axes[0].set(
        title="Monte Carlo anchors colored by $\\widehat q_i$",
        xlabel="$x_1$", ylabel="$x_2$", xlim=(0, 1), ylim=(0, 1), aspect="equal",
    )
    figure.colorbar(scatter, ax=axes[0], label="estimated committor $\\widehat q_i$")
    anchor_index = np.arange(1, len(values) + 1)
    axes[1].errorbar(
        anchor_index, values, yerr=errors, fmt="o", markersize=3.5,
        capsize=2, linewidth=0.8, color="tab:purple",
    )
    axes[1].set(
        title="Label estimates and one standard error",
        xlabel="anchor index", ylabel="$\\widehat q_i$",
        xlim=(0, len(values) + 1), ylim=(-0.05, 1.05),
    )
    axes[1].grid(alpha=0.25)
    figure.suptitle(
        f"Feynman–Kac data before training: {len(values)} anchors, "
        f"{data.metadata.get('paths_per_anchor', '?')} paths per anchor"
    )
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _plot_losses(method_directories: dict[str, Path], destination: Path) -> None:
    records_by_method = {
        method: _read_metrics(directory / "metrics.jsonl")
        for method, directory in method_directories.items()
    }
    components = sorted({
        key for records in records_by_method.values() for row in records for key in row
        if (key.endswith("_loss") or key.endswith("_mse"))
        and key not in ("loss", "objective")
        and not key.startswith("weighted_")
    })
    panels = [("objective", "total training objective")]
    panels.extend((name, name.replace("_", " ")) for name in components)
    figure, axes = plt.subplots(len(panels), 1, figsize=(8, 3.0 * len(panels)),
                                squeeze=False, constrained_layout=True)
    for axis, (field, label) in zip(axes[:, 0], panels):
        plotted = False
        methods_without_finite_values = []
        finite_values_across_methods = []
        for method, records in records_by_method.items():
            available = [row for row in records if field in row]
            if available:
                epochs = [row["epoch"] for row in available]
                values = np.asarray([float(row[field]) for row in available], dtype=float)
                finite = np.isfinite(values)
                if not np.any(finite):
                    methods_without_finite_values.append(method.replace("_", " "))
                    continue
                # Keep gaps where a run recorded NaN/Inf, rather than connecting
                # across them. Plot on linear axes first so all methods share a
                # scale decision for this panel.
                plot_values = np.where(finite, values, np.nan)
                finite_values_across_methods.extend(values[finite].tolist())
                axis.plot(epochs, plot_values, label=method.replace("_", " "))
                plotted = True
        if methods_without_finite_values:
            axis.text(
                0.01, 0.02,
                "No finite values: " + ", ".join(methods_without_finite_values),
                transform=axis.transAxes, va="bottom", fontsize="small",
            )
            plotted = True
        # Use a log scale only when every finite value in the shared panel is
        # safely positive. Zero/negative/subnormal values remain readable on a
        # linear scale and cannot trigger Matplotlib's empty-log-axis warning.
        if finite_values_across_methods and min(finite_values_across_methods) >= np.finfo(float).tiny:
            axis.set_yscale("log")
        axis.set(xlabel="epoch", ylabel=label)
        if plotted:
            axis.legend()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _reference_convergence(
    problem: PeriodicTwoWellCommittor, grid_sizes: list[int]
) -> tuple[PeriodicCommittorReference, dict[str, float | None]]:
    references = {}
    for size in grid_sizes:
        started = time.monotonic()
        print(f"Finite-difference reference: solving {size} x {size} grid on CPU.", flush=True)
        references[size] = solve_periodic_committor_fd(problem, size)
        print(
            f"Finite-difference reference: {size} x {size} complete "
            f"({time.monotonic() - started:.1f} s).",
            flush=True,
        )
    finest_size = max(grid_sizes)
    finest = references[finest_size]
    diagnostics: dict[str, float | None] = {}
    for size, coarse in references.items():
        if size == finest_size or finest_size % size:
            continue
        diagnostics.update(periodic_reference_grid_errors(coarse, finest))
    return finest, diagnostics

def _save_reference(output_dir: Path, reference: PeriodicCommittorReference) -> str:
    filename = f"reference_fd_{reference.values.shape[0]}.npz"
    np.savez_compressed(
        output_dir / filename,
        grid=reference.grid,
        values=reference.values,
        source=reference.source,
        target=reference.target,
        domain=reference.domain,
    )
    return filename


def _load_reference(path: Path) -> PeriodicCommittorReference:
    with np.load(path) as saved:
        values = saved["values"]
        return PeriodicCommittorReference(
            grid=saved["grid"],
            values=values,
            source=saved["source"],
            target=saved["target"],
            domain=saved["domain"],
            spacing=1.0 / values.shape[0],
        )


def _load_monte_carlo_labels(output_dir: Path) -> CommittorMonteCarloData:
    labels_path = output_dir / "monte_carlo_labels.npz"
    metadata_path = output_dir / "monte_carlo_labels.json"
    if not labels_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "CPU preparation is incomplete: the saved Monte Carlo labels are missing."
        )
    with np.load(labels_path) as saved:
        return CommittorMonteCarloData(
            anchors=torch.as_tensor(saved["anchors"], dtype=torch.float64),
            values=torch.as_tensor(saved["values"], dtype=torch.float64),
            standard_errors=torch.as_tensor(saved["standard_errors"], dtype=torch.float64),
            resolved_paths=torch.as_tensor(saved["resolved_paths"], dtype=torch.int64),
            total_paths=torch.as_tensor(saved["total_paths"], dtype=torch.int64),
            metadata=json.loads(metadata_path.read_text(encoding="utf-8")),
        )


def _prepare_cpu_artifacts(
    raw: dict[str, Any],
    output_dir: Path,
    *,
    interior_sampler: InteriorSampler | None = None,
) -> tuple[PeriodicCommittorReference, dict[str, float | None], CommittorMonteCarloData | None]:
    """Compute and save the CPU-only reference and shared Monte Carlo labels."""
    write_json(output_dir / "preparation_config.json", raw)
    problem = PeriodicTwoWellCommittor(**raw["problem"])
    grid_sizes = [int(size) for size in raw["reference_grid_sizes"]]
    print(f"CPU preparation phase 1/2: finite-difference grids {grid_sizes}.", flush=True)
    reference, fd_diagnostics = _reference_convergence(problem, grid_sizes)
    reference_file = _save_reference(output_dir, reference)
    print(f"Finite-difference reference saved: {output_dir / reference_file}", flush=True)

    shared_mc_data: CommittorMonteCarloData | None = None
    mc_settings = raw["training"].get("monte_carlo")
    if mc_settings is not None:
        print(
            "CPU preparation phase 2/2: generating the common Feynman-Kac labels. "
            f"Anchors={mc_settings['n_anchors']}, paths per anchor={mc_settings['paths_per_anchor']}, "
            f"dt={mc_settings['dt']}, step cap={mc_settings['max_steps']:,}, "
            f"required resolved fraction={mc_settings.get('minimum_resolved_fraction', 0.9):.0%}.",
            flush=True,
        )
        shared_mc_data = generate_committor_mc_data(
            problem, **mc_settings, interior_sampler=interior_sampler
        )
        np.savez_compressed(
            output_dir / "monte_carlo_labels.npz",
            anchors=shared_mc_data.anchors.numpy(),
            values=shared_mc_data.values.numpy(),
            standard_errors=shared_mc_data.standard_errors.numpy(),
            resolved_paths=shared_mc_data.resolved_paths.numpy(),
            total_paths=shared_mc_data.total_paths.numpy(),
        )
        write_json(output_dir / "monte_carlo_labels.json", shared_mc_data.metadata)
        plot_path = output_dir / "monte_carlo_anchors.png"
        _plot_monte_carlo_labels(problem, shared_mc_data, plot_path)
        print(f"Monte Carlo labels saved: {output_dir / 'monte_carlo_labels.npz'}", flush=True)
        print(f"Monte Carlo anchor plot saved: {plot_path}", flush=True)

    write_json(output_dir / "cpu_preparation.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "preparation_git_revision": _git_revision(),
        "preparation_schema_version": PREPARATION_SCHEMA_VERSION,
        "reference_file": reference_file,
        "reference_grid_sizes": grid_sizes,
        "reference_convergence": fd_diagnostics,
        "monte_carlo_file": "monte_carlo_labels.npz" if shared_mc_data is not None else None,
        "monte_carlo_metadata_file": "monte_carlo_labels.json" if shared_mc_data is not None else None,
        "monte_carlo_plot_file": "monte_carlo_anchors.png" if shared_mc_data is not None else None,
        "config_file": "preparation_config.json",
        "interior_sampler": _hook_name(
            interior_sampler, "uniform rejection on the punctured torus"
        ),
        "interior_sampler_fingerprint": _callable_fingerprint(interior_sampler),
    })
    print(f"CPU preparation complete: {output_dir}", flush=True)
    return reference, fd_diagnostics, shared_mc_data


def _load_cpu_preparation(
    raw: dict[str, Any],
    output_dir: Path,
    *,
    interior_sampler: InteriorSampler | None = None,
) -> tuple[PeriodicCommittorReference, dict[str, float | None], CommittorMonteCarloData | None]:
    """Load the finished CPU artifacts and reject incomplete or mismatched runs."""
    preparation_path = output_dir / "cpu_preparation.json"
    if not preparation_path.is_file():
        raise FileNotFoundError(
            f"No completed cpu_preparation.json in {output_dir}. "
            "Run the CPU preparation stage first."
        )
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    saved_config_path = output_dir / str(
        preparation.get("config_file", "preparation_config.json")
    )
    if not saved_config_path.is_file():
        saved_config_path = output_dir / "comparison_config.json"
    if saved_config_path.is_file():
        saved_config = json.loads(saved_config_path.read_text(encoding="utf-8"))
        if _preparation_inputs(saved_config) != _preparation_inputs(raw):
            raise ValueError(
                "The selected problem, reference grids, or Monte Carlo settings differ "
                "from the saved CPU preparation. Restore the saved preparation settings, "
                "or create a new folder for the changed preparation."
            )
    schema_version = preparation.get("preparation_schema_version", 1)
    if schema_version != PREPARATION_SCHEMA_VERSION:
        raise ValueError(
            "The saved CPU preparation uses an incompatible data format. "
            "Create a new preparation folder with the installed code."
        )
    expected_sampler = _hook_name(
        interior_sampler, "uniform rejection on the punctured torus"
    )
    saved_sampler_fingerprint = preparation.get("interior_sampler_fingerprint")
    if (
        saved_sampler_fingerprint is not None
        and saved_sampler_fingerprint != _callable_fingerprint(interior_sampler)
    ):
        raise ValueError(
            "The saved Monte Carlo anchors were generated with different sampler code. "
            "Restore the sampler from preparation or create a new preparation folder."
        )
    saved_sampler = preparation.get(
        "interior_sampler", "uniform rejection on the punctured torus"
    )
    if saved_sampler != expected_sampler:
        raise ValueError(
            "The prepared Monte Carlo anchors used a different interior sampler. "
            "Restore the sampling function from the preparation notebook or prepare a new run."
        )
    schema_version = preparation.get("preparation_schema_version", 1)
    if schema_version != PREPARATION_SCHEMA_VERSION:
        raise ValueError(
            "The saved CPU preparation uses an incompatible data format. "
            "Create a new preparation folder with the installed code."
        )
    if list(map(int, preparation.get("reference_grid_sizes", []))) != list(
        map(int, raw["reference_grid_sizes"])
    ):
        raise ValueError(
            "The saved finite-difference grids do not match the selected grid sizes. "
            "Restore the saved grid sizes or create a new preparation folder."
        )
    expected_reference = f"reference_fd_{max(map(int, raw['reference_grid_sizes']))}.npz"
    reference_file = output_dir / str(preparation.get("reference_file", ""))
    if preparation.get("reference_file") != expected_reference or not reference_file.is_file():
        raise FileNotFoundError(
            f"The prepared finite-difference reference is missing or does not match this configuration: "
            f"{output_dir / expected_reference}"
        )
    reference = _load_reference(reference_file)

    mc_settings = raw["training"].get("monte_carlo")
    shared_mc_data = None
    if mc_settings is not None:
        shared_mc_data = _load_monte_carlo_labels(output_dir)
        saved_mc = shared_mc_data.metadata
        if (
            int(saved_mc.get("anchors", -1)) != int(mc_settings["n_anchors"])
            or int(saved_mc.get("paths_per_anchor", -1)) != int(mc_settings["paths_per_anchor"])
            or float(saved_mc.get("dt", -1.0)) != float(mc_settings["dt"])
            or int(saved_mc.get("max_steps", -1)) != int(mc_settings["max_steps"])
            or int(saved_mc.get("seed", -1)) != int(mc_settings["seed"])
        ):
            raise ValueError("Saved Monte Carlo labels do not match the selected configuration.")
    elif preparation.get("monte_carlo_file") is not None:
        raise ValueError("Prepared folder has Monte Carlo labels, but this configuration does not.")
    diagnostics = preparation.get("reference_convergence", {})
    print(
        f"Loaded CPU-prepared reference and shared labels from {output_dir}; "
        "the long CPU preparation will not be repeated.",
        flush=True,
    )
    return reference, diagnostics, shared_mc_data


def _validate_comparison_config(raw: dict[str, Any]) -> None:
    """Reject misspelled or incomplete controls before CPU preparation or training."""
    def require_mapping(value: Any, label: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be a JSON object")
        return value

    def reject_unknown(section: dict[str, Any], allowed: set[str], label: str) -> None:
        unknown = set(section) - allowed
        if unknown:
            raise ValueError(f"Unknown {label} setting(s): {', '.join(sorted(unknown))}")

    raw = require_mapping(raw, "comparison configuration")
    reject_unknown(
        raw,
        {
            "seed", "device", "dtype", "methods", "problem", "model", "training",
            "reference_grid_sizes", "evaluation_batch_size",
        },
        "top-level",
    )
    for required in ("seed", "device", "dtype", "methods", "problem", "training"):
        if required not in raw:
            raise ValueError(f"Missing required comparison setting: {required}")
    if raw["dtype"] not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64")
    device = str(raw["device"])
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("device must be 'cpu' or a CUDA device such as 'cuda' or 'cuda:0'")
    if not isinstance(raw["methods"], list) or not raw["methods"]:
        raise ValueError("methods must be a nonempty list")
    if len(set(raw["methods"])) != len(raw["methods"]):
        raise ValueError("methods must not contain duplicates")
    unknown_methods = set(raw["methods"]) - set((*PAPER_METHODS, "naive_product", "penalty"))
    if unknown_methods:
        raise ValueError(f"Unknown method(s): {', '.join(sorted(unknown_methods))}")

    problem = require_mapping(raw["problem"], "problem")
    reject_unknown(
        problem,
        {"a", "b", "epsilon", "source_center", "target_center", "set_radius"},
        "problem",
    )
    if "model" in raw:
        model = require_mapping(raw["model"], "model")
        reject_unknown(
            model,
            {
                "architecture", "frequencies", "width", "depth", "activation",
                "activation_kwargs", "initialization", "gain",
            },
            "model",
        )
        architecture = str(model.get("architecture", "periodic_mlp")).lower()
        if architecture not in {"periodic_mlp", "periodic_fourier_mlp"}:
            raise ValueError("model.architecture must be 'periodic_mlp' or 'periodic_fourier_mlp'")
        for key in ("width", "depth"):
            value = float(model.get(key, 0))
            if value < 1 or not value.is_integer():
                raise ValueError(f"model.{key} must be a positive integer")
        activation_name = str(model.get("activation", "tanh"))
        try:
            activation = _activation(activation_name, model.get("activation_kwargs"))
            with torch.no_grad():
                activation.eval()(torch.zeros((2, int(model.get("width", 1)))))
        except Exception as error:
            raise ValueError(
                f"Could not construct or apply model.activation {activation_name!r}: {error}"
            ) from error
        if str(model.get("initialization", "xavier_normal")).lower() not in {
            "xavier_normal", "xavier_uniform",
        }:
            raise ValueError("model.initialization must be xavier_normal or xavier_uniform")
        if "gain" in model and float(model["gain"]) <= 0:
            raise ValueError("model.gain must be positive")
        if architecture == "periodic_fourier_mlp":
            frequencies = model.get("frequencies", [1, 2, 4])
            if not isinstance(frequencies, (list, tuple)) or not frequencies or any(
                float(value) < 1 or not float(value).is_integer() for value in frequencies
            ):
                raise ValueError("model.frequencies must be a nonempty list of positive integers")
        elif "frequencies" in model:
            raise ValueError("model.frequencies is only used with periodic_fourier_mlp")

    training = require_mapping(raw["training"], "training")
    reject_unknown(
        training,
        {
            "epochs", "collocation_points", "boundary_points", "boundary_weight",
            "learning_rate", "lr_gamma", "optimizer", "scheduler", "checkpoint_every",
            "log_interval_seconds", "log_every", "adaptive_loss_log_scale_init",
            "monte_carlo", "nitsche",
        },
        "training",
    )
    for key in ("epochs", "collocation_points", "checkpoint_every"):
        if key in training and int(training[key]) < 1:
            raise ValueError(f"training.{key} must be positive")
    if "boundary_points" in training and int(training["boundary_points"]) < 1:
        raise ValueError("training.boundary_points must be positive")
    if "learning_rate" in training and float(training["learning_rate"]) <= 0.0:
        raise ValueError("training.learning_rate must be positive")
    if "adaptive_loss_log_scale_init" in training and not np.isfinite(
        float(training["adaptive_loss_log_scale_init"])
    ):
        raise ValueError("training.adaptive_loss_log_scale_init must be finite")
    if "log_interval_seconds" in training and float(training["log_interval_seconds"]) <= 0.0:
        raise ValueError("training.log_interval_seconds must be positive")
    if "log_interval_seconds" in training and "log_every" in training:
        raise ValueError(
            "Set only training.log_interval_seconds for time-based reports; "
            "training.log_every is supported only by older configurations."
        )
    if "boundary_weight" in training and training.get("monte_carlo") is not None:
        raise ValueError(
            "training.boundary_weight is for legacy objectives; the paper comparison "
            "uses adaptive component weighting and method-specific boundary treatment."
        )

    optimizer = training.get("optimizer", {})
    if isinstance(optimizer, str):
        optimizer_name, optimizer_options = optimizer.lower(), {}
    else:
        optimizer_options = dict(require_mapping(optimizer, "training.optimizer"))
        optimizer_name = str(optimizer_options.pop("name", "adam")).lower()
    if optimizer_name not in {"adam", "adamw"}:
        raise ValueError("training.optimizer.name must be 'adam' or 'adamw'")
    reject_unknown(
        optimizer_options, {"betas", "eps", "weight_decay", "amsgrad"}, "optimizer"
    )
    if "betas" in optimizer_options:
        betas = optimizer_options["betas"]
        if not isinstance(betas, (list, tuple)) or len(betas) != 2 or any(
            not 0.0 <= float(beta) < 1.0 for beta in betas
        ):
            raise ValueError("training.optimizer.betas must contain two values in [0, 1)")
    if "eps" in optimizer_options and float(optimizer_options["eps"]) <= 0:
        raise ValueError("training.optimizer.eps must be positive")
    if "weight_decay" in optimizer_options and float(optimizer_options["weight_decay"]) < 0:
        raise ValueError("training.optimizer.weight_decay must be nonnegative")

    scheduler = training.get("scheduler")
    if scheduler is not None:
        if isinstance(scheduler, str):
            scheduler_name, scheduler_options = scheduler.lower(), {}
        else:
            scheduler_options = dict(require_mapping(scheduler, "training.scheduler"))
            scheduler_name = str(scheduler_options.pop("name", "exponential")).lower()
        if scheduler_name not in {"constant", "exponential", "step"}:
            raise ValueError("training.scheduler.name must be constant, exponential, or step")
        reject_unknown(scheduler_options, {"gamma", "step_size_epochs"}, "scheduler")
        if scheduler_name == "constant" and scheduler_options:
            raise ValueError("The constant scheduler accepts no additional settings")
        if scheduler_name == "exponential" and "step_size_epochs" in scheduler_options:
            raise ValueError("step_size_epochs is only used by the step scheduler")
        if scheduler_name == "step":
            if int(scheduler_options.get("step_size_epochs", 0)) < 1:
                raise ValueError("step scheduler requires positive step_size_epochs")
            if "gamma" not in scheduler_options:
                raise ValueError("step scheduler requires gamma")
        if "gamma" in scheduler_options and float(scheduler_options["gamma"]) <= 0:
            raise ValueError("scheduler.gamma must be positive")

    monte_carlo = training.get("monte_carlo")
    if monte_carlo is not None:
        monte_carlo = require_mapping(monte_carlo, "training.monte_carlo")
        if not set(raw["methods"]).issubset(PAPER_METHODS):
            raise ValueError(
                "Monte Carlo solution-value data is supported with the five paper methods only."
            )
        reject_unknown(
            monte_carlo,
            {
                "n_anchors", "paths_per_anchor", "dt", "max_steps", "seed",
                "minimum_resolved_fraction", "anchor_points",
                "progress_interval_fraction", "progress_interval_seconds",
            },
            "training.monte_carlo",
        )
        for key in ("n_anchors", "paths_per_anchor", "max_steps"):
            if int(monte_carlo.get(key, 0)) < 1:
                raise ValueError(f"training.monte_carlo.{key} must be positive")
        if float(monte_carlo.get("dt", 0.0)) <= 0.0:
            raise ValueError("training.monte_carlo.dt must be positive")
        resolution = float(monte_carlo.get("minimum_resolved_fraction", 0.9))
        if not 0.0 <= resolution <= 1.0:
            raise ValueError("training.monte_carlo.minimum_resolved_fraction must lie in [0, 1]")
        if not 0.0 < float(monte_carlo.get("progress_interval_fraction", 0.01)) <= 1.0:
            raise ValueError("training.monte_carlo.progress_interval_fraction must lie in (0, 1]")
        if float(monte_carlo.get("progress_interval_seconds", 10.0)) <= 0.0:
            raise ValueError("training.monte_carlo.progress_interval_seconds must be positive")

    nitsche = training.get("nitsche")
    if nitsche is not None:
        nitsche = require_mapping(nitsche, "training.nitsche")
        reject_unknown(
            nitsche,
            {"volume_points", "test_functions", "boundary_points", "length_scale", "gamma", "h"},
            "training.nitsche",
        )
        for key in ("volume_points", "test_functions", "boundary_points"):
            if key in nitsche and int(nitsche[key]) < 1:
                raise ValueError(f"training.nitsche.{key} must be positive")
        for key in ("length_scale", "gamma", "h"):
            if key in nitsche and float(nitsche[key]) <= 0.0:
                raise ValueError(f"training.nitsche.{key} must be positive")
    if "variational_nitsche" in raw["methods"] and monte_carlo is not None:
        required_nitsche = {
            "volume_points", "test_functions", "boundary_points", "length_scale", "gamma", "h"
        }
        missing = required_nitsche - set(nitsche or {})
        if missing:
            raise ValueError(
                "training.nitsche is missing required setting(s): "
                + ", ".join(sorted(missing))
            )

    if "reference_grid_sizes" in raw:
        sizes = raw["reference_grid_sizes"]
        if not isinstance(sizes, list) or not sizes or any(int(size) < 2 for size in sizes):
            raise ValueError("reference_grid_sizes must be a nonempty list of integers >= 2")
        if any(int(a) >= int(b) for a, b in zip(sizes, sizes[1:])):
            raise ValueError("reference_grid_sizes must be strictly increasing")
    if "evaluation_batch_size" in raw and int(raw["evaluation_batch_size"]) < 1:
        raise ValueError("evaluation_batch_size must be positive")


def run_periodic_comparison(
    config_path: Path,
    output_dir: Path,
    *,
    resume: bool = False,
    prepare_only: bool = False,
    train_prepared: bool = False,
    model_factory: ModelFactory | None = None,
    interior_sampler: InteriorSampler | None = None,
    optimizer_factory: OptimizerFactory | None = None,
    scheduler_factory: SchedulerFactory | None = None,
) -> None:
    """Prepare numerical references and run the configured neural comparisons.

    The default command performs CPU preparation and training in one process.
    Colab can instead use prepare_only=True in a CPU runtime and
    train_prepared=True later in a GPU runtime, so Monte Carlo generation does
    not occupy a GPU session.
    """
    if prepare_only and train_prepared:
        raise ValueError("choose either CPU preparation or training from prepared files")
    if prepare_only and resume:
        raise ValueError("CPU preparation cannot resume neural-network checkpoints")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_comparison_config(raw)
    methods = list(raw["methods"])
    if not methods:
        raise ValueError("comparison configuration needs at least one method")
    if str(raw.get("device", "cpu")).startswith("cuda") and not (prepare_only or train_prepared):
        raise ValueError(
            "A CUDA-configured comparison must be split into CPU preparation and GPU training. "
            "Run --prepare-cpu-only first, then use --train-prepared (and --resume if needed)."
        )
    if not prepare_only and model_factory is None and "model" not in raw:
        raise ValueError(
            "This run has no model configuration. Pass model_factory=your_function "
            "from Python, or add a model section for the built-in command-line model."
        )
    preparation_path = output_dir / "cpu_preparation.json"
    if output_dir.exists() and not (resume or train_prepared) and any(
        path.name != "console.log" for path in output_dir.iterdir()
    ):
        raise FileExistsError(f"{output_dir} already contains a run; choose a new directory or use --resume")
    if prepare_only and preparation_path.exists():
        raise FileExistsError(f"{output_dir} already has completed CPU preparation; choose a new run directory.")
    if train_prepared and not preparation_path.is_file():
        raise FileNotFoundError(
            f"No completed CPU preparation in {output_dir}. Run the CPU preparation stage first."
        )
    if train_prepared and not resume:
        existing_checkpoints = list(output_dir.glob("*/last.pt"))
        if existing_checkpoints:
            raise FileExistsError(
                f"{output_dir} already contains training checkpoints. "
                "Use the resume option to continue, or prepare a new run folder."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    config_record = output_dir / "comparison_config.json"
    if train_prepared and not config_record.is_file():
        raise FileNotFoundError(
            f"Prepared folder is missing its comparison_config.json: {output_dir}"
        )
    previous_config: dict[str, Any] | None = None
    if config_record.is_file():
        previous_config = json.loads(config_record.read_text(encoding="utf-8"))
        if previous_config != raw:
            if not train_prepared or resume:
                raise ValueError(
                    "The saved comparison configuration differs from the selected configuration."
                )
            if _preparation_inputs(previous_config) != _preparation_inputs(raw):
                raise ValueError(
                    "The selected problem, reference grids, or Monte Carlo settings differ "
                    "from the saved CPU preparation. Restore the saved preparation settings, "
                    "or create a new folder for the changed preparation."
                )
    else:
        write_json(config_record, raw)

    metadata_path = output_dir / "comparison_metadata.json"
    now = datetime.now(timezone.utc).isoformat()
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        revision = _git_revision()
        metadata = {
            "created_utc": now,
            "config_path": str(config_path),
            "git_revision": revision,
            "preparation_git_revision": revision,
        }

    if prepare_only:
        if metadata_path.is_file():
            raise FileExistsError(
                f"{output_dir} already has comparison metadata; choose a new run directory for CPU preparation."
            )
        write_json(metadata_path, metadata)
        _prepare_cpu_artifacts(raw, output_dir, interior_sampler=interior_sampler)
        print(
            "No neural network was trained. The CPU reference and Monte Carlo labels are ready "
            "for the separate GPU training stage.",
            flush=True,
        )
        return

    if train_prepared or (resume and preparation_path.is_file()):
        reference, fd_diagnostics, shared_mc_data = _load_cpu_preparation(
            raw, output_dir, interior_sampler=interior_sampler
        )
    else:
        reference, fd_diagnostics, shared_mc_data = _prepare_cpu_artifacts(
            raw, output_dir, interior_sampler=interior_sampler
        )

    if train_prepared and not resume:
        preparation_config_path = output_dir / "preparation_config.json"
        if not preparation_config_path.is_file():
            if previous_config is None:
                raise FileNotFoundError(
                    f"Prepared folder is missing its comparison_config.json: {output_dir}"
                )
            write_json(preparation_config_path, previous_config)
        preparation_record_path = output_dir / "cpu_preparation.json"
        preparation_record = json.loads(
            preparation_record_path.read_text(encoding="utf-8")
        )
        preparation_record["config_file"] = "preparation_config.json"
        write_json(preparation_record_path, preparation_record)
        write_json(config_record, raw)

    problem = PeriodicTwoWellCommittor(**raw["problem"])
    grid_sizes = [int(size) for size in raw["reference_grid_sizes"]]
    requested_device = torch.device(str(raw["device"]))
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The comparison configuration requests CUDA but no CUDA device is available.")
    revision = _git_revision()
    metadata["training_git_revision"] = revision
    metadata["runtime_hooks"] = {
        "model": _hook_name(model_factory, "repository model builder"),
        "interior_sampler": _hook_name(
            interior_sampler, "uniform rejection on the punctured torus"
        ),
        "optimizer": _hook_name(optimizer_factory, "repository optimizer builder"),
        "scheduler": _hook_name(scheduler_factory, "repository scheduler builder"),
    }
    metadata["runtime_hook_fingerprints"] = {
        "model": _callable_fingerprint(model_factory),
        "interior_sampler": _callable_fingerprint(interior_sampler),
        "optimizer": _callable_fingerprint(optimizer_factory),
        "scheduler": _callable_fingerprint(scheduler_factory),
    }
    metadata.setdefault("training_started_utc", now)
    metadata["git_revision"] = revision
    if resume:
        metadata["last_resumed_utc"] = now
        metadata.setdefault("resumes", []).append({
            "resumed_utc": now,
            "git_revision": revision,
        })
    write_json(metadata_path, metadata)

    dtype = _dtype(str(raw["dtype"]))
    batch_size = int(raw.get("evaluation_batch_size", 16_384))
    method_directories: dict[str, Path] = {}
    summary: dict[str, Any] = {
        "reference_grid_sizes": grid_sizes,
        "reference_convergence": fd_diagnostics,
        "methods": {},
    }

    print(f"GPU/CPU phase 3/3: training {len(methods)} configured methods on {requested_device}.", flush=True)
    for method in methods:
        run_directory = output_dir / method
        method_directories[method] = run_directory
        config = PeriodicCommittorConfig(
            seed=int(raw["seed"]),
            device=str(raw["device"]),
            dtype=str(raw["dtype"]),
            method=method,
            problem=raw["problem"],
            model=raw.get("model", {}),
            training=raw["training"],
        )
        method_resume = resume and (run_directory / "last.pt").exists()
        print(f"Starting {method} ({config.training['epochs']} epochs).", flush=True)
        final_training_metrics = run_periodic_committor(
            config,
            run_directory,
            resume=method_resume,
            mc_data=shared_mc_data,
            model_factory=model_factory,
            interior_sampler=interior_sampler,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
        )
        model = _build_periodic_committor_model(
            config, problem, model_factory
        ).to(device=requested_device, dtype=dtype)
        checkpoint = torch.load(run_directory / "best.pt", map_location=requested_device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        prediction = _evaluate(model, reference, device=requested_device, dtype=dtype, batch_size=batch_size)
        method_summary = {
            "best_epoch": int(checkpoint["epoch"]),
            "best_training_objective": float(checkpoint["best_loss"]),
            "best_training_loss": float(checkpoint["best_loss"]),
            "final_training_metrics": final_training_metrics,
            **periodic_reference_errors(prediction, reference),
            **_boundary_rmse(model, problem, device=requested_device, dtype=dtype),
        }
        summary["methods"][method] = method_summary
        _plot_method(method, prediction, reference, output_dir / f"committor_{method}.png")
        write_json(output_dir / "summary.json", summary)
        print(f"Completed {method}; results saved to {run_directory}.", flush=True)

    _plot_losses(method_directories, output_dir / "losses.png")
    write_json(output_dir / "summary.json", summary)
    print(f"Comparison complete: {output_dir}", flush=True)


def evaluate_saved_models(
    output_dir: Path,
    *,
    training_git_revision: str | None = None,
    model_factory: ModelFactory | None = None,
) -> Path:
    """Evaluate existing best checkpoints against the saved FDM reference on CPU.

    This performs no training and does not overwrite summary.json. It writes
    solution-error norms and true-boundary errors to saved_model_evaluation.json.
    """
    config_path = output_dir / "comparison_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing comparison_config.json in {output_dir}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_comparison_config(raw)
    if model_factory is None and "model" not in raw:
        raise ValueError(
            "This run used a Python model_factory. Re-run the notebook's model cell "
            "and pass that same function to evaluate_saved_models(..., model_factory=...)."
        )
    grid_sizes = [int(size) for size in raw["reference_grid_sizes"]]
    reference_path = output_dir / f"reference_fd_{max(grid_sizes)}.npz"
    if not reference_path.is_file():
        raise FileNotFoundError(f"Missing finite-difference reference: {reference_path}")
    with np.load(reference_path) as saved:
        values = saved["values"]
        reference = PeriodicCommittorReference(
            grid=saved["grid"],
            values=values,
            source=saved["source"],
            target=saved["target"],
            domain=saved["domain"],
            spacing=1.0 / values.shape[0],
        )

    problem = PeriodicTwoWellCommittor(**raw["problem"])
    dtype = _dtype(str(raw["dtype"]))
    device = torch.device("cpu")
    batch_size = int(raw.get("evaluation_batch_size", 16_384))
    current_model_fingerprint = _callable_fingerprint(model_factory)
    metadata_path = output_dir / "comparison_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    methods: dict[str, dict[str, Any]] = {}

    for method in raw["methods"]:
        method_metadata_path = output_dir / method / "metadata.json"
        method_metadata = (
            json.loads(method_metadata_path.read_text(encoding="utf-8"))
            if method_metadata_path.is_file()
            else {}
        )
        saved_model_fingerprint = method_metadata.get(
            "runtime_hook_fingerprints", {}
        ).get("model")
        if (
            saved_model_fingerprint is not None
            and saved_model_fingerprint != current_model_fingerprint
        ):
            raise ValueError(
                f"The model code for {method} differs from the saved checkpoint."
            )
        config = PeriodicCommittorConfig(
            seed=int(raw["seed"]),
            device="cpu",
            dtype=str(raw["dtype"]),
            method=method,
            problem=raw["problem"],
            model=raw.get("model", {}),
            training=raw["training"],
        )
        model = _build_periodic_committor_model(
            config, problem, model_factory
        ).to(device=device, dtype=dtype)
        checkpoint_path = output_dir / method / "best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing best checkpoint for {method}: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        prediction = _evaluate(
            model, reference, device=device, dtype=dtype, batch_size=batch_size
        )
        methods[method] = {
            "best_epoch": int(checkpoint["epoch"]),
            "best_training_objective": float(checkpoint["best_loss"]),
            "final_training_metrics": _saved_final_training_metrics(output_dir / method),
            **periodic_reference_errors(prediction, reference),
            **_boundary_rmse(model, problem, device=device, dtype=dtype),
        }
        print(f"Evaluated {method} checkpoint against FDM reference on CPU.", flush=True)

    destination = output_dir / "saved_model_evaluation.json"
    write_json(destination, {
        "training_git_revision": (
            training_git_revision
            or metadata.get("training_git_revision")
            or metadata.get("git_revision")
        ),
        "evaluation_git_revision": _git_revision(),
        "reference_file": reference_path.name,
        "reference_grid_size": int(reference.values.shape[0]),
        "norm_convention": (
            "Area-weighted finite-difference Sobolev norms. L2 uses exterior-domain "
            "nodes; H1 uses axial stencils contained in the domain; H2 uses full 3x3 "
            "stencils. H2 seminorm includes xx, xy, and yy once each. Relative errors "
            "use the matching FDM reference norm and mask."
        ),
        "methods": methods,
    })
    print(f"Saved solution-error report: {destination}", flush=True)
    return destination

def main() -> None:
    parser = argparse.ArgumentParser(description="Run and plot the periodic committor comparison.")
    parser.add_argument("config", type=Path, nargs="?", help="Comparison JSON configuration for training")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints, data, and figures")
    parser.add_argument("--resume", action="store_true", help="Resume methods with an existing last.pt")
    parser.add_argument(
        "--prepare-cpu-only", action="store_true",
        help="Solve and save the finite-difference reference and Monte Carlo labels, then stop before training",
    )
    parser.add_argument(
        "--train-prepared", action="store_true",
        help="Train using a completed --prepare-cpu-only folder without repeating CPU preparation",
    )
    parser.add_argument(
        "--evaluate-saved-models", action="store_true",
        help="Evaluate saved best.pt models against the FDM reference and true boundaries on CPU; never train",
    )
    parser.add_argument(
        "--training-git-revision",
        help="Original run revision when auditing an older run without revision metadata",
    )
    args = parser.parse_args()
    if args.evaluate_saved_models:
        if args.resume or args.prepare_cpu_only or args.train_prepared or args.config is not None:
            parser.error("--evaluate-saved-models uses only --output-dir")
        evaluate_saved_models(args.output_dir, training_git_revision=args.training_git_revision)
        return
    if args.training_git_revision is not None:
        parser.error("--training-git-revision can only be used with --evaluate-saved-models")
    if args.prepare_cpu_only and args.train_prepared:
        parser.error("choose either --prepare-cpu-only or --train-prepared")
    if args.prepare_cpu_only and args.resume:
        parser.error("--resume cannot be combined with --prepare-cpu-only")
    if args.config is None:
        parser.error("config is required for preparation or training")
    run_periodic_comparison(
        args.config,
        args.output_dir,
        resume=args.resume,
        prepare_only=args.prepare_cpu_only,
        train_prepared=args.train_prepared,
    )


if __name__ == "__main__":
    main()
