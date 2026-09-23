"""One-command periodic committor comparison with durable artifacts and plots."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .checkpointing import write_json
from .problems import PeriodicTwoWellCommittor
from .reference import PeriodicCommittorReference, periodic_reference_errors, solve_periodic_committor_fd
from .training import (
    PeriodicCommittorConfig,
    _build_periodic_committor_model,
    _dtype,
    run_periodic_committor,
)


def _git_revision() -> str | None:
    """Return the source revision when the runner is launched from a Git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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


def _boundary_rmse(prediction: np.ndarray, reference: PeriodicCommittorReference) -> dict[str, float]:
    return {
        "source_boundary_rmse": float(np.sqrt(np.mean(prediction[reference.source] ** 2))),
        "target_boundary_rmse": float(
            np.sqrt(np.mean((prediction[reference.target] - 1.0) ** 2))
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


def _plot_losses(method_directories: dict[str, Path], destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for method, run_directory in method_directories.items():
        records = _read_metrics(run_directory / "metrics.jsonl")
        if records:
            axis.semilogy([item["epoch"] for item in records], [item["loss"] for item in records], label=method.replace("_", " "))
    axis.set(xlabel="epoch", ylabel="training objective")
    axis.legend()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _reference_convergence(
    problem: PeriodicTwoWellCommittor, grid_sizes: list[int]
) -> tuple[PeriodicCommittorReference, dict[str, float]]:
    references = {size: solve_periodic_committor_fd(problem, size) for size in grid_sizes}
    finest_size = max(grid_sizes)
    finest = references[finest_size]
    diagnostics: dict[str, float] = {}
    for size, coarse in references.items():
        if size == finest_size or finest_size % size:
            continue
        stride = finest_size // size
        fine_values = finest.values[::stride, ::stride]
        fine_domain = finest.domain[::stride, ::stride]
        common = coarse.domain & fine_domain
        diagnostics[f"fd_l2_{size}_to_{finest_size}"] = float(
            np.sqrt(np.mean((coarse.values[common] - fine_values[common]) ** 2))
        )
    return finest, diagnostics


def run_periodic_comparison(config_path: Path, output_dir: Path, *, resume: bool = False) -> None:
    """Train all configured methods and save plots, checkpoints, and summary data."""

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    methods = list(raw["methods"])
    if not methods:
        raise ValueError("comparison configuration needs at least one method")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"{output_dir} already contains a run; choose a new directory or use --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "comparison_config.json", raw)
    write_json(output_dir / "comparison_metadata.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "git_revision": _git_revision(),
    })

    problem = PeriodicTwoWellCommittor(**raw["problem"])
    grid_sizes = [int(size) for size in raw["reference_grid_sizes"]]
    reference, fd_diagnostics = _reference_convergence(problem, grid_sizes)
    np.savez_compressed(
        output_dir / f"reference_fd_{reference.values.shape[0]}.npz",
        grid=reference.grid,
        values=reference.values,
        source=reference.source,
        target=reference.target,
        domain=reference.domain,
    )

    requested_device = torch.device(str(raw["device"]))
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The comparison configuration requests CUDA but no CUDA device is available.")
    dtype = _dtype(str(raw["dtype"]))
    batch_size = int(raw.get("evaluation_batch_size", 16_384))
    method_directories: dict[str, Path] = {}
    summary: dict[str, Any] = {
        "reference_grid_sizes": grid_sizes,
        "reference_convergence": fd_diagnostics,
        "methods": {},
    }

    for method in methods:
        run_directory = output_dir / method
        method_directories[method] = run_directory
        config = PeriodicCommittorConfig(
            seed=int(raw["seed"]),
            device=str(raw["device"]),
            dtype=str(raw["dtype"]),
            method=method,
            problem=raw["problem"],
            model=raw["model"],
            training=raw["training"],
        )
        method_resume = resume and (run_directory / "last.pt").exists()
        run_periodic_committor(config, run_directory, resume=method_resume)
        model = _build_periodic_committor_model(config, problem).to(device=requested_device, dtype=dtype)
        checkpoint = torch.load(run_directory / "best.pt", map_location=requested_device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        prediction = _evaluate(model, reference, device=requested_device, dtype=dtype, batch_size=batch_size)
        method_summary = {
            "best_epoch": int(checkpoint["epoch"]),
            "best_training_loss": float(checkpoint["best_loss"]),
            **periodic_reference_errors(prediction, reference),
            **_boundary_rmse(prediction, reference),
        }
        summary["methods"][method] = method_summary
        _plot_method(method, prediction, reference, output_dir / f"committor_{method}.png")
        write_json(output_dir / "summary.json", summary)

    _plot_losses(method_directories, output_dir / "losses.png")
    write_json(output_dir / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and plot the periodic committor comparison.")
    parser.add_argument("config", type=Path, help="Comparison JSON configuration")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints, data, and figures")
    parser.add_argument("--resume", action="store_true", help="Resume methods with an existing last.pt")
    args = parser.parse_args()
    run_periodic_comparison(args.config, args.output_dir, resume=args.resume)


if __name__ == "__main__":
    main()
