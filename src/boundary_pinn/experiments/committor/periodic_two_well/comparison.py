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

from ....checkpointing import write_json
from ....problems.committor import PeriodicTwoWellCommittor
from ....problems.committor.reference import PeriodicCommittorReference, periodic_reference_errors, solve_periodic_committor_fd
from ....problems.committor.training import (
    PeriodicCommittorConfig,
    _build_periodic_committor_model,
    _dtype,
    run_periodic_committor,
)


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
        source = problem.sample_boundary(
            "source", points_per_boundary, device="cpu", dtype=dtype, generator=generator
        ).to(device)
        target = problem.sample_boundary(
            "target", points_per_boundary, device="cpu", dtype=dtype, generator=generator
        ).to(device)
        source_error = model(source)
        target_error = model(target) - 1.0
        return {
            "source_boundary_rmse": float(source_error.square().mean().sqrt().cpu()),
            "target_boundary_rmse": float(target_error.square().mean().sqrt().cpu()),
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
    if output_dir.exists() and not resume and any(
        path.name != "console.log" for path in output_dir.iterdir()
    ):
        raise FileExistsError(f"{output_dir} already contains a run; choose a new directory or use --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_record = output_dir / "comparison_config.json"
    if resume and config_record.is_file():
        previous_config = json.loads(config_record.read_text(encoding="utf-8"))
        if previous_config != raw:
            raise ValueError("Cannot resume: the saved comparison configuration differs.")
    else:
        write_json(config_record, raw)

    metadata_path = output_dir / "comparison_metadata.json"
    now = datetime.now(timezone.utc).isoformat()
    if resume and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.setdefault("resumes", []).append({
            "resumed_utc": now,
            "git_revision": _git_revision(),
        })
    else:
        metadata = {
            "created_utc": now,
            "config_path": str(config_path),
            "git_revision": _git_revision(),
        }
    write_json(metadata_path, metadata)

    problem = PeriodicTwoWellCommittor(**raw["problem"])
    grid_sizes = [int(size) for size in raw["reference_grid_sizes"]]
    print(f"Solving finite-difference references on {grid_sizes} grids (CPU).", flush=True)
    reference, fd_diagnostics = _reference_convergence(problem, grid_sizes)
    np.savez_compressed(
        output_dir / f"reference_fd_{reference.values.shape[0]}.npz",
        grid=reference.grid,
        values=reference.values,
        source=reference.source,
        target=reference.target,
        domain=reference.domain,
    )
    print(f"Reference saved. Starting {len(methods)} model training runs.", flush=True)

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
        print(f"Starting {method} ({config.training['epochs']} epochs).", flush=True)
        run_periodic_committor(config, run_directory, resume=method_resume)
        model = _build_periodic_committor_model(config, problem).to(device=requested_device, dtype=dtype)
        checkpoint = torch.load(run_directory / "best.pt", map_location=requested_device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        prediction = _evaluate(model, reference, device=requested_device, dtype=dtype, batch_size=batch_size)
        method_summary = {
            "best_epoch": int(checkpoint["epoch"]),
            "best_training_loss": float(checkpoint["best_loss"]),
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



def audit_saved_boundaries(output_dir: Path) -> Path:
    """Check true boundary values of saved best checkpoints, using the CPU only."""
    raw = json.loads((output_dir / "comparison_config.json").read_text(encoding="utf-8"))
    problem = PeriodicTwoWellCommittor(**raw["problem"])
    dtype = _dtype(str(raw["dtype"]))
    device = torch.device("cpu")
    methods: dict[str, dict[str, float]] = {}
    for method in raw["methods"]:
        config = PeriodicCommittorConfig(
            seed=int(raw["seed"]),
            device="cpu",
            dtype=str(raw["dtype"]),
            method=method,
            problem=raw["problem"],
            model=raw["model"],
            training=raw["training"],
        )
        model = _build_periodic_committor_model(config, problem).to(device=device, dtype=dtype)
        checkpoint = torch.load(
            output_dir / method / "best.pt", map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state"])
        methods[method] = _boundary_rmse(model, problem, device=device, dtype=dtype)
        print(f"Checked {method} boundaries on CPU.", flush=True)

    metadata_path = output_dir / "comparison_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    destination = output_dir / "boundary_diagnostics.json"
    write_json(destination, {
        "points_per_boundary": 512,
        "sample_seed": 0,
        "training_git_revision": metadata.get("git_revision"),
        "audit_git_revision": _git_revision(),
        "methods": methods,
    })
    print(f"Boundary diagnostics saved: {destination}", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and plot the periodic committor comparison.")
    parser.add_argument("config", type=Path, nargs="?", help="Comparison JSON configuration for training")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints, data, and figures")
    parser.add_argument("--resume", action="store_true", help="Resume methods with an existing last.pt")
    parser.add_argument(
        "--audit-boundaries", action="store_true",
        help="Check saved best.pt models at the true boundaries using the CPU; never train",
    )
    args = parser.parse_args()
    if args.audit_boundaries:
        if args.resume or args.config is not None:
            parser.error("--audit-boundaries uses only --output-dir")
        audit_saved_boundaries(args.output_dir)
        return
    if args.config is None:
        parser.error("config is required for training")
    run_periodic_comparison(args.config, args.output_dir, resume=args.resume)


if __name__ == "__main__":
    main()
