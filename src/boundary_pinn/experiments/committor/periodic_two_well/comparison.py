"""One-command periodic committor comparison with durable artifacts and plots."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
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
from ....problems.committor.reference import (
    PeriodicCommittorReference,
    periodic_reference_errors,
    periodic_reference_grid_errors,
    solve_periodic_committor_fd,
)
from ....problems.committor.paper_methods import (
    CommittorMonteCarloData,
    generate_committor_mc_data,
)
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
        for method, records in records_by_method.items():
            available = [row for row in records if field in row]
            if available:
                epochs = [row["epoch"] for row in available]
                values = [float(row[field]) for row in available]
                if min(values) > 0.0:
                    axis.semilogy(epochs, values, label=method.replace("_", " "))
                else:
                    axis.plot(epochs, values, label=method.replace("_", " "))
                plotted = True
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
    raw: dict[str, Any], output_dir: Path
) -> tuple[PeriodicCommittorReference, dict[str, float | None], CommittorMonteCarloData | None]:
    """Compute and save the CPU-only reference and shared Monte Carlo labels."""
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
        shared_mc_data = generate_committor_mc_data(problem, **mc_settings)
        np.savez_compressed(
            output_dir / "monte_carlo_labels.npz",
            anchors=shared_mc_data.anchors.numpy(),
            values=shared_mc_data.values.numpy(),
            standard_errors=shared_mc_data.standard_errors.numpy(),
            resolved_paths=shared_mc_data.resolved_paths.numpy(),
            total_paths=shared_mc_data.total_paths.numpy(),
        )
        write_json(output_dir / "monte_carlo_labels.json", shared_mc_data.metadata)
        print(f"Monte Carlo labels saved: {output_dir / 'monte_carlo_labels.npz'}", flush=True)

    write_json(output_dir / "cpu_preparation.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "preparation_git_revision": _git_revision(),
        "reference_file": reference_file,
        "reference_grid_sizes": grid_sizes,
        "reference_convergence": fd_diagnostics,
        "monte_carlo_file": "monte_carlo_labels.npz" if shared_mc_data is not None else None,
        "monte_carlo_metadata_file": "monte_carlo_labels.json" if shared_mc_data is not None else None,
        "config_file": "comparison_config.json",
    })
    print(f"CPU preparation complete: {output_dir}", flush=True)
    return reference, fd_diagnostics, shared_mc_data


def _load_cpu_preparation(
    raw: dict[str, Any], output_dir: Path
) -> tuple[PeriodicCommittorReference, dict[str, float | None], CommittorMonteCarloData | None]:
    """Load the finished CPU artifacts and reject incomplete or mismatched runs."""
    preparation_path = output_dir / "cpu_preparation.json"
    if not preparation_path.is_file():
        raise FileNotFoundError(
            f"No completed cpu_preparation.json in {output_dir}. "
            "Run the CPU preparation stage first."
        )
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    prepared_revision = preparation.get("preparation_git_revision")
    active_revision = _git_revision()
    if prepared_revision and active_revision and prepared_revision != active_revision:
        raise ValueError(
            "The CPU reference and labels were prepared with a different repository revision. "
            "Run CPU preparation again from the currently installed branch before training."
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


def run_periodic_comparison(
    config_path: Path,
    output_dir: Path,
    *,
    resume: bool = False,
    prepare_only: bool = False,
    train_prepared: bool = False,
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
    methods = list(raw["methods"])
    if not methods:
        raise ValueError("comparison configuration needs at least one method")
    if str(raw.get("device", "cpu")).startswith("cuda") and not (prepare_only or train_prepared):
        raise ValueError(
            "A CUDA-configured comparison must be split into CPU preparation and GPU training. "
            "Run --prepare-cpu-only first, then use --train-prepared (and --resume if needed)."
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
    output_dir.mkdir(parents=True, exist_ok=True)

    config_record = output_dir / "comparison_config.json"
    if train_prepared and not config_record.is_file():
        raise FileNotFoundError(
            f"Prepared folder is missing its comparison_config.json: {output_dir}"
        )
    if config_record.is_file():
        previous_config = json.loads(config_record.read_text(encoding="utf-8"))
        if previous_config != raw:
            raise ValueError("The saved comparison configuration differs from the selected configuration.")
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
        _prepare_cpu_artifacts(raw, output_dir)
        print(
            "No neural network was trained. The CPU reference and Monte Carlo labels are ready "
            "for the separate GPU training stage.",
            flush=True,
        )
        return

    if train_prepared or (resume and preparation_path.is_file()):
        reference, fd_diagnostics, shared_mc_data = _load_cpu_preparation(raw, output_dir)
    else:
        reference, fd_diagnostics, shared_mc_data = _prepare_cpu_artifacts(raw, output_dir)

    problem = PeriodicTwoWellCommittor(**raw["problem"])
    grid_sizes = [int(size) for size in raw["reference_grid_sizes"]]
    requested_device = torch.device(str(raw["device"]))
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The comparison configuration requests CUDA but no CUDA device is available.")
    revision = _git_revision()
    metadata["training_git_revision"] = revision
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
            model=raw["model"],
            training=raw["training"],
        )
        method_resume = resume and (run_directory / "last.pt").exists()
        print(f"Starting {method} ({config.training['epochs']} epochs).", flush=True)
        final_training_metrics = run_periodic_committor(
            config, run_directory, resume=method_resume, mc_data=shared_mc_data
        )
        model = _build_periodic_committor_model(config, problem).to(device=requested_device, dtype=dtype)
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
    output_dir: Path, *, training_git_revision: str | None = None
) -> Path:
    """Evaluate existing best checkpoints against the saved FDM reference on CPU.

    This performs no training and does not overwrite summary.json. It writes
    solution-error norms and true-boundary errors to saved_model_evaluation.json.
    """
    config_path = output_dir / "comparison_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing comparison_config.json in {output_dir}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
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
    metadata_path = output_dir / "comparison_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    methods: dict[str, dict[str, Any]] = {}

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
