from pathlib import Path

import json
import nbformat
import numpy as np
import pytest
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "experiments" / "mean_escape_time" / "ou_ball" / "ou_mean_escape_time.ipynb",
    ROOT / "experiments" / "mean_escape_time" / "double_well" / "double_well_mean_escape_time.ipynb",
    ROOT / "experiments" / "first_passage" / "duffing_oscillator" / "duffing_mean_escape_time.ipynb",
]


REPLACEMENTS = {
    # OU disk
    "training_collocation_points = 512": "training_collocation_points = 32",
    "training_boundary_points = 128": "training_boundary_points = 16",
    "evaluation_interior_points = 512": "evaluation_interior_points = 32",
    "evaluation_boundary_points = 128": "evaluation_boundary_points = 16",
    "reference_grid_n = 200": "reference_grid_n = 24",
    "data_anchors = 24": "data_anchors = 3",
    "paths_per_anchor = 24": "paths_per_anchor = 3",
    "mc_dt = 5e-3": "mc_dt = 0.02",
    "mc_max_time = 2.0": "mc_max_time = 0.04",
    "trajectory_dt = 1e-3": "trajectory_dt = 0.02",
    "trajectory_max_time = 20.0": "trajectory_max_time = 0.04",
    # Double-well ellipse
    "training_collocation_points = 384": "training_collocation_points = 32",
    "training_boundary_points = 96": "training_boundary_points = 16",
    "evaluation_interior_points = 384": "evaluation_interior_points = 32",
    "evaluation_boundary_points = 96": "evaluation_boundary_points = 16",
    "data_anchors = 20": "data_anchors = 3",
    "paths_per_anchor = 20": "paths_per_anchor = 3",
    "mc_dt = 2e-3": "mc_dt = 0.02",
    "fem_refinements = 4": "fem_refinements = 2",
    "adam_epochs = 30": "adam_epochs = 1",
    "evaluation_grid_n = 200": "evaluation_grid_n = 24",
    # Duffing disk
    "training_collocation_count = 512": "training_collocation_count = 32",
    "training_boundary_count = 128": "training_boundary_count = 16",
    "evaluation_interior_count = 512": "evaluation_interior_count = 32",
    "evaluation_boundary_count = 128": "evaluation_boundary_count = 16",
    "lbfgs_interior_count = 512": "lbfgs_interior_count = 32",
    "lbfgs_boundary_count = 128": "lbfgs_boundary_count = 16",
    "mc_anchors = 20": "mc_anchors = 3",
    "mc_paths = 32": "mc_paths = 3",
    "mc_max_time = 5.0": "mc_max_time = 0.04",
    "adam_epochs = 40": "adam_epochs = 1",
    "n_test_functions = 50": "n_test_functions = 8",
    "evaluation_grid_n = 180": "evaluation_grid_n = 24",
    "trajectory_max_time = 5.0": "trajectory_max_time = 0.04",
}


def patched_notebook(path):
    notebook = nbformat.read(path, as_version=4)
    replacements_applied = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for old, new in REPLACEMENTS.items():
            replacements_applied += cell.source.count(old)
            cell.source = cell.source.replace(old, new)
    if replacements_applied < 8:
        raise AssertionError(
            "Tiny smoke-test settings did not match this notebook; "
            f"only {replacements_applied} setting overrides were applied."
        )
    return notebook


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_public_notebook_executes_with_tiny_settings(path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    notebook = patched_notebook(path)
    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()

    if path.name == "ou_mean_escape_time.ipynb":
        record_path = next(tmp_path.rglob("run_manifest.json"))
        arrays_path = record_path.parent / "numerical_results.npz"
        required = {
            "gx", "gy", "grid_points", "disk_mask", "reference_grid",
            "monte_carlo_anchors", "monte_carlo_exit_times",
            "monte_carlo_censored_fraction", "evaluation_interior",
            "evaluation_boundary", "run_seed",
        }
    elif path.name == "double_well_mean_escape_time.ipynb":
        record_path = next(tmp_path.rglob("run_manifest.json"))
        arrays_path = record_path.parent / "evaluation" / "fields_and_reference.npz"
        required = {"gx", "gy", "grid_points", "ellipse_mask", "reference"}
        with np.load(record_path.parent / "shared_data.npz", allow_pickle=False) as shared:
            assert {
                "anchors", "mean_exit_times", "censored_fraction",
                "evaluation_interior", "evaluation_boundary", "training_seed",
                "monte_carlo_seed",
            } <= set(shared.files)
    else:
        record_path = next(tmp_path.rglob("run_manifest.json"))
        arrays_path = record_path.parent / "numerical_results.npz"
        required = {
            "gx", "gy", "grid_points", "disk_mask",
            "monte_carlo_anchors", "monte_carlo_exit_times",
            "monte_carlo_censored_fraction", "training_interior",
            "training_boundary", "evaluation_interior", "evaluation_boundary",
            "lbfgs_interior", "lbfgs_boundary", "test_centers",
            "training_seed", "monte_carlo_seed",
        }
        methods = {"standard", "adapted", "over", "under", "variational"}
        required.update(
            key
            for method in methods
            for key in (
                f"prediction_{method}",
                f"anchor_prediction_{method}",
                f"signed_anchor_error_{method}",
                f"absolute_anchor_error_{method}",
                f"training_{method}_epoch",
            )
        )
        with np.load(record_path.parent / "training_points.npz", allow_pickle=False) as points:
            assert {
                "collocation", "evaluation", "label_points", "label_values",
                "label_censored_fraction", "training_interior", "training_boundary",
                "evaluation_interior", "evaluation_boundary", "lbfgs_interior",
                "lbfgs_boundary", "test_centers", "training_seed",
                "monte_carlo_seed",
            } <= set(points.files)

    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["schema_version"] == 2
    assert record["status"] == "complete"
    assert record["run_id"]
    assert record["created_utc"] and record["updated_utc"]
    assert "seed" in record and "monte_carlo_seed" in record
    assert arrays_path.is_file()
    assert str(arrays_path.relative_to(record_path.parent)) in record["artifacts"]
    with np.load(arrays_path, allow_pickle=False) as arrays:
        assert required <= set(arrays.files)
        methods = record.get("methods", record.get("settings", {}).get("methods", []))
        assert set(methods) == {"standard", "adapted", "over", "under", "variational"}
        if path.name != "duffing_mean_escape_time.ipynb":
            for method in methods:
                assert f"prediction_{method}" in arrays.files
                assert f"signed_error_{method}" in arrays.files
