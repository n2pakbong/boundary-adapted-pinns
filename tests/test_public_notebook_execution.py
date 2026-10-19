from pathlib import Path

import json
import re
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


TINY_SETTINGS = {
    "ou_mean_escape_time.ipynb": {
        "training_collocation_points": "32",
        "training_boundary_points": "16",
        "evaluation_interior_points": "32",
        "evaluation_boundary_points": "16",
        "reference_grid_n": "24",
        "data_anchor_fraction": "3 / 32",
        "paths_per_anchor": "3",
        "mc_dt": "0.02",
        "mc_max_time": "0.04",
        "adam_epochs": "1",
        "n_test_functions": "8",
        "trajectory_dt": "0.02",
        "trajectory_max_time": "0.04",
    },
    "double_well_mean_escape_time.ipynb": {
        "training_collocation_points": "32",
        "training_boundary_points": "16",
        "evaluation_interior_points": "32",
        "evaluation_boundary_points": "16",
        "evaluation_grid_n": "24",
        "data_anchor_fraction": "3 / 32",
        "paths_per_anchor": "3",
        "mc_dt": "0.02",
        "mc_max_time": "0.04",
        "fem_refinements": "2",
        "adam_epochs": "1",
        "n_test_functions": "8",
        "trajectory_dt": "0.02",
        "trajectory_max_time": "0.04",
    },
    "duffing_mean_escape_time.ipynb": {
        "training_collocation_count": "32",
        "training_boundary_count": "16",
        "evaluation_interior_count": "32",
        "evaluation_boundary_count": "16",
        "lbfgs_interior_count": "32",
        "lbfgs_boundary_count": "16",
        "mc_anchor_fraction": "3 / 32",
        "mc_paths": "3",
        "mc_dt": "0.02",
        "mc_max_time": "0.04",
        "adam_epochs": "1",
        "n_test_functions": "8",
        "evaluation_grid_n": "24",
        "trajectory_dt": "0.02",
        "trajectory_max_time": "0.04",
    },
}


def patched_notebook(path):
    notebook = nbformat.read(path, as_version=4)
    overrides = TINY_SETTINGS[path.name]
    for name, value in overrides.items():
        pattern = re.compile(rf"(?m)^([ ]*{re.escape(name)} *= *).*$")
        matches = 0
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            cell.source, count = pattern.subn(lambda match: match.group(1) + value, cell.source)
            matches += count
        if matches != 1:
            raise AssertionError(
                f"Expected one editable setting named {name!r} in {path.name}; found {matches}."
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
