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
    "SAVE_TO_DRIVE = True": "SAVE_TO_DRIVE = False",
    "RUN_TRAINING = False": "RUN_TRAINING = True",
    "pool_size, batch_size, adam_epochs = 512, 128, 40": "pool_size, batch_size, adam_epochs = 32, 8, 1",
    "data_anchors, paths_per_anchor = 24, 24": "data_anchors, paths_per_anchor = 3, 3",
    "mc_dt, mc_max_time = 5e-3, 2.0": "mc_dt, mc_max_time = 0.02, 0.04",
    "pool_size, batch_size, adam_epochs = 384, 96, 30": "pool_size, batch_size, adam_epochs = 32, 8, 1",
    "data_anchors, paths_per_anchor = 20, 20": "data_anchors, paths_per_anchor = 3, 3",
    "mc_dt, mc_max_time = 2e-3, 2.0": "mc_dt, mc_max_time = 0.02, 0.04",
    "fem_refinements = 4": "fem_refinements = 2",
    "collocation_count, batch_size = 512, 128": "collocation_count, batch_size = 32, 8",
    "adam_epochs, number_of_decays = 40, 20": "adam_epochs, number_of_decays = 1, 1",
    "mc_anchors, mc_paths, mc_dt, mc_max_time = 20, 32, 2e-3, 5.0": "mc_anchors, mc_paths, mc_dt, mc_max_time = 3, 3, 0.02, 0.04",
    "grid_n = 200": "grid_n = 24",
    "grid_n = 180": "grid_n = 24",
    "reference_grid_n = 200": "reference_grid_n = 24",
    "evaluation_grid_n = 200": "evaluation_grid_n = 24",
    "evaluation_grid_n = 180": "evaluation_grid_n = 24",
}


def patched_notebook(path):
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for old, new in REPLACEMENTS.items():
            cell.source = cell.source.replace(old, new)
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
        arrays_path = record_path.parent / "evaluation_field.npz"
        required = {
            "gx", "gy", "points", "mask", "field", "anchors", "prediction",
            "reference", "censored_fraction", "signed_anchor_error",
            "absolute_anchor_error",
        }
        with np.load(record_path.parent / "training_points.npz", allow_pickle=False) as points:
            assert {"collocation", "evaluation", "training_seed", "monte_carlo_seed"} <= set(points.files)

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
        if path.name != "duffing_mean_escape_time.ipynb":
            methods = record.get("methods", record.get("settings", {}).get("methods", []))
            for method in methods:
                assert f"prediction_{method}" in arrays.files
                assert f"signed_error_{method}" in arrays.files
