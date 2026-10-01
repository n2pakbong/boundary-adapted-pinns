from pathlib import Path

import nbformat
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
