import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "colab" / "periodic_committor.ipynb",
    ROOT / "experiments" / "mean_escape_time" / "ou_ball" / "ou_mean_escape_time.ipynb",
    ROOT / "experiments" / "mean_escape_time" / "double_well" / "double_well_mean_escape_time.ipynb",
    ROOT / "experiments" / "first_passage" / "duffing_oscillator" / "duffing_mean_escape_time.ipynb",
]


def python_source(cell):
    source = cell.get("source", "")
    if isinstance(source, list):
        source = "".join(source)
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            continue
        lines.append(line)
    return "\n".join(lines)


def test_primary_notebooks_are_valid_json_and_python():
    for path in NOTEBOOKS:
        with path.open(encoding="utf-8") as handle:
            notebook = json.load(handle)
        assert notebook["nbformat"] == 4
        assert isinstance(notebook.get("cells"), list) and notebook["cells"]
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = python_source(cell)
            try:
                ast.parse(source, filename=f"{path}:{index}")
            except SyntaxError as error:
                raise AssertionError(f"Invalid Python in {path}, cell {index}: {error}") from error


def test_new_notebook_runs_draw_fresh_seeds_and_unique_folders():
    sources = {
        path.name: "\n".join(
            python_source(cell)
            for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]
            if cell.get("cell_type") == "code"
        )
        for path in NOTEBOOKS
    }
    committor = sources["periodic_committor.ipynb"]
    assert "SEED = secrets.randbelow(2**31)" in committor
    assert "MONTE_CARLO_SEED = secrets.randbelow(2**31)" in committor
    assert "secrets.token_hex(4)" in committor

    ou = sources["ou_mean_escape_time.ipynb"]
    assert "SEED = secrets.randbelow(2**32)" in ou
    assert "secrets.token_hex(4)" in ou

    for filename in ("double_well_mean_escape_time.ipynb", "duffing_mean_escape_time.ipynb"):
        source = sources[filename]
        assert "SEED = secrets.randbits(32)" in source
        assert "MC_SEED = secrets.randbits(32)" in source
        assert "run_stamp" in source
