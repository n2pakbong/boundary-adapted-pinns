# Boundary-adapted PINNs

Research code for boundary-adapted neural networks on elliptic PDEs, motivated by mean escape time problems. The repository contains both the original paper experiment notebooks and a newer Python implementation. The committor is one example; it is not the scope of the repository.

## Start here

| If you want to… | Open… |
|---|---|
| Find the maintained Python implementation | `met_pinn/` |
| See the source notebooks for the 2026 paper experiments | `OU_process_2D/Models/` and `Double_Well_2D/Models/` |
| Run the newer periodic committor example in Colab | `colab/periodic_committor.ipynb` |
| Find the PDE definitions used by the newer implementation | `met_pinn/problems/` |

## Paper experiment source

The paper's numerical section studies an Ornstein–Uhlenbeck process in a ball and a two-dimensional double-well process. It compares standard, variational, over-enforced, under-enforced, and boundary-adapted PINNs. The original notebooks for these runs are kept in the problem folders above; they are the source record for the paper experiments.

- `OU_process_2D/Models/` contains the five OU method notebooks marked `[FINAL]`.
- `Double_Well_2D/Models/` contains the five double-well method notebooks marked `[FINAL]`. The additional `boundary_adapted_engd_cg.ipynb` records a distinct optimizer variant and is retained.
- `error_tables.ipynb` turns saved OU and double-well metrics into LaTeX tables. It expects `ou.txt` and `dw.txt` in its working directory.
- `plots.ipynb` is a small image-cropping helper for comparing prediction and reference panels.

The `[WIP]` double-well notebooks were drafts superseded by the corresponding `[FINAL]` notebooks and have been removed. The empty `exponential_pinn.ipynb` placeholder and empty Duffing placeholder files have also been removed.

`OU_process/` contains earlier OU parameter and curriculum experiments. `Duffin_oscillator/` contains a separate first-passage example. They are retained as distinct research material; neither is the current committor workflow.

## Maintained Python implementation

- `met_pinn/models.py`: neural architectures, including exact boundary-factor ansatzes.
- `met_pinn/problems/`: PDE operators, domain geometry, boundary data, and sampling for the current OU disk and periodic committor implementations.
- `met_pinn/training.py`: their training loops and restartable checkpoints.
- `met_pinn/periodic_comparison.py`: the complete periodic committor comparison, reference calculation, metrics, and figures.
- `met_pinn/reference.py`: the committor finite-difference reference solver.
- `met_pinn/cli.py`: the OU disk prototype command-line runner.
- `configs/`: versioned run settings; the periodic committor has one CPU preflight and one GPU pilot.
- `tests/`: tests for the current Python implementations.

This Python implementation is the starting point for future PDE work, not yet a general solver for arbitrary PDEs. Its training code still has problem-specific paths. For a new PDE, define the operator, domain, boundary conditions, and sampling in `met_pinn/problems/`; add a reference or validation method, a configuration under `configs/<problem>/`, and tests under `tests/`. Then connect it to the runner. The aim is to reuse the network, optimization, checkpointing, and plotting machinery instead of copying a full notebook for each PDE or method.

## Run the periodic committor example

Open the [Colab notebook from `main`](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/colab/periodic_committor.ipynb). Run setup, then the CPU preflight if this code version has not been checked, then the GPU pilot. Setup uses the saved `GITHUB_TOKEN` Colab Secret only to clone the private repository in a fresh runtime. The resume cell is only for a pilot interrupted before completion.

The CPU preflight runs the complete small pipeline without GPU computation. The GPU pilot trains three committor methods for 6,000 epochs each and saves results to Google Drive. This pilot is not a paper-scale result.

## Local checks

With Python 3.10 or later:

```bash
pip install -e '.[dev]'
pytest -q
```

These tests cover the current Python code. Each new PDE still needs its own mathematical validation and reference checks.

## License

MIT; see `LICENSE`.