# Boundary-adapted PINNs for elliptic Dirichlet problems

This repository accompanies [*Boundary-Adapted PINNs for Elliptic Dirichlet Problems: H²(Ω) A Priori Error Bounds with Application to Mean Escape Time Computation*](https://arxiv.org/abs/2607.19167).

The examples compare a standard PINN, three exact-boundary ansatzes, and a variational PINN with Nitsche boundary terms. The mean escape time notebooks retain the equations, loss terms, Monte Carlo labels, adaptive weights, architectures, optimization schedules, and error measures used for the paper experiments.

## Notebook order

| Open | Problem | What it shows |
|---|---|---|
| 1. [Periodic committor](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/colab/periodic_committor.ipynb) | Hitting probability on a periodic two-well domain | The most complete run, checkpoint, resume, and reporting workflow. Start here for the software workflow. |
| 2. [OU mean escape time](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/mean_escape_time/ou_ball/ou_mean_escape_time.ipynb) | Ornstein–Uhlenbeck process in a disk | The five paper methods against a known radial reference solution. |
| 3. [Double-well mean escape time](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/mean_escape_time/double_well/double_well_mean_escape_time.ipynb) | Anisotropic diffusion in a double-well potential | The five paper methods against a finite-element reference. |
| 4. [Duffing mean escape time](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/first_passage/duffing_oscillator/duffing_mean_escape_time.ipynb) | Stochastic Duffing oscillator in a disk | One boundary-adapted PINN, sample trajectories, and a Monte Carlo reference. |

Each notebook is self-contained: the mathematical problem, editable settings, method definitions, progress output, saved files, evaluation, and references are in the notebook. Use **Runtime → Change runtime type** in Colab before running setup. The `quick` setting checks the complete workflow with small counts; `paper` restores the recorded mean escape time settings and is computationally expensive.

## Results and reproducibility

The OU and double-well notebooks display the aggregate values retained from the original paper runs. Those values are historical outputs, not results produced by the reorganized notebooks. They predate the checkpoint-selection correction described in the notebooks. Rerun the paper-scale configurations in Colab before replacing or extending the reported tables.

The double-well notebook replaces the original FEniCS reference cell with a pure-Python `scikit-fem` implementation of the same weak problem. This removes the system-level FEniCS dependency. Existing paper values remain labeled as FEniCS-era outputs until the experiment is rerun.

## Local package and checks

The reusable periodic-committor and OU-disk code lives under `src/boundary_pinn/`. For a local CPU environment:

```bash
python -m pip install -e '.[dev]'
pytest -q
```

GitHub Actions performs CPU unit tests, notebook JSON checks, and Python syntax checks. These are static and small CPU checks; they do not reproduce the numerical experiments.

## References

The notebooks cite the accompanying paper and the numerical sources used for their methods and reference calculations. Machine-readable entries are collected in [`references.bib`](references.bib).

## License

MIT; see [`LICENSE`](LICENSE).
