# Boundary-adapted PINNs for elliptic Dirichlet problems

This repository accompanies [*Boundary-Adapted PINNs for Elliptic Dirichlet Problems: H²(Ω) A Priori Error Bounds with Application to Mean Escape Time Computation*](https://arxiv.org/abs/2607.19167).

The examples compare a standard PINN, three exact-boundary ansatzes, and a variational PINN with Nitsche boundary terms. The mean escape time notebooks retain the equations, loss terms, Monte Carlo data construction, adaptive weights, architectures, optimization schedules, and error measures used for the paper experiments.

## Notebook order

| Open | Problem | What it shows |
|---|---|---|
| 1. [Periodic committor](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/colab/periodic_committor.ipynb) | Hitting probability on a periodic two-well domain | The most complete run, checkpoint, resume, and reporting workflow. Start here for the software workflow. |
| 2. [OU mean escape time](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/mean_escape_time/ou_ball/ou_mean_escape_time.ipynb) | Ornstein–Uhlenbeck process in a disk | The five paper methods against a known radial reference solution. |
| 3. [Double-well mean escape time](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/mean_escape_time/double_well/double_well_mean_escape_time.ipynb) | Anisotropic diffusion in a double-well potential | The five paper methods against a finite-element reference. |
| 4. [Duffing mean escape time](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/first_passage/duffing_oscillator/duffing_mean_escape_time.ipynb) | Stochastic Duffing oscillator in a disk | One boundary-adapted PINN, sample trajectories, and a Monte Carlo reference. |

Each notebook is self-contained: the mathematical problem, editable settings, method definitions, progress output, saved files, evaluation, and references are in the notebook. The ready-to-run default is the committor notebook's shallow two-hidden-layer, width-64 `tanh` network. The visible model class or factory can be replaced by any scalar-output PyTorch architecture; it is not restricted to a preset menu. Use **Runtime → Change runtime type** in Colab before running setup. The `quick` setting checks the complete workflow with small counts; `paper` restores the recorded OU and double-well settings, while `full` restores the Duffing source architecture and training schedule. These runs are computationally expensive. The setup cells mount Google Drive by default so labels and checkpoints persist; set `SAVE_TO_DRIVE = False` to use temporary runtime storage.

## Results and reproducibility

The OU and double-well notebooks display the aggregate error and training-loss values retained from the original paper runs. Those values are historical outputs, not results produced by the reorganized notebooks. They predate the checkpoint-selection and variational array-shape corrections; the double-well values also predate the corrected ellipse arclength weights. Rerun the paper-scale configurations in Colab before replacing or extending the reported tables. The Duffing notebook has no bundled trained result and makes no reproduction claim.

The double-well notebook replaces the original FEniCS reference cell with a pure-Python `scikit-fem` implementation of the same weak problem. This removes the system-level FEniCS dependency. Existing paper values remain labeled as FEniCS-era outputs until the experiment is rerun.

## Local package and checks

Reusable periodic-committor, OU-disk, and single-trajectory code lives under `src/boundary_pinn/`. The trajectory helpers cover the OU, double-well, and Duffing systems and can be reused for explanatory plots. For a local CPU environment:

```bash
python -m pip install -e '.[dev]'
pytest -q
```

GitHub Actions performs CPU unit tests, notebook JSON and syntax checks, and tiny end-to-end executions of the three mean-escape-time notebooks. These checks exercise the code paths with reduced counts; they do not reproduce the numerical experiments.

## References

The notebooks cite the accompanying paper and the numerical sources used for their methods and reference calculations. Machine-readable entries are collected in [`references.bib`](references.bib).

## License

MIT; see [`LICENSE`](LICENSE).
