# Boundary-adapted PINNs for elliptic PDEs

This repository studies neural approximations for elliptic partial differential equations with Dirichlet boundary conditions. It accompanies [Boundary-Adapted PINNs for Elliptic Dirichlet Problems: H²(Ω) A Priori Error Bounds with Application to Mean Escape Time Computation](https://arxiv.org/abs/2607.19167).

The paper applies the method to mean escape time equations for an Ornstein–Uhlenbeck process in a ball and a two-dimensional double-well system. The original experiment notebooks include the derivations, explanations, references, training procedures, and result plots used in those studies. A periodic two-well committor is included as a separate example of a multiply connected Dirichlet problem.

## Navigate the repository

| Topic | Location |
|---|---|
| Shared neural architectures and numerical utilities | `src/boundary_pinn/` |
| Mean escape time equations and examples | `src/boundary_pinn/problems/mean_escape_time/` |
| Committor equations and examples | `src/boundary_pinn/problems/committor/` |
| Source notebooks for the OU-ball paper comparison | `experiments/mean_escape_time/ou_ball/notebooks/` |
| Source notebooks for the double-well paper comparison | `experiments/mean_escape_time/double_well/notebooks/` |
| Earlier OU parameter and training studies | `experiments/mean_escape_time/ou_process/` |
| Periodic two-well committor example | `experiments/committor/periodic_two_well/` and `colab/periodic_committor.ipynb` |
| Duffing oscillator first-passage studies | `experiments/first_passage/duffing_oscillator/` |
| Figures and error-table notebooks for the paper | `experiments/mean_escape_time/paper_results/` |

The OU-ball and double-well paper comparisons each retain five method notebooks: standard, variational, over-enforced, under-enforced, and boundary-adapted PINNs. The `paper_results/` folder contains the figure and error-table notebooks. The earlier OU-process studies retain a standard-PINN example, a boundary-fitted-PINN example, and parameter-influence plots. The Duffing first-passage example is presented through a boundary-adapted PINN, a penalty-PINN baseline, and a trajectory visualization.

## Python implementation

Install the package and its test dependencies in a Python 3.10+ environment:

```bash
pip install -e '.[dev]'
pytest -q
```

The package name is `boundary-pinn`; its import name is `boundary_pinn`. Shared network architectures and checkpoint utilities live at the package root. Problem definitions are grouped by equation type. Experiment-specific training and validation code remains identifiable with the problem it studies. GitHub Actions runs the CPU test suite on each pull request; these checks test the proposed branch without merging it and never request a GPU.

The current Python examples are an Ornstein–Uhlenbeck mean escape time problem on a disk and a periodic two-well committor. The code does not yet provide a turnkey solver for arbitrary PDEs. New PDE studies can reuse the shared architectures and numerical utilities while adding their own operator, domain, boundary conditions, experiment settings, tests, and reference solution.

## Running the periodic committor example

The periodic committor is the maintained PINN example for a PDE with two separated Dirichlet boundaries. Its one Colab notebook explains the equation, boundary construction, CPU checks, GPU pilot, saved outputs, and how to resume:

[Open the periodic committor notebook in Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/colab/periodic_committor.ipynb)

Run Setup and the CPU preflight before starting the GPU pilot. The pilot is a single-seed exploratory comparison, not a paper result. It writes the configuration, metrics, checkpoints, finite-difference reference, summary, and diagnostic plots to a timestamped folder in Google Drive. The notebook and experiment guide define all five paper formulations, their PDE, Monte Carlo data, standard boundary-residual, and Nitsche losses, and report both final loss components and absolute/relative L2, H1, and grid-estimated H2 solution errors against the finite-difference reference. The saved-checkpoint evaluation can be rerun on CPU, without repeating training.

For local CPU testing:

```bash
pip install -e '.[dev]'
pytest -q
```

For a small local end-to-end preflight:

```bash
periodic-committor experiments/committor/periodic_two_well/configs/cpu_preflight.json \
  --output-dir runs/periodic_committor/preflight
```

The experiment implementation is organized by purpose:

- `src/boundary_pinn/problems/committor/periodic_two_well.py` defines the PDE, periodic geometry, boundary sampling, and residual.
- `src/boundary_pinn/problems/committor/training.py` defines the five named formulations, adaptive loss objective, live component logs, and restartable checkpoints.
- `src/boundary_pinn/experiments/committor/periodic_two_well/comparison.py` runs the configured comparison, finite-difference reference, summary, and plots.
- `experiments/committor/periodic_two_well/configs/` contains the CPU preflight and GPU pilot settings.

See [the committor experiment guide](experiments/committor/periodic_two_well/README.md) for the output file map, metric definitions, provenance, and paper-result workflow.

## Citation

If this repository supports your work, cite the accompanying paper:

*Boundary-Adapted PINNs for Elliptic Dirichlet Problems: H²(Ω) A Priori Error Bounds with Application to Mean Escape Time Computation*, arXiv:2607.19167 (2026). [https://arxiv.org/abs/2607.19167](https://arxiv.org/abs/2607.19167).

## License

MIT; see [LICENSE](LICENSE).
