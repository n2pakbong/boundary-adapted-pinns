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

The `[FINAL]` OU-ball and double-well notebooks are preserved as the source records for the paper's comparisons. The additional double-well conjugate-gradient optimizer notebook and the distinct medium-size OU study are also retained. Earlier OU and Duffing experiments remain available as separate scientific material. Notebook Markdown, derivations, citations, code, and saved outputs have been kept with the notebooks.

## Python implementation

Install the package and its test dependencies in a Python 3.10+ environment:

```bash
pip install -e '.[dev]'
pytest -q
```

The package name is `boundary-pinn`; its import name is `boundary_pinn`. Shared network architectures and checkpoint utilities live at the package root. Problem definitions are grouped by equation type. Experiment-specific training and validation code remains identifiable with the problem it studies.

The current Python examples are an Ornstein–Uhlenbeck mean escape time problem on a disk and a periodic two-well committor. The code does not yet provide a turnkey solver for arbitrary PDEs. New PDE studies can reuse the shared architectures and numerical utilities while adding their own operator, domain, boundary conditions, experiment settings, tests, and reference solution.

## Running the periodic committor example

The complete notebook explains each stage and saves results to Google Drive:

[`colab/periodic_committor.ipynb`](colab/periodic_committor.ipynb)

The CPU preflight exercises the full training, reference, checkpoint, and plotting pipeline with tiny settings. It does not use a GPU and is a wiring check. The GPU pilot is a separate cell in the same notebook and runs only when started explicitly.

For a local CPU preflight:

```bash
periodic-committor experiments/committor/periodic_two_well/configs/preflight_comparison_cpu.json \
  --output-dir runs/periodic_committor/preflight
```

For larger runs, edit or copy a versioned configuration and choose a result directory outside the source tree. Each run records its configuration, code revision, metrics, checkpoints, reference data, and figures. The finite-difference reference is checked at multiple grid resolutions; it is a numerical comparator whose convergence should be examined before interpreting errors.

When this repository is private, the Colab notebook uses a `GITHUB_TOKEN` Colab Secret to fetch the source from GitHub. That token is only for repository access; it is not used for training or Google Drive. Public access to the repository removes the need for a token.

## Citation

If this repository supports your work, cite the accompanying paper:

> N. Pakbong, “Boundary-Adapted PINNs for Elliptic Dirichlet Problems: H²(Ω) A Priori Error Bounds with Application to Mean Escape Time Computation,” 2026. [arXiv:2607.19167](https://arxiv.org/abs/2607.19167).

## License

MIT; see [LICENSE](LICENSE).
