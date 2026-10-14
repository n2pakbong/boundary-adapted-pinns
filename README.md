# Boundary-Adapted PINNs for Elliptic Dirichlet Problems

[![arXiv:2607.19167](https://img.shields.io/badge/arXiv-2607.19167-b31b1b.svg)](https://arxiv.org/abs/2607.19167) [![Tests](https://github.com/n2pakbong/boundary-adapted-pinns/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/n2pakbong/boundary-adapted-pinns/actions/workflows/tests.yml) [![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository contains the code and experiment notebooks for our work on boundary-adapted physics-informed neural networks (PINNs). The paper studies second-order uniformly elliptic Dirichlet problems, proves a priori error bounds in $H^2(\Omega)$, and tests the method on mean escape time problems.

**Paper:** [arXiv:2607.19167](https://arxiv.org/abs/2607.19167) · [DOI: 10.48550/arXiv.2607.19167](https://doi.org/10.48550/arXiv.2607.19167)

## Citation

If you use the method or the experiments, please cite the paper:

```bibtex
@misc{tepakbong2026boundary,
  author = {Tepakbong, Nathanael and Fan, Jun and Zhou, Xiang and Zhou, Ding-Xuan},
  title = {{Boundary-Adapted PINNs for Elliptic Dirichlet Problems: $H^2(\Omega)$ A Priori Error Bounds with Application to Mean Escape Time Computation}},
  year = {2026},
  eprint = {2607.19167},
  archivePrefix = {arXiv},
  primaryClass = {math.NA},
  doi = {10.48550/arXiv.2607.19167},
  url = {https://arxiv.org/abs/2607.19167}
}
```

## The mathematical idea

A Dirichlet problem asks for a function $u$ that solves an equation inside a domain and takes prescribed values on its boundary:

$$
\begin{cases}
\mathscr L u=f & \text{in }\Omega,\\
u=g & \text{on }\partial\Omega.
\end{cases}
$$

A PINN represents $u$ with a neural network and trains it to reduce the equation's residual at sampled points. The boundary-adapted form used in the paper is

$$
u_\theta(x)=G(x)+\rho(x)v_\theta(x),
\qquad G=g \text{ on }\partial\Omega,\quad \rho=0 \text{ on }\partial\Omega.
$$

Here $G$ extends the boundary data into the domain, $\rho$ vanishes on the boundary, and $v_\theta$ is a neural network. Thus the boundary values hold for every $\theta$. The paper studies how the choice of $\rho$ affects approximation and $H^2(\Omega)$ error bounds.

## Start with an example

Each example has a guide beside its notebook or configuration. The guide explains the problem, how the reference is computed, and where the results are saved.

| Example | What it computes | Start here |
| --- | --- | --- |
| Ornstein–Uhlenbeck mean escape time | Expected exit time from a disk, compared with an analytic solution. | [Guide](experiments/mean_escape_time/ou_ball/README.md) · [Notebook](experiments/mean_escape_time/ou_ball/ou_mean_escape_time.ipynb) · [Open in Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/mean_escape_time/ou_ball/ou_mean_escape_time.ipynb) |
| Double-well mean escape time | Mean exit time for an anisotropic diffusion, with a finite-element reference. | [Guide](experiments/mean_escape_time/double_well/README.md) · [Notebook](experiments/mean_escape_time/double_well/double_well_mean_escape_time.ipynb) · [Open in Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/mean_escape_time/double_well/double_well_mean_escape_time.ipynb) |
| Periodic two-well committor | Probability of reaching target $B$ before target $A$, with Monte Carlo labels and a finite-difference reference. | [Guide](experiments/committor/periodic_two_well/README.md) · [Notebook](colab/periodic_committor.ipynb) · [Open in Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/colab/periodic_committor.ipynb) |
| Duffing oscillator first passage | A stochastic first-passage example with trajectory simulation and a PINN. | [Guide](experiments/first_passage/duffing_oscillator/README.md) · [Notebook](experiments/first_passage/duffing_oscillator/duffing_mean_escape_time.ipynb) |
| OU disk package example | A small CPU check of the problem and command-line runner. | [Guide and configuration](experiments/mean_escape_time/ou_disk/README.md) |

## Find and adapt the code

The reusable Python package is in `src/boundary_pinn/`. Its problem modules define the differential operator, domain and boundary geometry, samplers, and—where available—a reference solution. Training modules define the trial functions and losses. Experiment modules prepare shared data, run method comparisons, evaluate errors, and save results.

The `experiments/` directory groups notebooks, guides, and settings by problem. The `colab/` directory contains the periodic committor notebook. The `tests/` directory checks core calculations and notebook structure; the tests do not run the full GPU experiments or establish their numerical accuracy.

To adapt the code to another PDE, start with the closest problem module: define its operator and geometry, then implement its interior and boundary samplers. Add a reference solver if reference-based errors are needed, and connect the problem to an experiment runner. Existing JSON configurations change settings for supported problems; they do not define a new PDE or geometry by themselves. The [periodic committor guide](experiments/committor/periodic_two_well/README.md) traces one problem from its equation through training and saved results.

For Colab runs, choose the runtime before running setup, then follow the cells in order. The guides describe each notebook's save and resume options; for long runs, use its documented persistent-output setting and keep the printed run folder with the results.

## Run locally

Use Python 3.10 or newer:

```bash
python -m pip install -e ".[dev,notebooks]"
python -m pytest -q
```

The `notebooks` extra provides `scikit-fem` for the finite-element reference. For the base package, use `python -m pip install -e .`. The test suite is a CPU check, not a substitute for running or validating a full experiment.

## License

The code is released under the [MIT license](LICENSE).
