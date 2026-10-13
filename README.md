# Boundary-adapted PINNs for elliptic Dirichlet problems

[![arXiv:2607.19167](https://img.shields.io/badge/arXiv-2607.19167-b31b1b.svg)](https://arxiv.org/abs/2607.19167) [![Tests](https://github.com/n2pakbong/boundary-adapted-pinns/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/n2pakbong/boundary-adapted-pinns/actions/workflows/tests.yml) [![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository accompanies work on boundary-adapted physics-informed neural networks (PINNs) for second-order uniformly elliptic Dirichlet problems. The paper proves a priori $H^2(\Omega)$ error bounds, and the code and notebooks illustrate the method on mean escape time and a periodic committor.

**Start here:** [the equations and method](#the-problem-and-the-method) · [notebooks](#notebooks) · [how the code is organized](#how-the-code-is-organized) · [run locally](#run-locally) · [citation](#citation)

## Citation

To cite the method or results, cite the [paper on arXiv](https://arxiv.org/abs/2607.19167) or its [DOI](https://doi.org/10.48550/arXiv.2607.19167). To cite the software, use GitHub's **Cite this repository** link from [CITATION.cff](CITATION.cff).

<details>
<summary>BibTeX for the paper</summary>

```bibtex
@misc{tepakbong2026boundaryadapted,
  title = {Boundary-Adapted PINNs for Elliptic Dirichlet Problems: $H^2(\Omega)$ A Priori Error Bounds with Application to Mean Escape Time Computation},
  author = {Tepakbong, Nathanael and Fan, Jun and Zhou, Xiang and Zhou, Ding-Xuan},
  year = {2026},
  eprint = {2607.19167},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2607.19167},
  url = {https://arxiv.org/abs/2607.19167}
}
```

</details>
## The problem and the method

The paper considers second-order uniformly elliptic Dirichlet problems. Given an operator $\mathscr L$, forcing $f$, and boundary data $g$, the goal is to find $u^\star$ such that
$$
\begin{cases}
\mathscr L u^\star=f & \text{in }\Omega,\\
u^\star=g & \text{on }\partial\Omega.
\end{cases}
$$
A PINN approximates $u^\star$ by minimizing the equation's residual at sampled points. The boundary-adapted trial function is
$$
u_\theta(x)=g(x)+\rho(x)v_\theta(x),
$$
where $g$ extends the boundary data into the domain, $\rho$ vanishes on the boundary, and $v_\theta$ is a neural network. This construction makes $u_\theta=g$ on $\partial\Omega$ for every choice of network parameters.

The paper proves a priori $H^2(\Omega)$ error bounds. The $H^2$ norm measures error in the solution and its first and second derivatives.
## Notebooks

| Example | What it computes |
| --- | --- |
| [OU mean escape time](experiments/mean_escape_time/ou_ball/ou_mean_escape_time.ipynb) · [Open in Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/mean_escape_time/ou_ball/ou_mean_escape_time.ipynb) | Mean exit time from a disk, with an analytical reference. |
| [Double-well mean escape time](experiments/mean_escape_time/double_well/double_well_mean_escape_time.ipynb) · [Open in Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/mean_escape_time/double_well/double_well_mean_escape_time.ipynb) | Anisotropic diffusion, with a numerical reference. |
| [Periodic committor](colab/periodic_committor.ipynb) · [Open in Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/colab/periodic_committor.ipynb) · [Guide](experiments/committor/periodic_two_well/README.md) | Probability of reaching one target before another on a periodic domain. |
| [Duffing first passage](experiments/first_passage/duffing_oscillator/duffing_mean_escape_time.ipynb) | First-passage time for a stochastic oscillator. |

In Colab, select a CPU or GPU runtime and run the setup cells. The notebooks save outputs to Google Drive by default.

## How the code is organized

The reusable implementation is under `src/boundary_pinn/`: problem modules define the differential operator, geometry, boundary sampling, and reference solution; experiment modules prepare data, train models, and evaluate comparisons. The `experiments/` directory puts each problem's notebook, settings, and explanatory guide together. The `colab/` directory contains the periodic-committor entry notebook. Tests live in `tests/`.

For a full example of how these pieces connect, follow the [periodic committor guide](experiments/committor/periodic_two_well/README.md) to its [problem definition](src/boundary_pinn/problems/committor/periodic_two_well.py), [training methods](src/boundary_pinn/problems/committor/training.py), and [comparison runner](src/boundary_pinn/experiments/committor/periodic_two_well/comparison.py). For a mean-escape-time run, start with the [OU guide](experiments/mean_escape_time/ou_ball/README.md) beside its notebook.

To adapt the code to a new PDE, start from the closest problem module and update its operator, domain, and boundary samplers, then connect it to an experiment runner. Add a reference solution if you need reference-based error metrics. The [committor guide](experiments/committor/periodic_two_well/README.md) follows these pieces through one working example; changing a settings file alone does not add support for a new geometry or equation.

## Run locally

Use Python 3.10 or newer. Install the notebook and test dependencies, then run the CPU checks:

    python -m pip install -e ".[dev,notebooks]"
    python -m pytest -q

The base package installs with `python -m pip install -e .`. Tests check code behavior; they do not run full experiments or establish numerical accuracy.

## License

The code is released under the [MIT license](LICENSE).

**Last updated: 2026-10-14 03:19 +08:00.**
