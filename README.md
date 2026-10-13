# Boundary-adapted PINNs for elliptic Dirichlet problems

[![arXiv:2607.19167](https://img.shields.io/badge/arXiv-2607.19167-b31b1b.svg)](https://arxiv.org/abs/2607.19167) [![Tests](https://github.com/n2pakbong/boundary-adapted-pinns/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/n2pakbong/boundary-adapted-pinns/actions/workflows/tests.yml) [![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository pairs a mathematical paper with PyTorch code and runnable notebooks. The central question is how a physics-informed neural network should enforce a Dirichlet boundary condition if we want error control in the Sobolev space $H^2(\Omega)$, which measures errors in the solution and its first and second derivatives.

**Start:** [math and method](#the-mathematical-problem) · [notebooks](#notebooks) · [code map](#where-to-find-things) · [run locally](#run-locally) · [citation](#citation)

## Citation

For academic work using the method or results, cite the paper:

    @misc{tepakbong2026boundaryadapted,
      title = {Boundary-Adapted PINNs for Elliptic Dirichlet Problems: $H^2(\Omega)$ A Priori Error Bounds with Application to Mean Escape Time Computation},
      author = {Tepakbong, Nathanael and Fan, Jun and Zhou, Xiang and Zhou, Ding-Xuan},
      year = {2026},
      eprint = {2607.19167},
      archivePrefix = {arXiv},
      doi = {10.48550/arXiv.2607.19167},
      url = {https://arxiv.org/abs/2607.19167}
    }

To cite the software, use GitHub's **Cite this repository** link from [CITATION.cff](CITATION.cff). See the paper on [arXiv](https://arxiv.org/abs/2607.19167) or by [DOI](https://doi.org/10.48550/arXiv.2607.19167).

## The mathematical problem

For a diffusion $dX_t=b(X_t)\,dt+\sigma(X_t)\,dW_t$ in a domain $\Omega$, let $\tau_\Omega$ be the first time the process leaves $\Omega$. Its mean escape time is $\tau(x)=\mathbb E_x[\tau_\Omega]$. It solves the elliptic boundary-value problem
$$
\mathcal L\tau=-1\quad\text{in }\Omega,\qquad \tau=0\quad\text{on }\partial\Omega,
$$
Here $b$ is the drift, $\sigma$ is the diffusion matrix, and $W_t$ is standard Brownian motion. The generator is
$
\mathcal L f(x)=\sum_i b_i(x)\partial_i f(x)+\frac12\sum_{i,j}(\sigma\sigma^\top)_{ij}(x)\partial_{ij}f(x).
$
See the [paper](https://arxiv.org/html/2607.19167) for its assumptions and full error bounds.

A boundary-adapted network represents the approximation as
$$
\tau_\theta(x)=\rho(x)N_\theta(x),
$$
where $N_\theta$ is a neural network and $\rho$ is a smooth approximation of distance to the boundary. Since $\rho=0$ on $\partial\Omega$, the boundary value is satisfied exactly. The paper shows that vanishing on the boundary is not enough for its $H^2$ error theory: the distance factor must also be normalized so its inward normal derivative is one, $\partial_\nu\rho=1$. Here $\nu$ is the inward unit normal. This normalization keeps $\tau/\rho$ smooth up to the boundary under the paper's regularity assumptions. The theorem bounds minimizers of the sampled residual objective; it does not show that a training optimizer reaches one.

The mean-escape-time notebooks compare a standard PINN with a boundary penalty, the adapted factor $\rho$, stronger and weaker factors $\rho^{3/2}$ and $\rho^{1/2}$, and a variational PINN whose weak Nitsche loss imposes the boundary condition variationally. Their reported relative errors have the form
$$
E_{H^k}=\frac{\|\tau_\theta-\tau_{\mathrm{ref}}\|_{H^k(\Omega)}}{\|\tau_{\mathrm{ref}}\|_{H^k(\Omega)}},\qquad k=0,1,2,
$$
Here $\tau_{\mathrm{ref}}$ is the reference solution and $k=0$ gives the $L^2$ error. The $H^2$ comparison also measures second-derivative error. The OU example uses an analytical reference; the double-well example uses a numerical reference.

## Notebooks

| Notebook | Problem |
| --- | --- |
| [OU mean escape time](experiments/mean_escape_time/ou_ball/ou_mean_escape_time.ipynb) · [Open in Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/mean_escape_time/ou_ball/ou_mean_escape_time.ipynb) | Exit from a disk; analytical reference. |
| [Double-well mean escape time](experiments/mean_escape_time/double_well/double_well_mean_escape_time.ipynb) · [Open in Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/mean_escape_time/double_well/double_well_mean_escape_time.ipynb) | Anisotropic diffusion; numerical reference. |
| [Periodic committor](colab/periodic_committor.ipynb) · [Open in Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/colab/periodic_committor.ipynb) · [Guide](experiments/committor/periodic_two_well/README.md) | The probability of hitting target $B$ before target $A$: $\mathcal Lq=0$ away from the targets, with $q=0$ on $A$ and $q=1$ on $B$. |
| [Duffing first passage](experiments/first_passage/duffing_oscillator/duffing_mean_escape_time.ipynb) · [Open in Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/first_passage/duffing_oscillator/duffing_mean_escape_time.ipynb) | Exit time for a stochastic oscillator. |

In Colab, choose a CPU or GPU runtime and run the setup cells. Outputs are saved to Google Drive by default.

## Where to find things

The workflow is **notebook and settings → problem definition → experiment runner → metrics, plots, and checkpoints**.

| If you want to… | Start here |
| --- | --- |
| Change a network | [models.py](src/boundary_pinn/models.py) |
| Change a PDE, geometry, or sampler | [problem modules](src/boundary_pinn/problems/) |
| Trace preparation, training, evaluation, or plots | [experiment runners](src/boundary_pinn/experiments/) |
| Find notebooks and experiment settings | [experiments](experiments/) and [Colab notebooks](colab/) |
| Check behavior and dependencies | [tests](tests/), [GitHub Actions](.github/workflows/tests.yml), and [pyproject.toml](pyproject.toml) |

For a worked code path, follow the [committor guide](experiments/committor/periodic_two_well/README.md) to the [problem definition](src/boundary_pinn/problems/committor/periodic_two_well.py), [training methods](src/boundary_pinn/problems/committor/training.py), and [comparison runner](src/boundary_pinn/experiments/committor/periodic_two_well/comparison.py). The guide links the code that generates Monte Carlo labels and evaluates the reference solution.

## Run locally

Use Python 3.10 or newer. Install the notebook and test dependencies, then run the CPU checks:

    python -m pip install -e ".[dev,notebooks]"
    python -m pytest -q

The base package alone installs with `python -m pip install -e .`. Its optional notebook dependencies include scikit-fem. Tests check code behavior; they do not run full experiments or establish numerical accuracy.

## Reuse the code

Start with the problem module closest to your PDE. A new problem needs Python implementations of its operator, geometry, domain and boundary samplers, and reference solution, connected to an experiment runner; changing a settings file alone is not enough. The shared interface is still evolving.

The committor notebook generates fresh training and Monte Carlo seeds for a new run and saves them in its configuration. Its dated Drive folder contains the settings, reference and label data, metrics, figures, and checkpoints. The mean-escape-time aggregates shown in the notebooks are historical outputs; later code fixes mean a fresh run will not reproduce them exactly.

## License

The code is released under the [MIT license](LICENSE).

**Last updated: 2026-10-14 03:02 +08:00.**
