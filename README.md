# Boundary-adapted PINNs for elliptic Dirichlet problems

[![arXiv:2607.19167](https://img.shields.io/badge/arXiv-2607.19167-b31b1b.svg)](https://arxiv.org/abs/2607.19167) [![Tests](https://github.com/n2pakbong/boundary-adapted-pinns/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/n2pakbong/boundary-adapted-pinns/actions/workflows/tests.yml) [![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository studies how a physics-informed neural network (PINN) should impose Dirichlet boundary values when accuracy in the solution's derivatives matters. Its central construction builds the boundary condition into the network and uses a first-order normalized distance factor; the paper proves why that extra normalization matters for its $H^2(\Omega)$ error bounds. The repository contains the paper's mean-escape-time code and notebooks, plus a periodic committor example.

**Start here:** [the equations and method](#the-problem-and-the-method) · [notebooks](#notebooks) · [how the code is organized](#how-the-code-is-organized) · [run locally](#run-locally) · [citation](#citation)

## The problem and the method

For a diffusion in a domain $\Omega$,
$$
dX_t=b(X_t)\,dt+\sigma(X_t)\,dW_t,
$$
where $b$ is the drift, $\sigma$ is the diffusion matrix, and $W_t$ is Brownian motion, the generator is
$$
\mathcal L f(x)=\sum_i b_i(x)\partial_i f(x)+\frac12\sum_{i,j}(\sigma\sigma^\top)_{ij}(x)\partial_{ij}f(x).
$$
The mean escape time is the expected time until the process first leaves $\Omega$:
$$
\tau(x)=\mathbb E_x[\tau_\Omega],\qquad
\tau_\Omega=\inf\{t\geq0:X_t\notin\Omega\}.
$$
It solves the backward equation $\mathcal L\tau=-1$ in $\Omega$, with $\tau=0$ on its boundary.

A standard neural approximation is $N_\theta(x)$. The boundary-adapted approximation is
$$
\tau_\theta(x)=\rho(x)N_\theta(x),
$$
where $\rho$ vanishes on $\partial\Omega$. This makes the boundary value exact for every network parameter $\theta$. For the paper's $H^2$ error theory, however, vanishing alone is not enough: the smooth distance factor must also satisfy $\partial_\nu\rho=1$, where $\nu$ is the inward unit normal. This first-order normalization keeps $\tau/\rho$ regular up to the boundary under the paper's assumptions. The theorem concerns minimizers of the sampled residual objective; it does not guarantee that an optimizer finds one. See the [paper](https://arxiv.org/html/2607.19167) for the assumptions and bounds.

The notebooks compare a standard PINN with a boundary penalty, the normalized factor $\rho$, the stronger and weaker factors $\rho^{3/2}$ and $\rho^{1/2}$, and a variational PINN that imposes boundary data in a weak formulation. Mean-escape-time training also uses Monte Carlo solution values at interior points as data targets. The reported relative Sobolev error is
$$
E_{H^k}=\frac{\|\tau_\theta-\tau_{\mathrm{ref}}\|_{H^k(\Omega)}}{\|\tau_{\mathrm{ref}}\|_{H^k(\Omega)}},\qquad k=0,1,2,
$$
where $\tau_{\mathrm{ref}}$ is the reference solution. At $k=0$ this is relative $L^2$ error; $H^1$ and $H^2$ also measure first- and second-derivative error. The OU example uses an analytical reference, while the double-well example uses a numerical one.

The periodic committor asks a related but different question: what is the probability of reaching target $B$ before target $A$? Its solution satisfies
$$
q(x)=\mathbb P_x(\tau_B<\tau_A),\qquad
\mathcal Lq=0\text{ in }D=\Omega\setminus(A\cup B),\qquad
q|_A=0,\quad q|_B=1.
$$
The example uses the same boundary-adapted idea for these two boundary values.

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
## License

The code is released under the [MIT license](LICENSE).

**Last updated: 2026-10-14 03:09 +08:00.**
