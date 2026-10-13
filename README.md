# Boundary-adapted PINNs for elliptic Dirichlet problems

[![arXiv:2607.19167](https://img.shields.io/badge/arXiv-2607.19167-b31b1b.svg)](https://arxiv.org/abs/2607.19167) [![Tests](https://github.com/n2pakbong/boundary-adapted-pinns/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/n2pakbong/boundary-adapted-pinns/actions/workflows/tests.yml) [![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Runnable notebooks and a PyTorch package for [the paper](https://arxiv.org/abs/2607.19167) on boundary-adapted neural networks for elliptic Dirichlet problems and mean escape time.

**Start:** [open a notebook](#notebooks) · [run locally](#run-locally) · [find code by task](#where-to-find-things) · [cite the work](#citation)

## Citation

For academic work using the method or results, cite the paper. Copyable BibTeX:

    @misc{tepakbong2026boundaryadapted,
      title = {Boundary-Adapted PINNs for Elliptic Dirichlet Problems: $H^2(\Omega)$ A Priori Error Bounds with Application to Mean Escape Time Computation},
      author = {Tepakbong, Nathanael and Fan, Jun and Zhou, Xiang and Zhou, Ding-Xuan},
      year = {2026},
      eprint = {2607.19167},
      archivePrefix = {arXiv},
      doi = {10.48550/arXiv.2607.19167},
      url = {https://arxiv.org/abs/2607.19167}
    }

For the software citation, use GitHub's **Cite this repository** link, generated from [CITATION.cff](CITATION.cff). Paper links: [arXiv](https://arxiv.org/abs/2607.19167) · [DOI](https://doi.org/10.48550/arXiv.2607.19167).

## Notebooks

| Notebook | Problem | Open |
| --- | --- | --- |
| OU mean escape time | Exit from a disk; analytical reference. | [Source](experiments/mean_escape_time/ou_ball/ou_mean_escape_time.ipynb) · [Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/mean_escape_time/ou_ball/ou_mean_escape_time.ipynb) |
| Double-well mean escape time | Anisotropic diffusion; numerical reference. | [Source](experiments/mean_escape_time/double_well/double_well_mean_escape_time.ipynb) · [Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/mean_escape_time/double_well/double_well_mean_escape_time.ipynb) |
| Periodic committor | Probability of reaching one well before another. | [Source](colab/periodic_committor.ipynb) · [Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/colab/periodic_committor.ipynb) · [Guide](experiments/committor/periodic_two_well/README.md) |
| Duffing first passage | Exit time for a stochastic oscillator. | [Source](experiments/first_passage/duffing_oscillator/duffing_mean_escape_time.ipynb) · [Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/first_passage/duffing_oscillator/duffing_mean_escape_time.ipynb) |

In Colab, select a CPU or GPU runtime and run the notebook setup cells. Outputs are saved to Google Drive by default.

## Where to find things

A notebook or configuration selects a problem; the problem module supplies its equation and sampling; an experiment runner handles preparation, training, and evaluation; the run folder holds metrics, figures, and checkpoints.

| If you want to… | Start here |
| --- | --- |
| Change a network | [models.py](src/boundary_pinn/models.py) |
| Change an equation, domain, or sampler | [problem modules](src/boundary_pinn/problems/) |
| Trace training, evaluation, or plots | [experiment runners](src/boundary_pinn/experiments/) |
| Find a notebook, guide, or settings | [experiments](experiments/) and [Colab notebooks](colab/) |
| Check the code | [tests](tests/) and [GitHub Actions](.github/workflows/tests.yml) |
| Check dependencies and commands | [pyproject.toml](pyproject.toml) |

To follow one complete example, open the [periodic committor guide](experiments/committor/periodic_two_well/README.md), then trace its [problem definition](src/boundary_pinn/problems/committor/periodic_two_well.py), [training methods](src/boundary_pinn/problems/committor/training.py), and [comparison runner](src/boundary_pinn/experiments/committor/periodic_two_well/comparison.py). The guide links the label and weak-form implementation too.

## Run locally

The package requires Python 3.10 or newer. Install the test and notebook dependencies, then run the CPU checks:

    python -m pip install -e ".[dev,notebooks]"
    python -m pytest -q

For the base package alone, use `python -m pip install -e .`. The optional notebook dependencies include scikit-fem. The tests check code behavior; they do not run full experiments or verify numerical accuracy.

## Reuse the code

Start with the problem module closest to your PDE. A new problem currently needs its operator, geometry, domain and boundary samplers, and reference calculation connected to an experiment runner; this repository does not yet provide a one-config plugin interface. For the periodic committor, the guide explains each part and how the saved run is organized.

A new committor notebook run draws fresh training and Monte Carlo seeds and records them with its configuration. Its dated Drive folder includes the settings, reference and label data, metrics, plots, and checkpoints. The OU and double-well aggregate values shown in the notebooks are historical; later code fixes mean a fresh run will not reproduce them exactly.

## License

Code is released under the [MIT license](LICENSE).

**Last updated: 2026-10-14 03:00 +08:00.**
