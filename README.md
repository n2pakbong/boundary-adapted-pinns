# Boundary-adapted PINNs for elliptic Dirichlet problems

[![arXiv:2607.19167](https://img.shields.io/badge/arXiv-2607.19167-b31b1b.svg)](https://arxiv.org/abs/2607.19167) [![Tests](https://github.com/n2pakbong/boundary-adapted-pinns/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/n2pakbong/boundary-adapted-pinns/actions/workflows/tests.yml) [![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This repository accompanies the paper [*Boundary-Adapted PINNs for Elliptic Dirichlet Problems: $H^2(\Omega)$ A Priori Error Bounds with Application to Mean Escape Time Computation*](https://arxiv.org/abs/2607.19167). It contains the paper's mean-escape-time experiments, a shared PyTorch implementation, and additional notebooks that demonstrate the code on related problems.

**New here?** Read the paper for the mathematical results, open one of the two paper notebooks to see the numerical workflow, then use the code map below to trace a notebook into its implementation.

**Jump to:** [Experiments](#notebooks-and-experiments) · [Run the code](#run-in-colab-or-locally) · [Code map](#how-the-repository-is-organized) · [Reuse and extension](#reuse-and-extension) · [Citation](#cite-this-work)

## Cite this work

If you use the method or numerical results in academic work, cite the paper. Copyable BibTeX:

    @misc{tepakbong2026boundaryadapted,
      title = {Boundary-Adapted PINNs for Elliptic Dirichlet Problems: $H^2(\Omega)$ A Priori Error Bounds with Application to Mean Escape Time Computation},
      author = {Tepakbong, Nathanael and Fan, Jun and Zhou, Xiang and Zhou, Ding-Xuan},
      year = {2026},
      eprint = {2607.19167},
      archivePrefix = {arXiv},
      doi = {10.48550/arXiv.2607.19167},
      url = {https://arxiv.org/abs/2607.19167}
    }

To cite the code as software, use GitHub's **Cite this repository** link; it reads the software citation from [CITATION.cff](CITATION.cff). Cite the paper as well when your work uses its scientific results. The paper is available from [arXiv](https://arxiv.org/abs/2607.19167) and by [DOI](https://doi.org/10.48550/arXiv.2607.19167).

## Notebooks and experiments

The OU and double-well mean-escape-time studies are the paper's reported numerical examples. The periodic committor and Duffing notebooks are additional examples in this repository; they are not results reported in the paper.

| Example | Status | Open the notebook | What you can explore |
| --- | --- | --- | --- |
| **OU mean escape time** | Paper experiment | [Notebook source](experiments/mean_escape_time/ou_ball/ou_mean_escape_time.ipynb) · [Open in Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/mean_escape_time/ou_ball/ou_mean_escape_time.ipynb) | Five boundary-treatment methods and an analytical reference. |
| **Double-well mean escape time** | Paper experiment | [Notebook source](experiments/mean_escape_time/double_well/double_well_mean_escape_time.ipynb) · [Open in Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/mean_escape_time/double_well/double_well_mean_escape_time.ipynb) | Anisotropic diffusion and a numerical reference. |
| **Periodic committor** | Additional example | [Notebook source](colab/periodic_committor.ipynb) · [Open in Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/colab/periodic_committor.ipynb) · [Experiment guide](experiments/committor/periodic_two_well/README.md) | A prepare, train, evaluate, and resume workflow for a periodic two-well problem. |
| **Duffing first passage** | Additional example | [Notebook source](experiments/first_passage/duffing_oscillator/duffing_mean_escape_time.ipynb) · [Open in Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/first_passage/duffing_oscillator/duffing_mean_escape_time.ipynb) | A single-method stochastic-oscillator example. |

Colab notebooks save run outputs to Google Drive by default. The committor guide explains its target problem, five methods, reference calculation, saved files, and error measures.

## Run in Colab or locally

**Fastest route:** open one of the Colab links above, choose a CPU or GPU runtime, and run the notebook's setup cells. The notebook shows the problem-specific parameters and training code; use a copy of the notebook if you want to keep your edits.

**Local Python setup:** the package requires Python 3.10 or newer. Install the base package for imports, or include the optional development and notebook dependencies to run the checks and notebook workflows:

    python -m pip install -e .
    python -m pip install -e ".[dev,notebooks]"
    python -m pytest -q

The optional notebook dependencies include scikit-fem. The package and its command-line entry points are declared in [pyproject.toml](pyproject.toml). The CPU test suite checks software behavior; it does not run the long experiments or verify the paper's numerical accuracy.

## How the repository is organized

| Location | What lives there | Open it when you want to… |
| --- | --- | --- |
| [src/boundary_pinn/](src/boundary_pinn/) | The installable Python package. | Reuse or inspect the shared implementation. |
| [src/boundary_pinn/models.py](src/boundary_pinn/models.py) | Neural-network model definitions. | Change or compare network architectures. |
| [src/boundary_pinn/problems/](src/boundary_pinn/problems/) | Problem operators, geometry, sampling, and problem-specific training code. | See how a PDE and its boundary conditions enter the implementation. |
| [src/boundary_pinn/experiments/](src/boundary_pinn/experiments/) | Experiment runners, reference comparisons, evaluation, and plots. | Trace preparation, training, and result generation. |
| [experiments/](experiments/) | Paper notebooks, experiment documentation, and configurations. | Find a particular example or its settings. |
| [colab/](colab/) | Colab notebooks for added examples. | Run the periodic committor in Colab. |
| [tests/](tests/) and [.github/workflows/tests.yml](.github/workflows/tests.yml) | Automated CPU tests and their GitHub Actions workflow. | Check a code change before relying on it. |
| [CITATION.cff](CITATION.cff), [LICENSE](LICENSE), and [pyproject.toml](pyproject.toml) | Citation metadata, license, package requirements, and commands. | Check reuse terms or local setup. |

### Follow one experiment into the code

The periodic committor is the most fully documented end-to-end example. Start with its [Colab notebook](colab/periodic_committor.ipynb) and [experiment guide](experiments/committor/periodic_two_well/README.md), then follow these files:

| File | Role in the workflow |
| --- | --- |
| [periodic_two_well.py](src/boundary_pinn/problems/committor/periodic_two_well.py) | Defines the potential, PDE operator, periodic geometry, and samplers. |
| [paper_methods.py](src/boundary_pinn/problems/committor/paper_methods.py) | Implements Monte Carlo labels and the weak/Nitsche terms. |
| [training.py](src/boundary_pinn/problems/committor/training.py) | Defines trial functions, losses, training, logging, and checkpoints. |
| [comparison.py](src/boundary_pinn/experiments/committor/periodic_two_well/comparison.py) | Builds the reference, runs comparisons, and saves error summaries and figures. |
| [Configuration guide](experiments/committor/periodic_two_well/configs/README.md) | Explains the JSON settings and command-line preparation/training steps. |

## Reuse and extension

Shared neural-network and experiment code lives under src/boundary_pinn. Each problem keeps its own equation, geometry, samplers, and reference method under the problem modules; the experiment folders hold notebooks, settings, and guides. For a new PDE, start from the closest problem module and expect to implement its operator, domain and boundary samplers, reference solution, and connection to an experiment runner.

The problem interface is still evolving, so the package is not yet a plug-in system where any PDE can be added by filling in a single config. The committor guide gives a concrete example of the code that must agree. If you edit the notebook's custom model, sampler, optimizer, or schedule, save a copy of that notebook with the run: the saved run records callback names and fingerprints, but not the notebook-cell source.

For a new periodic-committor notebook run, the Colab workflow draws fresh training and Monte Carlo seeds and records them with the run configuration. The checked-in GPU pilot configuration is a single-seed exploratory preset; it is not a multi-run result or a source of run-to-run uncertainty estimates.

## Results and reproducibility

The aggregate OU and double-well values shown in the notebooks come from the original paper runs. The reorganized notebooks do not regenerate those exact runs: the current implementation includes later fixes to checkpoint selection, a variational-array shape, and double-well boundary quadrature. Treat those displayed values as historical paper outputs. Notebook quick runs and GitHub Actions tests check that the workflow runs; they do not reproduce the published numerical values or establish accuracy by themselves.

A committor run saves its settings, reference and Monte Carlo data, training metrics, error summaries, figures, and model checkpoints in a dated Google Drive folder. The [experiment guide](experiments/committor/periodic_two_well/README.md) describes the files and the steps for resuming or evaluating a run.

## License

The code is released under the [MIT license](LICENSE).

**Last updated: 2026-10-14 02:59 +08:00.**
