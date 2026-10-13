# Boundary-adapted PINNs for elliptic Dirichlet problems

Code and executable notebooks for the paper [*Boundary-Adapted PINNs for Elliptic Dirichlet Problems: $H^2(\Omega)$ A Priori Error Bounds with Application to Mean Escape Time Computation*](https://arxiv.org/abs/2607.19167). The repository contains the paper's mean-escape-time examples and additional worked examples, built around a reusable PyTorch package.

## Cite the paper and code

If you use the methods, code, or results in academic work, cite the paper:

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

To cite the software itself, use GitHub's **Cite this repository** link; its metadata is in [CITATION.cff](CITATION.cff). The paper is also available by [arXiv](https://arxiv.org/abs/2607.19167) and [DOI](https://doi.org/10.48550/arXiv.2607.19167).

## Choose an example

| Example | Notebook | What it covers |
| --- | --- | --- |
| **Paper: OU mean escape time** | [Open in Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/mean_escape_time/ou_ball/ou_mean_escape_time.ipynb) | Five-method comparison with an analytical reference. |
| **Paper: double-well mean escape time** | [Open in Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/mean_escape_time/double_well/double_well_mean_escape_time.ipynb) | Mean escape time with anisotropic diffusion and a numerical reference. |
| **Additional: periodic committor** | [Open in Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/colab/periodic_committor.ipynb) · [Experiment guide](experiments/committor/periodic_two_well/README.md) | Probability of reaching one well before another on a periodic domain. |
| **Additional: Duffing first passage** | [Open in Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/experiments/first_passage/duffing_oscillator/duffing_mean_escape_time.ipynb) | A single-method stochastic-oscillator example. |

The OU and double-well examples appear in the posted paper. The committor and Duffing notebooks are additional repository examples; their results are not reported as paper results.

## Run and adapt the code

For a quick start, open a notebook in Colab, select a CPU or GPU runtime, and run its setup cell. Notebook outputs are saved to Google Drive by default.

For local use, install Python 3.10 or newer and the package with its development tools, then run the test suite:

```bash
python -m pip install -e '.[dev]'
pytest -q
```

The repository is organized around:

```text
src/boundary_pinn/       shared PyTorch models, training, and experiment code
experiments/             paper examples, settings, and experiment-specific material
colab/                   Colab workflows, including the periodic committor
tests/                   automated checks
CITATION.cff             software citation metadata
```

The shared package is the starting point for reuse. Problem geometry, sampling, and settings still live with each experiment, so adapt the notebook and experiment code closest to your problem. The [committor guide](experiments/committor/periodic_two_well/README.md) walks through one example and links its main implementation files.

## Results and reproducibility

The aggregate OU and double-well values displayed by the notebooks are retained from the original paper runs. The reorganized notebooks do not regenerate those exact runs: the current code includes later fixes to checkpoint selection, a variational-array shape, and double-well boundary quadrature. Notebook quick runs and GitHub Actions checks verify that workflows run; they do not establish paper-level accuracy or reproduce the published values.

A full periodic-committor run draws fresh training and Monte Carlo seeds and records them in that run's configuration. Its saved results and settings go to a dated Google Drive folder.

The code is distributed under the [MIT license](LICENSE).

**Last updated: 2026-10-14 02:53 +08:00.**
