# Colab experiment workflow

Open the [periodic committor notebook on `main`](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/colab/periodic_committor.ipynb).

The notebook explains each stage and keeps the PDE and model code in the Python package.

1. **Setup and tests (CPU):** clones the private repository, installs it, and runs the fast geometry/boundary tests. A Colab Secret named `GITHUB_TOKEN` with Notebook access lets it clone the repository; the token is not saved to Drive or Git.
2. **CPU preflight:** runs all three methods for two epochs, solves small finite-difference references, and checks that checkpoints, summaries, and figures are produced. This uses no GPU computation.
3. **GPU pilot:** only after the preflight succeeds, trains all three methods for 6,000 epochs each and computes 128/256-grid references. It saves the configuration, code revision, runtime details, checkpoints, metrics, reference data, summary, and figures in Google Drive.
4. **Resume:** if the GPU pilot stops before completion, use the resume cell. It asks for the exact Drive folder printed when the pilot started and continues from its checkpoints.

Do not run the resume cell while a pilot is still running.