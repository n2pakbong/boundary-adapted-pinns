# Running the periodic two-well committor example

Open [`periodic_committor.ipynb`](periodic_committor.ipynb) in Google Colab. Run its cells in order:

1. **Setup and component tests (CPU):** fetches the latest `main` code, installs it, and runs tests. It does not train a PINN.
2. **CPU preflight:** runs the whole comparison pipeline for two epochs with a tiny network and small reference grids. It saves a separate result folder to Drive. This checks the pipeline; it does not assess scientific accuracy.
3. **GPU pilot:** trains the three methods for 6,000 epochs each and uses the selected Colab GPU. This cell starts only when you run it.
4. **Inspect or resume:** displays the saved results or continues an interrupted pilot from its existing Drive checkpoints.

While the repository is private, setup reads the `GITHUB_TOKEN` secret to fetch `main`. The token permits GitHub repository access only; it is not used for training and does not access Drive. If the repository is public, setup works without a token.
