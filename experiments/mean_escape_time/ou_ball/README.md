# Ornstein–Uhlenbeck mean escape time

Open [`ou_mean_escape_time.ipynb`](ou_mean_escape_time.ipynb). It is the single guided comparison for this problem and contains all five formulations used in the paper: standard, boundary-adapted, over-enforced, under-enforced, and variational/Nitsche PINNs.

The ready-to-run default is a two-hidden-layer, width-64 `tanh` network. The visible `make_base_network()` function accepts any scalar-output PyTorch model; the `paper` preset restores the architecture and training schedule recorded in the source experiment. The notebook also includes a reusable single-path simulator, the known radial solution, Monte Carlo labels, resumable training, and evaluation.

The experiment uses `sigma_squared = 2`. The retained paper-run table is historical output and is labeled with the code corrections that require a rerun.
