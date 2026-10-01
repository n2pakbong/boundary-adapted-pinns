# Duffing oscillator mean escape time

Open [`duffing_mean_escape_time.ipynb`](duffing_mean_escape_time.ipynb). It combines the trajectory study, Monte Carlo mean-exit estimate, and boundary-adapted PINN in one example with `zeta = 0.08`, `epsilon = 1e-5`, and disk radius `2`.

The ready-to-run default is a two-hidden-layer, width-64 `tanh` network. The visible `make_base_network()` function may return any scalar-output PyTorch model; the `full` preset restores the source four-layer ReQU network and training schedule. Parameters, samplers, losses, checkpoints, and evaluation settings remain ordinary editable Python.

No trained Duffing PINN result was stored in the previous notebooks. The public notebook therefore makes no reproduction claim and reports numerical values only after the reader runs it.
