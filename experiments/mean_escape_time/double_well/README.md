# Double-well mean escape time

Open [`double_well_mean_escape_time.ipynb`](double_well_mean_escape_time.ipynb). It is the single guided comparison for the five paper methods.

The ready-to-run default is a two-hidden-layer, width-64 `tanh` network. Readers can replace the visible `make_base_network()` factory with any scalar-output PyTorch model; the `paper` preset restores the recorded architecture and schedule. The notebook includes a reusable sample trajectory, shared Monte Carlo labels, resumable training, and comparison with a finite-element reference.

The original notebooks used FEniCS for the reference solution. The public notebook assembles the same P1 weak problem with `scikit-fem`, which installs directly in Colab. Historical values remain identified as earlier outputs until the corrected public workflow is rerun.
