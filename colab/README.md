# Running the periodic two-well committor example

Open [the notebook in Colab](periodic_committor.ipynb) and run its cells in order. It is ordinary Python: the network is a visible PyTorch class; activation, initialization, sampling, optimizer, and schedule are visible code you can edit.

## Change the experiment

- Edit `POTENTIAL_A`, `POTENTIAL_B`, `DIFFUSION_EPSILON`, the two centers, and `COMMON_CHORDAL_RADIUS` for the supported unit-torus example. The torus, equal-radius disk geometry, and cosine potential formula remain fixed. The code rejects overlapping or touching target sets.
- Edit `METHODS_TO_RUN` to choose which boundary-condition methods to train.
- Edit `PeriodicCommittorNetwork` to change the architecture. The example feeds a fully connected MLP with four periodic features: `sin(2πx1)`, `sin(2πx2)`, `cos(2πx1)`, and `cos(2πx2)`. The paper's mean-escape-time experiments used width-30 tanh MLPs with two or three hidden layers; its paper does not describe this torus committor or the periodic feature map. This notebook is an additional example using the boundary-adaptation methods.
- Change `activation()` to return any `torch.nn` module. Define a custom `nn.Module` in a notebook cell if you need one.
- Change `initialize_layer()` using any `torch.nn.init` function, or skip `.apply(initialize_layer)` to keep PyTorch's defaults.
- Change `sample_training_points()` to implement another interior distribution. Its current code samples uniformly from `[0,1)²` and rejects points in A or B, giving uniform area samples on D. The same function selects training collocation points and Monte Carlo anchors. A nonuniform distribution changes the empirical averages unless importance weights are added.
- Change `make_optimizer()` and `make_scheduler()` by calling PyTorch directly. The training loop calls `optimizer.step()` and `scheduler.step()` once per epoch.
- Change the ordinary count and schedule variables for epochs, collocation points, boundary points, Monte Carlo paths, reference grids, Nitsche quadrature, reporting, and checkpoints.

Boundary points use a separate geometry sampler: it draws a uniform radial angle and weights each point by its local arclength when computing boundary losses and Nitsche integrals. The implementation is in `src/boundary_pinn/problems/committor/periodic_two_well.py` and `paper_methods.py`; changing that law also requires matching quadrature weights.

Save a copy of the notebook with your edits alongside the run folder. The run's JSON configuration records scalar experiment settings and the names of custom Python functions; the source code for your custom model and sampling law remains in the notebook.

## Run sequence

1. **Setup and CPU tests:** choose a repository branch, install it, and run the test suite on CPU.
2. **CPU preflight:** run the tiny five-method pipeline check. Its files are temporary under `/content` and disappear when the Colab runtime resets. It does not train a scientifically meaningful model.
3. **CPU preparation:** generate finite-difference references and Monte Carlo labels. Watch the live output in the notebook cell. The completed reference, labels, metadata, and anchor plot are saved in a new dated Drive folder. No network training happens in this step.
4. **Training:** select the CPU or GPU runtime to match `DEVICE`, run Setup again after a runtime change, then run the training cell with the same Drive folder. It uses the model class and sampler, optimizer, and scheduler code defined above, prints epoch/loss/ETA progress, and saves per-epoch metrics and checkpoints in that folder. It does not repeat CPU preparation.
5. **Inspect, evaluate, or resume:** inspect the saved tables and plots, evaluate saved checkpoints on CPU, or resume an interrupted run using the same notebook functions.

The preflight uses temporary Colab storage. Production preparation and training results go to Drive. Keep the printed Drive path. After reconnecting, rerun the notebook's code cells that define your model and custom functions before evaluating or resuming. A resume checks callback names and repository revision, but cannot detect edits inside notebook cells; use the original code or start a new run folder.

While the repository is private, Setup reads the `GITHUB_TOKEN` Colab secret to fetch the selected branch. The token grants repository access; it is not used for training and does not access Drive.
