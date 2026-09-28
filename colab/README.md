# Running the periodic two-well committor example

Open [the notebook in Colab](periodic_committor.ipynb) and run its cells in order. The network is a visible PyTorch class near the top of the notebook, and the interior sampler is ordinary Python code beside it. Edit the model, sampling function, optimizer, and schedule directly.

## Change the experiment

- Edit `POTENTIAL_A`, `POTENTIAL_B`, `DIFFUSION_EPSILON`, the two centers, and `COMMON_CHORDAL_RADIUS` for the supported unit-torus example. The torus, equal-radius disk geometry, and cosine potential formula remain fixed. The code rejects overlapping or touching target sets.
- Edit `METHODS_TO_RUN` to choose which boundary-condition methods to train.
- Edit the visible `PeriodicCommittorNetwork` class to change its layers and activations. It uses four periodic features, two 64-unit tanh hidden layers, and a scalar output. The paper does not report a periodic committor model; it reports separate width-30 tanh networks for the OU (two hidden layers) and double-well (three hidden layers) mean-escape-time examples.
- `nn.Linear` uses PyTorch's default initialization. Add or change `torch.nn.init` calls directly in the class if you want different initialization.
- Edit `sample_training_points()` to choose another interior distribution. By default it samples uniformly from `[0,1)²` and rejects points in A or B, giving uniform area samples on D. The same function supplies training collocation points and Monte Carlo anchors. A nonuniform distribution changes empirical averages unless importance weights are added.
- Change `make_optimizer()` and `make_scheduler()` by calling PyTorch directly. The training loop calls `optimizer.step()` and `scheduler.step()` once per epoch.
- Change the ordinary count and schedule variables for epochs, collocation points, boundary points, Monte Carlo paths, reference grids, Nitsche quadrature, reporting, and checkpoints. `EVALUATION_BATCH_SIZE = 8192` means the network receives 8,192 grid points per call during the final error calculation; it does not affect training.

Boundary points use a separate geometry sampler: it draws a uniform radial angle and weights each point by its local arclength when computing boundary losses and Nitsche integrals. The implementation is in `src/boundary_pinn/problems/committor/periodic_two_well.py` and `paper_methods.py`; changing that law also requires matching quadrature weights.

Save a copy of the notebook with your edits alongside the run folder. The run's JSON configuration records scalar experiment settings and the names of custom Python functions; the source code for your custom model and sampling law remains in the notebook.

## Run sequence

1. **Choose the runtime and run Setup:** set `DEVICE` in the notebook, then choose GPU for `cuda` or CPU for `cpu` before Setup. Setup installs the selected branch and runs CPU tests in a child process with CUDA hidden; it leaves the notebook runtime attached.
2. **CPU preflight:** run the packaged tiny five-method check. It does not exercise your edited notebook code; its files are temporary under `/content`.
3. **Notebook code check:** run the short model, sampler, optimizer, and scheduler smoke check.
4. **CPU preparation:** generate the finite-difference reference and Monte Carlo labels. This work uses CPU even when Colab has a GPU attached. Watch the live output; the completed files are saved in a dated Drive folder.
5. **Training:** run the training cell in the same runtime and paste that Drive folder path. Training uses the device in `DEVICE`; it does not repeat CPU preparation.
6. **Inspect, evaluate, or resume:** inspect saved tables and plots, evaluate checkpoints on CPU, or resume an interrupted run using the same notebook functions.

If you manually change runtimes, Colab resets the live Python session. Rerun Setup and the notebook's code-definition and configuration cells, but do not repeat completed preparation saved on Drive.

Preflight outputs use temporary Colab storage; preparation and training results go to Drive. Keep the printed Drive path. After a runtime reset, rerun Setup with the same branch and rerun the cells that define your model and callbacks. Resume checks callback names and repository revision, but cannot detect edits inside notebook cells; use the original code or start a new run folder.

While the repository is private, Setup reads the `GITHUB_TOKEN` Colab secret to fetch the selected branch. The token grants repository access; it is not used for training and does not access Drive.
