# Periodic committor experiment settings

## Start here

`gpu_pilot.json` is the version-controlled five-method preset used for the reported pilot. In Colab, the **Set up a run one choice at a time** section presents the settings in separate, labelled controls for the problem, network, activation, initialization, optimizer, training schedule, sampling, and reference solve. The starting values match this preset. CPU preparation saves the fully resolved configuration as `comparison_config.json` in the Drive run folder; the later training stage reads that saved copy, so choices cannot silently change after preparation.

The defaults reproduce the successful pilot. All five methods use the same periodic MLP size, initialization seed, collocation budget, Monte Carlo anchors and labels, optimizer family, and epoch budget where the method permits. This keeps the comparison focused on the boundary representation and variational formulation. These values are a useful baseline, not the result of a hyperparameter sweep.

Earlier exploratory run folders used a three-method objective and may contain `training.log_every`, `training.lr_gamma`, or `training.boundary_weight`. Those fields remain readable for those runs. New five-method configurations should use `log_interval_seconds`, `scheduler`, and the paper's adaptive loss weighting; `boundary_weight` is rejected when the Monte Carlo paper objective is enabled.

## Main controls

| Setting | Pilot value | What it controls |
|---|---:|---|
| `methods` | all five methods | Selects which methods are trained: `simple_pinn`, `boundary_adapted`, `over_enforced`, `under_enforced`, and `variational_nitsche`. |
| `device`, `dtype` | `cuda`, `float64` | Select `cpu` or `cuda` training, and `float32` or `float64`. Setup, preflight, and reference/data preparation use CPU; the training stage uses the configured device. Full CPU training avoids allocating a GPU runtime but can be much slower. |
| `problem` | `a=1`, `b=0.6`, `epsilon=0.4`; centers `(0.25,0)` and `(0.75,0)`; radius `0.5` | Edit potential coefficients, diffusion strength, both set centers, and their common chordal radius on the fixed flat unit torus. The cosine potential formula and its phase centers stay fixed; moving set centers does not move its wells. The code rejects overlap and tangency. |
| `seed` | 7 | Seeds network initialization and training-point sampling. Each method starts from the same seed so the underlying MLP starts comparably. |
| `model.architecture` | `periodic_mlp` | Supported choices: `periodic_mlp` (first harmonic) and `periodic_fourier_mlp` (multiple harmonics), each followed by a fully connected MLP. |
| `model.frequencies` | `[1,2,4]` for the Fourier option | Positive integer harmonics in the sine/cosine embedding. Higher frequencies can represent finer periodic variation, while increasing derivative magnitudes in the strong-form loss. |
| `model.width`, `model.depth` | 64, 4 | Number of neurons per hidden layer and number of hidden layers. |
| `model.activation` | `tanh` | Select any module class exported by `torch.nn` by class name, or use a fully qualified import path for a custom `torch.nn.Module` class or PyTorch callable. Examples include `CELU`, `PReLU`, `ReLU`, `torch.nn.functional.gelu`, or `your_package.activations.Indicator`. Names are case-insensitive when using a bare `torch.nn` name. `SquaredReLU` is also available explicitly. The selected activation is applied to hidden layers; the output layer remains linear. |
| `model.activation_kwargs` | omitted | Optional JSON object passed as keyword arguments to an activation module constructor or callable. For example, use `{"beta": 2.0, "threshold": 12.0}` for Softplus or `{"approximate": "tanh"}` for `torch.nn.functional.gelu`. |
| `model.initialization`, `model.gain` | `xavier_normal`, 1 | Weight initialization for every linear layer, including the output layer. Supported schemes: `xavier_normal` and `xavier_uniform`. Biases are zero. |
| `training.epochs` | 6,000 per method | Number of optimizer steps for each selected method. There is no early stopping. |
| `training.collocation_points` | 1,024 per epoch | Fresh uniform-in-domain points used for the strong PDE residual at each epoch. |
| `training.boundary_points` | 256 per boundary | Fresh points on each boundary, used by the standard PINN. The exact-boundary methods have no pointwise boundary loss; Nitsche has its own quadrature count below. |
| `training.learning_rate` | 0.001 | Initial optimizer learning rate. |
| `training.optimizer` | Adam | Supported: `adam` and `adamw`. The object accepts `betas`, `eps`, `weight_decay`, and `amsgrad`. Pilot values are Adam's standard betas and epsilon, zero weight decay, and no AMSGrad. |
| `training.adaptive_loss_log_scale_init` | 0 | Initial value of every trainable adaptive loss log scale. The code converts these scales to the paper's coefficients; optimization learns them jointly with the network. |
| `training.scheduler` | exponential, `gamma=0.9999` | Updated once per epoch. Supported schedules: `constant`, `exponential`, and `step`; a step schedule also takes `step_size_epochs`. The pilot schedule decreases the learning rate from 0.001 to about 0.00055 over 6,000 epochs. |
| `training.log_interval_seconds` | 30 seconds | Wall-clock interval for live console reports. The first and final epochs are always reported. Shorten this to see more updates; lengthen it to reduce output. A full metric row is saved for every epoch regardless, so `losses.png` retains the complete loss history. |
| `training.checkpoint_every` | 100 epochs | How often the restartable `last.pt` checkpoint is refreshed. The best and final checkpoints are also saved. |

Bare activation names are resolved against torch.nn without a project-specific allowlist. A fully qualified import path can select any importable torch.nn.Module subclass or unary callable, including torch.nn.functional operations and a custom module such as your_package.activations.Indicator. Constructor or function keyword arguments are passed through from activation_kwargs. SquaredReLU computes max(0,x)^2 and is included because it is the activation used in the paper. ReLU, discontinuous functions, and other unsuitable choices remain available by deliberate selection. The strong-form objective differentiates the network twice; no smoothness or training-suitability filter is applied, so a poor choice may fail or train badly.

## Monte Carlo committor data

The pointwise data term uses one shared set of simulated committor labels for all selected methods. An anchor is one labelled input point. Its label is the fraction of simulated paths that hit target B before source A.

| Setting | Pilot value | Meaning |
|---|---:|---|
| `n_anchors` | 32 | Number of labelled input points. |
| `paths_per_anchor` | 256 | Independent Euler--Maruyama paths used for each label; 8,192 paths in total. More paths reduce binomial sampling noise in each label. |
| `dt` | 0.001 | Euler--Maruyama time step. It controls time-discretization error. |
| `max_steps` | 100,000 | Per-path step cap, corresponding to nominal time 100. |
| `minimum_resolved_fraction` | 0.90 | Every anchor must have at least this fraction of paths hit A or B. The run stops if an anchor falls below the threshold; censored paths are not assigned a fabricated label. |
| `seed` | 1007 | Independent seed for anchor selection and trajectory noise. |
| `progress_interval_fraction` | 0.01 | Report path-resolution progress at each 1% of the step cap, or sooner when the time interval is reached. |
| `progress_interval_seconds` | 10 | Maximum time to wait between progress reports. |
| `anchor_points` | omitted | Omit to sample anchors uniformly from the domain. A fixed point list is used only by the tiny preflight. |

These labels have Monte Carlo sampling error because they use finitely many paths. The saved metadata reports standard errors and path-resolution fractions. There is no separate additive-noise parameter: increase `paths_per_anchor` to reduce label sampling variance; change `dt` or `max_steps` only when changing the trajectory estimator itself.

## Nitsche and reference settings

The variational method uses `training.nitsche`: `volume_points=1024`, `test_functions=32`, `boundary_points=512`, `length_scale=0.12`, `gamma=10`, and `h=0.03`. These set the volume quadrature, Gaussian test functions, boundary quadrature, test-function width, stabilization strength, and boundary mesh scale, respectively. They are specific to the weak formulation; they do not add a boundary-residual loss to the exact-boundary networks.

`reference_grid_sizes=[128,256]` sets the two finite-difference grids. Errors use the 256 grid; the comparison between grids diagnoses reference sensitivity. `evaluation_batch_size=16384` controls how many reference-grid points are evaluated together.

The CPU preflight has deliberately tiny values in `cpu_preflight.json`. It checks code paths and outputs; its one-epoch results are not accuracy results.

For each run, preserve `comparison_config.json`, `comparison_metadata.json`, `cpu_preparation.json`, and Monte Carlo metadata with the results. To make a one-off notebook configuration reusable, copy its saved `comparison_config.json` into this directory under a descriptive name.
