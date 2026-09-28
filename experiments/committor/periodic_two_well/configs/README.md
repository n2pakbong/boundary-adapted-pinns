# Running from a JSON configuration

This page documents the versioned command-line presets, [`gpu_pilot.json`](gpu_pilot.json) and [`cpu_preflight.json`](cpu_preflight.json). They use the repository's built-in periodic MLP builder.

**The Colab notebook is a separate, more direct Python workflow.** It shows the model class, initializer, interior sampler, optimizer, and schedule as editable code and passes them to the runner. To use an arbitrary PyTorch architecture, activation, initializer, or sampler, edit those notebook functions; they are not encoded as a menu in this JSON file. Keep a copy of the edited notebook with the run because the JSON records values and callback names, not function source code.

## Main command-line settings

| Setting | Pilot value | Meaning |
|---|---:|---|
| `methods` | all five | Methods to train: `simple_pinn`, `boundary_adapted`, `over_enforced`, `under_enforced`, and `variational_nitsche`. |
| `device`, `dtype` | `cuda`, `float64` | Training device and floating-point precision. Preparation always runs on CPU. |
| `problem` | `a=1`, `b=0.6`, `epsilon=0.4`; centers `(0.25,0)` and `(0.75,0)`; radius `0.5` | Coefficients, target-set centers, and common chordal radius on the fixed flat unit torus. The cosine potential formula remains fixed. Overlapping or touching sets are rejected. |
| `seed` | 7 | Random seed for the built-in model and training samples. |
| `model.architecture` | `periodic_mlp` | Built-in options are `periodic_mlp` (one sine/cosine frequency) and `periodic_fourier_mlp` (multiple frequencies). Both apply an explicit periodic feature map before a fully connected MLP. |
| `model.width`, `model.depth` | 64, 2 | Hidden-layer width and number of hidden layers for the built-in MLP. |
| `model.activation`, `model.activation_kwargs` | `tanh` | A `torch.nn` module name or importable callable path, plus optional constructor keyword arguments. This setting applies only to the built-in MLP. For a custom activation class, define it in Python and use the notebook workflow. |
| `model.initialization` | `pytorch_default` | Use the initialization from PyTorch's `nn.Linear` layers. Optional Xavier normal/uniform initialization is available with `model.gain`; for custom `torch.nn.init` code, edit the notebook class. |
| `training.epochs` | 6,000 | Optimizer updates per selected method. |
| `training.collocation_points` | 1,024 | Interior PDE points resampled at each epoch. The built-in law proposes points uniformly on `[0,1)^2` and rejects points in A or B, producing uniform area samples on D. |
| `training.boundary_points` | 256 per boundary | Radial-angle boundary samples used by the simple PINN. Arclength weights correct the boundary integral. Nitsche has a separate boundary quadrature count. |
| `training.learning_rate` | 0.001 | Initial learning rate for the built-in Adam-family optimizer. |
| `training.optimizer`, `training.scheduler` | Adam, exponential | The built-in configuration supports Adam/AdamW and constant/exponential/step schedules. To instantiate another PyTorch optimizer or scheduler directly, use the notebook's Python functions. |
| `training.log_interval_seconds`, `training.checkpoint_every` | 30 seconds, 100 epochs | Live reporting interval and restart-checkpoint interval. Every epoch's metrics are saved. |

The user-defined interior sampler in the notebook is called both for PDE points and for Monte Carlo anchors. A nonuniform distribution changes the loss averages unless importance weights are added. Boundary points are a separate law: uniform radial angles with local arclength weights.

## Monte Carlo labels

The five-method pilot uses a shared set of committor labels generated from Euler--Maruyama paths.

| Setting | Pilot value | Meaning |
|---|---:|---|
| `n_anchors` | 32 | Number of labeled input locations. |
| `paths_per_anchor` | 256 | Paths used for each label. More paths reduce binomial sampling noise. |
| `dt` | 0.001 | Euler--Maruyama time step. |
| `max_steps` | 100,000 | Maximum steps per path. |
| `minimum_resolved_fraction` | 0.90 | Minimum fraction of each anchor's paths that must hit A or B before the cap. |
| `seed` | 1007 | Seed for anchor selection and trajectory noise. |
| `progress_interval_fraction`, `progress_interval_seconds` | 0.01, 10 | Report after this fraction of the step cap or time interval, whichever comes first. |
| `anchor_points` | omitted | If omitted, use the selected interior sampler. A fixed point list is used by the tiny preflight. |

The saved metadata includes standard errors and path-resolution fractions. The finite path count, time step, and censoring cap affect the estimator.

## Nitsche and reference settings

`training.nitsche` contains the volume-point, Gaussian test-function, boundary-point, test-width, stabilization, and boundary-scale settings used only by the variational method. The finite-difference reference grids default to `[128, 256]`; the finer grid is used for reported errors, and their difference estimates reference sensitivity. The GPU pilot and Colab notebook both use `evaluation_batch_size = 8192`: the network evaluates 8,192 grid points at a time when calculating errors after training.

## Prepare and train

The preflight uses tiny values from `cpu_preflight.json`; it checks code paths, not accuracy. For a terminal pilot run, both stages use the same output folder:

```bash
RUN_DIR=runs/periodic_committor/gpu_pilot
periodic-committor experiments/committor/periodic_two_well/configs/gpu_pilot.json \
  --output-dir "$RUN_DIR" --prepare-cpu-only
# If device is cuda, run the next command in a GPU-enabled environment; if device is cpu, keep using CPU:
periodic-committor experiments/committor/periodic_two_well/configs/gpu_pilot.json \
  --output-dir "$RUN_DIR" --train-prepared
```

The Colab notebook runs these stages through the Python API so its visible model and callback functions remain active during preparation, training, evaluation, and resume. It saves the resolved scalar configuration, code revision, references, labels, metrics, and checkpoints in the Drive run folder. It records callback names; save the edited notebook code alongside the folder for reproducibility.
