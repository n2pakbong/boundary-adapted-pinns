# Periodic committor configurations

Each configuration fully specifies the methods, equation, model, optimizer, sampling budgets, and reference grids.

- `cpu_preflight.json` runs the five methods for one epoch on CPU with tiny networks, one fixed near-boundary Monte Carlo anchor, and 32/64-grid references. It checks that each method and saved output path works; it does not test numerical accuracy.
- `gpu_pilot.json` is a single-seed five-method exploratory comparison. It uses 6,000 epochs per method, 32 common anchors with 256 Euler--Maruyama paths per anchor (`dt=0.001`, maximum 100,000 steps), and 128/256 finite-difference reference grids. The Monte Carlo labels are generated once on CPU. Network optimization uses the configured GPU.

The methods are the standard PINN with a sampled boundary-residual loss; the boundary-adapted, over-enforced (`rho^(3/2)`), and under-enforced (`rho^(1/2)`) exact-boundary PINNs; and the variational/Nitsche PINN. The three exact-boundary methods have no separate boundary loss. The Nitsche method trains the paper's squared interior-plus-flux loss and squared Nitsche boundary-contribution loss (symmetric consistency plus stabilization), alongside the shared pointwise Monte Carlo data term. This differs from the standard PINN's pointwise boundary residual.

The Colab notebook runs Setup and the CPU preflight, then prepares the production finite-difference reference and Monte Carlo labels in a CPU runtime. Only after those saved inputs are complete does it use a GPU for network training and solution evaluation.

For a terminal run, use the same output directory for both stages:

```bash
RUN_DIR=runs/periodic_committor/gpu_pilot
periodic-committor experiments/committor/periodic_two_well/configs/gpu_pilot.json \
  --output-dir "$RUN_DIR" --prepare-cpu-only
# Run this second command in the GPU-enabled environment:
periodic-committor experiments/committor/periodic_two_well/configs/gpu_pilot.json \
  --output-dir "$RUN_DIR" --train-prepared
```

The second command reuses `cpu_preparation.json`, the finite-difference reference, and the shared Monte Carlo labels; it does not repeat CPU preparation. The same folder receives checkpoints, live epoch records, final losses, solution errors, and plots. Use `--resume` with `--train-prepared` after an interrupted training run. See the parent experiment guide for the equations, estimator details, metrics, and output schema.
