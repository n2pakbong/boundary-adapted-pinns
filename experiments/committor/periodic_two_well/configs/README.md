# Periodic committor configurations

Each configuration fully specifies the methods, equation, model, optimizer, sampling budgets, and reference grids.

- `cpu_preflight.json` runs the five methods for one epoch on CPU with tiny networks, one fixed near-boundary Monte Carlo anchor, and 32/64-grid references. It checks that each method and saved output path works; it does not test numerical accuracy.
- `gpu_pilot.json` is a single-seed five-method exploratory comparison. It uses 6,000 epochs per method, 32 common anchors with 256 Euler--Maruyama paths per anchor (`dt=0.001`, maximum 100,000 steps), and 128/256 finite-difference reference grids. The Monte Carlo labels are generated once on CPU. Network optimization uses the configured GPU.

The methods are the standard PINN with a sampled boundary-residual loss; the boundary-adapted, over-enforced (`rho^(3/2)`), and under-enforced (`rho^(1/2)`) exact-boundary PINNs; and the variational/Nitsche PINN. The three exact-boundary methods have no separate boundary loss. The variational method has weak-residual, Nitsche-stabilization, and pointwise Monte Carlo data terms.

The main Colab notebook runs the CPU preflight first. A full comparison outside Colab can be launched with:

```bash
periodic-committor experiments/committor/periodic_two_well/configs/gpu_pilot.json \
  --output-dir runs/periodic_committor/gpu_pilot
```

See the parent experiment guide for the equations, precise loss definitions, estimator details, and output schema.
