# Periodic committor configurations

The three smoke configurations differ only in their treatment of the two
Dirichlet components:

- `boundary_adapted_smoke.json`: exact boundary lift plus the normalized
  two-component first-order factor.
- `naive_product_smoke.json`: the same lift with an unnormalized product
  factor.  It has exact values but is the deliberately weaker ablation.
- `penalty_smoke.json`: an unconstrained network with sampled boundary-loss
  penalties.

Run each from a fresh Colab runtime after installing the project:

```bash
met-pinn configs/periodic_committor/boundary_adapted_smoke.json \
  --run-dir runs/periodic_committor/boundary_adapted_smoke
```

The smoke files check wiring and restart artifacts only.  Before reporting a
comparison, copy them to a named paper configuration, increase training length,
record the Colab hardware/runtime details, and perform the reference-grid
convergence check described in `docs/periodic_committor.md`.
