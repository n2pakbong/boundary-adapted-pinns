# Periodic committor run configurations

Each JSON file records the parameters for one run:

- `cpu_preflight.json` runs all three methods for two epochs with tiny networks and 32/64 reference grids. It checks the complete pipeline; it is not a scientific accuracy run.
- `gpu_pilot.json` runs all three methods for 6,000 epochs each with the 128/256 reference grids.
- `boundary_adapted_smoke.json`, `naive_product_smoke.json`, and `penalty_smoke.json` are individual short checks of the three ansatzes.

The Colab notebook runs the CPU preflight before the GPU pilot. To start the full comparison outside Colab:

```bash
periodic-committor experiments/committor/periodic_two_well/configs/gpu_pilot.json \
  --output-dir runs/periodic_committor/gpu_pilot
```
