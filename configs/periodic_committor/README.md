# Periodic committor configurations

## Run order

1. `preflight_comparison_cpu.json` runs the complete three-method pipeline on CPU: reference solve, training, checkpoints, summary, and figures. It is deliberately tiny (two epochs) and establishes that the full command works without spending GPU time. It is not a scientific result.
2. `overnight_comparison.json` is the first GPU pilot: three methods, 6,000 epochs each, and 128/256-grid references. Run it only after the CPU preflight completes successfully.
3. A later paper configuration should be created only after inspecting this pilot's runtime, reference-grid discrepancy, and numerical error.

The individual `*_smoke.json` files test each architecture in isolation. They are retained for development; ordinary users should use the CPU preflight followed by the GPU pilot.

The preflight command is:

```bash
met-pinn-periodic-comparison \
  configs/periodic_committor/preflight_comparison_cpu.json \
  --output-dir runs/periodic_committor/preflight
```

The runner writes the exact configuration, metadata, JSONL metrics, `last.pt`/ `best.pt` checkpoints, reference data, `summary.json`, and figures in the output directory.
