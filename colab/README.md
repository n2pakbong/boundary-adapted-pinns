# Colab use

The experiment runner is deliberately notebook-independent. In a fresh Colab runtime, after cloning the branch, install the package and run a named configuration:

```bash
pip install -e '.[dev]'
met-pinn configs/ou_disk/boundary_factor_smoke.json --run-dir artifacts/ou_smoke
```

To resume after an interruption, rerun the same command with `--resume`. The run directory contains the configuration, metadata, append-only metrics, `last.pt`, and `best.pt`.

The smoke configuration uses CPU and is only a wiring check. GPU configurations and paper-scale budgets will be added only after the OU validation protocol is agreed.
