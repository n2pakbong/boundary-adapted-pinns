# OU mean escape time on a disk

This Python example defines an Ornstein–Uhlenbeck mean escape time problem on a two-dimensional disk. Its boundary-factor smoke configuration checks the analytic problem definition and training runner on a small CPU run.

```bash
pip install -e '.[dev]'
boundary-pinn experiments/mean_escape_time/ou_disk/configs/boundary_factor_smoke.json \
  --run-dir runs/ou_disk/boundary_factor_smoke
```

This small package example is separate from the guided OU paper comparison in [`../ou_ball/ou_mean_escape_time.ipynb`](../ou_ball/ou_mean_escape_time.ipynb).
