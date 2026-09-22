# Boundary-adapted PINNs for elliptic Dirichlet problems

Research code supporting [*Boundary-Adapted PINNs for Elliptic Dirichlet Problems: H²(Ω) A Priori Error Bounds with Application to Mean Escape Time Computation*](https://arxiv.org/abs/2607.19167).

The paper studies PINN ansatzes that impose homogeneous Dirichlet data exactly through a smooth, first-order-normalized distance approximation. The repository is being rebuilt from its original Colab notebooks into a reproducible experiment suite. The legacy notebooks remain intact while this work proceeds.

## Status

The current rebuild work is on `rebuild/committor-ready-core`.

- The original OU, double-well, and Duffing notebooks remain at the repository root as historical experiment records.
- `met_pinn/` contains the new reusable implementation; its first target is the two-dimensional OU disk benchmark.
- Double-well and Duffing are not being migrated at this stage.
- The first new problem is a periodic-domain committor with two interior
  Dirichlet components; its mathematical specification is in
  [`docs/periodic_committor.md`](docs/periodic_committor.md).

`MIGRATION.md` records the boundary between preserved legacy material and the new code.

## Quick start

Use a fresh Python environment with Python 3.10 or later. The intended execution environment is Google Colab for GPU runs.

```bash
git clone --branch rebuild/committor-ready-core git@github.com:n2pakbong/boundary-adapted-pinns.git
cd boundary-adapted-pinns
pip install -e '.[dev]'
pytest -q
met-pinn configs/ou_disk/boundary_factor_smoke.json --run-dir artifacts/ou_smoke
```

The smoke configuration is deliberately small and CPU-based. It verifies the runner; it is not a paper-scale experiment.

To resume an interrupted run:

```bash
met-pinn configs/ou_disk/boundary_factor_smoke.json --run-dir artifacts/ou_smoke --resume
```

Every new run writes its configuration, runtime metadata, append-only metrics, a complete restart checkpoint (`last.pt`), and the current best checkpoint (`best.pt`) into its run directory.

## Repository guide

| Location | Contents |
|---|---|
| `met_pinn/problems/` | PDE/SDE problem specifications, domains, boundary factors, and validation hooks. |
| `met_pinn/models.py` | Plain and boundary-factor neural ansatzes. |
| `met_pinn/training.py` | Restartable strong-form experiment runners. |
| `configs/` | Versioned configurations for individual runs. |
| `tests/` | Mathematical and implementation-level checks. |
| `colab/` | Colab usage notes. |
| `docs/periodic_committor.md` | Periodic committor problem, comparisons, and reference-grid protocol. |
| `references.bib` | Bibliography for research material cited by the new experiments. |
| `legacy/` | Provenance index for unchanged historical notebooks. |
| `OU_process*`, `Double_Well_2D`, `Duffin_oscillator` | Preserved legacy notebooks and embedded outputs. |

## Reproducibility policy

Paper-facing results must be generated from a versioned configuration and retain their seed, validation rule, code revision, logs, and restartable checkpoints. Large transient checkpoints and arrays are excluded by default; selected archival results will be added deliberately with their provenance.

## Legacy material

The legacy notebooks have scientific value, but they are not a drop-in reproducible release: they depend on Colab cell order, unpinned packages, and in some cases missing local log files. They are preserved rather than silently rewritten. New results should use `met_pinn/` and `configs/`.

## License

MIT; see [LICENSE](LICENSE).
