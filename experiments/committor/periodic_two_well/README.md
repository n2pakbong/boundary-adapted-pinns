# Periodic two-well committor

This is the repository's worked example of a committor PDE on a periodic two-dimensional domain with two Dirichlet boundaries. The Colab notebook is the user-facing workflow; this guide maps the code, settings, saved files, and numerical checks.

[Open the experiment notebook in Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/colab/periodic_committor.ipynb).

## PDE and boundary construction

For overdamped Langevin dynamics on the flat torus,
$$
dX_t=-\nabla V(X_t)\,dt+\sqrt{2\varepsilon}\,dW_t \pmod 1,
$$
with
$$
V(x_1,x_2)=a\bigl[1-\cos(4\pi(x_1-1/4))\bigr]
+b\bigl[1-\cos(2\pi x_2)\bigr],
$$
the forward committor is $q(x)=\mathbb{P}_x(\tau_B<\tau_A)$. It solves
$$
-\nabla V\cdot\nabla q+\varepsilon\Delta q=0
\quad\text{in }D=\mathbb T^2\setminus(A\cup B),\qquad
q|_{\partial A}=0,\quad q|_{\partial B}=1.
$$
The periodic sets $A$ and $B$ are centered at $(1/4,0)$ and $(3/4,0)$. The default parameters are $a=1$, $b=0.6$, $\varepsilon=0.4$, and $r=0.5$.

The smooth periodic geometry uses the defining function $\omega_j(x)=4\sum_{\ell=1}^2\sin^2(\pi(x_\ell-c_{j\ell}))-r^2$ and the normalized factor $d_j=\omega_j/\sqrt{\omega_j^2+|\nabla\omega_j|^2}$. The lift
$$
g=\frac{d_A}{d_A+d_B}
$$
sets the two boundary values. The boundary-adapted factor
$$
\rho=\frac{d_Ad_B}{\sqrt{d_A^2+d_B^2}}
$$
vanishes on both boundaries with unit first-order scaling. The network ansatz is $q_\theta=g+\rho N_\theta$. The comparison includes the same lift with the unnormalized product $d_Ad_B$, and a periodic MLP trained with a sampled boundary penalty.

## Where the code lives

- `src/boundary_pinn/problems/committor/periodic_two_well.py`: potential, geometry, level-set boundary sampling, and differential operator.
- `src/boundary_pinn/problems/committor/training.py`: network construction, PDE and boundary losses, progress printing, metrics, and checkpoints.
- `src/boundary_pinn/problems/committor/reference.py`: sparse finite-difference reference on periodic grids and grid-based error calculations.
- `src/boundary_pinn/experiments/committor/periodic_two_well/comparison.py`: orchestration, output metadata, comparison summary, plots, and CPU-only boundary audit.
- `experiments/committor/periodic_two_well/configs/cpu_preflight.json`: tiny CPU wiring check.
- `experiments/committor/periodic_two_well/configs/gpu_pilot.json`: single-seed exploratory GPU settings.
- `colab/periodic_committor.ipynb`: derivation, references, setup, preflight, training, inspection, audit, and resume instructions.

## Run sequence

In Colab, use the cells in this order:

1. **Setup and CPU tests.** Fetches the selected `main` revision, installs the code, and runs CPU tests. The printed revision identifies that checkout.
2. **CPU preflight.** Exercises the full pipeline with tiny settings. It writes only to temporary Colab storage and uses no GPU.
3. **GPU pilot.** Runs the three methods for 6,000 epochs each and computes the 128- and 256-grid finite-difference references. Training progress is printed every 20 epochs with the method, epoch, total objective, PDE loss, and boundary loss. The run saves to a dated Drive folder.
4. **Inspect.** Displays `summary.json` and preview plots from that Drive folder.
5. **Boundary audit (CPU only).** Loads each `best.pt` and writes actual-boundary errors to `boundary_diagnostics.json`. It does not retrain or replace the original summary.
6. **Resume (only after interruption).** Reuses `last.pt` and the same output folder. Do not run resume while training is active.

The notebook gives the full Drive path at the start of training. Keep that folder as the run record. The setup token is only used to fetch a private repository; the training code never reads it.

For local CPU tests and preflight:

```bash
pip install -e '.[dev]'
pytest -q
periodic-committor experiments/committor/periodic_two_well/configs/cpu_preflight.json \
  --output-dir runs/periodic_committor/preflight
```

## Files saved for each GPU run

The pilot output folder contains:

| File | What it is for |
|---|---|
| `comparison_config.json` | Complete problem, model, training, device, seed, and method settings used for this comparison. |
| `comparison_metadata.json` | Run date, config path, and Git revision. Keep the Git revision with the run when tracing older runs that predate this metadata field. |
| `reference_fd_256.npz` | Finest finite-difference grid, reference values, source/target masks, and domain mask. Use it with saved checkpoints for later figures or analysis. |
| `summary.json` | Per-method best epoch and loss, discrete $L^2$ and $H^1$-seminorm errors against the 256-grid reference, plus the 128-to-256 reference comparison. |
| `losses.png` | Training objective versus epoch. This is an optimization diagnostic, not a solution-accuracy plot. |
| `committor_*.png` | Preview of the learned field, finite-difference reference, and pointwise difference for each method. These are diagnostic plots, not paper-ready figures. |
| `<method>/metrics.jsonl` | One record per logging interval: epoch, total objective, PDE loss, boundary loss, and learning rate. |
| `<method>/best.pt` | Best model weights, epoch, and best training objective, used for evaluation and boundary audit. |
| `<method>/last.pt` | Most recent model, optimizer, scheduler, and random-number state, used only to resume. Saved at epoch 1, every 100 epochs, and at the final epoch. |
| `<method>/config.json`, `metadata.json` | Method-specific settings and Python/PyTorch/device information. |

For the existing pilot made before Git-revision metadata was added, the notebook's CPU audit asks for the SHA printed by that run's Setup cell and stores it with the corrected boundary diagnostics. New runs save the revision automatically. The audit writes a separate JSON file and leaves the historical `summary.json` unchanged.

## How to read the comparisons

The two exact-boundary methods minimize the PDE residual. The penalty baseline minimizes the PDE residual plus a weighted boundary penalty. For that reason, the three training-objective curves are not comparable by final height. Use the grid-based solution errors in `summary.json` to compare predictions, and inspect `reference_convergence` before deciding whether the finite-difference reference is sufficiently resolved. These checks are numerical evidence, not an exact error bound.

The current pilot is one seed and is exploratory. It can guide the next experiment; it does not establish a paper conclusion. For a paper-facing result, repeat with several seeds, document the reference-grid study and validation rule, and choose publication-quality figures from the saved arrays and checkpoints.

Keep full checkpoints and arrays in Google Drive. The notebook itself is source code, not the results archive. When selected results are ready to share, add the small figure/table files, configuration, and provenance to GitHub deliberately; do not commit transient checkpoints or arrays.

## References

1. Yuehaw Khoo, Jianfeng Lu, and Lexing Ying. *Solving for high dimensional committor functions using artificial neural networks*. arXiv:1802.10275, 2018.
2. Zezheng Song, Maria K. Cameron, and Haizhao Yang. *A Finite Expression Method for Solving High-Dimensional Committor Problems*. arXiv:2306.12268, 2023.
3. Yuan Gao, Tiejun Li, Xiaoguang Li, and Jian-Guo Liu. *Transition path theory for Langevin dynamics on manifold: optimal control and data-driven solver*. arXiv:2010.09988, 2020.

The BibTeX entries are in [`references.bib`](../../../references.bib).
