# Periodic two-well committor

This is the repository's worked example of a committor PDE on a periodic two-dimensional domain with two Dirichlet boundaries. The Colab notebook is the user-facing workflow; this guide maps the mathematics, code, settings, saved files, and numerical checks.

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

The smooth periodic geometry uses the defining function
$\omega_j(x)=4\sum_{\ell=1}^2\sin^2(\pi(x_\ell-c_{j\ell}))-r^2$
and normalized factor $d_j=\omega_j/\sqrt{\omega_j^2+|\nabla\omega_j|^2}$.
The lift $g=d_A/(d_A+d_B)$ sets the two boundary values. The boundary-adapted
factor $\rho=d_Ad_B/\sqrt{d_A^2+d_B^2}$ vanishes on both boundaries with
unit first-order scaling. The network ansatz is $q_\theta=g+\rho N_\theta$.
The comparison includes the same lift with the unnormalized product $d_Ad_B$,
and a periodic MLP trained with sampled boundary penalties.

## Where the code lives

- `src/boundary_pinn/problems/committor/periodic_two_well.py`: potential, geometry, level-set boundary sampling, and differential operator.
- `src/boundary_pinn/problems/committor/training.py`: network construction, PDE and boundary losses, live progress, and checkpoints.
- `src/boundary_pinn/problems/committor/reference.py`: finite-difference reference solutions and discrete $L^2$, $H^1$, and $H^2$ error calculations.
- `src/boundary_pinn/experiments/committor/periodic_two_well/comparison.py`: training orchestration, output metadata, comparisons, plots, and CPU-only saved-checkpoint evaluation.
- `experiments/committor/periodic_two_well/configs/cpu_preflight.json`: tiny CPU wiring check.
- `experiments/committor/periodic_two_well/configs/gpu_pilot.json`: single-seed exploratory GPU settings.
- `colab/periodic_committor.ipynb`: derivation, references, CPU checks, GPU pilot, output inspection, and CPU evaluation of saved models.

## Run sequence

In Colab, use the cells in this order:

1. **Setup and CPU tests.** Choose a Git branch; press Enter to use `main`. Setup fetches that revision, installs the package, runs mathematical and tiny training tests on CPU, and prints the exact code revision. To preview an open pull request, enter its branch name here.
2. **CPU preflight.** Exercises the full pipeline with tiny settings. It uses temporary Colab storage only and does not use a GPU or Google Drive.
3. **GPU pilot.** Runs the three methods for 6,000 epochs each, then solves the 128- and 256-grid finite-difference references on CPU. Training progress prints the method, epoch, objective, PDE loss, and boundary loss every 20 epochs. Checkpoints and outputs go in one dated Drive folder.
4. **Inspect.** Displays the saved objective and solution-preview plots. If the run predates the full norm report, this step tells you to run Step 5.
5. **Saved-model evaluation (CPU only).** Reloads each `best.pt`, evaluates its solution on the saved finite-difference reference, and samples the actual source and target boundaries. It does no training and uses no GPU. It displays the error table and writes `saved_model_evaluation.json` in the same run folder.
6. **Resume (only after interruption).** Reuses `last.pt` and the same output folder. Do not run resume while training is active.

The GPU cell prints the complete Drive path of the run folder. Keep that folder as the run record. The setup token is used only to fetch a private GitHub branch; the training code never reads it.

For local CPU checks:

```bash
pip install -e '.[dev]'
pytest -q
periodic-committor experiments/committor/periodic_two_well/configs/cpu_preflight.json \
  --output-dir runs/periodic_committor/preflight
```

## What each loss contains

Let $R_\theta(x)=\mathcal Lq_\theta(x)$ be the PDE residual at an interior collocation point. All methods use the mean squared residual

$$
L_{\mathrm{PDE}}=\frac{1}{N_D}\sum_{i=1}^{N_D}R_\theta(x_i)^2.
$$

The **boundary-adapted** and **naïve-product** ansatzes impose both Dirichlet values exactly, so their optimized objective is $L_{\mathrm{PDE}}$.

The **penalty** network does not impose the values exactly. It samples $N_A$ source-boundary points and $N_B$ target-boundary points and adds

$$
L_{\partial}=\frac{1}{N_A}\sum_{i=1}^{N_A}q_\theta(x_i^A)^2
+\frac{1}{N_B}\sum_{i=1}^{N_B}(q_\theta(x_i^B)-1)^2,
\qquad
L_{\mathrm{total}}=L_{\mathrm{PDE}}+\lambda L_{\partial}.
$$

For the supplied pilot configuration, $\lambda=100$. The plotted quantity and the checkpoint's best objective are method-specific: for the penalty method they include $100L_{\partial}$; for the exact-boundary methods the boundary term is zero. `metrics.jsonl` records the objective, PDE term, and unweighted boundary term separately. The loss is the quantity optimized during training; the solution-error table below is the accuracy comparison.

## Solution error report

The finite-difference solution on the 256 grid is the target approximation for the learned-solution errors. Both future GPU runs (in `summary.json`) and CPU reevaluation of an existing run (in `saved_model_evaluation.json`) report:

| Field | Meaning |
|---|---|
| `grid_l2_abs`, `grid_l2_rel` | Absolute and relative discrete $L^2(D)$ error. |
| `grid_h1_abs`, `grid_h1_rel` | Absolute and relative full $H^1(D)$ error, combining the value and both first derivatives. |
| `grid_h1_seminorm_abs`, `grid_h1_seminorm_rel` | Absolute and relative $H^1$ seminorm error from the two first derivatives. |
| `grid_h2_abs`, `grid_h2_rel` | Absolute and relative full $H^2(D)$ error, combining values, first derivatives, and second derivatives. |
| `grid_h2_seminorm_abs`, `grid_h2_seminorm_rel` | Absolute and relative $H^2$ seminorm error from $q_{x_1x_1}$, $q_{x_1x_2}$, and $q_{x_2x_2}$. |

The absolute norms approximate integrals with the grid quadrature weight $h^2$. L2 uses all grid nodes in the exterior domain. H1 uses nodes whose centered axial stencil stays in the domain; H2 uses nodes whose full $3\times3$ stencil stays in the domain. Each relative error divides by the corresponding norm of the 256-grid FDM target evaluated on the same nodes. The H2 seminorm convention includes each of the three listed second derivatives once.

The solver also checks FDM resolution in `summary.json` under `reference_convergence`. Entries such as `fd_grid_h1_rel_128_to_256` compare the two FDM solutions and their derivatives, with each grid's derivatives computed at its own spacing. These differences estimate reference-grid sensitivity; they are not a rigorous error bound. The older `grid_l2` and `grid_h1_seminorm` fields remain as RMS diagnostics for compatibility with earlier run summaries. Use the explicit `*_abs` and `*_rel` fields for the Sobolev error table.

Boundary RMSE is evaluated on sampled points of the actual level sets, independently of errors on the reference grid. Exact-boundary representations should have boundary RMSE close to floating-point precision.

## Files saved for each GPU run

| File | What it contains |
|---|---|
| `comparison_config.json` | Complete problem, model, training, device, seed, and method settings. |
| `comparison_metadata.json` | Run date and code revision. |
| `reference_fd_256.npz` | FDM grid, values, boundary masks, and domain mask. |
| `summary.json` | Per-method epoch, training objective, full absolute/relative L2, H1, and H2 errors, true-boundary RMSE, and FDM grid-convergence estimates. |
| `saved_model_evaluation.json` | CPU reevaluation of existing checkpoints; adds complete solution-error norms and boundary RMSE without changing the original summary. |
| `losses.png` | Method-specific training objectives. |
| `committor_*.png` | Diagnostic previews of predictions, FDM reference, and pointwise differences; not paper-ready figures. |
| `<method>/metrics.jsonl` | Epoch, total objective, PDE loss, boundary loss, and learning rate. |
| `<method>/best.pt` | Best model weights and training objective. |
| `<method>/last.pt` | Most recent model, optimizer, scheduler, and random-number state, for resuming. |
| `<method>/config.json`, `metadata.json` | Method-specific configuration and Python/PyTorch/device information. |

For an old run that lacks its code revision in metadata, the CPU evaluation cell asks for the SHA printed by that run's Setup cell. It records the value in the evaluation JSON. New runs save the code revision automatically.

Keep full checkpoints and arrays in Drive. The notebook is source code, not the results archive. For a paper-facing result, use multiple seeds, document validation and reference resolution, and retain selected small figures, tables, configuration, and provenance in GitHub. The run folder remains the complete record.

## References

1. Yuehaw Khoo, Jianfeng Lu, and Lexing Ying. *Solving for high dimensional committor functions using artificial neural networks*. arXiv:1802.10275, 2018.
2. Zezheng Song, Maria K. Cameron, and Haizhao Yang. *A Finite Expression Method for Solving High-Dimensional Committor Problems*. arXiv:2306.12268, 2023.
3. Yuan Gao, Tiejun Li, Xiaoguang Li, and Jian-Guo Liu. *Transition path theory for Langevin dynamics on manifold: optimal control and data-driven solver*. arXiv:2010.09988, 2020.

The BibTeX entries are in [`references.bib`](../../../references.bib).
