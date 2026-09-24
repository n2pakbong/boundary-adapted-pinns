# Periodic two-well committor

[Open the periodic committor notebook in Colab](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/colab/periodic_committor.ipynb).

This page explains the equation, every trained method, each loss, the Monte Carlo data estimator, and the files saved by the experiment. It is intended to be readable without knowing the paper or the repository internals.

## Problem

On the flat torus $\mathbb T^2=[0,1)^2$, consider overdamped Langevin dynamics

$$
dX_t=-\nabla V(X_t)\,dt+\sqrt{2\varepsilon}\,dW_t\pmod 1,
\qquad
V(x_1,x_2)=a[1-\cos(4\pi(x_1-1/4))]+b[1-\cos(2\pi x_2)].
$$

Let $A$ and $B$ be the two periodic disks centred at $(1/4,0)$ and $(3/4,0)$. The forward committor is

$$
q(x)=\mathbb P_x(\tau_B<\tau_A),
\qquad
\mathcal Lq=-\nabla V\cdot\nabla q+\varepsilon\Delta q=0\quad\text{in }D=\mathbb T^2\setminus(A\cup B),
$$

with $q=0$ on $\partial A$ and $q=1$ on $\partial B$. The example therefore solves a probability problem, not a mean escape time problem. The default values are $a=1$, $b=0.6$, $\varepsilon=0.4$, and chordal radius $r=0.5$.

## Boundary geometry and network representations

For $j\in\{A,B\}$, with centre $c_j$, define

$$
\omega_j(x)=4\sum_{k=1}^2\sin^2(\pi(x_k-c_{jk}))-r^2,
\qquad
d_j(x)=\frac{\omega_j(x)}{\sqrt{\omega_j(x)^2+|\nabla\omega_j(x)|^2}}.
$$

The zero level set of $\omega_j$ is the boundary, $d_j=0$ there, and $|\nabla d_j|=1$ there. In $D$, both $d_A$ and $d_B$ are positive. The code uses the lift and the first-order-normalized two-boundary factor

$$
g(x)=\frac{d_A(x)}{d_A(x)+d_B(x)},
\qquad
\rho(x)=\frac{d_A(x)d_B(x)}{\sqrt{d_A(x)^2+d_B(x)^2}}.
$$

Thus $g=0$ on $\partial A$, $g=1$ on $\partial B$, and $\rho=0$ on both. The exact-boundary family is

$$
q_\theta(x)=g(x)+\rho(x)^\alpha N_\theta(x).
$$

The exponent $\alpha$ defines three distinct members of that family: boundary-adapted $(\alpha=1)$, over-enforced $(\alpha=3/2)$, and under-enforced $(\alpha=1/2)$. All three satisfy both boundary values by construction. None has a separate boundary-residual loss. The periodic network $N_\theta$ receives sine/cosine coordinates, so its output is periodic.

## Methods and exact objectives

Every method uses the same collocation and Monte Carlo anchor sets within a comparison. Write

$$
\widehat J_{\rm pde}(\theta)=\frac1{N_D}\sum_{i=1}^{N_D}\left(\mathcal Lq_\theta(x_i)\right)^2,
\qquad
\widehat J_{\rm data}(\theta)=\frac1{N_{\rm data}}\sum_{i=1}^{N_{\rm data}}\left(q_\theta(z_i)-\widehat q_i\right)^2.
$$

The five methods are:

| Method | Trial function | Raw loss components | Boundary treatment |
|---|---|---|---|
| `simple_pinn` | $q_\theta=N_\theta$ | $\widehat J_{\rm pde},\widehat J_{\rm data},\widehat J_{\rm bc}$ | Standard sampled Dirichlet residual penalty. |
| `boundary_adapted` | $q_\theta=g+\rho N_\theta$ | $\widehat J_{\rm pde},\widehat J_{\rm data}$ | Both values exact; no boundary penalty. |
| `over_enforced` | $q_\theta=g+\rho^{3/2}N_\theta$ | $\widehat J_{\rm pde},\widehat J_{\rm data}$ | Both values exact; no boundary penalty. |
| `under_enforced` | $q_\theta=g+\rho^{1/2}N_\theta$ | $\widehat J_{\rm pde},\widehat J_{\rm data}$ | Both values exact; no boundary penalty. |
| `variational_nitsche` | $q_\theta=N_\theta$ | $\widehat J_{\rm var},\widehat J_{\rm bc},\widehat J_{\rm data}$ | Weak imposition with Nitsche consistency, symmetry, and stabilization terms. |

For the standard PINN only,

$$
\widehat J_{\rm bc}=\frac1{N_A}\sum_i q_\theta(x_i^A)^2+
\frac1{N_B}\sum_i(q_\theta(x_i^B)-1)^2.
$$

This boundary-residual penalty is part of the ordinary unconstrained PINN. It is not applied to the three exact-boundary representations. The variational method has a different boundary stabilization term, described below; it is part of Nitsche's weak boundary treatment, not a penalty added to a boundary-adapted ansatz.

### Monte Carlo solution-value regularization

The paper's mean-escape-time notebooks use simulated solution values as a pointwise regularizer. Here the corresponding committor target is the probability of hitting $B$ before $A$. Starting from a common anchor $z_i\in D$, the code simulates

$$
X_{n+1}=\bigl[X_n-\nabla V(X_n)\,\Delta t+\sqrt{2\varepsilon\Delta t}\,\xi_n\bigr]\pmod 1,
\qquad \xi_n\sim\mathcal N(0,I_2),
$$

until the path first enters $A$ or $B$, or reaches the configured step cap. The label is

$$
\widehat q_i=\frac{\#\{\text{paths entering }B\text{ first from }z_i\}}
{\#\{\text{paths from }z_i\text{ resolved by the step cap}\}}.
$$

The code records each anchor's resolved-path fraction and the conditional binomial standard error calculated from resolved paths. During simulation, Colab streams the step count against the cap, the number of paths that reached either set, the number of anchors meeting the resolution threshold, minimum and mean anchor resolution, and elapsed time. It prints a final resolution summary before training starts. It stops with an error if the configured minimum resolved fraction is missed; censored trajectories are never silently treated as source hits. If some paths remain unresolved below that threshold, the conditional estimate can still have truncation bias. The time step and path cap also introduce estimator choices: the discrete Euler--Maruyama target has time-discretization and Monte Carlo error. Those settings and diagnostics are saved in `monte_carlo_labels.json`. The same saved anchors and labels are used by all five methods.

This is the committor analogue of the pointwise Monte Carlo regularizer in the mean-escape-time experiments. It is an extension to this example, not a claim that the committor experiment reproduces the paper's mean-escape-time outputs.

### Adaptive balancing and the logged losses

The paper writes adaptive loss balancing with the component notation $\widehat J_i$ and a positive trainable scale $\lambda_i$ [Tepakbong et al. (2026), Eq. (33)](https://arxiv.org/abs/2607.19167):

$$
\mathcal J(\theta,\{\lambda_i\}_{i\in\mathcal I})
=\sum_{i\in\mathcal I}\frac{1}{2\lambda_i^2}\widehat J_i(\theta)
+\sum_{i\in\mathcal I}\log\lambda_i,
\qquad \lambda_i>0.
$$

The active set $\mathcal I$ contains only the components trained by the selected method:

- simple_pinn: $\{\mathrm{pde},\mathrm{bc},\mathrm{data}\}$, with $\widehat J_{\rm bc}$ the sampled Dirichlet residual.
- boundary_adapted, over_enforced, and under_enforced: $\{\mathrm{pde},\mathrm{data}\}$. These exact-boundary methods have no boundary-residual component.
- variational_nitsche: $\{\mathrm{var},\mathrm{bc},\mathrm{data}\}$, with $\widehat J_{\rm var}$ the Nitsche weak-residual loss and $\widehat J_{\rm bc}$ the Nitsche stabilization loss. Here $\widehat J_{\rm bc}$ is part of the variational boundary treatment, not the pointwise boundary residual used by simple_pinn.

In this committor example, $\widehat J_{\rm pde}$ is the mean-square committor PDE residual and $\widehat J_{\rm data}$ is the mean-square difference from the simulated committor labels $\widehat q_i$. The implementation keeps its descriptive names for the Nitsche terms: nitsche_weak_loss is $\widehat J_{\rm var}$, and nitsche_penalty_loss is $\widehat J_{\rm bc}$. The latter is the Nitsche stabilization term only.

**How the saved code scale matches the paper's $\lambda_i$.** To preserve existing checkpoint parameter names, the implementation stores $s_i=\log\sigma_i$ and computes each weighted component with multiplier $e^{-2s_i}$. The correspondence is $\lambda_i=e^{s_i}/\sqrt{2}$, so $e^{-2s_i}=1/(2\lambda_i^2)$. Consequently, the implementation's total objective differs from the displayed paper objective only by the constant $|\mathcal I|\log(2)/2$. This constant has no effect on the optimization; the effective component weights are exactly the paper coefficients. The raw losses and their reported values are unchanged. The trainable-scale idea is the uncertainty-weighted balancing described by [Kendall, Gal, and Cipolla (2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html).

At the configured logging interval, training prints the epoch, objective, and active raw loss components. Each record, including weighted components and effective weights, is appended to metrics.jsonl. losses.png plots the objective and raw component histories. final_training_metrics.json and summary.json save final component values and the best epoch/objective.

### Variational/Nitsche formulation

The drift is a gradient, so with $m(x)=e^{-V(x)/\varepsilon}$,

$$
\mathcal Lq=0\quad\Longleftrightarrow\quad \nabla\cdot(\varepsilon m\nabla q)=0.
$$

For a periodic Gaussian test function centred at $c_j$,

$$
\phi_j(x)=\exp\!\left[-\frac{4\sum_{k=1}^2\sin^2(\pi(x_k-c_{jk}))}{2\ell^2}\right],
$$

integration by parts gives a volume term and a boundary flux term. With $n_D$ the outward normal of $D$ (pointing into the two removed sets), the symmetric Nitsche weak residual used for each test is

$$
R_j=\varepsilon\int_D m\nabla q_\theta\cdot\nabla\phi_j\,dx
-\varepsilon\int_{\partial D}m\,\partial_{n_D}q_\theta\,\phi_j\,ds
-\varepsilon\int_{\partial D}m\,(q_\theta-g)\,\partial_{n_D}\phi_j\,ds,
$$

where $g=0$ on $\partial A$ and $g=1$ on $\partial B$. The first boundary term is the consistency (normal-flux) term; the second is the symmetric Nitsche term. The stabilization moment is

$$
P_j=\frac{\gamma\varepsilon}{h}\int_{\partial D}m\,(q_\theta-g)\phi_j\,ds.
$$

The implementation defines the variational and boundary components by $\widehat J_{\rm var}=N_\phi^{-1}\sum_jR_j^2$ and $\widehat J_{\rm bc}=N_\phi^{-1}\sum_jP_j^2$. It optimizes these two terms with $\widehat J_{\rm data}$ using the named log-scale objective above. `volume_term_mse`, `boundary_flux_term_mse`, `nitsche_consistency_term_mse`, and `nitsche_penalty_term_mse` are logged as diagnostics; these are squared component moments, while `nitsche_weak_loss` is the square of their signed weak residual sum. Volume integrals use uniform-in-domain Monte Carlo points with the estimated area of $D$; boundary integrals use radial-angle samples weighted by the curve's arclength Jacobian. The values of $\ell,\gamma,h$ and both sample counts are in the versioned config.

Nitsche's method is a weak boundary-enforcement method. Its stabilization term is intentional and belongs to this variational method; it is not attached to `boundary_adapted`, `over_enforced`, or `under_enforced`.

## Check the equations against the implementation

| Mathematical object | Implementation |
|---|---|
| Periodic potential, generator, boundary sets, and factor $\rho$ | [periodic_two_well.py](../../../src/boundary_pinn/problems/committor/periodic_two_well.py) |
| Euler--Maruyama labels and weak/Nitsche integrals | [paper_methods.py](../../../src/boundary_pinn/problems/committor/paper_methods.py) |
| Trial functions, named losses, trainable scales, and epoch logging | [training.py](../../../src/boundary_pinn/problems/committor/training.py) |
| Reference solve, error norms, summary files, and plots | [comparison.py](../../../src/boundary_pinn/experiments/committor/periodic_two_well/comparison.py) |

These files implement this committor example; the notebook calls the version-controlled runner. To add another PDE, its problem module supplies the operator, domain sampler, boundary sampler, and reference solver, while the experiment configuration selects the model and training budget. The repository is still building that reusable PDE interface; this example is not a finished general-purpose PDE plugin system.

## Accuracy against the finite-difference reference

Training losses describe optimization. Solution accuracy is reported separately against the finest configured finite-difference approximation. `summary.json` and CPU-only `saved_model_evaluation.json` give absolute and relative grid estimates of $L^2$, full $H^1$, the $H^1$ seminorm, full $H^2$, and the $H^2$ seminorm. Boundary RMSE is evaluated on actual level-set points. The 128-to-256 reference-grid comparison estimates grid sensitivity; it is not an error bound.

The notebook displays the Sobolev error table and the final training-loss table. `losses.png` and per-method `metrics.jsonl` preserve training progress and curves; these are diagnostics, not substitutes for solution errors or publication figures.

## Run sequence and compute used

1. **Setup and tests:** install the selected repository revision and run unit tests on CPU.
2. **CPU preflight:** run the five-method code with fixed near-boundary Monte Carlo paths, tiny networks, one epoch, and coarse references. This checks the wiring, not scientific accuracy. It writes only to temporary Colab storage.
3. **CPU experiment preparation:** solve both finite-difference reference grids and generate the shared Feynman–Kac labels with `--prepare-cpu-only`. This is the potentially long CPU stage. Its step progress and resolution diagnostics are printed while it runs. The reference, labels, configuration, metadata, and log are saved in the experiment's dated Drive folder so the work survives a runtime restart.
4. **GPU training and solution evaluation:** after CPU preparation succeeds, switch Colab to a GPU runtime and use `--train-prepared` with that same folder. This loads the saved reference and labels; it does not repeat the finite-difference solves or Monte Carlo paths. Network training and inference for the solution-error tables use GPU. Only start this stage when ready to spend GPU time.
5. **Inspect:** read final training components, Sobolev errors, loss curves, and solution previews in the notebook.
6. **Evaluate saved models:** recompute solution errors from saved checkpoints on CPU; this does not retrain.

Changing the Colab runtime clears its variables and local files. After changing to GPU, rerun Setup to install the same branch and paste the path printed by CPU preparation into the GPU cell. Setup skips rerunning unit tests in the GPU runtime; the CPU tests and preflight already passed before preparation. The GPU stage checks the saved preflight marker and repository revision. It reuses the saved preparation folder on Drive and never repeats the long CPU work. The supplied GPU config is a single-seed exploratory run, not a final paper result; a paper study requires multiple seeds, reference-grid and Monte Carlo convergence checks, and matched compute budgets.

For a terminal run, use the same output directory for both stages:

```bash
RUN_DIR=runs/periodic_committor/gpu_pilot
periodic-committor experiments/committor/periodic_two_well/configs/gpu_pilot.json \
  --output-dir "$RUN_DIR" --prepare-cpu-only
# Run this second command in the GPU-enabled environment:
periodic-committor experiments/committor/periodic_two_well/configs/gpu_pilot.json \
  --output-dir "$RUN_DIR" --train-prepared
```

## Code and saved files

| Path | Role |
|---|---|
| `src/boundary_pinn/problems/committor/periodic_two_well.py` | Potential, PDE operator, periodic geometry, and samplers. |
| `src/boundary_pinn/problems/committor/paper_methods.py` | Euler--Maruyama committor labels, Nitsche weak residual, and quadrature geometry. |
| `src/boundary_pinn/problems/committor/training.py` | Five ansatzes, adaptive objective, progress logs, and restart checkpoints. |
| `src/boundary_pinn/experiments/committor/periodic_two_well/comparison.py` | Reference solve, shared data generation, orchestration, error tables, and plots. |
| `experiments/committor/periodic_two_well/configs/cpu_preflight.json` | CPU-only pipeline smoke test. |
| `experiments/committor/periodic_two_well/configs/gpu_pilot.json` | Explicit single-seed five-method exploratory settings. |
| `colab/periodic_committor.ipynb` | Colab workflow, derivation, method definitions, and visible output tables. |

Each dated GPU run folder contains `comparison_config.json`, `comparison_metadata.json`, the FDM reference `.npz`, shared `monte_carlo_labels.npz` and `.json`, `summary.json`, `losses.png`, diagnostic solution figures, and one folder per method. Each method folder contains its `metrics.jsonl`, `final_training_metrics.json`, `best.pt`, and resumable `last.pt`.

Older completed folders from the three-method pilot are preserved as-is. Those runs used `boundary_adapted`, the naïve product, and a fixed-weight `penalty` baseline, without Monte Carlo solution-value data. CPU evaluation reports their saved solution errors and recorded final losses, marks the missing data term as absent, and does not alter or overwrite the old run. Their results should not be described as the five-method paper-aligned comparison.

## References

- Raissi, M., Perdikaris, P., and Karniadakis, G. E. “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.” *Journal of Computational Physics* 378 (2019), 686–707. [doi:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045).
- Sukumar, N. and Srivastava, A. “Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks.” *Computer Methods in Applied Mechanics and Engineering* 389 (2022), 114333. [doi:10.1016/j.cma.2021.114333](https://doi.org/10.1016/j.cma.2021.114333).
- Berrone, S., Canuto, C., Pintore, M., and Sukumar, N. “Enforcing Dirichlet boundary conditions in physics-informed neural networks and variational physics-informed neural networks.” *Heliyon* 9(8) (2023), e18820. [doi:10.1016/j.heliyon.2023.e18820](https://doi.org/10.1016/j.heliyon.2023.e18820).
- Tepakbong, N., Fan, J., Zhou, X., and Zhou, D.-X. “Boundary-Adapted PINNs for Elliptic Dirichlet Problems: $H^2(\Omega)$ A Priori Error Bounds with Application to Mean Escape Time Computation.” arXiv:2607.19167 (2026). [arXiv](https://arxiv.org/abs/2607.19167).
- Khoo, Y., Lu, J., and Ying, L. “Solving for high dimensional committor functions using artificial neural networks.” [arXiv:1802.10275](https://arxiv.org/abs/1802.10275) (2018).
- Song, Z., Cameron, M. K., and Yang, H. “A Finite Expression Method for Solving High-Dimensional Committor Problems.” [arXiv:2306.12268](https://arxiv.org/abs/2306.12268) (2023).

BibTeX entries are in [`references.bib`](../../../references.bib).
