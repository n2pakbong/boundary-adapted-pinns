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
| `variational_nitsche` | $q_\theta=N_\theta$ | $\widehat J_{\rm pde},\widehat J_{\rm bc},\widehat J_{\rm data}$ | Weak imposition with Nitsche consistency, symmetry, and stabilization terms. |

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

The code records each anchor's resolved-path fraction and the conditional binomial standard error calculated from resolved paths. During simulation, Colab reports step-cap progress, paths resolved to (A) and (B), still-active paths, anchors meeting the resolution threshold, minimum and mean anchor resolution, and elapsed time. Reports occur at the configured fraction of the step cap or time interval, whichever comes first. After labels finish, `monte_carlo_anchors.png` plots the anchor locations and estimated labels with standard errors. It prints a final resolution summary before training starts. It stops with an error if the configured minimum resolved fraction is missed; censored trajectories are never silently treated as source hits. If some paths remain unresolved below that threshold, the conditional estimate can still have truncation bias. The time step and path cap also introduce estimator choices: the discrete Euler--Maruyama target has time-discretization and Monte Carlo error. Those settings and diagnostics are saved in `monte_carlo_labels.json`. The same saved anchors and labels are used by all five methods.

This is the committor analogue of the pointwise Monte Carlo regularizer in the mean-escape-time experiments. It is an extension to this example, not a claim that the committor experiment reproduces the paper's mean-escape-time outputs.

### Adaptive balancing and the logged losses

The paper denotes the total adaptive objective by $\widehat J$ and its active empirical loss components by $\widehat J_i$. For each component, $\lambda_i>0$ is a trainable scale [Tepakbong et al. (2026), Eq. (33)](https://arxiv.org/abs/2607.19167):

$$
\widehat J(\theta,\{\lambda_i\}_{i\in\mathcal I})
=\sum_{i\in\mathcal I}\frac{1}{2\lambda_i^2}\widehat J_i(\theta)
+\sum_{i\in\mathcal I}\log\lambda_i.
$$

The active components match the paper's method-specific choices:

- simple_pinn: $\mathcal I=\{\mathrm{pde},\mathrm{bc},\mathrm{data}\}$; $\widehat J_{\rm bc}$ is the sampled Dirichlet residual.
- boundary_adapted, over_enforced, and under_enforced: $\mathcal I=\{\mathrm{pde},\mathrm{data}\}$; the boundary values are exact and there is no boundary-residual loss.
- variational_nitsche: $\mathcal I=\{\mathrm{pde},\mathrm{bc},\mathrm{data}\}$; $\widehat J_{\rm pde}$ is the interior-plus-flux loss and $\widehat J_{\rm bc}$ is the Nitsche boundary contribution, including symmetric consistency and stabilization. This is not the pointwise boundary loss used by simple_pinn.

For this committor, $\widehat J_{\rm pde}$ uses the residual $\mathcal Lq_\theta=0$, and $\widehat J_{\rm data}$ compares $q_\theta(z_i)$ with the simulated committor label $\widehat q_i$. In the Nitsche method, the code records these three optimized components as pde_loss, boundary_loss, and data_loss.

**How the saved code scale matches the paper's $\lambda_i$.** The implementation stores its trainable scales in log form, $s_i=\log\sigma_i$, and multiplies each loss by $e^{-2s_i}$. The correspondence is $\lambda_i=e^{s_i}/\sqrt{2}$, so $e^{-2s_i}=1/(2\lambda_i^2)$. Therefore the total code objective differs from the displayed paper objective only by the constant $|\mathcal I|\log(2)/2$. This constant has zero gradient; the component weights exactly equal the paper's coefficients. The raw losses and effective weights are reported separately. The trainable-scale idea follows [Kendall, Gal, and Cipolla (2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html).

Training reports are controlled by `training.log_interval_seconds`, not a fixed epoch count. The first epoch, each elapsed-time interval, and the final epoch are printed with elapsed time, estimated remaining time, objective, and active raw loss components. `metrics.jsonl` records every epoch, including raw and weighted components and effective adaptive weights; `losses.png` plots the complete per-epoch history. Console messages remain time-sampled. `training.checkpoint_every` independently controls restartable-checkpoint frequency. `final_training_metrics.json` and `summary.json` save final component values and the best epoch/objective.

### Variational/Nitsche formulation

Because the drift is a gradient, with $m(x)=e^{-V(x)/\varepsilon}$ the committor equation is equivalent to

$$
\mathcal Lq=0\quad\Longleftrightarrow\quad \nabla\cdot(\varepsilon m\nabla q)=0.
$$

For periodic Gaussian test functions $\phi_j$ centred at $c_j$, let $n_D$ be the outward normal of the exterior domain $D$, pointing into the removed sets. The implementation separates the Nitsche residual into the paper's interior-plus-flux contribution and boundary contribution:

$$
A_j=
\varepsilon\int_Dm\nabla q_\theta\cdot\nabla\phi_j\,dx
-\varepsilon\int_{\partial D}m\,\partial_{n_D}q_\theta\,\phi_j\,ds.
$$

The symmetric consistency term and stabilization moment are

$$
C_j=-\varepsilon\int_{\partial D}m(q_\theta-g)\,\partial_{n_D}\phi_j\,ds,
\qquad
P_j=\frac{\gamma\varepsilon}{h}\int_{\partial D}m(q_\theta-g)\phi_j\,ds,
$$

where $g=0$ on $\partial A$ and $g=1$ on $\partial B$. They form the Nitsche boundary contribution $B_j=C_j+P_j$, and the full weak residual is $R_j=A_j+B_j$.

The trained losses use the same split as the paper:

$$
\widehat J_{\rm pde}=\frac1{N_\phi}\sum_{j=1}^{N_\phi}A_j^2,
\qquad
\widehat J_{\rm bc}=\frac1{N_\phi}\sum_{j=1}^{N_\phi}B_j^2.
$$

These are optimized with the common pointwise committor-label loss $\widehat J_{\rm data}$. In the saved training records, these components are pde_loss, boundary_loss, and data_loss; their adaptive weights are the paper's $\lambda_{\rm pde}$, $\lambda_{\rm bc}$, and $\lambda_{\rm data}$ coefficients. The pointwise boundary-residual loss is not used by this Nitsche method. The separately logged volume, flux, consistency, stabilization, and full-residual mean squares are diagnostics, not additional adaptive losses. The Nitsche formulation follows [Liao and Ming (2021)](https://doi.org/10.4208/cicp.OA-2020-0219) and the variational PINN discussion in [Berrone et al. (2023)](https://doi.org/10.1016/j.heliyon.2023.e18820).

The values of $\ell,\gamma,h$ and the volume, test-function, and boundary sample counts are in the versioned config.

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

## Choosing and changing experiment settings

The values used for the successful five-method pilot are in [`gpu_pilot.json`](configs/gpu_pilot.json). A field-by-field guide, including the Monte Carlo sample count and noise, optimizer options, initialization, reporting interval, and reference grids, is in [the configuration guide](configs/README.md).

The periodic network uses the same sine/cosine coordinate embedding and fully connected MLP across methods. The default has width 64, depth 4, `tanh`, Xavier-normal weights with gain 1, zero biases, and seed 7. This keeps the network core comparable while changing the boundary ansatz. Smooth `tanh` supports the strong-form residual's second derivatives. Width and depth set the network's capacity. These defaults reproduce the successful pilot; they have not been selected by a hyperparameter sweep.

The `device` and `dtype` settings select CPU or CUDA and float32 or float64. Full CPU training avoids allocating a GPU runtime but can be substantially slower.

The training preset uses Adam with learning rate 0.001 and exponential decay `gamma=0.9999` for 6,000 epochs. The 1,024 PDE collocation points are resampled each epoch. The standard PINN uses 256 fresh points on each Dirichlet boundary. The exact-boundary methods have no pointwise boundary loss. Nitsche instead uses its explicitly listed volume, test-function, and boundary-quadrature counts.

The committor data term uses 32 anchor points, each labelled with 256 independent paths (8,192 paths total). Finite path counts create binomial sampling noise. The time step and path cap also affect the estimator through discretization and censoring. All five methods reuse the same anchors and labels. The preparation plot shows the anchor locations, labels, and estimated standard errors before training.

In Colab, the **Experiment settings** cell loads the tracked preset and accepts overrides without changing training source. CPU preparation saves the resolved configuration in the run folder. After the runtime restarts, training reads that saved configuration. The temporary `/content` copy is used only to hand the selected settings to CPU preparation; the durable copy is `comparison_config.json` in the Drive run folder. To reuse an override, copy that file into `configs/` under a descriptive name and point `PRESET_CONFIG_PATH` in the notebook settings cell at it. The configuration guide lists every field and supported value.

## Run sequence and compute used

1. **Setup and tests:** install the selected repository revision and run unit tests on CPU.
2. **CPU preflight:** run the five-method code with fixed near-boundary Monte Carlo paths, tiny networks, one epoch, and coarse references. This checks the wiring, not scientific accuracy. It writes only to temporary Colab storage.
3. **CPU experiment preparation:** solve both finite-difference reference grids and generate the shared Feynman–Kac labels with `--prepare-cpu-only`. This is the potentially long CPU stage. Its step progress and resolution diagnostics are printed while it runs. The reference, labels, configuration, metadata, and log are saved in the experiment's dated Drive folder so the work survives a runtime restart.
4. **Training and solution evaluation:** after CPU preparation succeeds, use `--train-prepared` with that same folder. If `device` is `cuda`, switch Colab to a GPU runtime first; if it is `cpu`, stay on CPU. This loads the saved reference and labels; it does not repeat the finite-difference solves or Monte Carlo paths.
5. **Inspect:** read final training components, Sobolev errors, loss curves, and solution previews in the notebook.
6. **Evaluate saved models:** recompute solution errors from saved checkpoints on CPU; this does not retrain.

CPU preparation files record the repository revision. Reuse them only with the same code revision; if the code changes, rerun CPU preparation before training. The training runner checks the saved revision and refuses mismatched preparation. Resuming a checkpoint also requires the same objective version; if the version is incompatible, use a new dated run folder rather than combining different training objectives.

Changing the Colab runtime clears its variables and local files. If the saved configuration selects `cuda`, change Colab to GPU, rerun Setup with the same branch, and paste the path printed by CPU preparation into the training cell. If it selects `cpu`, stay on CPU and use that same training cell. Setup skips rerunning unit tests in the GPU runtime; the CPU tests and preflight already passed before preparation. Both paths reuse the saved preparation folder on Drive and never repeat the long CPU work. The supplied five-method preset uses CUDA and is a single-seed exploratory run, not a final paper result; a paper study requires multiple seeds, reference-grid and Monte Carlo convergence checks, and matched compute budgets.

For a terminal run, use the same output directory for both stages. CPU preparation always runs on CPU; run the training command on CPU or GPU to match the configuration's `device` value:

```bash
RUN_DIR=runs/periodic_committor/gpu_pilot
periodic-committor experiments/committor/periodic_two_well/configs/gpu_pilot.json \
  --output-dir "$RUN_DIR" --prepare-cpu-only
# If device is cuda, run this command in a GPU-enabled environment; if device is cpu, keep using CPU:
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

Each dated experiment folder contains `comparison_config.json`, `comparison_metadata.json`, the FDM reference `.npz`, shared `monte_carlo_labels.npz` and `.json`, `summary.json`, `losses.png`, diagnostic solution figures, and one folder per method. Each method folder contains its `metrics.jsonl`, `final_training_metrics.json`, `best.pt`, and resumable `last.pt`.

## References

- Liao, Y. and Ming, P. “Deep Nitsche Method: Deep Ritz Method with Essential Boundary Conditions.” *Communications in Computational Physics* 29(5) (2021), 1365–1384. [doi:10.4208/cicp.OA-2020-0219](https://doi.org/10.4208/cicp.OA-2020-0219).
- Raissi, M., Perdikaris, P., and Karniadakis, G. E. “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.” *Journal of Computational Physics* 378 (2019), 686–707. [doi:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045).
- Sukumar, N. and Srivastava, A. “Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks.” *Computer Methods in Applied Mechanics and Engineering* 389 (2022), 114333. [doi:10.1016/j.cma.2021.114333](https://doi.org/10.1016/j.cma.2021.114333).
- Berrone, S., Canuto, C., Pintore, M., and Sukumar, N. “Enforcing Dirichlet boundary conditions in physics-informed neural networks and variational physics-informed neural networks.” *Heliyon* 9(8) (2023), e18820. [doi:10.1016/j.heliyon.2023.e18820](https://doi.org/10.1016/j.heliyon.2023.e18820).
- Tepakbong, N., Fan, J., Zhou, X., and Zhou, D.-X. “Boundary-Adapted PINNs for Elliptic Dirichlet Problems: $H^2(\Omega)$ A Priori Error Bounds with Application to Mean Escape Time Computation.” arXiv:2607.19167 (2026). [arXiv](https://arxiv.org/abs/2607.19167).
- Khoo, Y., Lu, J., and Ying, L. “Solving for high dimensional committor functions using artificial neural networks.” [arXiv:1802.10275](https://arxiv.org/abs/1802.10275) (2018).
- Song, Z., Cameron, M. K., and Yang, H. “A Finite Expression Method for Solving High-Dimensional Committor Problems.” [arXiv:2306.12268](https://arxiv.org/abs/2306.12268) (2023).

BibTeX entries are in [`references.bib`](../../../references.bib).
