# Periodic two-well committor

[Open the Colab notebook](https://colab.research.google.com/github/n2pakbong/boundary-adapted-pinns/blob/main/colab/periodic_committor.ipynb).

This page explains the equation, every trained method, each loss, the Monte Carlo data estimator, and the files saved by the experiment. It is intended to be readable without knowing the paper or the repository internals.

## Problem

On the flat torus \(\mathbb T^2=[0,1)^2\), consider overdamped Langevin dynamics

\[
dX_t=-\nabla V(X_t)\,dt+\sqrt{2\varepsilon}\,dW_t\pmod 1,
\qquad
V(x_1,x_2)=a[1-\cos(4\pi(x_1-1/4))]+b[1-\cos(2\pi x_2)].
\]

Let \(A\) and \(B\) be the two periodic disks centred at \((1/4,0)\) and \((3/4,0)\). The forward committor is

\[
q(x)=\mathbb P_x(\tau_B<\tau_A),
\qquad
\mathcal Lq=-\nabla V\cdot\nabla q+\varepsilon\Delta q=0\quad\text{in }D=\mathbb T^2\setminus(A\cup B),
\]

with \(q=0\) on \(\partial A\) and \(q=1\) on \(\partial B\). The example therefore solves a probability problem, not a mean escape time problem. The default values are \(a=1\), \(b=0.6\), \(\varepsilon=0.4\), and chordal radius \(r=0.5\).

## Boundary geometry and network representations

For \(j\in\{A,B\}\), with centre \(c_j\), define

\[
\omega_j(x)=4\sum_{k=1}^2\sin^2(\pi(x_k-c_{jk}))-r^2,
\qquad
d_j(x)=\frac{\omega_j(x)}{\sqrt{\omega_j(x)^2+|\nabla\omega_j(x)|^2}}.
\]

The zero level set of \(\omega_j\) is the boundary, \(d_j=0\) there, and \(|\nabla d_j|=1\) there. In \(D\), both \(d_A\) and \(d_B\) are positive. The code uses the lift and the first-order-normalized two-boundary factor

\[
g(x)=\frac{d_A(x)}{d_A(x)+d_B(x)},
\qquad
\rho(x)=\frac{d_A(x)d_B(x)}{\sqrt{d_A(x)^2+d_B(x)^2}}.
\]

Thus \(g=0\) on \(\partial A\), \(g=1\) on \(\partial B\), and \(\rho=0\) on both. The exact-boundary family is

\[
q_\theta(x)=g(x)+\rho(x)^\alpha N_\theta(x).
\]

The exponent \(\alpha\) defines three distinct members of that family: boundary-adapted \((\alpha=1)\), over-enforced \((\alpha=3/2)\), and under-enforced \((\alpha=1/2)\). All three satisfy both boundary values by construction. None has a separate boundary-residual loss. The periodic network \(N_\theta\) receives sine/cosine coordinates, so its output is periodic.

## Methods and exact objectives

Every method uses the same collocation and Monte Carlo anchor sets within a comparison. Write

\[
L_{\rm PDE}=\frac1{N_D}\sum_{i=1}^{N_D}\left(\mathcal Lq_\theta(x_i)\right)^2,
\qquad
L_{\rm MC}=\frac1{N_M}\sum_{i=1}^{N_M}\left(q_\theta(z_i)-\widehat q_i\right)^2.
\]

The five methods are:

| Method | Trial function | Raw loss components | Boundary treatment |
|---|---|---|---|
| `simple_pinn` | \(q_\theta=N_\theta\) | \(L_{\rm PDE},L_{\rm MC},L_{\partial}\) | Standard sampled Dirichlet residual penalty. |
| `boundary_adapted` | \(q_\theta=g+\rho N_\theta\) | \(L_{\rm PDE},L_{\rm MC}\) | Both values exact; no boundary penalty. |
| `over_enforced` | \(q_\theta=g+\rho^{3/2}N_\theta\) | \(L_{\rm PDE},L_{\rm MC}\) | Both values exact; no boundary penalty. |
| `under_enforced` | \(q_\theta=g+\rho^{1/2}N_\theta\) | \(L_{\rm PDE},L_{\rm MC}\) | Both values exact; no boundary penalty. |
| `variational_nitsche` | \(q_\theta=N_\theta\) | \(L_{\rm weak},L_{\rm Nitsche},L_{\rm MC}\) | Weak imposition with Nitsche consistency, symmetry, and stabilization terms. |

For the standard PINN only,

\[
L_{\partial}=\frac1{N_A}\sum_i q_\theta(x_i^A)^2+
\frac1{N_B}\sum_i(q_\theta(x_i^B)-1)^2.
\]

This boundary-residual penalty is part of the ordinary unconstrained PINN. It is not applied to the three exact-boundary representations. The variational method has a different boundary stabilization term, described below; it is part of Nitsche's weak boundary treatment, not a penalty added to a boundary-adapted ansatz.

### Monte Carlo solution-value regularization

The paper's mean-escape-time notebooks use simulated solution values as a pointwise regularizer. Here the corresponding committor target is the probability of hitting \(B\) before \(A\). Starting from a common anchor \(z_i\in D\), the code simulates

\[
X_{n+1}=\bigl[X_n-\nabla V(X_n)\,\Delta t+\sqrt{2\varepsilon\Delta t}\,\xi_n\bigr]\pmod 1,
\qquad \xi_n\sim\mathcal N(0,I_2),
\]

until the path first enters \(A\) or \(B\), or reaches the configured step cap. The label is

\[
\widehat q_i=\frac{\#\{\text{paths entering }B\text{ first from }z_i\}}
{\#\{\text{paths from }z_i\text{ resolved by the step cap}\}}.
\]

The code records each anchor's resolved-path fraction and the conditional binomial standard error calculated from resolved paths. It stops with an error if the configured minimum resolved fraction is missed; censored trajectories are never silently treated as source hits. If some paths remain unresolved below that threshold, the conditional estimate can still have truncation bias. The time step and path cap also introduce estimator choices: the discrete Euler--Maruyama target has time-discretization and Monte Carlo error. Those settings and diagnostics are saved in `monte_carlo_labels.json`. The same saved anchors and labels are used by all five methods.

This is the committor analogue of the pointwise Monte Carlo regularizer in the mean-escape-time experiments. It is an extension to this example, not a claim that the committor experiment reproduces the paper's mean-escape-time outputs.

### Adaptive balancing and the logged losses

The paper's loss balancing is applied to the raw components listed above. Each component \(L_c\) has a trainable scalar \(s_c\), and the optimized objective is

\[
J(\theta,s)=\sum_{c}\left(e^{-2s_c}L_c(\theta)+s_c\right).
\]

The effective weight is \(e^{-2s_c}\). For example, `simple_pinn` uses the PDE, Monte Carlo data, and boundary-residual components in this sum; `boundary_adapted` uses only the PDE and Monte Carlo data components. The optimizer learns the weights along with the network. The objective includes the trainable \(s_c\) offsets and can be negative; it is an optimization quantity, not a physical residual norm. The objective, every raw component, each weighted component, and each current effective weight are written to `metrics.jsonl` during training. `losses.png` plots the objective and the raw component histories separately (using a linear axis for any series that takes nonpositive values). `final_training_metrics.json` and `summary.json` save the final values and the best epoch/objective.

### Variational/Nitsche formulation

The drift is a gradient, so with \(m(x)=e^{-V(x)/\varepsilon}\),

\[
\mathcal Lq=0\quad\Longleftrightarrow\quad \nabla\cdot(\varepsilon m\nabla q)=0.
\]

For a periodic Gaussian test function centred at \(c_j\),

\[
\phi_j(x)=\exp\!\left[-\frac{4\sum_{k=1}^2\sin^2(\pi(x_k-c_{jk}))}{2\ell^2}\right],
\]

integration by parts gives a volume term and a boundary flux term. With \(n_D\) the outward normal of \(D\) (pointing into the two removed sets), the symmetric Nitsche weak residual used for each test is

\[
R_j=\varepsilon\int_D m\nabla q_\theta\cdot\nabla\phi_j\,dx
-\varepsilon\int_{\partial D}m\,\partial_{n_D}q_\theta\,\phi_j\,ds
-\varepsilon\int_{\partial D}m\,(q_\theta-g)\,\partial_{n_D}\phi_j\,ds,
\]

where \(g=0\) on \(\partial A\) and \(g=1\) on \(\partial B\). The first boundary term is the consistency (normal-flux) term; the second is the symmetric Nitsche term. The stabilization moment is

\[
P_j=\frac{\gamma\varepsilon}{h}\int_{\partial D}m\,(q_\theta-g)\phi_j\,ds.
\]

The implementation minimizes \(L_{\rm weak}=N_\phi^{-1}\sum_jR_j^2\) and \(L_{\rm Nitsche}=N_\phi^{-1}\sum_jP_j^2\), alongside \(L_{\rm MC}\), with the same trainable loss-balancing formula above. `volume_term_mse`, `boundary_flux_term_mse`, `nitsche_consistency_term_mse`, and `nitsche_penalty_term_mse` are logged as diagnostics; these are squared component moments, while `nitsche_weak_loss` is the square of their signed weak residual sum. Volume integrals use uniform-in-domain Monte Carlo points with the estimated area of \(D\); boundary integrals use radial-angle samples weighted by the curve's arclength Jacobian. The values of \(\ell,\gamma,h\) and both sample counts are in the versioned config.

Nitsche's method is a weak boundary-enforcement method. Its stabilization term is intentional and belongs to this variational method; it is not attached to `boundary_adapted`, `over_enforced`, or `under_enforced`.

## Accuracy against the finite-difference reference

Training losses describe optimization. Solution accuracy is reported separately against the finest configured finite-difference approximation. `summary.json` and CPU-only `saved_model_evaluation.json` give absolute and relative grid estimates of \(L^2\), full \(H^1\), the \(H^1\) seminorm, full \(H^2\), and the \(H^2\) seminorm. Boundary RMSE is evaluated on actual level-set points. The 128-to-256 reference-grid comparison estimates grid sensitivity; it is not an error bound.

The notebook displays the Sobolev error table and the final training-loss table. `losses.png` and per-method `metrics.jsonl` preserve training progress and curves; these are diagnostics, not substitutes for solution errors or publication figures.

## Run sequence and compute used

1. **Setup and tests:** install the selected repository revision and run unit tests on CPU.
2. **CPU preflight:** run the five-method code with fixed near-boundary Monte Carlo paths, tiny networks, one epoch, and coarse references. This tests wiring, not scientific accuracy. It writes only to temporary Colab storage.
3. **GPU comparison:** calculate the configured Monte Carlo labels on CPU once, then train the five networks on GPU. Finite differences, plotting, and saved-model evaluation use CPU. The output path is printed before training.
4. **Inspect:** read final training components, Sobolev errors, loss curves, and solution previews.
5. **Evaluate saved models:** recompute solution errors from saved checkpoints on CPU; this does not retrain.

The supplied GPU config is a single-seed exploratory run. It is not a final paper result; establishing one requires multiple seeds, adequate Monte Carlo and reference-grid convergence checks, and an explicit comparison of budgets. Running the CPU preflight does not start any GPU training.

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
- Tepakbong, N., Fan, J., Zhou, X., and Zhou, D.-X. “Boundary-Adapted PINNs for Elliptic Dirichlet Problems: \(H^2(\Omega)\) A Priori Error Bounds with Application to Mean Escape Time Computation.” arXiv:2607.19167 (2026). [arXiv](https://arxiv.org/abs/2607.19167).
- Khoo, Y., Lu, J., and Ying, L. “Solving for high dimensional committor functions using artificial neural networks.” arXiv:1802.10275 (2018).
- Song, Z., Cameron, M. K., and Yang, H. “A Finite Expression Method for Solving High-Dimensional Committor Problems.” arXiv:2306.12268 (2023).

BibTeX entries are in [`references.bib`](../../../references.bib).
