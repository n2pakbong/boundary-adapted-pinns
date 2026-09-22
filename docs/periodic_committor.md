# Periodic two-well committor experiment

This is the first post-legacy example in the rebuilt suite.  It is deliberately
two-dimensional and controlled: it tests the boundary construction without
claiming an expensive molecular-dynamics benchmark.

## Problem

On the flat torus \(\mathbb T^2=[0,1)^2\), consider overdamped Langevin
dynamics

\[
 dX_t=-\nabla V(X_t)\,dt+\sqrt{2\varepsilon}\,dW_t \pmod 1,
\]

with

\[
 V(x_1,x_2)=a[1-\cos(4\pi(x_1-1/4))]+b[1-\cos(2\pi x_2)].
\]

For periodic disks \(A\) and \(B\), centered at \((1/4,0)\) and
\((3/4,0)\), respectively, the forward committor satisfies

\[
 -\nabla V\mathbin\cdot\nabla q+\varepsilon\Delta q=0
 \quad\text{in }D=\mathbb T^2\setminus(A\cup B),\qquad
 q|_{\partial A}=0,\quad q|_{\partial B}=1.
\]

The default \(a=1\), \(b=0.6\), and \(\varepsilon=0.4\) make this a
moderately separated two-channel problem, not a rare-event stress test.

## Periodic geometry and the architecture

There is no exterior boundary of the torus.  The relevant boundary is instead
the two-component boundary of the punctured domain \(D\).  For a center
\(c_j\), the code represents a periodic set by

\[
 \omega_j(x)=4\sum_{\ell=1}^2\sin^2(\pi(x_\ell-c_{j\ell}))-r^2.
\]

This is smooth, periodic, and negative inside the set.  Its normalized form

\[
 d_j(x)=\frac{\omega_j(x)}{\sqrt{\omega_j(x)^2+|\nabla\omega_j(x)|^2}}
\]

is suitable for automatic differentiation and has unit first-order behavior at
its own boundary.  With \(d_A,d_B>0\) in \(D\), define

\[
 g=\frac{d_A}{d_A+d_B},\qquad
 \rho=\frac{d_A d_B}{\sqrt{d_A^2+d_B^2}},\qquad
 q_\theta=g+\rho N_\theta.
\]

Thus the network satisfies both boundary values exactly, and \(\rho\) has the
desired first-order normalization on either boundary component.  The
`naive_product` comparison retains the same lift but uses \(d_A d_B\), which
has exact boundary values without that normalization.  The `penalty` baseline
is an unconstrained-output MLP trained with sampled boundary penalties.  All
three neural networks use the periodic input embedding
\((\sin(2\pi x_1),\sin(2\pi x_2),\cos(2\pi x_1),\cos(2\pi x_2))\), so the
network output itself is exactly torus-periodic rather than merely being
trained on a periodic geometry.

## Why this is a useful first committor case

Neural variational formulations of the committor problem are established in the
literature, including the solver of Khoo, Lu, and Ying [1].  Finite-expression
methods provide a separate recent neural approach; the correct author order is
Song, Cameron, and Yang [2].  The present experiment is not intended to replay
either paper's benchmark.  Instead, it isolates the question relevant here:
does a boundary-adapted representation improve the strong-form PINN comparison
when the Dirichlet set boundaries are periodic and multiply connected?

The torus is also a useful stepping stone toward angular collective variables:
Gao, Li, Li, and Liu discuss committors for Langevin dynamics on manifolds and
include torus-valued molecular examples [3].  Nothing in this repository yet
claims an alanine-dipeptide result.

## Running and validating

For the complete three-method comparison, open
[`colab/periodic_committor.ipynb`](../colab/periodic_committor.ipynb) in Colab,
choose a GPU, and use **Runtime → Run all**.  The notebook writes every result
to Google Drive and supplies a resume cell for an interrupted runtime.

The underlying command is also available for a non-notebook environment:

```bash
pip install -e '.[dev]'
met-pinn-periodic-comparison configs/periodic_committor/overnight_comparison.json \
  --output-dir runs/periodic_committor/overnight
```

The comparison runs the boundary-adapted, naive-product, and penalty methods;
then saves `committor_*.png`, `losses.png`, `summary.json`, and the reference
grid as well as each method's `config.json`, `metadata.json`, `metrics.jsonl`,
`best.pt`, and `last.pt`. `last.pt` includes optimizer, scheduler, and
random-number state for an unattended restart. Add `--resume` with the same
output directory to continue from `last.pt`.

`met_pinn.reference.solve_periodic_committor_fd` provides a low-dimensional
central-difference comparator.  It is *not* a declared ground truth: run at
multiple grid resolutions and retain the convergence evidence with every paper
configuration.  Report its grid-based L2 and H1-seminorm errors only after the
reference itself is stable at the displayed precision.

## References

[1] Yuehaw Khoo, Jianfeng Lu, and Lexing Ying. *Solving for high dimensional
committor functions using artificial neural networks*. arXiv:1802.10275, 2018.

[2] Zezheng Song, Maria K. Cameron, and Haizhao Yang. *A Finite Expression
Method for Solving High-Dimensional Committor Problems*. arXiv:2306.12268,
2023.

[3] Yuan Gao, Tiejun Li, Xiaoguang Li, and Jian-Guo Liu. *Transition path
theory for Langevin dynamics on manifold: optimal control and data-driven
solver*. arXiv:2010.09988, 2020.

The corresponding BibTeX entries are in [`references.bib`](../references.bib).
