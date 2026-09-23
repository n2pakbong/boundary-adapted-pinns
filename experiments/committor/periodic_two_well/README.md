# Periodic two-well committor

This example solves a committor equation on the flat torus with two interior target sets. It tests boundary-adapted representations on a periodic, multiply connected domain.

## Problem

For overdamped Langevin dynamics
[
dX_t=-\nabla V(X_t)\,dt+\sqrt{2\varepsilon}\,dW_t\pmod 1,
]
use the periodic potential
[
V(x_1,x_2)=a[1-\cos(4\pi(x_1-1/4))]+b[1-\cos(2\pi x_2)].
]
The periodic disks (A) and (B), centered at ((1/4,0)) and ((3/4,0)), define the source and target. The forward committor solves
[
-\nabla V\cdot\nabla q+\varepsilon\Delta q=0
\quad\text{in }D=\mathbb T^2\setminus(A\cup B),\qquad
q|_{\partial A}=0,\quad q|_{\partial B}=1.
]
Default parameters are (a=1), (b=0.6), and (arepsilon=0.4).

## Boundary construction and methods

For a periodic disk centered at (c_j), define
[
\omega_j(x)=4\sum_{\ell=1}^2\sin^2(\pi(x_\ell-c_{j\ell}))-r^2,
\qquad
d_j(x)=\frac{\omega_j(x)}{\sqrt{\omega_j(x)^2+|\nabla\omega_j(x)|^2}}.
]
The normalized factors (d_A,d_B) vanish on their own boundary and have unit first-order behavior there. With
[
g=\frac{d_A}{d_A+d_B},\qquad
\rho=\frac{d_Ad_B}{\sqrt{d_A^2+d_B^2}},
\]
the boundary-adapted ansatz (q_\theta=g+\rho N_\theta) satisfies both Dirichlet values exactly and uses the normalized two-boundary factor.

The comparison includes three methods:

- **Boundary-adapted:** exact lift and normalized factor (ho).
- **Naïve product:** the same lift with the unnormalized product (d_Ad_B).
- **Penalty:** a periodic MLP trained with sampled boundary penalties.

Each network uses a sine/cosine embedding of the torus coordinates, so it is periodic by construction.

## Run the experiment

Open [`colab/periodic_committor.ipynb`](../../../colab/periodic_committor.ipynb). Its setup and tests use the CPU, followed by a tiny end-to-end CPU preflight. A separate cell runs the three-method GPU pilot for 6,000 epochs per method. It writes checkpoints, metrics, reference data, summaries, and figures to Google Drive.

The same comparison can be run from a Python environment:

```bash
pip install -e '.[dev]'
periodic-committor experiments/committor/periodic_two_well/configs/gpu_pilot.json \
  --output-dir runs/periodic_committor/gpu_pilot
```

Use `--resume` with the same output directory to continue an interrupted run from its saved `last.pt` checkpoint.

The finite-difference reference is a numerical comparator, not exact ground truth. Comparing its results on multiple grid resolutions shows whether reference discretization error is small relative to the PINN errors.

## References

[1] Yuehaw Khoo, Jianfeng Lu, and Lexing Ying. *Solving for high dimensional committor functions using artificial neural networks*. arXiv:1802.10275, 2018.

[2] Zezheng Song, Maria K. Cameron, and Haizhao Yang. *A Finite Expression Method for Solving High-Dimensional Committor Problems*. arXiv:2306.12268, 2023.

[3] Yuan Gao, Tiejun Li, Xiaoguang Li, and Jian-Guo Liu. *Transition path theory for Langevin dynamics on manifold: optimal control and data-driven solver*. arXiv:2010.09988, 2020.

The corresponding BibTeX entries are in [`references.bib`](../../../references.bib).
