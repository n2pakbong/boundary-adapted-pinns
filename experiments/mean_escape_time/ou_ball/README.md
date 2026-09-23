# Ornstein–Uhlenbeck mean escape time in a ball

These notebooks contain the two-dimensional OU-ball comparison reported in the boundary-adapted PINN paper. They solve the mean escape time Dirichlet problem and compare five approaches: standard PINN, variational PINN, over-enforced PINN, under-enforced PINN, and boundary-adapted PINN.

The five notebooks correspond to the methods compared in the paper: standard PINN, variational PINN, over-enforced PINN, under-enforced PINN, and boundary-adapted PINN. Their derivations, explanations, citations, training procedures, and saved outputs remain with the experiment records.

The notebooks use the original Colab workflow. Their full training cells may require GPU time. The separately maintained Python OU-disk example in `src/boundary_pinn/problems/mean_escape_time/ou_disk.py` is a small implementation example; it is not a replacement for these paper notebooks.
