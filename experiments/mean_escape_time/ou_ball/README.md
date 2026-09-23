# Ornstein–Uhlenbeck mean escape time in a ball

These notebooks contain the two-dimensional OU-ball comparison reported in the boundary-adapted PINN paper. They solve the mean escape time Dirichlet problem and compare five approaches: standard PINN, variational PINN, over-enforced PINN, under-enforced PINN, and boundary-adapted PINN.

These five notebooks are the source records for the paper's comparison: standard PINN, variational PINN, over-enforced PINN, under-enforced PINN, and boundary-adapted PINN. Their derivations, explanations, citations, training procedures, and saved outputs accompany the experiment.

The notebooks use the original Colab workflow. Their full training cells may require GPU time. The separately maintained Python OU-disk example in `src/boundary_pinn/problems/mean_escape_time/ou_disk.py` is a small implementation example; it is not a replacement for these paper notebooks.
