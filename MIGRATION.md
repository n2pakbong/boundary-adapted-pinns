# Rebuild branch: scope and provenance

This branch starts a reproducible implementation without changing the legacy notebooks at the repository root. Those notebooks, their outputs, and their tracked PyTorch checkpoints remain the historical record for arXiv:2607.19167.

The first executable target is the two-dimensional OU disk benchmark. It provides an analytic validation route and exercises the boundary-adapted ansatz used in the paper. The double-well notebooks remain a preserved reference only: they will not be migrated in this stage. The Duffing material is out of scope for this stage.

The first new scientific problem is now a periodic-domain committor problem.
Its two periodic target sets, exact non-homogeneous boundary lift, and
normalized two-component factor are documented in
`docs/periodic_committor.md`.  The three-way comparison uses a normalized exact
factor, an exact but unnormalized product factor, and a boundary-penalty MLP.
The shared separation between `problems`, `models`, and `training` keeps this
addition a focused experiment rather than another notebook fork.

Every new long run must write a self-contained directory containing:

- the exact configuration and runtime metadata;
- append-only metrics;
- a complete restart checkpoint (`last.pt`) on a fixed cadence; and
- the best checkpoint (`best.pt`).

No result is treated as a paper result until its configuration, seed, validation rule, and code revision are recorded.

## Validation status of this initial commit

The source files compile and the JSON configuration has been checked. The
execution environment used to create this branch has neither PyTorch, SciPy,
nor pytest, so the OU and periodic-committor unit tests, the finite-difference
reference, and smoke runs have not been executed here. They are required checks
in a Colab runtime before this scaffold is used for a scientific run.
