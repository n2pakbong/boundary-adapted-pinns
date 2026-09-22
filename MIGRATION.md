# Rebuild branch: scope and provenance

This branch starts a reproducible implementation without changing the legacy notebooks at the repository root. Those notebooks, their outputs, and their tracked PyTorch checkpoints remain the historical record for arXiv:2607.19167.

The first executable target is the two-dimensional OU disk benchmark. It provides an analytic validation route and exercises the boundary-adapted ansatz used in the paper. The double-well notebooks remain a preserved reference only: they will not be migrated in this stage. The Duffing material is out of scope for this stage.

The next new scientific problem will be a periodic-domain committor problem. It is deliberately not implemented here: its PDE, source/target sets, periodic representation, and validation data must be specified before code is added. The shared separation between `problems`, `models`, and `training` is intended to make that addition principled rather than another notebook fork.

Every new long run must write a self-contained directory containing:

- the exact configuration and runtime metadata;
- append-only metrics;
- a complete restart checkpoint (`last.pt`) on a fixed cadence; and
- the best checkpoint (`best.pt`).

No result is treated as a paper result until its configuration, seed, validation rule, and code revision are recorded.

## Validation status of this initial commit

The source files compile and the JSON configuration has been checked. The
execution environment used to create this branch has neither PyTorch nor
pytest, so the OU unit tests and smoke run have not been executed here. They
are required checks in a Colab runtime before this scaffold is used for a
scientific run.
