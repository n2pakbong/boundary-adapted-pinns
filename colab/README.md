# Colab use

Open [`periodic_committor.ipynb`](periodic_committor.ipynb) in Colab, choose a
GPU runtime, and use **Runtime → Run all**. It installs the branch, tests it,
runs the full three-method periodic-committor comparison, and saves checkpoints,
reference data, summaries, and figures in Google Drive.

The notebook prints the exact Drive run directory before training. If Colab
disconnects, reopen the notebook, rerun its installation cell, copy that
directory into the supplied resume cell, and run the command with `--resume`.
