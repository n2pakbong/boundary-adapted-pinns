# Colab notebooks

Start with [periodic_committor.ipynb](periodic_committor.ipynb) for the committor workflow. The repository root links the OU, double-well, and Duffing notebooks.

## Saving experiment runs

The OU, double-well, and Duffing notebooks draw fresh seeds for each new run and create a separate run folder. They save to Google Drive by default; set SAVE_TO_DRIVE = False to use the runtime's local runs folder. Local files disappear when Colab deletes the runtime. Choose RUN_MODE = "resume" or "evaluate" only with the exact folder path printed for the run you want.

The committor notebook uses a staged workflow and saves preparation and training results to a dated Google Drive folder. Keep the entire run folder and the notebook source used for that run. The folder contains the reference, Monte Carlo labels, checkpoints, per-epoch metrics, evaluated arrays, figures, and run settings.

Download or copy the complete run folder when you need to archive it. The individual metrics files alone are not a substitute for the saved numerical arrays.

## Runtime and setup

Choose the runtime before running setup, then follow the notebook cells in order. The supplied default network in the experiment notebooks is a two-hidden-layer, width-64 tanh model. The model factories accept other scalar-output PyTorch architectures; the paper or full presets restore each experiment's recorded architecture and schedule. Quick settings exercise the workflow, while paper-scale runs require substantially more computation.
