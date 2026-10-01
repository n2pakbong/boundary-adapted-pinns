# Colab notebooks

Start with [`periodic_committor.ipynb`](periodic_committor.ipynb) for the full comparison workflow. The repository root then links the OU, double-well, and Duffing notebooks in reading order.

The repository is designed to be publicly readable. Setup cells need no GitHub access token. The experiment notebooks mount Google Drive by default so long runs can be resumed; set `SAVE_TO_DRIVE = False` to use temporary runtime storage.

Choose the runtime before setup, edit the visible Python configuration, and run the cells in order. The supplied default in every notebook is the same shallow width-64 `tanh` pattern used by the committor notebook. The model factories accept arbitrary scalar-output PyTorch architectures, and the paper/full presets retain the recorded experiment architectures and schedules. The quick settings exercise the workflow; paper-scale runs require substantially more computation.
