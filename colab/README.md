# Running the periodic two-well committor example

Open the [committor notebook in Colab](periodic_committor.ipynb) and run its cells in order.

1. **Setup and CPU tests.** Fetch the selected Git branch, install the repository, and run the component tests. This stage does not train a network.
2. **CPU preflight.** Run the small one-epoch, five-method pipeline check. Its temporary files go under Colab's /content storage and disappear with that runtime; the preflight does not write results to Drive and does not measure scientific accuracy.
3. **Choose the experiment.** Edit the labelled controls beside the explanations for the problem, methods, network, training, Monte Carlo labels, and reference solve. The final review cell assembles the choices and prints a readable summary; no nested override dictionary needs editing. The starting values match [`gpu_pilot.json`](../experiments/committor/periodic_two_well/configs/gpu_pilot.json).
4. **CPU preparation.** Generate the finite-difference reference and shared Monte Carlo labels. This is the longer CPU-only stage. Its outputs and progress log are saved in a new dated Google Drive run folder.
5. **Training.** The `gpu_pilot.json` preset selects five methods for 6,000 epochs each. If you change the notebook controls, the selected methods and duration follow those choices. Training uses the saved device setting: CUDA by default, or CPU if you select CPU. A GPU runtime is needed only for CUDA training.
6. **Inspect, evaluate, or resume.** Read the saved metrics and plots, evaluate saved checkpoints on CPU, or resume interrupted training from its Drive folder.

Keep the Drive folder from CPU preparation: it contains the resolved configuration, code revision, reference, Monte Carlo labels, and progress metadata required by training. The training stage reads those prepared files rather than generating them again.

While the repository is private, setup reads the GITHUB_TOKEN Colab secret to fetch the selected branch. The token permits repository access only; it is not used for training and does not access Drive. If the repository is made public, setup works without a token.
