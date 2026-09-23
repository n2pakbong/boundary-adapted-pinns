# Repository development status

## Paper experiment notebooks

The paper-facing notebooks remain under `OU_process_2D/Models/` and `Double_Well_2D/Models/`. They record the original OU-ball and double-well comparisons. The `[WIP]` double-well drafts were removed because matching `[FINAL]` notebooks exist. No paper method notebook was removed.

`OU_process/` contains earlier OU explorations, while `Duffin_oscillator/` contains a separate first-passage example. These are retained as distinct research material.

## Python implementation

`met_pinn/` is the newer structured implementation. It currently contains an OU disk boundary-factor prototype and the periodic committor comparison. It does not yet replace the original paper notebooks or provide an arbitrary-PDE runner.

The intended extension point is `met_pinn/problems/`: each PDE supplies its operator, domain and boundary data, and collocation sampling. Shared model, optimization, checkpointing, and evaluation code should remain outside those problem modules. Each problem should have its own configuration, tests, and reference validation.

## Current Colab example

`colab/periodic_committor.ipynb` runs the periodic committor example. It is one example in the project and does not define the scope of future PDE work.