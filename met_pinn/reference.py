"""Small finite-difference reference solvers used for two-dimensional checks.

These solvers are deliberately limited to low dimensions.  They provide a
transparent numerical comparator for PINN experiments, not an unqualified
ground truth: grid convergence must be checked for every reported reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class PeriodicCommittorReference:
    """Finite-difference solution and masks on a uniform periodic grid."""

    grid: np.ndarray
    values: np.ndarray
    source: np.ndarray
    target: np.ndarray
    domain: np.ndarray
    spacing: float


def _periodic_omega(
    x: np.ndarray, y: np.ndarray, center: tuple[float, float], radius: float
) -> np.ndarray:
    return 4.0 * (
        np.sin(np.pi * (x - center[0])) ** 2 + np.sin(np.pi * (y - center[1])) ** 2
    ) - radius**2


def solve_periodic_committor_fd(problem: object, grid_size: int = 128) -> PeriodicCommittorReference:
    """Solve the toroidal committor by second-order central differences.

    ``problem`` is intentionally duck-typed to avoid a run-time torch import
    in this NumPy/SciPy reference path; it must provide the fields used below.
    Boundary-set grid nodes are held at their prescribed values.  The operator
    is central-difference discretized and therefore should be used only after
    a resolution study at the chosen parameters.
    """

    if grid_size < 8:
        raise ValueError("grid_size must be at least 8")
    h = 1.0 / grid_size
    axis = np.arange(grid_size, dtype=float) * h
    x, y = np.meshgrid(axis, axis, indexing="ij")
    source = _periodic_omega(x, y, problem.source_center, problem.set_radius) <= 0.0
    target = _periodic_omega(x, y, problem.target_center, problem.set_radius) <= 0.0
    if np.any(source & target):
        raise ValueError("source and target sets overlap on this grid")
    domain = ~(source | target)
    unknown_index = -np.ones((grid_size, grid_size), dtype=int)
    unknown_index[domain] = np.arange(domain.sum())
    matrix = lil_matrix((domain.sum(), domain.sum()), dtype=float)
    rhs = np.zeros(domain.sum(), dtype=float)
    grad_x = 4.0 * np.pi * problem.a * np.sin(4.0 * np.pi * (x - 0.25))
    grad_y = 2.0 * np.pi * problem.b * np.sin(2.0 * np.pi * y)
    diffusion = problem.epsilon / h**2

    for i, j in zip(*np.where(domain)):
        row = unknown_index[i, j]
        matrix[row, row] = -4.0 * diffusion
        neighbours = (
            ((i + 1) % grid_size, j, diffusion - grad_x[i, j] / (2.0 * h)),
            ((i - 1) % grid_size, j, diffusion + grad_x[i, j] / (2.0 * h)),
            (i, (j + 1) % grid_size, diffusion - grad_y[i, j] / (2.0 * h)),
            (i, (j - 1) % grid_size, diffusion + grad_y[i, j] / (2.0 * h)),
        )
        for ni, nj, coefficient in neighbours:
            neighbour_row = unknown_index[ni, nj]
            if neighbour_row >= 0:
                matrix[row, neighbour_row] += coefficient
            elif target[ni, nj]:
                rhs[row] -= coefficient

    solution = spsolve(matrix.tocsr(), rhs)
    values = np.zeros((grid_size, grid_size), dtype=float)
    values[target] = 1.0
    values[domain] = solution
    return PeriodicCommittorReference(
        grid=np.stack((x, y), axis=-1),
        values=values,
        source=source,
        target=target,
        domain=domain,
        spacing=h,
    )


def periodic_reference_errors(
    prediction: np.ndarray, reference: PeriodicCommittorReference
) -> dict[str, float]:
    """Return discrete L2 and H1 errors on stencils wholly in the open domain."""

    if prediction.shape != reference.values.shape:
        raise ValueError("prediction must have the reference grid shape")
    domain = reference.domain
    h = reference.spacing
    l2 = np.sqrt(np.mean((prediction[domain] - reference.values[domain]) ** 2))
    safe = domain.copy()
    for axis in (0, 1):
        safe &= np.roll(domain, 1, axis=axis) & np.roll(domain, -1, axis=axis)
    if not np.any(safe):
        raise ValueError("no full interior stencils are available for H1 error")
    gradient_error_square = np.zeros_like(prediction, dtype=float)
    for axis in (0, 1):
        pred_derivative = (np.roll(prediction, -1, axis=axis) - np.roll(prediction, 1, axis=axis)) / (2.0 * h)
        ref_derivative = (
            np.roll(reference.values, -1, axis=axis) - np.roll(reference.values, 1, axis=axis)
        ) / (2.0 * h)
        gradient_error_square += (pred_derivative - ref_derivative) ** 2
    h1_seminorm = np.sqrt(np.mean(gradient_error_square[safe]))
    return {"grid_l2": float(l2), "grid_h1_seminorm": float(h1_seminorm)}
