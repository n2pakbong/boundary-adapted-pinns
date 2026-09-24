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


def _stencil_masks(domain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return masks for first- and second-derivative stencils inside the PDE domain."""
    first = domain.copy()
    for shift in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        first &= np.roll(domain, shift=shift, axis=(0, 1))

    # The mixed derivative uses the four diagonal nodes; the pure second
    # derivatives use the axial nodes. Requiring the full 3x3 stencil makes
    # every term in the H2 norm use the same set of physical grid points.
    second = domain.copy()
    for shift_x in (-1, 0, 1):
        for shift_y in (-1, 0, 1):
            second &= np.roll(domain, shift=(shift_x, shift_y), axis=(0, 1))
    return first, second


def _grid_derivatives(values: np.ndarray, spacing: float) -> tuple[np.ndarray, ...]:
    """Compute periodic centered first and second derivatives on one grid."""
    dx = (np.roll(values, -1, axis=0) - np.roll(values, 1, axis=0)) / (2.0 * spacing)
    dy = (np.roll(values, -1, axis=1) - np.roll(values, 1, axis=1)) / (2.0 * spacing)
    dxx = (np.roll(values, -1, axis=0) - 2.0 * values + np.roll(values, 1, axis=0)) / spacing**2
    dyy = (np.roll(values, -1, axis=1) - 2.0 * values + np.roll(values, 1, axis=1)) / spacing**2
    dxy = (
        np.roll(np.roll(values, -1, axis=0), -1, axis=1)
        - np.roll(np.roll(values, -1, axis=0), 1, axis=1)
        - np.roll(np.roll(values, 1, axis=0), -1, axis=1)
        + np.roll(np.roll(values, 1, axis=0), 1, axis=1)
    ) / (4.0 * spacing**2)
    return dx, dy, dxx, dxy, dyy


def _weighted_norm(field: np.ndarray, mask: np.ndarray, spacing: float) -> float:
    """Nodal quadrature approximation to an L2 norm on a masked 2D domain."""
    return float(np.sqrt(spacing**2 * np.sum(np.square(field[mask]))))


def _relative_error(error_norm: float, reference_norm: float) -> float | None:
    """Return a relative error; use null when the reference norm is zero."""
    if reference_norm == 0.0:
        return 0.0 if error_norm == 0.0 else None
    return float(error_norm / reference_norm)


def _sobolev_error_metrics(
    prediction: np.ndarray,
    reference_values: np.ndarray,
    spacing: float,
    domain: np.ndarray,
    first_stencil: np.ndarray,
    second_stencil: np.ndarray,
    prediction_derivatives: tuple[np.ndarray, ...] | None = None,
    reference_derivatives: tuple[np.ndarray, ...] | None = None,
) -> dict[str, float | None]:
    """Compute discrete L2, H1, and H2 errors and their reference-relative values."""
    if prediction_derivatives is None:
        prediction_derivatives = _grid_derivatives(prediction, spacing)
    if reference_derivatives is None:
        reference_derivatives = _grid_derivatives(reference_values, spacing)

    error = prediction - reference_values
    derivative_errors = tuple(
        pred - ref for pred, ref in zip(prediction_derivatives, reference_derivatives)
    )
    pred_dx, pred_dy, pred_dxx, pred_dxy, pred_dyy = prediction_derivatives
    ref_dx, ref_dy, ref_dxx, ref_dxy, ref_dyy = reference_derivatives
    error_dx, error_dy, error_dxx, error_dxy, error_dyy = derivative_errors

    l2_error = _weighted_norm(error, domain, spacing)
    l2_reference = _weighted_norm(reference_values, domain, spacing)

    h1_semi_error = _weighted_norm(
        np.sqrt(np.square(error_dx) + np.square(error_dy)), first_stencil, spacing
    )
    h1_semi_reference = _weighted_norm(
        np.sqrt(np.square(ref_dx) + np.square(ref_dy)), first_stencil, spacing
    )
    h1_error = _weighted_norm(
        np.sqrt(np.square(error) + np.square(error_dx) + np.square(error_dy)),
        first_stencil,
        spacing,
    )
    h1_reference = _weighted_norm(
        np.sqrt(np.square(reference_values) + np.square(ref_dx) + np.square(ref_dy)),
        first_stencil,
        spacing,
    )

    h2_semi_error = _weighted_norm(
        np.sqrt(np.square(error_dxx) + np.square(error_dxy) + np.square(error_dyy)),
        second_stencil,
        spacing,
    )
    h2_semi_reference = _weighted_norm(
        np.sqrt(np.square(ref_dxx) + np.square(ref_dxy) + np.square(ref_dyy)),
        second_stencil,
        spacing,
    )
    h2_error = _weighted_norm(
        np.sqrt(
            np.square(error)
            + np.square(error_dx)
            + np.square(error_dy)
            + np.square(error_dxx)
            + np.square(error_dxy)
            + np.square(error_dyy)
        ),
        second_stencil,
        spacing,
    )
    h2_reference = _weighted_norm(
        np.sqrt(
            np.square(reference_values)
            + np.square(ref_dx)
            + np.square(ref_dy)
            + np.square(ref_dxx)
            + np.square(ref_dxy)
            + np.square(ref_dyy)
        ),
        second_stencil,
        spacing,
    )

    # Preserve the two historical RMS fields so existing analysis scripts keep
    # working. The new *_abs values are the area-weighted Sobolev norms.
    legacy_l2 = float(np.sqrt(np.mean(np.square(error[domain]))))
    legacy_h1_semi = float(
        np.sqrt(np.mean((np.square(error_dx) + np.square(error_dy))[first_stencil]))
    )
    return {
        "grid_l2_abs": l2_error,
        "grid_l2_rel": _relative_error(l2_error, l2_reference),
        "grid_h1_abs": h1_error,
        "grid_h1_rel": _relative_error(h1_error, h1_reference),
        "grid_h1_seminorm_abs": h1_semi_error,
        "grid_h1_seminorm_rel": _relative_error(h1_semi_error, h1_semi_reference),
        "grid_h2_abs": h2_error,
        "grid_h2_rel": _relative_error(h2_error, h2_reference),
        "grid_h2_seminorm_abs": h2_semi_error,
        "grid_h2_seminorm_rel": _relative_error(h2_semi_error, h2_semi_reference),
        "grid_l2": legacy_l2,
        "grid_h1_seminorm": legacy_h1_semi,
    }


def periodic_reference_errors(
    prediction: np.ndarray, reference: PeriodicCommittorReference
) -> dict[str, float | None]:
    """Compare a grid prediction with the FDM solution in discrete Sobolev norms.

    L2 uses every exterior-domain grid node. H1 derivatives use nodes whose
    axial centered-difference stencil lies entirely in the domain. H2 uses
    nodes whose complete 3x3 stencil lies in the domain. The quadrature weight
    is h^2. The H2 seminorm convention is
    sqrt(||e_xx||_L2^2 + ||e_xy||_L2^2 + ||e_yy||_L2^2).
    Relative errors divide by the matching norm of the FDM reference on the
    same grid and stencil mask. A relative error is null if that norm is zero.
    """
    if prediction.shape != reference.values.shape:
        raise ValueError("prediction must have the reference grid shape")
    first_stencil, second_stencil = _stencil_masks(reference.domain)
    if not np.any(first_stencil):
        raise ValueError("no full interior stencils are available for H1 error")
    if not np.any(second_stencil):
        raise ValueError("no full interior stencils are available for H2 error")
    return _sobolev_error_metrics(
        prediction,
        reference.values,
        reference.spacing,
        reference.domain,
        first_stencil,
        second_stencil,
    )


def periodic_reference_grid_errors(
    coarse: PeriodicCommittorReference, fine: PeriodicCommittorReference
) -> dict[str, float | None]:
    """Estimate FDM grid change by comparing coarse and fine solutions in Sobolev norms.

    Fine-grid derivatives are computed using the fine spacing before the fine
    arrays are sampled at coarse-grid nodes. All norms use common domain and
    stencil masks and the coarse-grid quadrature weight.
    """
    coarse_size = coarse.values.shape[0]
    fine_size = fine.values.shape[0]
    if coarse.values.ndim != 2 or coarse.values.shape[0] != coarse.values.shape[1]:
        raise ValueError("reference grids must be square")
    if fine.values.shape[0] != fine.values.shape[1] or fine_size % coarse_size:
        raise ValueError("fine grid size must be an integer multiple of the coarse grid size")
    stride = fine_size // coarse_size
    if not np.allclose(coarse.grid, fine.grid[::stride, ::stride], rtol=0.0, atol=1.0e-12):
        raise ValueError("coarse and fine grids do not share aligned periodic nodes")

    coarse_first, coarse_second = _stencil_masks(coarse.domain)
    fine_first, fine_second = _stencil_masks(fine.domain)
    fine_domain = fine.domain[::stride, ::stride]
    first_stencil = coarse_first & fine_first[::stride, ::stride]
    second_stencil = coarse_second & fine_second[::stride, ::stride]
    if not np.any(first_stencil) or not np.any(second_stencil):
        raise ValueError("no common coarse/fine derivative stencils are available")
    common_domain = coarse.domain & fine_domain

    coarse_derivatives = _grid_derivatives(coarse.values, coarse.spacing)
    fine_derivatives_full = _grid_derivatives(fine.values, fine.spacing)
    fine_derivatives = tuple(item[::stride, ::stride] for item in fine_derivatives_full)
    metrics = _sobolev_error_metrics(
        coarse.values,
        fine.values[::stride, ::stride],
        coarse.spacing,
        common_domain,
        first_stencil,
        second_stencil,
        prediction_derivatives=coarse_derivatives,
        reference_derivatives=fine_derivatives,
    )
    return {
        f"fd_{name}_{coarse_size}_to_{fine_size}": value
        for name, value in metrics.items()
        if name.startswith("grid_") and name not in {"grid_l2", "grid_h1_seminorm"}
    }
