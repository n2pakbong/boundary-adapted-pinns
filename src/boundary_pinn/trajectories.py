"""Reusable Euler--Maruyama trajectories for the paper's exit-time examples.

The functions in this module generate one path for explanation and plotting.
The notebooks contain their own vectorized simulators for Monte Carlo labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


Vector = NDArray[np.float64]


@dataclass(frozen=True)
class ExitTrajectory:
    """One stopped path, including its initial and final points."""

    points: NDArray[np.float64]
    times: NDArray[np.float64]
    exited: bool

    @property
    def exit_time(self) -> float:
        """Return the stopping time or the time cap for a censored path."""

        return float(self.times[-1])


def euler_maruyama_exit_trajectory(
    drift: Callable[[Vector], ArrayLike],
    noise_matrix: ArrayLike,
    inside: Callable[[Vector], bool],
    start: ArrayLike,
    *,
    dt: float,
    max_time: float,
    seed: int | None = None,
) -> ExitTrajectory:
    """Simulate one Euler--Maruyama path until it leaves a domain.

    ``noise_matrix`` is the coefficient multiplying a standard Brownian
    increment, so one update is ``x + drift(x) * dt + G @ N(0, dt I)``.
    """

    if dt <= 0 or max_time <= 0:
        raise ValueError("dt and max_time must be positive")
    point = np.asarray(start, dtype=float).copy()
    if point.ndim != 1:
        raise ValueError("start must be a one-dimensional point")
    noise = np.asarray(noise_matrix, dtype=float)
    if noise.shape != (point.size, point.size):
        raise ValueError("noise_matrix must be square and match the point dimension")

    rng = np.random.default_rng(seed)
    points = [point.copy()]
    times = [0.0]
    exited = not bool(inside(point))
    max_steps = int(np.ceil(max_time / dt))

    for step in range(1, max_steps + 1):
        if exited:
            break
        step_dt = min(dt, max_time - times[-1])
        if step_dt <= 0:
            break
        drift_value = np.asarray(drift(point), dtype=float)
        if drift_value.shape != point.shape:
            raise ValueError("drift(point) must have the same shape as point")
        point = point + drift_value * step_dt + noise @ rng.normal(size=point.size) * np.sqrt(step_dt)
        points.append(point.copy())
        times.append(min(step * dt, max_time))
        exited = not bool(inside(point))

    return ExitTrajectory(np.asarray(points), np.asarray(times), exited)


def simulate_ou_exit_trajectory(
    start: ArrayLike = (0.0, 0.0),
    *,
    theta: float = 3.0,
    sigma_squared: float = 2.0,
    radius: float = 2.0,
    dt: float = 1e-3,
    max_time: float = 20.0,
    seed: int | None = None,
) -> ExitTrajectory:
    """Simulate ``dX = -theta X dt + sqrt(sigma_squared) dB`` in a disk."""

    dimension = np.asarray(start).size
    return euler_maruyama_exit_trajectory(
        lambda point: -theta * point,
        np.sqrt(sigma_squared) * np.eye(dimension),
        lambda point: bool(np.dot(point, point) < radius**2),
        start,
        dt=dt,
        max_time=max_time,
        seed=seed,
    )


def simulate_double_well_exit_trajectory(
    start: ArrayLike = (0.0, 0.0),
    *,
    potential_height: float = 5.0,
    well_width: float = 0.1,
    diffusion_epsilon: float = 1.2,
    ellipse_axes: tuple[float, float] = (2.0, 1.5),
    dt: float = 1e-3,
    max_time: float = 20.0,
    seed: int | None = None,
) -> ExitTrajectory:
    """Simulate the anisotropic double-well diffusion used in the notebook."""

    a, b = ellipse_axes

    def drift(point: Vector) -> Vector:
        x, y = point
        d1 = (x - 1.0) ** 2 + (y - 0.5) ** 2
        d2 = (x + 1.0) ** 2 + (y + 0.5) ** 2
        e1, e2 = np.exp(-d1 / well_width**2), np.exp(-d2 / well_width**2)
        f1, f2 = 1.0 - e1, 1.0 - e2
        dphi_x = potential_height * (
            2 * (x - 1.0) / well_width**2 * e1 * f2
            + f1 * 2 * (x + 1.0) / well_width**2 * e2
        )
        dphi_y = potential_height * (
            2 * (y - 0.5) / well_width**2 * e1 * f2
            + f1 * 2 * (y + 0.5) / well_width**2 * e2
        )
        return np.array([-dphi_x, -dphi_y])

    noise = np.diag([np.sqrt(2 * diffusion_epsilon), np.sqrt(diffusion_epsilon)])
    return euler_maruyama_exit_trajectory(
        drift,
        noise,
        lambda point: bool((point[0] / a) ** 2 + (point[1] / b) ** 2 < 1.0),
        start,
        dt=dt,
        max_time=max_time,
        seed=seed,
    )


def simulate_duffing_exit_trajectory(
    start: ArrayLike = (0.0, 0.0),
    *,
    zeta: float = 0.08,
    epsilon: float = 1e-5,
    radius: float = 2.0,
    dt: float = 2e-3,
    max_time: float = 20.0,
    seed: int | None = None,
) -> ExitTrajectory:
    """Simulate the stochastic Duffing oscillator used in the notebook."""

    def drift(point: Vector) -> Vector:
        x1, x2 = point
        return np.array([x2, -x1 - x1**3 - 2 * zeta * x2])

    noise = np.diag([np.sqrt(2 * epsilon), np.sqrt(2 * zeta)])
    return euler_maruyama_exit_trajectory(
        drift,
        noise,
        lambda point: bool(np.dot(point, point) < radius**2),
        start,
        dt=dt,
        max_time=max_time,
        seed=seed,
    )
