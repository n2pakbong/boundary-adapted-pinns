import numpy as np
import pytest

from boundary_pinn.trajectories import (
    euler_maruyama_exit_trajectory,
    simulate_double_well_exit_trajectory,
    simulate_duffing_exit_trajectory,
    simulate_ou_exit_trajectory,
)


@pytest.mark.parametrize(
    "simulator",
    [simulate_ou_exit_trajectory, simulate_double_well_exit_trajectory, simulate_duffing_exit_trajectory],
)
def test_paper_trajectory_helpers_are_reproducible_and_stop_at_cap(simulator):
    first = simulator(dt=0.01, max_time=0.03, seed=17)
    second = simulator(dt=0.01, max_time=0.03, seed=17)
    np.testing.assert_allclose(first.points, second.points)
    np.testing.assert_allclose(first.times, second.times)
    assert first.points.shape[0] == first.times.shape[0]
    assert first.points.shape[1] == 2
    assert first.exit_time <= 0.03


def test_generic_trajectory_stops_after_first_exit():
    result = euler_maruyama_exit_trajectory(
        lambda point: np.array([1.0]),
        np.zeros((1, 1)),
        lambda point: bool(point[0] < 0.15),
        [0.0],
        dt=0.1,
        max_time=1.0,
        seed=1,
    )
    assert result.exited
    assert result.exit_time == pytest.approx(0.2)
    np.testing.assert_allclose(result.points[:, 0], [0.0, 0.1, 0.2])


def test_generic_trajectory_validates_shapes_and_steps():
    with pytest.raises(ValueError, match="positive"):
        euler_maruyama_exit_trajectory(lambda point: point, np.eye(2), lambda point: True, [0, 0], dt=0, max_time=1)
    with pytest.raises(ValueError, match="noise_matrix"):
        euler_maruyama_exit_trajectory(lambda point: point, np.eye(1), lambda point: True, [0, 0], dt=0.1, max_time=1)
