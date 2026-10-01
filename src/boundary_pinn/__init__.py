"""Neural-network tools and PDE benchmarks for boundary-adapted PINNs."""

from .models import BoundaryFactorPINN, LiftedBoundaryFactorPINN, MLP, PeriodicMLP
from .trajectories import (
    ExitTrajectory,
    euler_maruyama_exit_trajectory,
    simulate_double_well_exit_trajectory,
    simulate_duffing_exit_trajectory,
    simulate_ou_exit_trajectory,
)

__all__ = [
    "BoundaryFactorPINN",
    "ExitTrajectory",
    "LiftedBoundaryFactorPINN",
    "MLP",
    "PeriodicMLP",
    "euler_maruyama_exit_trajectory",
    "simulate_double_well_exit_trajectory",
    "simulate_duffing_exit_trajectory",
    "simulate_ou_exit_trajectory",
]
