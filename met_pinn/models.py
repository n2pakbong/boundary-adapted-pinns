"""Network architectures shared across problem families."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def _activation(name: str) -> Callable[[], nn.Module]:
    if name == "tanh":
        return nn.Tanh
    if name == "relu":
        return nn.ReLU
    raise ValueError(f"Unknown activation {name!r}; use 'tanh' or 'relu'.")


class MLP(nn.Module):
    """Scalar fully connected network with an explicit input/output dimension."""

    def __init__(self, input_dim: int, width: int, depth: int, activation: str = "tanh"):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least one hidden layer")
        make_activation = _activation(activation)
        layers: list[nn.Module] = [nn.Linear(input_dim, width), make_activation()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), make_activation()))
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class PeriodicMLP(nn.Module):
    """An MLP composed with a sine/cosine embedding of the unit torus."""

    def __init__(self, network: MLP):
        super().__init__()
        self.network = network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * torch.pi * x
        features = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        return self.network(features)


class BoundaryFactorPINN(nn.Module):
    """``u(x)=rho(x)^exponent N(x)`` with a supplied boundary factor ``rho``."""

    def __init__(
        self,
        network: nn.Module,
        boundary_factor: Callable[[torch.Tensor], torch.Tensor],
        exponent: float = 1.0,
    ):
        super().__init__()
        if exponent <= 0:
            raise ValueError("boundary-factor exponent must be positive")
        self.network = network
        self.boundary_factor = boundary_factor
        self.exponent = exponent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rho = self.boundary_factor(x)
        # All current exact-boundary problems sample within their closed domain.
        return rho.clamp_min(0.0).pow(self.exponent) * self.network(x)


class LiftedBoundaryFactorPINN(nn.Module):
    """``u(x)=g(x)+rho(x)N(x)`` for non-homogeneous Dirichlet data.

    ``g`` supplies the boundary values and ``rho`` vanishes on every Dirichlet
    component.  Keeping the lifting and factor as explicit callables makes the
    architecture useful for multiply-connected problems without baking a
    particular PDE geometry into the network implementation.
    """

    def __init__(
        self,
        network: nn.Module,
        lifting: Callable[[torch.Tensor], torch.Tensor],
        boundary_factor: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.network = network
        self.lifting = lifting
        self.boundary_factor = boundary_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lifting(x) + self.boundary_factor(x) * self.network(x)
