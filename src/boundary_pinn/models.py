"""Network architectures shared across problem families."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn


class SquaredReLU(nn.Module):
    """Squared rectified linear unit, x -> max(0, x)^2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x).square()


class _CallableActivation(nn.Module):
    """Wrap a stateless PyTorch callable as a module with configured arguments."""

    def __init__(self, function: Callable[..., torch.Tensor], kwargs: Mapping[str, Any]):
        super().__init__()
        self.function = function
        self.kwargs = dict(kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.function(x, **self.kwargs)


def _activation(name: str, kwargs: Mapping[str, Any] | None = None) -> nn.Module:
    """Build an activation from a PyTorch module name or importable callable path.

    Bare names resolve against torch.nn. Qualified paths can select any
    importable nn.Module subclass or unary tensor callable; constructor or
    function keyword arguments are passed through from the run configuration.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("activation must be a nonempty module name or import path")

    if kwargs is None:
        options: dict[str, Any] = {}
    elif isinstance(kwargs, Mapping):
        options = dict(kwargs)
    else:
        raise TypeError("activation_kwargs must be a mapping of constructor arguments")

    module_name = name.strip()
    if module_name.lower() == "squaredrelu":
        activation_type: type[nn.Module] | None = SquaredReLU
    else:
        bare_name = module_name[3:] if module_name.startswith("nn.") else module_name
        target: Any = None
        if "." not in bare_name:
            for namespace in (nn, nn.modules.activation):
                candidate = getattr(namespace, bare_name, None)
                if isinstance(candidate, type) and issubclass(candidate, nn.Module):
                    target = candidate
                    break
                candidate = next(
                    (
                        value
                        for exported_name, value in vars(namespace).items()
                        if exported_name.lower() == bare_name.lower()
                        and isinstance(value, type)
                        and issubclass(value, nn.Module)
                    ),
                    None,
                )
                if candidate is not None:
                    target = candidate
                    break
        else:
            module_path, _, attribute_path = bare_name.rpartition(".")
            try:
                target = importlib.import_module(module_path)
                for attribute in attribute_path.split("."):
                    target = getattr(target, attribute)
            except (ImportError, AttributeError) as error:
                raise ValueError(
                    f"Could not resolve activation import path {name!r}"
                ) from error

        if isinstance(target, type) and issubclass(target, nn.Module):
            activation_type = target
        elif callable(target) and not isinstance(target, type):
            return _CallableActivation(target, options)
        else:
            raise ValueError(
                f"Activation {name!r} must name a PyTorch nn.Module class "
                "or an importable callable."
            )

    try:
        return activation_type(**options)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Could not construct activation {name!r} with arguments {options!r}: {error}"
        ) from error


class MLP(nn.Module):
    """Scalar fully connected network with an explicit input/output dimension."""

    def __init__(
        self,
        input_dim: int,
        width: int,
        depth: int,
        activation: str = "tanh",
        activation_kwargs: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least one hidden layer")
        layers: list[nn.Module] = [
            nn.Linear(input_dim, width),
            _activation(activation, activation_kwargs),
        ]
        for _ in range(depth - 1):
            layers.extend(
                (
                    nn.Linear(width, width),
                    _activation(activation, activation_kwargs),
                )
            )
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class PeriodicMLP(nn.Module):
    """An MLP composed with one or more integer-frequency torus features.

    Frequency one reproduces the original four features exactly. Multiple
    positive integer frequencies add periodic Fourier features while preserving
    unit-periodicity in each coordinate.
    """

    def __init__(self, network: MLP, frequencies: tuple[float, ...] = (1.0,)):
        super().__init__()
        if not frequencies or any(
            float(value) <= 0.0 or not float(value).is_integer() for value in frequencies
        ):
            raise ValueError("frequencies must be a nonempty sequence of positive integers")
        self.network = network
        self.frequencies = tuple(float(value) for value in frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frequencies == (1.0,):
            phase = 2.0 * torch.pi * x
            features = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        else:
            frequencies = torch.as_tensor(self.frequencies, device=x.device, dtype=x.dtype)
            phase = 2.0 * torch.pi * x.unsqueeze(-2) * frequencies.view(
                *((1,) * (x.ndim - 1)), -1, 1
            )
            features = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1).flatten(
                start_dim=-2
            )
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
