"""A two-well committor problem on the flat two-torus.

The reaction coordinate is periodic in each component.  The two target sets
are periodic disks represented by smooth trigonometric defining functions, so
the same expressions can be differentiated directly by PyTorch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PeriodicTwoWellCommittor:
    """Committor for reversible overdamped Langevin dynamics on ``[0, 1)^2``.

    The process is ``dX=-grad(V)dt+sqrt(2 epsilon)dW`` with periodic
    identification.  The committor is zero on ``A`` and one on ``B``.  The
    scalar ``set_radius`` is measured in the chordal coordinates of
    :meth:`defining_function`; values below one give two small, smooth sets.
    """

    a: float = 1.0
    b: float = 0.6
    epsilon: float = 0.4
    source_center: tuple[float, float] = (0.25, 0.0)
    target_center: tuple[float, float] = (0.75, 0.0)
    set_radius: float = 0.5

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0 < self.set_radius < 1:
            raise ValueError("set_radius must lie in (0, 1) for this sampler")
        for name, center in (
            ("source_center", self.source_center),
            ("target_center", self.target_center),
        ):
            if len(center) != 2:
                raise ValueError(f"{name} must contain two torus coordinates")
            if any(not math.isfinite(float(coordinate)) for coordinate in center):
                raise ValueError(f"{name} must contain finite coordinates")

        # For equal-radius product-chordal balls, the wrapped midpoint
        # minimizes the larger of the two distances to the centers.
        wrapped_delta = tuple(
            (float(target) - float(source) + 0.5) % 1.0 - 0.5
            for source, target in zip(self.source_center, self.target_center, strict=True)
        )
        midpoint_radius = 2.0 * math.sqrt(
            sum(math.sin(math.pi * delta / 2.0) ** 2 for delta in wrapped_delta)
        )
        if midpoint_radius <= self.set_radius + 1.0e-12:
            raise ValueError(
                "source and target periodic disks overlap or touch; "
                "choose disjoint centers or a smaller set_radius"
            )

    @staticmethod
    def _center_tensor(
        center: tuple[float, float], x: torch.Tensor
    ) -> torch.Tensor:
        return torch.as_tensor(center, device=x.device, dtype=x.dtype)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        """Periodic two-channel potential with minima near the two sets."""

        return self.a * (1.0 - torch.cos(4.0 * torch.pi * (x[..., 0] - 0.25))) + self.b * (
            1.0 - torch.cos(2.0 * torch.pi * x[..., 1])
        )

    def potential_gradient(self, x: torch.Tensor) -> torch.Tensor:
        return torch.column_stack(
            (
                4.0 * torch.pi * self.a * torch.sin(4.0 * torch.pi * (x[..., 0] - 0.25)),
                2.0 * torch.pi * self.b * torch.sin(2.0 * torch.pi * x[..., 1]),
            )
        )

    def defining_function(
        self, x: torch.Tensor, center: tuple[float, float]
    ) -> torch.Tensor:
        """Periodic defining function, negative inside the corresponding set."""

        displacement = x - self._center_tensor(center, x)
        chordal_square = 4.0 * torch.sin(torch.pi * displacement).square().sum(dim=-1)
        return chordal_square - self.set_radius**2

    def defining_gradient(
        self, x: torch.Tensor, center: tuple[float, float]
    ) -> torch.Tensor:
        displacement = x - self._center_tensor(center, x)
        return 4.0 * torch.pi * torch.sin(2.0 * torch.pi * displacement)

    def normalized_defining_function(
        self, x: torch.Tensor, center: tuple[float, float]
    ) -> torch.Tensor:
        """A first-order-normalized defining function, positive on ``D``."""

        omega = self.defining_function(x, center)
        gradient = self.defining_gradient(x, center)
        return omega / torch.sqrt(omega.square() + gradient.square().sum(dim=-1))

    def source_factor(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalized_defining_function(x, self.source_center)

    def target_factor(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalized_defining_function(x, self.target_center)

    def lifting(self, x: torch.Tensor) -> torch.Tensor:
        """A smooth lift which is zero on ``A`` and one on ``B``."""

        source = self.source_factor(x)
        target = self.target_factor(x)
        return source / (source + target)

    def boundary_adapted_factor(self, x: torch.Tensor) -> torch.Tensor:
        """First-order factor for both components of the Dirichlet boundary."""

        source = self.source_factor(x)
        target = self.target_factor(x)
        return source * target / torch.sqrt(source.square() + target.square())

    def naive_product_factor(self, x: torch.Tensor) -> torch.Tensor:
        """Exact-boundary baseline lacking first-order normalization."""

        return self.source_factor(x) * self.target_factor(x)

    def source_mask(self, x: torch.Tensor) -> torch.Tensor:
        return self.defining_function(x, self.source_center) <= 0.0

    def target_mask(self, x: torch.Tensor) -> torch.Tensor:
        return self.defining_function(x, self.target_center) <= 0.0

    def domain_mask(self, x: torch.Tensor) -> torch.Tensor:
        return ~(self.source_mask(x) | self.target_mask(x))

    def sample_interior(
        self,
        n: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Uniform rejection samples from the punctured periodic domain."""

        accepted: list[torch.Tensor] = []
        remaining = n
        for _ in range(100):
            proposal = torch.rand(
                max(2 * remaining, 32), 2, device=device, dtype=dtype, generator=generator
            )
            keep = proposal[self.domain_mask(proposal)]
            if len(keep):
                take = keep[:remaining]
                accepted.append(take)
                remaining -= len(take)
            if remaining == 0:
                return torch.cat(accepted, dim=0)
        raise RuntimeError("could not sample enough interior points; check the set geometry")

    def sample_boundary(
        self,
        which: str,
        n: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample the requested periodic level set by radial bisection.

        A Euclidean radial ray is only used to parameterize the trigonometric
        curve; bisection ensures the returned points satisfy ``omega=0``.
        """

        if which == "source":
            center = self.source_center
        elif which == "target":
            center = self.target_center
        else:
            raise ValueError("which must be 'source' or 'target'")
        angle = 2.0 * torch.pi * torch.rand(
            n, device=device, dtype=dtype, generator=generator
        )
        direction = torch.column_stack((torch.cos(angle), torch.sin(angle)))
        lower = torch.zeros(n, device=device, dtype=dtype)
        upper = torch.full((n,), 0.24, device=device, dtype=dtype)
        center_tensor = self._center_tensor(center, direction)
        for _ in range(48):
            midpoint = (lower + upper) / 2.0
            point = center_tensor + midpoint[:, None] * direction
            outside = self.defining_function(point, center) > 0.0
            upper = torch.where(outside, midpoint, upper)
            lower = torch.where(outside, lower, midpoint)
        return torch.remainder(center_tensor + lower[:, None] * direction, 1.0)

    def generator(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        grad_u = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            grad_u[:, 0], x, grad_outputs=torch.ones_like(grad_u[:, 0]), create_graph=True
        )[0][:, 0]
        u_yy = torch.autograd.grad(
            grad_u[:, 1], x, grad_outputs=torch.ones_like(grad_u[:, 1]), create_graph=True
        )[0][:, 1]
        return -(self.potential_gradient(x) * grad_u).sum(dim=-1) + self.epsilon * (u_xx + u_yy)

    def residual(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Strong-form residual ``L(model)`` for the committor equation."""

        x = x.detach().clone().requires_grad_(True)
        return self.generator(model(x).reshape(-1), x)
