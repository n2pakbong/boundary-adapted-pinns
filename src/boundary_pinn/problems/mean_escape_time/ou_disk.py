"""Two-dimensional Ornstein--Uhlenbeck mean-exit-time problem in a disk."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OUDiskProblem:
    """Solve ``L u = -1`` in ``{|x| < radius}``, with ``u=0`` on its boundary.

    Here ``L u = -theta x . grad(u) + sigma^2 Delta(u)/2``.  The class keeps
    only the problem data and exact geometry; method and training choices live
    elsewhere.
    """

    radius: float = 2.0
    theta: float = 1.0
    sigma: float = 1.0

    def signed_distance_factor(self, x: torch.Tensor) -> torch.Tensor:
        """First-order-normalized defining function used by the legacy PINNs.

        It is positive inside the disk, vanishes on the boundary, and has
        outward normal derivative -1 on the boundary.
        """

        return (self.radius**2 - x.square().sum(dim=-1)) / (2.0 * self.radius)

    def sample_interior(
        self,
        n: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw points uniformly in area from the open disk."""

        angle = 2.0 * torch.pi * torch.rand(
            n, device=device, dtype=dtype, generator=generator
        )
        radius = self.radius * torch.sqrt(
            torch.rand(n, device=device, dtype=dtype, generator=generator)
        )
        return torch.column_stack((radius * torch.cos(angle), radius * torch.sin(angle)))

    def sample_boundary(
        self,
        n: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw points uniformly in arclength from the boundary circle."""

        angle = 2.0 * torch.pi * torch.rand(
            n, device=device, dtype=dtype, generator=generator
        )
        return self.radius * torch.column_stack((torch.cos(angle), torch.sin(angle)))

    def generator(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the OU generator of scalar network values ``u(x)``."""

        grad_u = torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )[0]
        u_xx = torch.autograd.grad(
            grad_u[:, 0],
            x,
            grad_outputs=torch.ones_like(grad_u[:, 0]),
            create_graph=True,
        )[0][:, 0]
        u_yy = torch.autograd.grad(
            grad_u[:, 1],
            x,
            grad_outputs=torch.ones_like(grad_u[:, 1]),
            create_graph=True,
        )[0][:, 1]
        drift = -self.theta * (x * grad_u).sum(dim=-1)
        diffusion = 0.5 * self.sigma**2 * (u_xx + u_yy)
        return drift + diffusion

    def residual(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Strong-form residual ``L(model) + 1`` for the mean-exit-time BVP."""

        x = x.detach().clone().requires_grad_(True)
        u = model(x).reshape(-1)
        return self.generator(u, x) + 1.0
