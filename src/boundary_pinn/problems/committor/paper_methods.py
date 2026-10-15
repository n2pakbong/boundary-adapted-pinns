"""Monte Carlo labels and weak Nitsche terms for the periodic committor.

The pointwise labels approximate q(x)=P_x(tau_B < tau_A) by independent
Euler--Maruyama paths.  The weak method uses the reversible divergence form
of the committor equation and Gaussian test functions on the flat torus.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from .periodic_two_well import (
    InteriorSampler,
    PeriodicTwoWellCommittor,
    sample_interior_points,
)


@dataclass(frozen=True)
class CommittorMonteCarloData:
    """Common interior anchors and Monte Carlo estimates used by all methods."""

    anchors: torch.Tensor
    values: torch.Tensor
    standard_errors: torch.Tensor
    resolved_paths: torch.Tensor
    total_paths: torch.Tensor
    metadata: dict[str, float | int | str]


def _omega_numpy(
    points: np.ndarray, center: tuple[float, float], radius: float
) -> np.ndarray:
    displacement = points - np.asarray(center, dtype=np.float64)
    return 4.0 * np.square(np.sin(np.pi * displacement)).sum(axis=-1) - radius**2


def generate_committor_mc_data(
    problem: PeriodicTwoWellCommittor,
    *,
    n_anchors: int,
    paths_per_anchor: int,
    dt: float,
    max_steps: int,
    seed: int,
    minimum_resolved_fraction: float = 0.9,
    anchor_points: list[list[float]] | None = None,
    interior_sampler: InteriorSampler | None = None,
    progress_interval_fraction: float = 0.01,
    progress_interval_seconds: float = 10.0,
) -> CommittorMonteCarloData:
    """Estimate committor values by capped, vectorized Euler--Maruyama paths.

    Paths start at common interior anchors and evolve as
    ``X[n+1] = X[n] - grad(V(X[n])) dt + sqrt(2 epsilon dt) Z[n] (mod 1)``.
    A path is labelled one if it enters B before A, and zero if it enters A
    first.  The function fails if any anchor has too many paths unresolved at
    ``max_steps``; silently treating censored paths as zero would bias q.
    """

    if n_anchors < 1 or paths_per_anchor < 1 or max_steps < 1 or dt <= 0:
        raise ValueError("Monte Carlo counts and dt must be positive")
    if not 0.0 <= minimum_resolved_fraction <= 1.0:
        raise ValueError("minimum_resolved_fraction must lie in [0, 1]")
    if not 0.0 < progress_interval_fraction <= 1.0:
        raise ValueError("progress_interval_fraction must lie in (0, 1]")
    if progress_interval_seconds <= 0.0:
        raise ValueError("progress_interval_seconds must be positive")

    if anchor_points is None:
        anchor_generator = torch.Generator(device="cpu").manual_seed(int(seed))
        anchors = sample_interior_points(
            problem,
            n_anchors,
            device="cpu",
            dtype=torch.float64,
            generator=anchor_generator,
            sampler=interior_sampler,
        ).numpy()
        anchor_source = (
            "uniform_rejection_from_domain"
            if interior_sampler is None
            else f"custom:{interior_sampler.__module__}.{interior_sampler.__qualname__}"
        )
    else:
        anchors = np.asarray(anchor_points, dtype=np.float64)
        if anchors.shape != (n_anchors, 2):
            raise ValueError("anchor_points must have shape (n_anchors, 2)")
        anchor_tensor = torch.as_tensor(anchors, dtype=torch.float64)
        if not torch.all(problem.domain_mask(anchor_tensor)):
            raise ValueError("all supplied Monte Carlo anchors must lie in D")
        anchor_source = "explicit_cpu_preflight_points"
    states = np.broadcast_to(
        anchors[:, None, :], (n_anchors, paths_per_anchor, 2)
    ).copy()
    alive = np.ones((n_anchors, paths_per_anchor), dtype=bool)
    successes = np.zeros(n_anchors, dtype=np.int64)
    resolved = np.zeros(n_anchors, dtype=np.int64)
    rng = np.random.default_rng(seed + 1)
    noise_scale = np.sqrt(2.0 * problem.epsilon * dt)

    started = time.monotonic()
    next_time_report = started + progress_interval_seconds
    progress_interval = max(1, int(np.ceil(max_steps * progress_interval_fraction)))
    total_paths = n_anchors * paths_per_anchor
    for step in range(max_steps):
        anchor_ids, path_ids = np.nonzero(alive)
        if anchor_ids.size == 0:
            break
        x = states[anchor_ids, path_ids]
        drift = np.column_stack((
            -4.0 * np.pi * problem.a * np.sin(4.0 * np.pi * (x[:, 0] - 0.25)),
            -2.0 * np.pi * problem.b * np.sin(2.0 * np.pi * x[:, 1]),
        ))
        x = np.remainder(x + drift * dt + noise_scale * rng.standard_normal(x.shape), 1.0)
        states[anchor_ids, path_ids] = x
        hit_a = _omega_numpy(x, problem.source_center, problem.set_radius) <= 0.0
        hit_b = _omega_numpy(x, problem.target_center, problem.set_radius) <= 0.0
        # The two sets are disjoint for the configured geometry.
        hit_b &= ~hit_a
        hit = hit_a | hit_b
        if np.any(hit):
            np.add.at(resolved, anchor_ids[hit], 1)
            np.add.at(successes, anchor_ids[hit_b], 1)
            alive[anchor_ids[hit], path_ids[hit]] = False
        now = time.monotonic()
        at_step_milestone = (step + 1) % progress_interval == 0
        at_time_milestone = now >= next_time_report
        if max_steps >= 1_000 and (at_step_milestone or at_time_milestone):
            fractions_now = resolved / float(paths_per_anchor)
            anchors_ready = int(np.count_nonzero(fractions_now >= minimum_resolved_fraction))
            elapsed = now - started
            resolved_b = int(successes.sum())
            resolved_a = int(resolved.sum()) - resolved_b
            active_now = int(alive.sum())
            print(
                "Feynman-Kac paths | "
                f"step cap {step + 1:,}/{max_steps:,} "
                f"({100.0 * (step + 1) / max_steps:.1f}%) | "
                f"resolved A/B {resolved_a:,}/{resolved_b:,} of {total_paths:,} | "
                f"active {active_now:,} | "
                f"anchors at ≥{minimum_resolved_fraction:.0%}: {anchors_ready}/{n_anchors} | "
                f"resolution min/mean {100.0 * fractions_now.min():.1f}%/"
                f"{100.0 * fractions_now.mean():.1f}% | elapsed {elapsed / 60.0:.1f} min",
                flush=True,
            )
            if at_time_milestone:
                next_time_report = now + progress_interval_seconds

    unresolved = paths_per_anchor - resolved
    fractions = resolved / float(paths_per_anchor)
    if np.any(resolved == 0):
        raise RuntimeError(
            "No Monte Carlo paths resolved at one or more anchors. Increase max_steps; "
            "a label with zero resolved paths is not defined."
        )
    if np.any(fractions < minimum_resolved_fraction):
        worst = float(fractions.min())
        raise RuntimeError(
            "Monte Carlo committor labels are too censored: the least-resolved "
            f"anchor has {worst:.1%} resolved paths, below the configured "
            f"minimum {minimum_resolved_fraction:.1%}. Increase max_steps or "
            "revise the explicitly recorded estimator settings."
        )
    elapsed_seconds = time.monotonic() - started
    print(
        "Feynman-Kac labels complete | "
        f"paths hit A or B: {int(resolved.sum()):,}/{n_anchors * paths_per_anchor:,} | "
        f"anchor resolution minimum/mean: {100.0 * fractions.min():.1f}%/"
        f"{100.0 * fractions.mean():.1f}% | elapsed {elapsed_seconds / 60.0:.1f} min",
        flush=True,
    )
    estimates = successes / resolved
    standard_errors = np.sqrt(estimates * (1.0 - estimates) / resolved)
    metadata: dict[str, float | int | str] = {
        "seed": int(seed),
        "anchors": int(n_anchors),
        "anchor_source": anchor_source,
        "paths_per_anchor": int(paths_per_anchor),
        "dt": float(dt),
        "max_steps": int(max_steps),
        "nominal_time_cap": float(dt * max_steps),
        "minimum_resolved_fraction": float(minimum_resolved_fraction),
        "mean_resolved_fraction": float(fractions.mean()),
        "minimum_observed_resolved_fraction": float(fractions.min()),
        "unresolved_paths": int(unresolved.sum()),
        "generation_elapsed_seconds": float(elapsed_seconds),
        "progress_interval_fraction": float(progress_interval_fraction),
        "progress_interval_seconds": float(progress_interval_seconds),
    }
    return CommittorMonteCarloData(
        anchors=torch.as_tensor(anchors, dtype=torch.float64),
        values=torch.as_tensor(estimates, dtype=torch.float64),
        standard_errors=torch.as_tensor(standard_errors, dtype=torch.float64),
        resolved_paths=torch.as_tensor(resolved, dtype=torch.int64),
        total_paths=torch.full((n_anchors,), paths_per_anchor, dtype=torch.int64),
        metadata=metadata,
    )


def _gaussian_tests(
    points: torch.Tensor, centers: torch.Tensor, length_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Periodic Gaussian tests and their gradients, shaped [test, point, ...]."""

    displacement = points.unsqueeze(0) - centers.unsqueeze(1)
    distance_square = 4.0 * torch.sin(torch.pi * displacement).square().sum(dim=-1)
    grad_distance_square = 4.0 * torch.pi * torch.sin(2.0 * torch.pi * displacement)
    tests = torch.exp(-distance_square / (2.0 * length_scale**2))
    gradients = -tests.unsqueeze(-1) * grad_distance_square / (2.0 * length_scale**2)
    return tests, gradients


def sample_boundary_geometry(
    problem: PeriodicTwoWellCommittor,
    which: str,
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return points, the outward normal of D, and arclength quadrature weights.

    The existing sampler draws a uniform radial angle.  The weight
    ``(2 pi / n) |dx/dtheta|`` converts that sampling to an arclength integral.
    The domain's outward normal points into the corresponding excluded set.
    """

    center = problem.source_center if which == "source" else problem.target_center
    points = problem.sample_boundary(
        which, n, device=device, dtype=dtype, generator=generator
    )
    center_t = torch.as_tensor(center, device=device, dtype=dtype)
    displacement = torch.remainder(points - center_t + 0.5, 1.0) - 0.5
    radius = displacement.norm(dim=-1).clamp_min(torch.finfo(dtype).eps)
    angle = torch.atan2(displacement[:, 1], displacement[:, 0])
    radial = torch.column_stack((torch.cos(angle), torch.sin(angle)))
    tangent = torch.column_stack((-torch.sin(angle), torch.cos(angle)))
    grad_omega = problem.defining_gradient(points, center)
    denominator = (grad_omega * radial).sum(dim=-1)
    dr_dtheta = -radius * (grad_omega * tangent).sum(dim=-1) / denominator
    speed = torch.sqrt(radius.square() + dr_dtheta.square())
    normal = -grad_omega / grad_omega.norm(dim=-1, keepdim=True)
    weights = (2.0 * torch.pi / n) * speed
    return points, normal, weights


def estimate_domain_area(
    problem: PeriodicTwoWellCommittor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    points: int = 4096,
    seed: int | None = None,
) -> float:
    """Estimate |D| by periodic-boundary quadrature with a caller-supplied budget."""

    if points < 1:
        raise ValueError("area quadrature needs at least one point")
    n = int(points)
    areas = []
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(seed))
    for which, center in (("source", problem.source_center), ("target", problem.target_center)):
        points, _, _ = sample_boundary_geometry(
            problem, which, n, device=device, dtype=dtype, generator=generator
        )
        center_t = torch.as_tensor(center, device=device, dtype=dtype)
        displacement = torch.remainder(points - center_t + 0.5, 1.0) - 0.5
        radius_square = displacement.square().sum(dim=-1)
        # Uniform-angle polar integration: area = 1/2 integral r(theta)^2 dtheta.
        areas.append(float(torch.pi * radius_square.mean().detach().cpu()))
    domain_area = 1.0 - sum(areas)
    if not 0.0 < domain_area <= 1.0:
        raise RuntimeError("invalid area estimate for the punctured torus")
    return domain_area


def nitsche_committor_losses(
    model: torch.nn.Module,
    problem: PeriodicTwoWellCommittor,
    *,
    volume_points: torch.Tensor,
    test_centers: torch.Tensor,
    boundary_points: int,
    domain_area: float,
    length_scale: float,
    gamma: float,
    h: float,
    boundary_samples: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Evaluate paper-aligned interior-plus-flux and Nitsche boundary loss components.

    The reversible density is m=exp(-V/epsilon), so
    Lq=0 is equivalent to div(epsilon*m*grad(q))=0.  The returned first dict contains signed per-test contributions; the second
    contains the two squared loss components and diagnostic means.
    """

    if length_scale <= 0 or gamma <= 0 or h <= 0:
        raise ValueError("Nitsche length_scale, gamma, and h must be positive")
    x = volume_points.detach().clone().requires_grad_(True)
    q = model(x).reshape(-1)
    grad_q = torch.autograd.grad(q.sum(), x, create_graph=True)[0]
    phi, grad_phi = _gaussian_tests(x, test_centers, length_scale)
    density = torch.exp(-problem.potential(x) / problem.epsilon)
    volume = domain_area * problem.epsilon * (
        density.unsqueeze(0) * (grad_q.unsqueeze(0) * grad_phi).sum(dim=-1)
    ).mean(dim=1)

    flux = torch.zeros_like(volume)
    consistency = torch.zeros_like(volume)
    penalty = torch.zeros_like(volume)
    for which, value in (("source", 0.0), ("target", 1.0)):
        if boundary_samples is None:
            xb, normal, weights = sample_boundary_geometry(
                problem, which, boundary_points, device=x.device, dtype=x.dtype
            )
        else:
            xb, normal, weights = boundary_samples[which]
            xb = xb.to(device=x.device, dtype=x.dtype)
            normal = normal.to(device=x.device, dtype=x.dtype)
            weights = weights.to(device=x.device, dtype=x.dtype)
            if xb.shape != (boundary_points, 2) or normal.shape != (boundary_points, 2):
                raise ValueError(f"fixed {which} boundary data has the wrong point count")
            if weights.shape != (boundary_points,):
                raise ValueError(f"fixed {which} boundary weights have the wrong point count")
        xb = xb.detach().clone().requires_grad_(True)
        qb = model(xb).reshape(-1)
        grad_qb = torch.autograd.grad(qb.sum(), xb, create_graph=True)[0]
        phi_b, grad_phi_b = _gaussian_tests(xb, test_centers, length_scale)
        density_b = torch.exp(-problem.potential(xb) / problem.epsilon)
        q_error = qb - value
        normal_flux = (grad_qb * normal).sum(dim=-1)
        normal_test_derivative = (grad_phi_b * normal.unsqueeze(0)).sum(dim=-1)
        weighted_density = weights * problem.epsilon * density_b
        flux = flux - (weighted_density.unsqueeze(0) * normal_flux.unsqueeze(0) * phi_b).sum(dim=1)
        consistency = consistency - (
            weighted_density.unsqueeze(0) * q_error.unsqueeze(0) * normal_test_derivative
        ).sum(dim=1)
        penalty = penalty + (
            (gamma / h) * weighted_density.unsqueeze(0) * q_error.unsqueeze(0) * phi_b
        ).sum(dim=1)

    pde_component = volume + flux
    boundary_component = consistency + penalty
    weak_residual = pde_component + boundary_component
    terms = {
        "volume_term": volume,
        "boundary_flux_term": flux,
        "nitsche_consistency_term": consistency,
        "nitsche_penalty_term": penalty,
        "pde_component": pde_component,
        "boundary_component": boundary_component,
        "weak_residual": weak_residual,
    }
    losses = {
        "pde_loss": pde_component.square().mean(),
        "boundary_loss": boundary_component.square().mean(),
        "full_weak_residual_mse": weak_residual.square().mean(),
        "volume_term_mse": volume.square().mean(),
        "boundary_flux_term_mse": flux.square().mean(),
        "nitsche_consistency_term_mse": consistency.square().mean(),
        "nitsche_penalty_term_mse": penalty.square().mean(),
    }
    return terms, losses
