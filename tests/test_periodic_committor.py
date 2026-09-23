import torch

from met_pinn.models import LiftedBoundaryFactorPINN, MLP, PeriodicMLP
from met_pinn.problems import PeriodicTwoWellCommittor


def _problem() -> PeriodicTwoWellCommittor:
    return PeriodicTwoWellCommittor()


def test_periodic_geometry_is_invariant_under_integer_shifts() -> None:
    problem = _problem()
    x = torch.tensor([[0.11, 0.92], [0.68, 0.14]], dtype=torch.float64)
    shifted = x + torch.tensor([1.0, -2.0], dtype=torch.float64)
    assert torch.allclose(problem.source_factor(x), problem.source_factor(shifted))
    assert torch.allclose(problem.target_factor(x), problem.target_factor(shifted))
    assert torch.allclose(problem.lifting(x), problem.lifting(shifted))


def test_periodic_network_is_invariant_under_integer_shifts() -> None:
    network = PeriodicMLP(MLP(input_dim=4, width=8, depth=2)).to(dtype=torch.float64)
    x = torch.tensor([[0.11, 0.92], [0.68, 0.14]], dtype=torch.float64)
    shifted = x + torch.tensor([1.0, -2.0], dtype=torch.float64)
    assert torch.allclose(network(x), network(shifted))


def test_boundary_samples_satisfy_periodic_level_set() -> None:
    problem = _problem()
    for which, center in (("source", problem.source_center), ("target", problem.target_center)):
        boundary = problem.sample_boundary(which, 32, device="cpu", dtype=torch.float64)
        assert torch.max(torch.abs(problem.defining_function(boundary, center))) < 1.0e-10


def test_lifted_ansatz_has_exact_committor_values_on_both_components() -> None:
    problem = _problem()
    model = LiftedBoundaryFactorPINN(
        MLP(input_dim=2, width=8, depth=2), problem.lifting, problem.boundary_adapted_factor
    ).to(dtype=torch.float64)
    source = problem.sample_boundary("source", 16, device="cpu", dtype=torch.float64)
    target = problem.sample_boundary("target", 16, device="cpu", dtype=torch.float64)
    assert torch.max(torch.abs(model(source))) < 1.0e-10
    assert torch.max(torch.abs(model(target) - 1.0)) < 1.0e-10


def test_normalized_factor_has_unit_gradient_on_both_boundaries() -> None:
    problem = _problem()
    for which in ("source", "target"):
        x = problem.sample_boundary(which, 16, device="cpu", dtype=torch.float64).requires_grad_(True)
        factor = problem.boundary_adapted_factor(x)
        gradient = torch.autograd.grad(factor.sum(), x)[0]
        assert torch.allclose(gradient.norm(dim=-1), torch.ones(16, dtype=torch.float64), atol=2.0e-8)
