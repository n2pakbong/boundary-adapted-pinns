import torch

from met_pinn.models import BoundaryFactorPINN, MLP
from met_pinn.problems import OUDiskProblem


def test_boundary_factor_has_unit_first_order_normalisation() -> None:
    problem = OUDiskProblem(radius=2.0)
    x = torch.tensor([[2.0, 0.0]], dtype=torch.float64, requires_grad=True)
    rho = problem.signed_distance_factor(x)
    gradient = torch.autograd.grad(rho, x)[0]
    assert torch.allclose(rho, torch.zeros_like(rho))
    # On the boundary the outward normal is (1, 0), so d_n rho=-1.
    assert torch.allclose(gradient[:, 0], torch.tensor([-1.0], dtype=torch.float64))


def test_boundary_factor_network_is_zero_on_boundary() -> None:
    problem = OUDiskProblem(radius=2.0)
    model = BoundaryFactorPINN(MLP(2, width=4, depth=1), problem.signed_distance_factor)
    x = problem.sample_boundary(12, device="cpu", dtype=torch.float32)
    assert torch.allclose(model(x), torch.zeros(12), atol=1e-6)


def test_ou_generator_of_quadratic_function() -> None:
    problem = OUDiskProblem(radius=2.0, theta=1.5, sigma=0.5)
    x = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.float64, requires_grad=True)
    u = x.square().sum(dim=-1)
    actual = problem.generator(u, x)
    expected = -2.0 * problem.theta * x.square().sum(dim=-1) + 2.0 * problem.sigma**2
    assert torch.allclose(actual, expected)
