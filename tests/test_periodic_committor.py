import numpy as np
import math
import pytest
import torch

from boundary_pinn.models import LiftedBoundaryFactorPINN, MLP, PeriodicMLP, SquaredReLU
from boundary_pinn.problems.committor import PeriodicTwoWellCommittor
from boundary_pinn.problems.committor.periodic_two_well import (
    _device_request_matches,
    sample_interior_points,
)
from boundary_pinn.problems.committor.reference import (
    PeriodicCommittorReference,
    periodic_reference_errors,
    periodic_reference_grid_errors,
)


def _problem() -> PeriodicTwoWellCommittor:
    return PeriodicTwoWellCommittor()


def test_periodic_committor_rejects_overlapping_sets() -> None:
    with pytest.raises(ValueError, match="overlap or touch"):
        PeriodicTwoWellCommittor(
            source_center=(0.25, 0.0),
            target_center=(0.26, 0.0),
            set_radius=0.1,
        )


def test_periodic_committor_rejects_touching_sets() -> None:
    touching_radius = 2.0 * math.sin(math.pi * 0.1 / 2.0)
    with pytest.raises(ValueError, match="overlap or touch"):
        PeriodicTwoWellCommittor(
            source_center=(0.0, 0.0),
            target_center=(0.1, 0.0),
            set_radius=touching_radius,
        )


def test_periodic_committor_accepts_disjoint_chordal_disks() -> None:
    # The center distance is below twice the radius, but their midpoint
    # distance still exceeds the radius, so these chordal disks are disjoint.
    problem = PeriodicTwoWellCommittor(
        source_center=(0.0, 0.0),
        target_center=(0.1, 0.0),
        set_radius=0.31,
    )
    assert problem.source_center == (0.0, 0.0)


def test_squared_relu_is_available_as_an_explicit_activation() -> None:
    values = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    expected = torch.tensor([0.0, 0.0, 0.0, 0.25, 4.0])
    assert torch.equal(SquaredReLU()(values), expected)

    from boundary_pinn.problems.committor.training import _initialize_mlp

    network = _initialize_mlp(
        {"width": 4, "depth": 1, "activation": "SquaredReLU"},
        input_dim=2,
    )
    assert any(isinstance(module, SquaredReLU) for module in network.modules())


def test_functional_pytorch_activation_import_path_accepts_kwargs() -> None:
    from boundary_pinn.models import _activation

    activation = _activation("torch.nn.functional.gelu", {"approximate": "tanh"})
    values = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.isfinite(activation(values)).all()


def test_custom_indicator_module_is_available_by_import_path(monkeypatch) -> None:
    import sys
    import types

    from boundary_pinn.models import _activation

    module = types.ModuleType("test_user_activation")

    class Indicator(torch.nn.Module):
        def forward(self, values):
            return (values > 0).to(values.dtype)

    module.Indicator = Indicator
    monkeypatch.setitem(sys.modules, "test_user_activation", module)
    activation = _activation("test_user_activation.Indicator")
    values = torch.tensor([-1.0, 0.0, 1.0])
    expected = torch.tensor([0.0, 0.0, 1.0])
    assert torch.equal(activation(values), expected)


def test_user_interior_sampler_must_return_points_inside_the_domain() -> None:
    problem = _problem()
    calls = []

    def sampler(problem, n, *, device, dtype, generator=None):
        calls.append(n)
        return problem.sample_interior(
            n, device=device, dtype=dtype, generator=generator
        )

    points = sample_interior_points(
        problem, 8, device="cpu", dtype=torch.float64, sampler=sampler
    )
    assert points.shape == (8, 2)
    assert calls == [8]

    def invalid_sampler(problem, n, *, device, dtype, generator=None):
        return torch.tensor([0.25, 0.0], device=device, dtype=dtype).expand(n, 2).clone()

    with pytest.raises(ValueError, match="outside the open domain"):
        sample_interior_points(
            problem, 2, device="cpu", dtype=torch.float64, sampler=invalid_sampler
        )


@pytest.mark.parametrize(
    ("actual", "requested", "expected"),
    [
        (torch.device("cuda:0"), torch.device("cuda"), True),
        (torch.device("cuda:1"), torch.device("cuda"), True),
        (torch.device("cuda:1"), torch.device("cuda:0"), False),
        (torch.device("cpu"), torch.device("cpu"), True),
        (torch.device("cpu"), torch.device("cuda"), False),
    ],
)
def test_unindexed_device_request_matches_current_device(actual, requested, expected) -> None:
    assert _device_request_matches(actual, requested) is expected


@pytest.mark.parametrize(
    ("activation", "kwargs", "expected_type"),
    [
        ("CELU", {"alpha": 0.5}, torch.nn.CELU),
        ("PReLU", {"num_parameters": 1, "init": 0.25}, torch.nn.PReLU),
        ("Softplus", {"beta": 2.0, "threshold": 12.0}, torch.nn.Softplus),
    ],
)
def test_torch_nn_activation_modules_accept_constructor_kwargs(
    activation, kwargs, expected_type
) -> None:
    from boundary_pinn.problems.committor.training import _initialize_mlp

    network = _initialize_mlp(
        {
            "width": 8,
            "depth": 2,
            "activation": activation,
            "activation_kwargs": kwargs,
            "initialization": "xavier_normal",
        },
        input_dim=4,
    )
    activation_layers = [
        module for module in network.modules() if isinstance(module, expected_type)
    ]
    assert len(activation_layers) == 2
    assert torch.isfinite(network(torch.randn(5, 4))).all()
    if activation == "PReLU":
        assert torch.allclose(activation_layers[0].weight, torch.tensor([0.25]))


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


def _smooth_reference(grid_size: int) -> PeriodicCommittorReference:
    axis = np.arange(grid_size, dtype=float) / grid_size
    x, y = np.meshgrid(axis, axis, indexing="ij")
    values = np.sin(2.0 * np.pi * x) + 0.3 * np.cos(2.0 * np.pi * y)
    domain = np.ones_like(values, dtype=bool)
    return PeriodicCommittorReference(
        grid=np.stack((x, y), axis=-1),
        values=values,
        source=np.zeros_like(domain),
        target=np.zeros_like(domain),
        domain=domain,
        spacing=1.0 / grid_size,
    )


def test_solution_errors_include_absolute_and_relative_l2_h1_h2_norms() -> None:
    reference = _smooth_reference(64)
    exact = periodic_reference_errors(reference.values.copy(), reference)
    names = (
        "grid_l2_abs", "grid_l2_rel",
        "grid_h1_abs", "grid_h1_rel",
        "grid_h1_seminorm_abs", "grid_h1_seminorm_rel",
        "grid_h2_abs", "grid_h2_rel",
        "grid_h2_seminorm_abs", "grid_h2_seminorm_rel",
    )
    assert all(exact[name] == pytest.approx(0.0) for name in names)

    prediction = reference.values + 0.05
    errors = periodic_reference_errors(prediction, reference)
    assert errors["grid_l2_abs"] == pytest.approx(0.05)
    assert errors["grid_l2_rel"] > 0.0
    assert errors["grid_h1_abs"] > 0.0
    assert errors["grid_h2_abs"] > 0.0
    assert errors["grid_h1_seminorm_abs"] == pytest.approx(0.0, abs=1.0e-12)
    assert errors["grid_h2_seminorm_abs"] == pytest.approx(0.0, abs=1.0e-10)


def test_fdm_grid_comparison_uses_derivatives_from_each_resolution() -> None:
    report = periodic_reference_grid_errors(
        _smooth_reference(32), _smooth_reference(64)
    )
    assert report["fd_grid_l2_abs_32_to_64"] == pytest.approx(0.0, abs=1.0e-12)
    assert report["fd_grid_l2_rel_32_to_64"] == pytest.approx(0.0, abs=1.0e-12)
    assert report["fd_grid_h1_abs_32_to_64"] > 0.0
    assert report["fd_grid_h2_abs_32_to_64"] > 0.0


def test_periodic_comparison_cpu_smoke(tmp_path, capsys) -> None:
    import json

    from boundary_pinn.experiments.committor.periodic_two_well.comparison import (
        run_periodic_comparison,
    )

    config = {
        "seed": 7,
        "device": "cpu",
        "dtype": "float64",
        "methods": ["boundary_adapted", "naive_product", "penalty"],
        "problem": {
            "a": 1.0,
            "b": 0.6,
            "epsilon": 0.4,
            "source_center": [0.25, 0.0],
            "target_center": [0.75, 0.0],
            "set_radius": 0.5,
        },
        "model": {"width": 4, "depth": 1, "activation": "tanh", "gain": 1.0},
        "training": {
            "epochs": 1,
            "collocation_points": 8,
            "boundary_points": 4,
            "boundary_weight": 100.0,
            "learning_rate": 0.001,
            "lr_gamma": 1.0,
            "checkpoint_every": 1,
            "log_every": 1,
        },
        "reference_grid_sizes": [32, 64],
        "evaluation_batch_size": 512,
    }
    config_path = tmp_path / "cpu_smoke.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "console.log").write_text("capture started\n", encoding="utf-8")

    run_periodic_comparison(config_path, output_dir)

    assert (output_dir / "reference_fd_64.npz").is_file()
    assert (output_dir / "summary.json").is_file()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((output_dir / "comparison_metadata.json").read_text(encoding="utf-8"))
    assert "git_revision" in metadata
    assert set(summary["methods"]) == set(config["methods"])


def test_comparison_accepts_plain_pytorch_model_sampler_optimizer_and_scheduler(
    tmp_path, capsys
) -> None:
    import json

    from boundary_pinn.experiments.committor.periodic_two_well.comparison import (
        run_periodic_comparison,
    )

    class UserPeriodicNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)
            )

        def forward(self, x):
            phase = 2.0 * torch.pi * x
            features = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
            return self.layers(features)

    sampler_calls = []
    optimizer_calls = []
    scheduler_calls = []

    def make_model():
        model = UserPeriodicNet()
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                torch.nn.init.zeros_(layer.bias)
        return model

    def sample_points(problem, n, *, device, dtype, generator=None):
        sampler_calls.append(n)
        return problem.sample_interior(
            n, device=device, dtype=dtype, generator=generator
        )

    def make_optimizer(parameters):
        optimizer_calls.append(True)
        return torch.optim.Adam(parameters, lr=1.0e-3)

    def make_scheduler(optimizer):
        scheduler_calls.append(True)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    config = {
        "seed": 7,
        "device": "cpu",
        "dtype": "float64",
        "methods": ["simple_pinn", "boundary_adapted", "naive_product"],
        "problem": {
            "a": 1.0,
            "b": 0.6,
            "epsilon": 0.4,
            "source_center": [0.25, 0.0],
            "target_center": [0.75, 0.0],
            "set_radius": 0.5,
        },
        "training": {
            "epochs": 1,
            "collocation_points": 8,
            "boundary_points": 4,
            "boundary_weight": 10.0,
            "learning_rate": 1.0e-3,
            "lr_gamma": 1.0,
            "checkpoint_every": 1,
            "log_every": 1,
        },
        "reference_grid_sizes": [16, 32],
        "evaluation_batch_size": 256,
    }
    config_path = tmp_path / "user_model.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = tmp_path / "user_model_run"

    run_periodic_comparison(
        config_path,
        output_dir,
        model_factory=make_model,
        interior_sampler=sample_points,
        optimizer_factory=make_optimizer,
        scheduler_factory=make_scheduler,
    )

    assert sampler_calls
    assert optimizer_calls == [True] * len(config["methods"])
    assert scheduler_calls == [True] * len(config["methods"])
    assert (output_dir / "simple_pinn" / "best.pt").is_file()
    metadata = json.loads((output_dir / "simple_pinn" / "metadata.json").read_text())
    assert metadata["runtime_hooks"]["model"].endswith("make_model")
    assert metadata["runtime_hooks"]["interior_sampler"].endswith("sample_points")
    for method in config["methods"]:
        assert (output_dir / method / "last.pt").is_file()
        assert (output_dir / method / "best.pt").is_file()
    assert (output_dir / "losses.png").is_file()
    from boundary_pinn.experiments.committor.periodic_two_well.comparison import (
        evaluate_saved_models,
    )
    evaluation_path = evaluate_saved_models(
        output_dir,
        training_git_revision="test-revision",
        model_factory=make_model,
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert evaluation_path.name == "saved_model_evaluation.json"
    assert evaluation["training_git_revision"] == "test-revision"
    assert set(evaluation["methods"]) == set(config["methods"])
    for method in config["methods"]:
        assert evaluation["methods"][method]["grid_l2_abs"] >= 0.0
        assert evaluation["methods"][method]["grid_l2_rel"] >= 0.0
        assert evaluation["methods"][method]["grid_h1_abs"] >= 0.0
        assert evaluation["methods"][method]["grid_h2_abs"] >= 0.0
    for method in ("boundary_adapted", "naive_product"):
        assert evaluation["methods"][method]["source_boundary_rmse"] < 1e-8
        assert evaluation["methods"][method]["target_boundary_rmse"] < 1e-8
    captured = capsys.readouterr().out
    for method in config["methods"]:
        assert f"{method} | epoch 1/1" in captured
    with pytest.raises(FileExistsError, match="already contains a run"):
        run_periodic_comparison(config_path, output_dir, model_factory=make_model)


@pytest.mark.parametrize("activation", ["tanh", "relu", "gelu", "silu", "softplus"])
@pytest.mark.parametrize("initialization", ["xavier_normal", "xavier_uniform"])
def test_user_selectable_activation_and_initialization_are_finite(activation, initialization) -> None:
    from boundary_pinn.problems.committor.training import _initialize_mlp

    network = _initialize_mlp({
        "width": 8,
        "depth": 2,
        "activation": activation,
        "initialization": initialization,
        "gain": 1.0,
    }, input_dim=12)
    network = network.to(dtype=torch.float64)
    values = network(torch.randn(5, 12, dtype=torch.float64))
    assert values.shape == (5,)
    assert torch.isfinite(values).all()


def test_periodic_fourier_mlp_is_periodic_for_multiple_integer_frequencies() -> None:
    network = PeriodicMLP(
        MLP(input_dim=12, width=8, depth=2),
        frequencies=(1, 2, 4),
    ).to(dtype=torch.float64)
    points = torch.tensor([[0.11, 0.92], [0.68, 0.14]], dtype=torch.float64)
    shifted = points + torch.tensor([2.0, -3.0], dtype=torch.float64)
    assert torch.allclose(network(points), network(shifted), atol=1.0e-12)


def test_optimizer_and_learning_rate_schedule_are_configurable() -> None:
    from boundary_pinn.problems.committor.training import (
        _build_optimizer,
        _build_scheduler,
    )

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    training = {
        "learning_rate": 0.01,
        "optimizer": {
            "name": "adamw",
            "betas": [0.8, 0.95],
            "eps": 1.0e-7,
            "weight_decay": 0.001,
            "amsgrad": True,
        },
        "scheduler": {
            "name": "step",
            "step_size_epochs": 5,
            "gamma": 0.5,
        },
    }
    optimizer = _build_optimizer([parameter], training)
    scheduler = _build_scheduler(optimizer, training)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert optimizer.param_groups[0]["betas"] == pytest.approx((0.8, 0.95))
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.001)
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert scheduler.step_size == 5
    assert scheduler.gamma == pytest.approx(0.5)


def test_monte_carlo_committor_labels_are_reproducible_and_uncensored() -> None:
    from boundary_pinn.problems.committor.paper_methods import generate_committor_mc_data

    problem = _problem()
    settings = {
        "n_anchors": 1,
        "paths_per_anchor": 8,
        "dt": 0.01,
        "max_steps": 20,
        "seed": 19,
        "minimum_resolved_fraction": 0.5,
        "anchor_points": [[0.34, 0.0]],
    }
    first = generate_committor_mc_data(problem, **settings)
    second = generate_committor_mc_data(problem, **settings)
    assert torch.equal(first.anchors, second.anchors)
    assert torch.equal(first.values, second.values)
    assert first.values.shape == (1,)
    assert 0.0 <= first.values.item() <= 1.0
    assert first.metadata["minimum_observed_resolved_fraction"] >= 0.5


def test_paper_method_ansatzes_enforce_boundary_only_when_they_should() -> None:
    from boundary_pinn.problems.committor.training import (
        PeriodicCommittorConfig,
        _build_periodic_committor_model,
    )

    problem = _problem()
    model_config = {"width": 8, "depth": 2, "activation": "tanh", "gain": 1.0}
    for method in ("boundary_adapted", "over_enforced", "under_enforced"):
        torch.manual_seed(7)
        config = PeriodicCommittorConfig(7, "cpu", "float64", method, {}, model_config, {})
        model = _build_periodic_committor_model(config, problem).to(dtype=torch.float64)
        source = problem.sample_boundary("source", 12, device="cpu", dtype=torch.float64)
        target = problem.sample_boundary("target", 12, device="cpu", dtype=torch.float64)
        assert torch.max(torch.abs(model(source))) < 1e-6
        assert torch.max(torch.abs(model(target) - 1.0)) < 1e-6

    config = PeriodicCommittorConfig(
        7, "cpu", "float64", "simple_pinn", {}, model_config, {}
    )
    simple = _build_periodic_committor_model(config, problem).to(dtype=torch.float64)
    source = problem.sample_boundary("source", 12, device="cpu", dtype=torch.float64)
    target = problem.sample_boundary("target", 12, device="cpu", dtype=torch.float64)
    assert torch.max(torch.abs(simple(source))) > 1e-6
    assert torch.max(torch.abs(simple(target) - 1.0)) > 1e-6


def test_nitsche_paper_losses_split_interior_flux_and_boundary_terms() -> None:
    from boundary_pinn.problems.committor.paper_methods import nitsche_committor_losses

    problem = _problem()
    model = PeriodicMLP(MLP(input_dim=4, width=8, depth=2)).to(dtype=torch.float64)
    volume = problem.sample_interior(8, device="cpu", dtype=torch.float64)
    centers = problem.sample_interior(2, device="cpu", dtype=torch.float64)
    terms, losses = nitsche_committor_losses(
        model,
        problem,
        volume_points=volume,
        test_centers=centers,
        boundary_points=8,
        domain_area=0.95,
        length_scale=0.15,
        gamma=10.0,
        h=0.1,
    )
    expected_pde = terms["volume_term"] + terms["boundary_flux_term"]
    expected_boundary = terms["nitsche_consistency_term"] + terms["nitsche_penalty_term"]
    assert torch.allclose(terms["pde_component"], expected_pde)
    assert torch.allclose(terms["boundary_component"], expected_boundary)
    assert torch.allclose(terms["weak_residual"], expected_pde + expected_boundary)
    assert torch.allclose(losses["pde_loss"], expected_pde.square().mean())
    assert torch.allclose(losses["boundary_loss"], expected_boundary.square().mean())
    assert torch.isfinite(losses["full_weak_residual_mse"])
    objective = losses["pde_loss"] + losses["boundary_loss"]
    objective.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

def test_five_paper_methods_report_every_final_loss_component_on_cpu(tmp_path, monkeypatch, capsys) -> None:
    import json

    from boundary_pinn.experiments.committor.periodic_two_well import comparison

    methods = [
        "simple_pinn", "boundary_adapted", "over_enforced", "under_enforced",
        "variational_nitsche",
    ]
    config = {
        "seed": 7,
        "device": "cpu",
        "dtype": "float64",
        "methods": methods,
        "problem": {
            "a": 1.0, "b": 0.6, "epsilon": 0.4,
            "source_center": [0.25, 0.0], "target_center": [0.75, 0.0],
            "set_radius": 0.5,
        },
        "model": {"width": 4, "depth": 1, "activation": "tanh", "gain": 1.0},
        "training": {
            "epochs": 1, "collocation_points": 8, "boundary_points": 4,
            "learning_rate": 0.001, "lr_gamma": 1.0,
            "checkpoint_every": 1, "log_interval_seconds": 0.01,
            "monte_carlo": {
                "n_anchors": 1, "paths_per_anchor": 8, "dt": 0.01,
                "max_steps": 20, "seed": 19,
                "minimum_resolved_fraction": 0.5,
                "progress_interval_fraction": 0.01,
                "progress_interval_seconds": 0.01,
                "anchor_points": [[0.34, 0.0]],
            },
            "nitsche": {
                "volume_points": 8, "test_functions": 2, "boundary_points": 4,
                "length_scale": 0.15, "gamma": 10.0, "h": 0.1,
            },
        },
        "reference_grid_sizes": [32, 64],
        "evaluation_batch_size": 512,
    }
    config_path = tmp_path / "paper_cpu.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = tmp_path / "paper_run"
    output_dir.mkdir()
    comparison.run_periodic_comparison(config_path, output_dir, prepare_only=True)
    assert (output_dir / "cpu_preparation.json").is_file()
    assert (output_dir / "reference_fd_64.npz").is_file()
    assert (output_dir / "monte_carlo_labels.npz").is_file()
    assert not (output_dir / "simple_pinn").exists()

    def unexpected_cpu_preparation(*_args, **_kwargs):
        pytest.fail("prepared GPU stage must reuse the CPU reference and labels")

    monkeypatch.setattr(comparison, "_reference_convergence", unexpected_cpu_preparation)
    monkeypatch.setattr(comparison, "generate_committor_mc_data", unexpected_cpu_preparation)
    comparison.run_periodic_comparison(config_path, output_dir, train_prepared=True)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["methods"]) == set(methods)
    assert (output_dir / "monte_carlo_labels.npz").is_file()
    assert (output_dir / "monte_carlo_labels.json").is_file()
    assert (output_dir / "monte_carlo_anchors.png").is_file()
    assert (output_dir / "losses.png").is_file()
    assert "ETA " in capsys.readouterr().out
    for method in methods:
        final = summary["methods"][method]["final_training_metrics"]
        raw = final["raw_loss_components"]
        assert "data_loss" in raw
        assert final["final_epoch"] == 1
        assert (output_dir / method / "final_training_metrics.json").is_file()
        assert (output_dir / method / "metrics.jsonl").is_file()
        records = [
            json.loads(line)
            for line in (output_dir / method / "metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert records[-1]["session_elapsed_seconds"] >= 0.0
        assert records[-1]["estimated_remaining_seconds"] >= 0.0
    assert "boundary_loss" in summary["methods"]["simple_pinn"]["final_training_metrics"]["raw_loss_components"]
    for method in ("boundary_adapted", "over_enforced", "under_enforced"):
        assert "boundary_loss" not in summary["methods"][method]["final_training_metrics"]["raw_loss_components"]
        assert summary["methods"][method]["source_boundary_rmse"] < 1e-6
        assert summary["methods"][method]["target_boundary_rmse"] < 1e-6
    nitsche = summary["methods"]["variational_nitsche"]["final_training_metrics"]["raw_loss_components"]
    assert {"pde_loss", "boundary_loss", "data_loss", "full_weak_residual_mse",
            "volume_term_mse", "boundary_flux_term_mse",
            "nitsche_consistency_term_mse", "nitsche_penalty_term_mse"} <= set(nitsche)
    assert set(summary["methods"]["variational_nitsche"]["final_training_metrics"]["adaptive_loss_weights"]) == {
        "pde_loss", "boundary_loss", "data_loss"
    }


def test_cpu_preparation_uses_no_cuda_even_when_training_config_requests_it(tmp_path, monkeypatch) -> None:
    import json

    from boundary_pinn.experiments.committor.periodic_two_well import comparison

    config = {
        "seed": 7,
        "device": "cuda",
        "dtype": "float64",
        "methods": ["simple_pinn"],
        "problem": {
            "a": 1.0, "b": 0.6, "epsilon": 0.4,
            "source_center": [0.25, 0.0], "target_center": [0.75, 0.0],
            "set_radius": 0.5,
        },
        "model": {"width": 4, "depth": 1, "activation": "tanh", "gain": 1.0},
        "training": {"epochs": 1, "collocation_points": 8, "boundary_points": 4},
        "reference_grid_sizes": [16, 32],
        "evaluation_batch_size": 128,
    }
    config_path = tmp_path / "gpu_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = tmp_path / "prepared"
    output_dir.mkdir()

    def unexpected_cuda_check():
        pytest.fail("CPU preparation must not query or initialize CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda_check)
    with pytest.raises(ValueError, match="split into CPU preparation and GPU training"):
        comparison.run_periodic_comparison(config_path, tmp_path / "unsafe_one_step")
    comparison.run_periodic_comparison(config_path, output_dir, prepare_only=True)

    preparation = json.loads((output_dir / "cpu_preparation.json").read_text(encoding="utf-8"))
    assert preparation["reference_file"] == "reference_fd_32.npz"
    assert not (output_dir / "simple_pinn").exists()
    assert not (output_dir / "summary.json").exists()


def test_loss_plot_handles_nonfinite_and_zero_series_without_log_warning(tmp_path) -> None:
    import warnings

    from boundary_pinn.experiments.committor.periodic_two_well.comparison import _plot_losses

    method_dir = tmp_path / "method"
    method_dir.mkdir()
    (method_dir / "metrics.jsonl").write_text(
        '{"epoch": 1, "objective": Infinity, "pde_loss": 0.0}\n'
        '{"epoch": 2, "objective": Infinity, "pde_loss": 0.0}\n',
        encoding="utf-8",
    )
    positive_method_dir = tmp_path / "positive_method"
    positive_method_dir.mkdir()
    (positive_method_dir / "metrics.jsonl").write_text(
        '{"epoch": 1, "objective": 1.0, "pde_loss": 1.0}\n'
        '{"epoch": 2, "objective": 2.0, "pde_loss": 0.0}\n',
        encoding="utf-8",
    )
    destination = tmp_path / "losses.png"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _plot_losses({"method": method_dir, "positive_method": positive_method_dir}, destination)

    assert destination.is_file()
    assert not any("no positive values" in str(warning.message).lower() for warning in caught)



def test_wall_clock_controls_console_only_and_metrics_keep_every_epoch(tmp_path, capsys) -> None:
    import json

    from boundary_pinn.problems.committor.training import (
        PeriodicCommittorConfig,
        run_periodic_committor,
    )

    config = PeriodicCommittorConfig(
        seed=5,
        device="cpu",
        dtype="float64",
        method="boundary_adapted",
        problem={
            "a": 1.0,
            "b": 0.6,
            "epsilon": 0.4,
            "source_center": [0.25, 0.0],
            "target_center": [0.75, 0.0],
            "set_radius": 0.5,
        },
        model={
            "width": 4,
            "depth": 1,
            "activation": "tanh",
            "initialization": "xavier_normal",
        },
        training={
            "epochs": 3,
            "collocation_points": 8,
            "learning_rate": 1e-3,
            "optimizer": {"name": "adam"},
            "scheduler": {"name": "constant"},
            "checkpoint_every": 3,
            "log_interval_seconds": 3600,
        },
    )
    run_periodic_committor(config, tmp_path / "run")

    records = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [record["epoch"] for record in records] == [1, 2, 3]
    output = capsys.readouterr().out
    assert "epoch 1/3" in output
    assert "epoch 2/3" not in output
    assert "epoch 3/3" in output


def test_comparison_settings_reject_silent_or_incomplete_controls() -> None:
    from boundary_pinn.experiments.committor.periodic_two_well.comparison import (
        _validate_comparison_config,
    )

    valid = {
        "seed": 7,
        "device": "cpu",
        "dtype": "float64",
        "methods": ["boundary_adapted"],
        "problem": {
            "a": 1.0,
            "b": 0.6,
            "epsilon": 0.4,
            "source_center": [0.25, 0.0],
            "target_center": [0.75, 0.0],
            "set_radius": 0.5,
        },
        "model": {"width": 8, "depth": 2, "activation": "tanh"},
        "training": {
            "epochs": 2,
            "collocation_points": 8,
            "learning_rate": 1e-3,
            "scheduler": {
                "name": "step",
                "step_size_epochs": 1,
                "gamma": 0.5,
            },
        },
        "reference_grid_sizes": [32, 64],
    }
    _validate_comparison_config(valid)

    typo = {**valid, "training": {**valid["training"], "log_intervel_seconds": 10}}
    with pytest.raises(ValueError, match="Unknown training setting"):
        _validate_comparison_config(typo)

    missing_gamma = {
        **valid,
        "training": {
            **valid["training"],
            "scheduler": {"name": "step", "step_size_epochs": 1},
        },
    }
    with pytest.raises(ValueError, match="requires gamma"):
        _validate_comparison_config(missing_gamma)

    unsupported_device = {**valid, "device": "mps"}
    with pytest.raises(ValueError, match="device must be 'cpu' or a CUDA device"):
        _validate_comparison_config(unsupported_device)


@pytest.mark.parametrize(
    ("activation", "activation_kwargs"),
    [
        ("Tanh", {}),
        ("ReLU", {}),
        ("SquaredReLU", {}),
        ("PReLU", {"num_parameters": 1, "init": 0.25}),
        ("CELU", {"alpha": 1.0}),
        ("torch.nn.functional.gelu", {"approximate": "tanh"}),
    ],
)
def test_comparison_config_accepts_pytorch_activations_and_arguments(
    activation: str, activation_kwargs: dict[str, object]
) -> None:
    from boundary_pinn.experiments.committor.periodic_two_well.comparison import (
        _validate_comparison_config,
    )

    config = {
        "seed": 7,
        "device": "cpu",
        "dtype": "float64",
        "methods": ["boundary_adapted"],
        "problem": {
            "a": 1.0,
            "b": 0.6,
            "epsilon": 0.4,
            "source_center": [0.25, 0.0],
            "target_center": [0.75, 0.0],
            "set_radius": 0.5,
        },
        "model": {
            "width": 8,
            "depth": 2,
            "activation": activation,
            "activation_kwargs": activation_kwargs,
        },
        "training": {
            "epochs": 2,
            "collocation_points": 8,
            "learning_rate": 1e-3,
            "scheduler": {"name": "constant"},
        },
        "reference_grid_sizes": [32, 64],
    }

    _validate_comparison_config(config)


def test_comparison_config_accepts_importable_custom_module_and_function(monkeypatch) -> None:
    import sys
    import types

    from boundary_pinn.experiments.committor.periodic_two_well.comparison import (
        _validate_comparison_config,
    )

    module = types.ModuleType("test_custom_activation_config")

    class CustomSmoothActivation(torch.nn.Module):
        def forward(self, values):
            return torch.tanh(values) + values.square() * 0.01

    module.CustomSmoothActivation = CustomSmoothActivation

    def custom_activation_function(values, beta=1.0):
        return torch.tanh(beta * values)

    module.custom_activation_function = custom_activation_function
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = {
        "seed": 7,
        "device": "cpu",
        "dtype": "float64",
        "methods": ["boundary_adapted"],
        "problem": {
            "a": 1.0,
            "b": 0.6,
            "epsilon": 0.4,
            "source_center": [0.25, 0.0],
            "target_center": [0.75, 0.0],
            "set_radius": 0.5,
        },
        "model": {
            "width": 8,
            "depth": 2,
            "activation": "test_custom_activation_config.CustomSmoothActivation",
            "activation_kwargs": {},
        },
        "training": {
            "epochs": 2,
            "collocation_points": 8,
            "learning_rate": 1e-3,
            "scheduler": {"name": "constant"},
        },
        "reference_grid_sizes": [32, 64],
    }

    _validate_comparison_config(config)
    function_config = {
        **config,
        "model": {
            **config["model"],
            "activation": "test_custom_activation_config.custom_activation_function",
            "activation_kwargs": {"beta": 0.75},
        },
    }
    _validate_comparison_config(function_config)
