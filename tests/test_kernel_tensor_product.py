import pytest
import torch
from e3nn import o3

from we3nn import (
    RestrictedSphericalHarmonics,
    gspaces,
    nn,
    planar_o3,
    restrict_o3,
)


def _check(module, x, filters, weights, atol=3e-5):
    output = module(x, filters, weights)
    for element in module.in1_type.fibergroup.elements:
        actual = module(
            module.in1_type.transform_fibers(x, element),
            module.in2_type.transform_fibers(filters, element),
            weights,
        )
        expected = module.out_type.transform_fibers(output, element)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=atol)


def _replace_with_legacy_paths(module):
    grouped = module.tensor_product
    module.tensor_product = nn.TensorProduct(
        grouped.in1_type,
        grouped.in2_type,
        grouped.out_type,
        instructions=list(grouped.instructions),
        internal_weights=False,
        shared_weights=grouped.shared_weights,
    ).double()
    module.weight_numel = module.tensor_product.weight_numel
    return module


def test_native_finite_kernel_tensor_product_with_per_sample_reduced_weights():
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(6).fibergroup)
    node = nn.FieldType(space, [space.irrep(1, 1)])
    filter_type = nn.FieldType(space, [space.irrep(1, 2)])
    output = nn.FieldType(space, [space.irrep(0, 0), space.irrep(1, 1)])
    module = nn.KernelTensorProduct(node, filter_type, output)
    x = torch.randn(8, node.size)
    filters = torch.randn(8, filter_type.size)
    weights = torch.randn(8, module.weight_numel, requires_grad=True)
    _check(module, x, filters, weights)
    module(x, filters, weights).square().mean().backward()
    assert weights.grad is not None


def test_restricted_o3_filter_uses_full_finite_group_couplings():
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(6).fibergroup)
    embedding = planar_o3(space.fibergroup)
    node_rep = restrict_o3(o3.Irrep("1o"), embedding)
    filter_rep = restrict_o3(o3.Irrep("2e"), embedding)
    output_rep = restrict_o3(o3.Irrep("1o"), embedding)
    node = nn.FieldType(space, [node_rep])
    filter_type = nn.FieldType(space, [filter_rep])
    output = nn.FieldType(space, [output_rep])
    module = nn.SphericalKernelTensorProduct(node, filter_type, output)
    x = torch.randn(3, node.size)
    filters = torch.randn(3, filter_type.size)
    weights = torch.randn(3, module.weight_numel)
    _check(module, x, filters, weights, atol=8e-5)
    # O(3) has at most one path for this irrep triple, while D6 restriction
    # generally exposes several reduced matrix elements.
    assert module.weight_numel > 1


def test_spherical_kernel_tensor_product_samples_physical_kernel_basis_from_points():
    torch.manual_seed(19)
    group = gspaces.flipRot2dOnR2(6).fibergroup
    space = gspaces.no_base_space(group)
    harmonics = RestrictedSphericalHarmonics(
        group,
        degrees=[0, 1, 2],
        normalization="component",
    )
    module = nn.SphericalKernelTensorProduct(
        space.irrep(1, 1),
        harmonics,
        space.irrep(1, 1),
    ).double()
    points = torch.randn(7, 3, dtype=torch.float64)
    points = points / torch.linalg.vector_norm(points, dim=-1, keepdim=True)
    features = torch.randn(7, 2, dtype=torch.float64)
    weights = torch.randn(7, module.weight_numel, dtype=torch.float64)

    sampled = module.sample_kernel_basis(points)
    assert sampled.shape == (7, module.weight_numel, 2, 2)
    from_basis = torch.einsum("npoi,ni,np->no", sampled, features, weights)
    torch.testing.assert_close(
        from_basis,
        module.forward_from_points(features, points, weights),
        atol=2e-12,
        rtol=2e-12,
    )
    torch.testing.assert_close(module.sample(points), sampled)
    assert module.embedding is harmonics.embedding
    assert module.degrees == (0, 1, 2)


def test_spherical_kernel_tensor_product_rejects_unrestricted_finite_filters():
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(6).fibergroup)
    with pytest.raises(TypeError, match=r"restricted O\(3\)"):
        nn.SphericalKernelTensorProduct(
            space.irrep(1, 1),
            space.irrep(1, 1),
            space.irrep(0, 0),
        )


def test_grouped_kernel_matches_legacy_paths_basis_outputs_and_gradients():
    torch.manual_seed(31)
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(6).fibergroup)
    scalar, e1, e2 = space.trivial_repr, space.irrep(1, 1), space.irrep(1, 2)
    features_type = nn.FieldType(space, [e1, scalar, e1])
    filter_type = nn.FieldType(space, [e2, scalar, e2])
    output_type = nn.FieldType(space, [e1, scalar, e1])
    grouped = nn.KernelTensorProduct(features_type, filter_type, output_type).double()
    legacy = _replace_with_legacy_paths(
        nn.KernelTensorProduct(features_type, filter_type, output_type)
    )
    assert grouped.weight_numel == legacy.weight_numel

    features_new = torch.randn(2, features_type.size, dtype=torch.float64, requires_grad=True)
    filters_new = torch.randn(2, filter_type.size, dtype=torch.float64, requires_grad=True)
    weights_new = torch.randn(2, grouped.weight_numel, dtype=torch.float64, requires_grad=True)
    features_old = features_new.detach().clone().requires_grad_()
    filters_old = filters_new.detach().clone().requires_grad_()
    weights_old = weights_new.detach().clone().requires_grad_()

    actual = grouped(features_new, filters_new, weights_new)
    expected = legacy(features_old, filters_old, weights_old)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for new, old in (
        (features_new, features_old),
        (filters_new, filters_old),
        (weights_new, weights_old),
    ):
        torch.testing.assert_close(new.grad, old.grad, atol=2e-11, rtol=2e-11)

    grouped_basis = grouped.sample_kernel_basis(filters_new.detach())
    legacy_basis = legacy.sample_kernel_basis(filters_old.detach())
    torch.testing.assert_close(grouped_basis, legacy_basis, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        torch.einsum("...poi,...i,...p->...o", grouped_basis, features_new.detach(), weights_new.detach()),
        actual.detach(),
        atol=2e-12,
        rtol=2e-12,
    )
    _check(grouped, features_new.detach(), filters_new.detach(), weights_new.detach(), atol=2e-10)


@pytest.mark.parametrize("regular_position", ["output", "left", "right", "all"])
def test_grouped_regular_kernel_basis_matches_legacy_paths(regular_position):
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(4).fibergroup)
    vector, regular = space.irrep(1, 1), space.regular_repr
    feature_rep = regular if regular_position in {"left", "all"} else vector
    filter_rep = regular if regular_position in {"right", "all"} else vector
    output_rep = regular if regular_position in {"output", "all"} else vector
    features_type = nn.FieldType(space, [feature_rep, feature_rep])
    filter_type = nn.FieldType(space, [filter_rep, filter_rep])
    output_type = nn.FieldType(space, [output_rep, output_rep])
    grouped = nn.KernelTensorProduct(features_type, filter_type, output_type).double()
    legacy = _replace_with_legacy_paths(
        nn.KernelTensorProduct(features_type, filter_type, output_type)
    )
    filters = torch.randn(3, filter_type.size, dtype=torch.float64)
    torch.testing.assert_close(
        grouped.sample_kernel_basis(filters),
        legacy.sample_kernel_basis(filters),
        atol=3e-12,
        rtol=3e-12,
    )


def test_grouped_spherical_kernel_matches_legacy_from_points_and_gradients():
    torch.manual_seed(37)
    group = gspaces.flipRot2dOnR2(6).fibergroup
    space = gspaces.no_base_space(group)
    harmonics = RestrictedSphericalHarmonics(
        group, degrees=[0, 1, 2], normalization="component"
    )
    features_type = nn.FieldType(space, [space.irrep(1, 1)] * 2)
    output_type = nn.FieldType(space, [space.irrep(1, 1), space.trivial_repr])
    grouped = nn.SphericalKernelTensorProduct(
        features_type, harmonics, output_type
    ).double()
    legacy = _replace_with_legacy_paths(
        nn.SphericalKernelTensorProduct(features_type, harmonics, output_type)
    )
    features_new = torch.randn(2, features_type.size, dtype=torch.float64, requires_grad=True)
    points_new = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    weights_new = torch.randn(2, grouped.weight_numel, dtype=torch.float64, requires_grad=True)
    features_old = features_new.detach().clone().requires_grad_()
    points_old = points_new.detach().clone().requires_grad_()
    weights_old = weights_new.detach().clone().requires_grad_()
    actual = grouped.forward_from_points(features_new, points_new, weights_new)
    expected = legacy.forward_from_points(features_old, points_old, weights_old)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for new, old in (
        (features_new, features_old),
        (points_new, points_old),
        (weights_new, weights_old),
    ):
        torch.testing.assert_close(new.grad, old.grad, atol=3e-11, rtol=3e-11)
    torch.testing.assert_close(
        grouped.sample_kernel_basis(points_new.detach()),
        legacy.sample_kernel_basis(points_old.detach()),
        atol=3e-12,
        rtol=3e-12,
    )
