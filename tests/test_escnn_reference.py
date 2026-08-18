"""Numerical convention checks, enabled by the optional ``reference`` extra."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.escnn

try:
    from escnn import gspaces as escnn_gspaces, nn as escnn_nn
except ImportError as error:
    pytest.skip(f"complete escnn reference installation unavailable: {error}", allow_module_level=True)

from we3nn import cyclic_group, dihedral_group, gspaces, nn
from we3nn import clebsch_gordan, subspace_diagnostics
from we3nn import (
    RestrictedSphericalHarmonics,
    intertwiner_basis,
    tensor_product_representation,
    direct_sum,
    planar_o3,
    restrict_o3,
    find_representation_intertwiner,
)
from e3nn import o3


@pytest.mark.parametrize("n", [3, 4, 6, 9])
def test_cyclic_representation_matrices_match_escnn(n):
    reference = escnn_gspaces.rot2dOnR2(N=n).fibergroup
    ours = cyclic_group(n)
    for frequency in range(n // 2 + 1):
        for reference_element, element in zip(reference.elements, ours.elements):
            np.testing.assert_allclose(
                ours.irrep(frequency)(element).numpy(),
                reference.irrep(frequency)(reference_element),
                atol=1e-14,
            )
    for reference_element, element in zip(reference.elements, ours.elements):
        np.testing.assert_allclose(
            ours.regular_repr(element).numpy(),
            reference.regular_representation(reference_element),
            atol=1e-14,
        )


@pytest.mark.parametrize("n", [3, 4, 6, 9])
def test_dihedral_representation_matrices_match_escnn(n):
    reference = escnn_gspaces.flipRot2dOnR2(N=n).fibergroup
    ours = dihedral_group(n)
    for irrep_id in ours.irrep_ids():
        for reference_element, element in zip(reference.elements, ours.elements):
            np.testing.assert_allclose(
                ours.irrep(*irrep_id)(element).numpy(),
                reference.irrep(*irrep_id)(reference_element),
                atol=1e-14,
            )
    for reference_element, element in zip(reference.elements, ours.elements):
        np.testing.assert_allclose(
            ours.regular_repr(element).numpy(),
            reference.regular_representation(reference_element),
            atol=1e-14,
        )


def test_d6_linear_has_same_number_of_trainable_degrees_of_freedom_as_escnn():
    esc_space = escnn_gspaces.no_base_space(escnn_gspaces.flipRot2dOnR2(N=6).fibergroup)
    our_space = gspaces.no_base_space(gspaces.flipRot2dOnR2(N=6).fibergroup)
    esc_in = escnn_nn.FieldType(esc_space, 10 * [esc_space.irrep(0, 0)] + 2 * [esc_space.irrep(1, 1)])
    our_in = nn.FieldType(our_space, 10 * [our_space.irrep(0, 0)] + 2 * [our_space.irrep(1, 1)])
    esc_out = escnn_nn.FieldType(esc_space, 4 * [esc_space.regular_repr])
    our_out = nn.FieldType(our_space, 4 * [our_space.regular_repr])
    esc_layer = escnn_nn.Linear(esc_in, esc_out, bias=True)
    our_layer = nn.WELinear(our_in, our_out, bias=True)
    assert sum(parameter.numel() for parameter in our_layer.parameters()) == sum(
        parameter.numel() for parameter in esc_layer.parameters()
    )


def test_d6_homomorphism_spaces_match_escnn_as_subspaces():
    from escnn import group as escnn_group

    reference = escnn_gspaces.flipRot2dOnR2(N=6).fibergroup
    ours = dihedral_group(6)
    pairs = [
        (ours.trivial_irrep, ours.trivial_irrep, reference.irrep(0, 0), reference.irrep(0, 0)),
        (ours.trivial_irrep, ours.irrep(1, 1), reference.irrep(0, 0), reference.irrep(1, 1)),
        (ours.irrep(1, 1), ours.trivial_irrep, reference.irrep(1, 1), reference.irrep(0, 0)),
        (ours.irrep(1, 1), ours.irrep(1, 1), reference.irrep(1, 1), reference.irrep(1, 1)),
        (ours.regular_repr, ours.irrep(1, 1), reference.regular_representation, reference.irrep(1, 1)),
        (ours.irrep(1, 1), ours.regular_repr, reference.irrep(1, 1), reference.regular_representation),
        (ours.regular_repr, ours.regular_repr, reference.regular_representation, reference.regular_representation),
        (
            direct_sum([ours.regular_repr, ours.regular_repr]),
            ours.regular_repr,
            escnn_group.directsum([reference.regular_representation, reference.regular_representation]),
            reference.regular_representation,
        ),
    ]
    for our_in, our_out, ref_in, ref_out in pairs:
        our_basis = intertwiner_basis(our_in, our_out)
        reference_basis = torch.from_numpy(escnn_group.homomorphism_space(ref_in, ref_out)).to(torch.float64)
        assert our_basis.shape == reference_basis.shape
        diagnostics = subspace_diagnostics(our_basis, reference_basis)
        assert diagnostics["projector_error"] < 2e-8, diagnostics


def test_d6_regular_relu_forward_matches_escnn():
    reference_space = escnn_gspaces.no_base_space(escnn_gspaces.flipRot2dOnR2(N=6).fibergroup)
    ours_space = gspaces.no_base_space(gspaces.flipRot2dOnR2(N=6).fibergroup)
    reference_type = escnn_nn.FieldType(reference_space, [reference_space.regular_repr])
    ours_type = nn.FieldType(ours_space, [ours_space.regular_repr])
    x = torch.randn(17, 12)
    actual = nn.ReLU(ours_type)(x)
    expected = escnn_nn.ReLU(reference_type)(escnn_nn.GeometricTensor(x, reference_type)).tensor
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("degree", range(5))
def test_restricted_o3_representations_match_escnn_up_to_one_basis(degree):
    from escnn import group as escnn_group

    ours_group = dihedral_group(6)
    ours = restrict_o3(
        o3.Irrep(degree, (-1) ** degree),
        planar_o3(ours_group),
    )
    parent = escnn_group.o3_group()
    reference_group, _, _ = parent.subgroup(("cone", 6))
    reference = parent.irrep(degree % 2, degree).restrict(("cone", 6))
    wrapped = ours_group.representation(
        {
            element: reference(reference_element)
            for element, reference_element in zip(ours_group.elements, reference_group.elements)
        },
        name=f"escnn-restricted-l{degree}",
    )
    for element in ours_group.elements:
        np.testing.assert_allclose(ours.character(element), wrapped.character(element), atol=2e-8)
    transform = find_representation_intertwiner(wrapped, ours, atol=2e-7)
    for element in ours_group.elements:
        torch.testing.assert_close(
            ours(element) @ transform,
            transform @ wrapped(element),
            atol=2e-7,
            rtol=2e-7,
        )


def test_spherical_kernel_tensor_product_sampled_kernel_space_matches_escnn():
    from escnn import group as escnn_group
    from escnn.kernels import kernels_O3_subgroup_act_R3

    maximum_frequency = 2
    ours_group = dihedral_group(6)
    ours_space = gspaces.no_base_space(ours_group)
    harmonics = RestrictedSphericalHarmonics(
        ours_group,
        degrees=range(maximum_frequency + 1),
        normalization="component",
    )
    ours = nn.SphericalKernelTensorProduct(
        ours_space.irrep(1, 1),
        harmonics,
        ours_space.irrep(1, 1),
    ).double()

    parent = escnn_group.o3_group(maximum_frequency)
    reference_group, _, _ = parent.subgroup(("cone", 6))
    reference_irrep = reference_group.irrep(1, 1)
    reference = kernels_O3_subgroup_act_R3(
        reference_irrep,
        reference_irrep,
        ("cone", 6),
        radii=[1.0],
        sigma=[0.6],
        maximum_frequency=maximum_frequency,
    )

    points = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
            [-2.0, 1.0, 0.5],
            [0.2, -0.7, 1.3],
            [1.0, 1.0, -1.0],
        ],
        dtype=torch.float64,
    )
    points = points / torch.linalg.vector_norm(points, dim=-1, keepdim=True)
    ours_sampled = ours.sample_kernel_basis(points).permute(1, 0, 2, 3)
    reference_sampled = reference.sample(points.float()).double().permute(1, 0, 2, 3)

    assert ours_sampled.shape == reference_sampled.shape
    diagnostics = subspace_diagnostics(ours_sampled, reference_sampled)
    assert diagnostics["projector_error"] < 5e-6, diagnostics
    assert diagnostics["largest_principal_angle"] < 5e-6, diagnostics


@pytest.mark.parametrize("kind,n", [("cyclic", 5), ("cyclic", 6), ("dihedral", 5), ("dihedral", 6)])
def test_all_irrep_cg_spaces_match_escnn(kind, n):
    reference = (
        escnn_gspaces.rot2dOnR2(N=n).fibergroup
        if kind == "cyclic"
        else escnn_gspaces.flipRot2dOnR2(N=n).fibergroup
    )
    ours = cyclic_group(n) if kind == "cyclic" else dihedral_group(n)
    for left in ours.irreps():
        for right in ours.irreps():
            for output in ours.irreps():
                ours_basis = clebsch_gordan(left, right, output)
                reference_raw = reference._clebsh_gordan_coeff(left.id, right.id, output.id)
                reference_basis = torch.from_numpy(reference_raw).permute(2, 3, 0, 1).to(torch.float64)
                assert ours_basis.shape == reference_basis.shape, (
                    kind, n, left.id, right.id, output.id, ours_basis.shape, reference_basis.shape
                )
                diagnostics = subspace_diagnostics(ours_basis, reference_basis)
                assert diagnostics["projector_error"] < 2e-8, diagnostics


def _physical_basis(layer):
    parameters = list(layer.parameters())
    with torch.no_grad():
        saved = [parameter.clone() for parameter in parameters]
        for parameter in parameters:
            parameter.zero_()
        columns = []
        for parameter in parameters:
            for flat_index in range(parameter.numel()):
                parameter.reshape(-1)[flat_index] = 1.0
                weight, bias = layer.expand_parameters()
                columns.append(torch.cat((weight.flatten(), bias.flatten())))
                parameter.reshape(-1)[flat_index] = 0.0
        for parameter, value in zip(parameters, saved):
            parameter.copy_(value)
    return torch.stack(columns, dim=1)


def test_synchronized_d6_linear_forward_matches_escnn():
    torch.manual_seed(10)
    esc_space = escnn_gspaces.no_base_space(escnn_gspaces.flipRot2dOnR2(N=6).fibergroup)
    our_space = gspaces.no_base_space(gspaces.flipRot2dOnR2(N=6).fibergroup)
    esc_in = escnn_nn.FieldType(esc_space, 3 * [esc_space.irrep(0, 0)] + 2 * [esc_space.irrep(1, 1)])
    our_in = nn.FieldType(our_space, 3 * [our_space.irrep(0, 0)] + 2 * [our_space.irrep(1, 1)])
    esc_out = escnn_nn.FieldType(esc_space, 2 * [esc_space.regular_repr])
    our_out = nn.FieldType(our_space, 2 * [our_space.regular_repr])
    esc_layer = escnn_nn.Linear(esc_in, esc_out, bias=True).double()
    our_layer = nn.WELinear(our_in, our_out, bias=True).double()
    esc_weight, esc_bias = esc_layer.expand_parameters()
    target = torch.cat((esc_weight.flatten(), esc_bias.flatten())).detach()
    physical_basis = _physical_basis(our_layer)
    coefficients = torch.linalg.lstsq(physical_basis, target).solution
    residual = torch.linalg.vector_norm(physical_basis @ coefficients - target)
    assert float(residual) < 6e-8
    with torch.no_grad():
        offset = 0
        for parameter in our_layer.parameters():
            parameter.copy_(coefficients[offset:offset + parameter.numel()].reshape_as(parameter))
            offset += parameter.numel()
    x = torch.randn(37, our_in.size, dtype=torch.float64)
    ours_output = our_layer(x)
    reference_output = esc_layer(escnn_nn.GeometricTensor(x, esc_in)).tensor
    # escnn constructs this basis in float32 before module.double(), so the
    # synchronized physical operator retains a few e-8 of source rounding.
    torch.testing.assert_close(ours_output, reference_output, atol=5e-8, rtol=5e-8)


def _synchronize_linear(our_layer, esc_layer):
    esc_weight, esc_bias = esc_layer.expand_parameters()
    if esc_bias is None:
        esc_bias = esc_weight.new_zeros(our_layer.out_type.size)
    target = torch.cat((esc_weight.flatten(), esc_bias.flatten())).detach()
    physical_basis = _physical_basis(our_layer)
    coefficients = torch.linalg.lstsq(physical_basis, target).solution
    residual = torch.linalg.vector_norm(physical_basis @ coefficients - target)
    assert float(residual) < 2e-6
    with torch.no_grad():
        offset = 0
        for parameter in our_layer.parameters():
            parameter.copy_(coefficients[offset:offset + parameter.numel()].reshape_as(parameter))
            offset += parameter.numel()


def test_synchronized_d6_message_trunk_and_heads_match_escnn_at_every_stage():
    torch.manual_seed(21)
    hidden, scalar_inputs = 8, 14
    esc_space = escnn_gspaces.no_base_space(escnn_gspaces.flipRot2dOnR2(N=6).fibergroup)
    our_space = gspaces.no_base_space(gspaces.flipRot2dOnR2(N=6).fibergroup)
    esc_in = escnn_nn.FieldType(esc_space, scalar_inputs * [esc_space.irrep(0, 0)] + 2 * [esc_space.irrep(1, 1)])
    our_in = nn.FieldType(our_space, scalar_inputs * [our_space.irrep(0, 0)] + 2 * [our_space.irrep(1, 1)])
    esc_hidden = escnn_nn.FieldType(esc_space, (hidden // 2) * [esc_space.regular_repr])
    our_hidden = nn.FieldType(our_space, (hidden // 2) * [our_space.regular_repr])
    esc_scalar = escnn_nn.FieldType(esc_space, hidden * [esc_space.irrep(0, 0)])
    our_scalar = nn.FieldType(our_space, hidden * [our_space.irrep(0, 0)])
    esc_vector = escnn_nn.FieldType(esc_space, [esc_space.irrep(1, 1)])
    our_vector = nn.FieldType(our_space, [our_space.irrep(1, 1)])
    esc_z = escnn_nn.FieldType(esc_space, [esc_space.irrep(0, 0)])
    our_z = nn.FieldType(our_space, [our_space.irrep(0, 0)])

    esc_trunk_linears = [escnn_nn.Linear(esc_in, esc_hidden)] + [escnn_nn.Linear(esc_hidden, esc_hidden) for _ in range(3)]
    our_trunk_linears = [nn.WELinear(our_in, our_hidden)] + [nn.WELinear(our_hidden, our_hidden) for _ in range(3)]
    esc_scalar_head, our_scalar_head = escnn_nn.Linear(esc_hidden, esc_scalar), nn.WELinear(our_hidden, our_scalar)
    esc_vector_linears = [escnn_nn.Linear(esc_hidden, esc_hidden), escnn_nn.Linear(esc_hidden, esc_vector)]
    our_vector_linears = [nn.WELinear(our_hidden, our_hidden), nn.WELinear(our_hidden, our_vector)]
    esc_z_linears = [escnn_nn.Linear(esc_hidden, esc_hidden), escnn_nn.Linear(esc_hidden, esc_z)]
    our_z_linears = [nn.WELinear(our_hidden, our_hidden), nn.WELinear(our_hidden, our_z)]
    all_pairs = list(zip(our_trunk_linears, esc_trunk_linears)) + [
        (our_scalar_head, esc_scalar_head),
        *zip(our_vector_linears, esc_vector_linears),
        *zip(our_z_linears, esc_z_linears),
    ]
    for ours_layer, esc_layer in all_pairs:
        ours_layer.double()
        esc_layer.double()
        _synchronize_linear(ours_layer, esc_layer)

    tensor = torch.randn(31, our_in.size, dtype=torch.float64)
    ours_value = tensor
    esc_value = escnn_nn.GeometricTensor(tensor, esc_in)
    for index, (ours_layer, esc_layer) in enumerate(zip(our_trunk_linears, esc_trunk_linears)):
        ours_value, esc_value = ours_layer(ours_value), esc_layer(esc_value)
        torch.testing.assert_close(ours_value, esc_value.tensor, atol=2e-6, rtol=2e-6)
        if index < 3:
            ours_value = nn.ReLU(our_hidden)(ours_value)
            esc_value = escnn_nn.ReLU(esc_hidden)(esc_value)
            torch.testing.assert_close(ours_value, esc_value.tensor, atol=2e-6, rtol=2e-6)

    torch.testing.assert_close(
        our_scalar_head(ours_value),
        esc_scalar_head(esc_value).tensor,
        atol=2e-6,
        rtol=2e-6,
    )
    for ours_lines, esc_lines, ours_type, esc_type in (
        (our_vector_linears, esc_vector_linears, our_hidden, esc_hidden),
        (our_z_linears, esc_z_linears, our_hidden, esc_hidden),
    ):
        ours_head = nn.ELU(ours_type)(ours_lines[0](ours_value))
        esc_head = escnn_nn.ELU(esc_type)(esc_lines[0](esc_value))
        torch.testing.assert_close(ours_head, esc_head.tensor, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(
            ours_lines[1](ours_head),
            esc_lines[1](esc_head).tensor,
            atol=2e-6,
            rtol=2e-6,
        )
