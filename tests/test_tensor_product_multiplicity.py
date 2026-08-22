import pytest
import torch

from we3nn import gspaces, nn
from we3nn.clebsch_gordan import full_coupling_basis


def _legacy_from(grouped):
    return nn.TensorProduct(
        grouped.in1_type,
        grouped.in2_type,
        grouped.out_type,
        instructions=list(grouped.instructions),
        internal_weights=False,
        shared_weights=grouped.shared_weights,
    ).double()


@pytest.mark.parametrize("kind,n", [("cyclic", 5), ("cyclic", 6), ("dihedral", 5), ("dihedral", 6)])
def test_grouped_default_matches_legacy_paths_outputs_and_gradients(kind, n):
    base = gspaces.rot2dOnR2(n) if kind == "cyclic" else gspaces.flipRot2dOnR2(n)
    space = gspaces.no_base_space(base.fibergroup)
    scalar = space.trivial_repr
    vector = space.irrep(1) if kind == "cyclic" else space.irrep(1, 1)
    other = space.irrep(2) if kind == "cyclic" else space.irrep(1, 2)
    # Equal representations are deliberately non-contiguous.
    left = nn.FieldType(space, [vector, scalar, vector])
    right = nn.FieldType(space, [scalar, vector, other, vector])
    output = nn.FieldType(space, [other, scalar, vector, other])
    grouped = nn.TensorProduct(
        left, right, output, internal_weights=False, shared_weights=False
    ).double()
    legacy = _legacy_from(grouped)
    assert grouped.weight_numel == legacy.weight_numel

    left_new = torch.randn(2, left.size, dtype=torch.float64, requires_grad=True)
    right_new = torch.randn(2, right.size, dtype=torch.float64, requires_grad=True)
    weight_new = torch.randn(2, grouped.weight_numel, dtype=torch.float64, requires_grad=True)
    left_old = left_new.detach().clone().requires_grad_()
    right_old = right_new.detach().clone().requires_grad_()
    weight_old = weight_new.detach().clone().requires_grad_()

    actual = grouped(left_new, right_new, weight_new)
    expected = legacy(left_old, right_old, weight_old)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(left_new.grad, left_old.grad, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(right_new.grad, right_old.grad, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(weight_new.grad, weight_old.grad, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("regular_position", ["output", "left", "right", "both_inputs", "all"])
def test_grouped_regular_multiplicities_match_legacy_paths(regular_position):
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(4).fibergroup)
    vector, regular = space.irrep(1, 1), space.regular_repr
    left_rep = regular if regular_position in {"left", "both_inputs", "all"} else vector
    right_rep = regular if regular_position in {"right", "both_inputs", "all"} else vector
    output_rep = regular if regular_position in {"output", "all"} else vector
    left = nn.FieldType(space, [left_rep, vector, left_rep])
    right = nn.FieldType(space, [right_rep, right_rep])
    output = nn.FieldType(space, [output_rep, vector, output_rep])
    grouped = nn.TensorProduct(left, right, output, internal_weights=False).double()
    legacy = _legacy_from(grouped)
    x = torch.randn(3, left.size, dtype=torch.float64)
    y = torch.randn(3, right.size, dtype=torch.float64)
    weights = torch.randn(grouped.weight_numel, dtype=torch.float64)
    torch.testing.assert_close(
        grouped(x, y, weights), legacy(x, y, weights), atol=3e-12, rtol=3e-12
    )


def test_homogeneous_multiplicity_is_one_execution_block():
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(6).fibergroup)
    e1, e2 = space.irrep(1, 1), space.irrep(1, 2)
    multiplicity = 16
    left = nn.FieldType(space, [e1] * multiplicity)
    right = nn.FieldType(space, [e1] * multiplicity)
    output = nn.FieldType(space, [e2] * multiplicity)
    product = nn.FullyConnectedTensorProduct(left, right, output)
    coupling_count = full_coupling_basis(e1, e1, e2).shape[0]
    assert product.weight_numel == multiplicity**3 * coupling_count
    assert len(product.blocks) == 1
    assert len(product.paths) == 0
    assert len(product.instructions) == multiplicity**3
    assert sum(
        buffer.numel() > 0
        for name, buffer in product.named_buffers()
        if name.endswith("coupling_basis")
    ) == 1
    x = torch.randn(2, left.size, requires_grad=True)
    y = torch.randn(2, right.size, requires_grad=True)
    product(x, y).square().mean().backward()
    product.check_equivariance()


def test_128_multiplicity_constructs_one_block_without_path_objects():
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(6).fibergroup)
    e1, e2 = space.irrep(1, 1), space.irrep(1, 2)
    multiplicity = 128
    product = nn.FullyConnectedTensorProduct(
        nn.FieldType(space, [e1] * multiplicity),
        nn.FieldType(space, [e1] * multiplicity),
        nn.FieldType(space, [e2] * multiplicity),
    )
    coupling_count = full_coupling_basis(e1, e1, e2).shape[0]
    assert len(product.blocks) == 1
    assert len(product.paths) == 0
    assert len(product.instructions) == multiplicity**3
    assert product.weight_numel == multiplicity**3 * coupling_count
    # The only parameter is the mathematically required multiplicity tensor.
    assert sum(parameter.numel() for parameter in product.parameters()) == product.weight_numel
    x = torch.randn(1, product.in1_type.size, requires_grad=True)
    y = torch.randn(1, product.in2_type.size, requires_grad=True)
    result = product(x, y)
    assert result.shape == (1, product.out_type.size)
    result.square().mean().backward()
    assert x.grad is not None and y.grad is not None
    assert product.blocks[0].weight.grad is not None


def test_legacy_internal_path_checkpoint_loads_into_grouped_blocks():
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(6).fibergroup)
    scalar, e1, e2 = space.trivial_repr, space.irrep(1, 1), space.irrep(1, 2)
    left = nn.FieldType(space, [e1, scalar, e1])
    right = nn.FieldType(space, [e2, e1])
    output = nn.FieldType(space, [scalar, e2, scalar])
    grouped = nn.TensorProduct(left, right, output).double()
    legacy = nn.TensorProduct(
        left,
        right,
        output,
        instructions=list(grouped.instructions),
    ).double()
    grouped.load_state_dict(legacy.state_dict())
    x = torch.randn(3, left.size, dtype=torch.float64)
    y = torch.randn(3, right.size, dtype=torch.float64)
    torch.testing.assert_close(grouped(x, y), legacy(x, y), atol=2e-12, rtol=2e-12)
