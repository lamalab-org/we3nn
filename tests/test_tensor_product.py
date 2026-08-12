import pytest
import torch

from e3nn_WE import gspaces, nn


def _assert_equivariant(module, input1, input2, atol=3e-10):
    output = module(input1, input2)
    for element in input1.type.fibergroup.elements:
        actual = module(input1.transform_fibers(element), input2.transform_fibers(element)).tensor
        expected = output.transform_fibers(element).tensor
        torch.testing.assert_close(actual, expected, atol=atol, rtol=atol)


@pytest.mark.parametrize("kind,n", [("cyclic", 5), ("cyclic", 6), ("dihedral", 5), ("dihedral", 6)])
def test_irrep_fully_connected_tensor_product(kind, n):
    base = gspaces.rot2dOnR2(n) if kind == "cyclic" else gspaces.flipRot2dOnR2(n)
    space = gspaces.no_base_space(base.fibergroup)
    irreps = list(space.fibergroup.irreps())
    in1 = nn.FieldType(space, irreps)
    in2 = nn.FieldType(space, list(reversed(irreps)))
    out = nn.FieldType(space, irreps)
    module = nn.FullyConnectedTensorProduct(in1, in2, out).double()
    x = nn.GeometricTensor(torch.randn(4, in1.size, dtype=torch.float64, requires_grad=True), in1)
    y = nn.GeometricTensor(torch.randn(4, in2.size, dtype=torch.float64, requires_grad=True), in2)
    _assert_equivariant(module, x, y)
    module(x, y).tensor.square().mean().backward()
    assert x.tensor.grad is not None and y.tensor.grad is not None
    assert all(parameter.grad is not None for parameter in module.parameters())


@pytest.mark.parametrize("regular_position", ["output", "left", "right", "both_inputs", "all"])
def test_analytic_regular_tensor_product_paths(regular_position):
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(4).fibergroup)
    vector = space.irrep(1, 1)
    regular = space.regular_repr
    left_rep = regular if regular_position in {"left", "both_inputs", "all"} else vector
    right_rep = regular if regular_position in {"right", "both_inputs", "all"} else vector
    output_rep = regular if regular_position in {"output", "all"} else vector
    in1, in2, out = (
        nn.FieldType(space, [left_rep]),
        nn.FieldType(space, [right_rep]),
        nn.FieldType(space, [output_rep]),
    )
    module = nn.TensorProduct(in1, in2, out).double()
    x = nn.GeometricTensor(torch.randn(3, in1.size, dtype=torch.float64), in1)
    y = nn.GeometricTensor(torch.randn(3, in2.size, dtype=torch.float64), in2)
    _assert_equivariant(module, x, y)


def test_external_shared_and_unshared_weights():
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(5).fibergroup)
    type_ = nn.FieldType(space, [space.irrep(1, 1)])
    out = nn.FieldType(space, [space.irrep(0, 0), space.irrep(1, 2)])
    x = nn.GeometricTensor(torch.randn(6, type_.size), type_)
    y = nn.GeometricTensor(torch.randn(6, type_.size), type_)
    shared = nn.TensorProduct(type_, type_, out, internal_weights=False)
    shared_weight = torch.randn(shared.weight_numel)
    result = shared(x, y, shared_weight)
    for element in space.fibergroup.elements:
        torch.testing.assert_close(
            shared(x.transform_fibers(element), y.transform_fibers(element), shared_weight).tensor,
            result.transform_fibers(element).tensor,
            atol=2e-5,
            rtol=2e-5,
        )
    unshared = nn.TensorProduct(type_, type_, out, internal_weights=False, shared_weights=False)
    unshared_weight = torch.randn(6, unshared.weight_numel)
    assert unshared(x, y, unshared_weight).tensor.shape == (6, out.size)


def test_instruction_subset_and_shape_errors():
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(6).fibergroup)
    scalar = nn.FieldType(space, [space.irrep(0, 0)])
    vector = nn.FieldType(space, [space.irrep(1, 1)])
    module = nn.TensorProduct(scalar, vector, vector, instructions=[(0, 0, 0)])
    assert len(module.instructions) == 1
    assert module.evaluate_output_shape((2, 3, 1), (2, 3, 2)) == (2, 3, 2)
    with pytest.raises(ValueError, match="matching"):
        module(
            nn.GeometricTensor(torch.randn(2, 1), scalar),
            nn.GeometricTensor(torch.randn(3, 2), vector),
        )


def test_full_tensor_product_product_basis_equivariance():
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(6).fibergroup)
    left = nn.FieldType(space, [space.irrep(1, 1), space.irrep(0, 0)])
    right = nn.FieldType(space, [space.irrep(1, 2)])
    module = nn.FullTensorProduct(left, right)
    x = nn.GeometricTensor(torch.randn(5, left.size), left)
    y = nn.GeometricTensor(torch.randn(5, right.size), right)
    _assert_equivariant(module, x, y, atol=2e-5)


def test_tensor_product_gradcheck_inputs_and_external_weights():
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(3).fibergroup)
    vector = nn.FieldType(space, [space.irrep(1, 1)])
    scalar = nn.FieldType(space, [space.irrep(0, 0)])
    module = nn.TensorProduct(vector, vector, scalar, internal_weights=False, shared_weights=False).double()
    left = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    right = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(2, module.weight_numel, dtype=torch.float64, requires_grad=True)

    def function(left_tensor, right_tensor, weight_tensor):
        return module(
            nn.GeometricTensor(left_tensor, vector),
            nn.GeometricTensor(right_tensor, vector),
            weight_tensor,
        ).tensor

    assert torch.autograd.gradcheck(function, (left, right, weights), atol=1e-6, rtol=1e-5)


def test_explicit_coupling_instruction_and_unweighted_path():
    from e3nn_WE.nn import TensorProductInstruction

    group = gspaces.rot2dOnR2(5).fibergroup
    vector = group.irrep(1)
    instruction = TensorProductInstruction(0, 0, 0, coupling=0, has_weight=False)
    product = nn.TensorProduct(
        vector,
        vector,
        group.trivial_irrep,
        instructions=[instruction],
    )
    left, right = torch.randn(6, 2), torch.randn(6, 2)
    assert product(left, right).shape == (6, 1)
    assert product.weight_numel == 0
    product.check_equivariance()
