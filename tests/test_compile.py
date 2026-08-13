import pytest
import torch

from e3nn_WE import gspaces, nn


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_linear_and_tensor_product_compile_smoke_outputs_and_gradients():
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(4).fibergroup)
    vector = nn.FieldType(space, [space.irrep(1, 1)])
    regular = nn.FieldType(space, [space.regular_repr])
    linear = nn.Linear(vector, regular)
    product = nn.TensorProduct(vector, vector, regular)
    compiled_linear = torch.compile(linear, backend="eager")
    compiled_product = torch.compile(product, backend="eager")
    left_tensor = torch.randn(3, 2, requires_grad=True)
    right_tensor = torch.randn(3, 2, requires_grad=True)
    left, right = left_tensor, right_tensor
    torch.testing.assert_close(compiled_linear(left), linear(left))
    torch.testing.assert_close(compiled_product(left, right), product(left, right))
    (compiled_linear(left).sum() + compiled_product(left, right).sum()).backward()
    assert left_tensor.grad is not None and right_tensor.grad is not None


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_wigner_eckart_compile_smoke_outputs_and_gradients():
    group = gspaces.flipRot2dOnR2(4).fibergroup
    vector, scalar = group.irrep(1, 1), group.irrep(0, 0)
    module = nn.WignerEckartTensorProduct(vector, vector, scalar).double()
    compiled = torch.compile(module, backend="eager")
    left = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    right = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(4, module.weight_numel, dtype=torch.float64, requires_grad=True)
    eager = module(left, right, weights)
    actual = compiled(left, right, weights)
    torch.testing.assert_close(actual, eager)
    actual.sum().backward()
    assert left.grad is not None and right.grad is not None and weights.grad is not None


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_d6_message_layer_compile_smoke_outputs_and_gradients():
    from mp_example import EquivariantMPLayer

    module = EquivariantMPLayer(3, 4, torch.nn.ReLU()).double()
    compiled = torch.compile(module, backend="eager")
    h = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    pos = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    anchor = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
    edges = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    eager = module(h, pos, anchor, edges)
    actual = compiled(h, pos, anchor, edges)
    torch.testing.assert_close(actual[0], eager[0])
    torch.testing.assert_close(actual[1], eager[1])
    (actual[0].sum() + actual[1].sum()).backward()
    assert h.grad is not None and pos.grad is not None and anchor.grad is not None
