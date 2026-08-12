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
    left = nn.GeometricTensor(left_tensor, vector)
    right = nn.GeometricTensor(right_tensor, vector)
    torch.testing.assert_close(compiled_linear(left).tensor, linear(left).tensor)
    torch.testing.assert_close(compiled_product(left, right).tensor, product(left, right).tensor)
    (compiled_linear(left).tensor.sum() + compiled_product(left, right).tensor.sum()).backward()
    assert left_tensor.grad is not None and right_tensor.grad is not None
