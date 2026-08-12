import torch
from e3nn import o3

from e3nn_WE import gspaces, nn, planar_o3, restrict_o3


def _check(module, x, filters, weights, atol=3e-5):
    output = module(x, filters, weights)
    for element in x.type.fibergroup.elements:
        actual = module(x.transform_fibers(element), filters.transform_fibers(element), weights).tensor
        expected = output.transform_fibers(element).tensor
        torch.testing.assert_close(actual, expected, atol=atol, rtol=atol)


def test_native_finite_wigner_eckart_with_per_sample_reduced_weights():
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(6).fibergroup)
    node = nn.FieldType(space, [space.irrep(1, 1)])
    filter_type = nn.FieldType(space, [space.irrep(1, 2)])
    output = nn.FieldType(space, [space.irrep(0, 0), space.irrep(1, 1)])
    module = nn.WignerEckartTensorProduct(node, filter_type, output)
    x = nn.GeometricTensor(torch.randn(8, node.size), node)
    filters = nn.GeometricTensor(torch.randn(8, filter_type.size), filter_type)
    weights = torch.randn(8, module.weight_numel, requires_grad=True)
    _check(module, x, filters, weights)
    module(x, filters, weights).tensor.square().mean().backward()
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
    module = nn.RestrictedWignerEckartTensorProduct(node, filter_type, output)
    x = nn.GeometricTensor(torch.randn(3, node.size), node)
    filters = nn.GeometricTensor(torch.randn(3, filter_type.size), filter_type)
    weights = torch.randn(3, module.weight_numel)
    _check(module, x, filters, weights, atol=8e-5)
    # O(3) has at most one path for this irrep triple, while D6 restriction
    # generally exposes several reduced matrix elements.
    assert module.weight_numel > 1
