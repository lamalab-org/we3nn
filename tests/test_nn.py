import pytest
import torch

from e3nn_WE import gspaces, nn


def _types(kind: str, n: int):
    base = gspaces.rot2dOnR2(n) if kind == "cyclic" else gspaces.flipRot2dOnR2(n)
    space = gspaces.no_base_space(base.fibergroup)
    vector = space.irrep(1) if kind == "cyclic" else space.irrep(1, 1)
    input_type = nn.FieldType(space, 3 * [space.trivial_repr] + 2 * [vector])
    regular_type = nn.FieldType(space, 3 * [space.regular_repr])
    return space, input_type, regular_type


def _assert_module_equivariant(module, input_type, group, *, atol=2e-5):
    reference = next(module.parameters(), None)
    if reference is None:
        reference = next(module.buffers())
    x = torch.randn(5, input_type.size, device=reference.device, dtype=reference.dtype)
    y = module(x)
    for element in group.elements:
        actual = module(input_type.transform_fibers(x, element))
        expected = module.out_type.transform_fibers(y, element)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=atol)


@pytest.mark.parametrize("kind,n", [("cyclic", 3), ("cyclic", 4), ("cyclic", 9), ("dihedral", 3), ("dihedral", 4), ("dihedral", 6)])
def test_linear_and_regular_nonlinearity_equivariance(kind, n):
    space, input_type, regular_type = _types(kind, n)
    model = nn.SequentialModule(
        nn.Linear(input_type, regular_type),
        nn.ReLU(regular_type),
        nn.Linear(regular_type, regular_type),
        nn.ELU(regular_type),
        nn.Linear(regular_type, input_type),
    )
    _assert_module_equivariant(model, input_type, space.fibergroup)
    assert model.evaluate_output_shape((11, input_type.size)) == (11, input_type.size)


def test_linear_gradients_reach_every_parameter():
    _, input_type, regular_type = _types("dihedral", 6)
    layer = nn.Linear(input_type, regular_type)
    x = torch.randn(7, input_type.size, requires_grad=True)
    layer(x).square().mean().backward()
    assert x.grad is not None
    assert all(parameter.grad is not None for parameter in layer.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in layer.parameters())


def test_bias_is_invariant_and_nontrivial_outputs_have_no_bias():
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(6).fibergroup)
    vector_type = nn.FieldType(space, [space.irrep(1, 1)])
    layer = nn.Linear(vector_type, vector_type, bias=True)
    assert sum(parameter.numel() for parameter in layer.bias_parameters) == 0
    _assert_module_equivariant(layer, vector_type, space.fibergroup)


def test_pointwise_rejects_vector_irrep():
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(6).fibergroup)
    vector_type = nn.FieldType(space, [space.irrep(1, 1)])
    with pytest.raises(ValueError, match="permutation"):
        nn.ReLU(vector_type)


def test_type_and_shape_errors_are_early():
    space, input_type, regular_type = _types("dihedral", 6)
    with pytest.raises(ValueError, match="last dimension"):
        nn.Linear(input_type, regular_type)(torch.randn(2, input_type.size + 1))


def test_large_group_regular_setup_uses_quadratic_permutation_indices():
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(128).fibergroup)
    type_ = nn.FieldType(space, [space.regular_repr])
    layer = nn.Linear(type_, type_)
    pair = layer._pairs[0]
    assert pair.basis.numel() == 0
    assert pair.relative.shape == (256, 256)
    assert pair.coefficients.numel() == 256
    assert sum(buffer.numel() * buffer.element_size() for buffer in layer.buffers()) < 600_000
    _assert_module_equivariant(layer, type_, space.fibergroup, atol=3e-5)


def test_double_rebuilds_intertwiner_basis_at_float64_precision():
    space = gspaces.no_base_space(gspaces.rot2dOnR2(3).fibergroup)
    input_type = nn.FieldType(space, [space.irrep(1), space.regular_repr, space.irrep(0)])
    output_type = nn.FieldType(space, [space.regular_repr, space.irrep(1)])
    layer = nn.Linear(input_type, output_type).double()
    x = torch.randn(4, input_type.size, dtype=torch.float64)
    y = layer(x)
    for element in space.fibergroup.elements:
        torch.testing.assert_close(
            layer(input_type.transform_fibers(x, element)),
            output_type.transform_fibers(y, element),
            atol=2e-12,
            rtol=2e-12,
        )


def test_valid_parameterless_zero_map_tracks_dtype_and_device():
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(6).fibergroup)
    input_type = nn.FieldType(space, [space.irrep(1, 1)])
    output_type = nn.FieldType(space, [space.irrep(1, 2)])
    layer = nn.Linear(input_type, output_type, bias=True).double()
    assert sum(parameter.numel() for parameter in layer.parameters()) == 0
    x = torch.randn(3, input_type.size, dtype=torch.float64)
    assert torch.equal(layer(x), torch.zeros(3, output_type.size, dtype=torch.float64))
    _assert_module_equivariant(layer, input_type, space.fibergroup, atol=1e-12)


def test_generic_and_structured_linear_backends_span_same_maps():
    space = gspaces.no_base_space(gspaces.flipRot2dOnR2(4).fibergroup)
    in_type = nn.FieldType(space, [space.irrep(0, 0), space.irrep(1, 1)])
    out_type = nn.FieldType(space, [space.irrep(1, 1), space.irrep(0, 0)])
    modules = [
        nn.Linear(in_type, out_type, bias=False, backend="structured").double(),
        nn.Linear(in_type, out_type, bias=False, backend="generic").double(),
    ]
    bases = []
    for module in modules:
        columns = []
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.zero_()
            for parameter in module.parameters():
                for index in range(parameter.numel()):
                    parameter.reshape(-1)[index] = 1
                    columns.append(module.expand_parameters()[0].flatten().clone())
                    parameter.reshape(-1)[index] = 0
        bases.append(torch.linalg.qr(torch.stack(columns, dim=1), mode="reduced").Q)
    torch.testing.assert_close(bases[0] @ bases[0].T, bases[1] @ bases[1].T, atol=1e-10, rtol=1e-10)


def test_no_grad_linear_cache_invalidates_after_parameter_update():
    _, input_type, regular_type = _types("dihedral", 6)
    layer = nn.Linear(input_type, regular_type)
    x = torch.randn(4, input_type.size)
    with torch.no_grad():
        before = layer(x)
        assert layer._inference_weight.numel() > 0
        next(layer.parameters()).add_(0.25)
        after = layer(x)
    assert not torch.allclose(before, after)
    with torch.enable_grad():
        torch.testing.assert_close(after, layer(x))
